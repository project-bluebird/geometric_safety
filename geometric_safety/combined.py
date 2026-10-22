"""Combined lateral and vertical safety checks."""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal

import numba
import numpy as np

from geometric_safety.heading import (
    HeadingAssuranceResult,
    HeadingMotion,
    _require_nonnegative_finite,
    _validate_heading_query,
    certify_heading_separation,
)
from geometric_safety.relevant_aircraft import (
    TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
    TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
    _geometry_within_numerical_domain,
    _lateral_parameters_within_numerical_range,
    _validate_deviation_envelope,
    catch_up_projection_interval,
    catch_up_projection_interval_with_turns,
)
from geometric_safety.util import _validate_heading_and_rate, latlon_to_local_xy
from geometric_safety.vertical import (
    DEFAULT_VERTICAL_RATE_FPM,
    DEFAULT_VERTICAL_ROUNDING_FL,
    DEFAULT_VERTICAL_SEPARATION_FL,
    VERTICAL_OVERLAP_NEVER_RESOLVES_S,
    _conservative_vertical_resolution_time_s,
    time_to_vertical_overlap_resolution,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CombinedHeadingAssuranceResult:
    """Combined acceptance over the requested horizon and its supporting checks.

    vertical_resolution_time_s is a time from which the declared altitude bands
    stay separated, or None when no usable resolution time was established.
    It can exceed the requested horizon. lateral_horizon_s is the interval end
    passed to the heading solver, capped at the requested horizon.

    lateral_result contains the heading solver's result over that interval.
    It is None only when vertical separation suffices from time zero, in which
    case lateral_horizon_s is zero. Its diagnostics and inflation describe only
    the lateral interval, not the entire combined encounter. unresolved does
    not establish a conflict; unsupported comes from a required lateral check.
    """

    status: Literal["certified_safe", "unresolved", "unsupported"]
    reason: str
    vertical_resolution_time_s: float | None
    lateral_horizon_s: float
    lateral_result: HeadingAssuranceResult | None


def certify_heading_separation_with_vertical(
    relative_position_m: tuple[float, float],
    a: HeadingMotion,
    b: HeadingMotion,
    *,
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    minimum_vertical_rate_fpm: float,
    separation_threshold_m: float,
    horizon_s: float,
    required_vertical_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    vertical_rounding_fl: int | np.integer = DEFAULT_VERTICAL_ROUNDING_FL,
    relative_position_error_m: float = 0.0,
    relative_velocity_error_mps: float = 0.0,
) -> CombinedHeadingAssuranceResult:
    """Certify lateral or vertical separation throughout [0, horizon_s].

    Each aircraft progresses monotonically from its current flight level to
    its selected level at no less than minimum_vertical_rate_fpm, with no
    overshoot, then stays there. Rates may differ and vary above that shared
    minimum. The lower bound applies immediately until the selected level is
    reached; a response delay or ramp below it is outside this contract. A zero
    minimum supplies no progress guarantee, but separated initial bands still
    suffice. Flight levels are exact inputs to this model, without an altitude
    error allowance. Lateral speed bounds must include climb/descent changes.

    Remaining altitude bands enclose all faster progress and shrink over time.
    The vertical calculation proposes a time at which they become separated.
    Exact verification of the unrounded gap can move this time later. The
    heading solver checks up to the verified time, capped at the requested
    horizon. If no usable vertical time is found, it checks the whole
    horizon. Already separated bands suffice without a lateral check, even
    when the declared horizontal law is unsupported by the heading solver.

    Positions use the same A-minus-B Cartesian east/north frame as
    certify_heading_separation. Invalid finite/nonnegative inputs raise
    ValueError even when vertical separation alone would suffice. Negative
    minimum rates are rejected; direction follows current and selected levels.
    vertical_rounding_fl is a nonnegative integer; zero disables rounding.
    The grid tolerance cannot reduce the requested vertical gap. Required
    lateral checks retain the heading solver's numerical limitations.
    """
    _validate_heading_query(
        relative_position_m,
        separation_threshold_m,
        horizon_s,
        relative_position_error_m,
        relative_velocity_error_mps,
    )
    for name, level in (
        ("a_current_fl", a_current_fl),
        ("a_selected_fl", a_selected_fl),
        ("b_current_fl", b_current_fl),
        ("b_selected_fl", b_selected_fl),
    ):
        if not math.isfinite(level):
            raise ValueError(f"{name} must be finite")
    _require_nonnegative_finite("minimum_vertical_rate_fpm", minimum_vertical_rate_fpm)
    _require_nonnegative_finite("required_vertical_gap_fl", required_vertical_gap_fl)
    if (
        isinstance(vertical_rounding_fl, bool)
        or not isinstance(vertical_rounding_fl, Integral)
        or vertical_rounding_fl < 0
    ):
        raise ValueError("vertical_rounding_fl must be a nonnegative integer")

    resolution_s = _conservative_vertical_resolution_time_s(
        a_current_fl,
        a_selected_fl,
        b_current_fl,
        b_selected_fl,
        minimum_vertical_rate_fpm,
        required_vertical_gap_fl,
        int(vertical_rounding_fl),
    )
    if resolution_s == 0.0:
        return CombinedHeadingAssuranceResult(
            status="certified_safe",
            reason="declared altitude bands are separated throughout the horizon",
            vertical_resolution_time_s=0.0,
            lateral_horizon_s=0.0,
            lateral_result=None,
        )
    lateral_horizon_s = float(
        _combined_projection_horizon_s(
            horizon_s, VERTICAL_OVERLAP_NEVER_RESOLVES_S if resolution_s is None else resolution_s
        )
    )
    lateral_result = certify_heading_separation(
        relative_position_m,
        a,
        b,
        separation_threshold_m=separation_threshold_m,
        horizon_s=lateral_horizon_s,
        relative_position_error_m=relative_position_error_m,
        relative_velocity_error_mps=relative_velocity_error_mps,
    )
    if lateral_result.status != "certified_safe":
        reason = f"required lateral check: {lateral_result.reason}"
    elif lateral_horizon_s < horizon_s:
        reason = "lateral separation certified until the altitude bands separate"
    else:
        reason = "lateral separation certified throughout the requested horizon"
    return CombinedHeadingAssuranceResult(
        status=lateral_result.status,
        reason=reason,
        vertical_resolution_time_s=resolution_s,
        lateral_horizon_s=lateral_horizon_s,
        lateral_result=lateral_result,
    )


@numba.njit(cache=True, fastmath=True)
def _combined_projection_horizon_s(projection_time_s: float, vertical_resolution_time_s: float) -> float:
    """Return the lateral horizon that must remain clear before vertical resolution."""
    # If vertical never resolves, the lateral solver has to certify the whole requested
    # projection window. Otherwise it only needs to certify up to vertical resolution.
    if vertical_resolution_time_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S:
        return projection_time_s
    if vertical_resolution_time_s < projection_time_s:
        return vertical_resolution_time_s
    return projection_time_s


@numba.njit(cache=True, fastmath=False)
def catch_up_projection_interval_with_vertical(
    a_lat: float,
    a_lon: float,
    a_heading: float,
    a_speed_kt: float,
    a_current_fl: float,
    a_selected_fl: float,
    b_lat: float,
    b_lon: float,
    b_heading: float,
    b_speed_kt: float,
    b_current_fl: float,
    b_selected_fl: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    vertical_rate_fpm: float = DEFAULT_VERTICAL_RATE_FPM,
    required_vertical_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    vertical_rounding_fl: int = DEFAULT_VERTICAL_ROUNDING_FL,
) -> tuple[bool, float, float, float]:
    """
    Evaluate straight-line lateral safety with an independent vertical overlap check.

    The vertical model estimates the first time at which the remaining cleared-to-selected
    level bands are separated by `required_vertical_gap_fl`. The lateral solver is then
    only required to certify the interval before that vertical resolution time. If the
    bands are already resolved, the function returns safe with lateral diagnostics at
    `t=0`. If the bands never resolve, lateral safety is checked over the full projection
    horizon and `vertical_resolution_time_s` is `VERTICAL_OVERLAP_NEVER_RESOLVES_S`.

    Parameters
    ----------
    a_lat, a_lon : float
        Aircraft A current position.
    a_heading : float
        Aircraft A heading in degrees.
    a_speed_kt : float
        Aircraft A nominal speed in knots.
    a_current_fl : float
        Aircraft A current flight level.
    a_selected_fl : float
        Aircraft A selected flight level.
    b_lat, b_lon : float
        Aircraft B current position.
    b_heading : float
        Aircraft B heading in degrees.
    b_speed_kt : float
        Aircraft B nominal speed in knots.
    b_current_fl : float
        Aircraft B current flight level.
    b_selected_fl : float
        Aircraft B selected flight level.
    separation_threshold_m : float
        Required lateral spacing in metres.
    speed_diff_kt : float
        Symmetric lateral speed uncertainty in knots.
    projection_time_s : float
        Total projection horizon in seconds.
    vertical_rate_fpm : float, optional
        Shared climb/descent rate magnitude in feet per minute, used for both
        aircraft. Negative values are treated as magnitudes.
    required_vertical_gap_fl : float, optional
        Required gap between the two remaining vertical bands, in flight levels.
    vertical_rounding_fl : int, optional
        Flight-level rounding step used before measuring the vertical gap. The lower
        compared level is rounded upward and the higher compared level downward.
        Values less than or equal to zero use unrounded floating-point band edges.

    Returns
    -------
    tuple[bool, float, float, float]
        `(is_safe, lateral_min_distance_m, lateral_closest_time_s,
        vertical_resolution_time_s)`.

        `vertical_resolution_time_s` estimates the first time at which the vertical bands are
        resolved, or `VERTICAL_OVERLAP_NEVER_RESOLVES_S` if they never resolve. If the
        vertical bands are already resolved, the lateral distance/time pair describes
        the current geometry at `t=0`.

    Notes
    -----
    The lateral horizon ends at the estimated vertical resolution time, or at the
    full projection horizon if the bands do not resolve. The timing calculation
    uses floating-point band edges and, with rounding enabled, a 1e-9 FL tolerance
    per edge. It does not independently verify the unrounded gap. Use
    `certify_heading_separation_with_vertical` for exact verification of that gap
    before shortening the horizon.

    Pass a conservative low vertical rate that both aircraft meet or exceed while
    progressing immediately towards their selected levels, without overshoot, and
    remaining there on arrival. Lateral speed bounds must hold during climb/descent;
    altitude-driven lateral speed changes are not modelled by this wrapper.
    """
    if not _lateral_parameters_within_numerical_range(
        a_speed_kt, b_speed_kt, speed_diff_kt, projection_time_s, separation_threshold_m
    ):
        return False, math.nan, math.nan, math.nan
    if not math.isfinite(a_heading) or not math.isfinite(b_heading):
        return False, math.nan, math.nan, math.nan
    vertical_resolution_time_s = time_to_vertical_overlap_resolution(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        required_gap_fl=required_vertical_gap_fl,
        rounded=vertical_rounding_fl,
    )

    # If the bands are already distinct by the required gap, the pair is vertically
    # safe from the start. Only the current positions are needed for diagnostics.
    if vertical_resolution_time_s == 0.0:
        position = latlon_to_local_xy(b_lat, b_lon, a_lat, a_lon)
        if not _geometry_within_numerical_domain(position, 0.0, separation_threshold_m):
            return False, math.nan, math.nan, vertical_resolution_time_s
        return True, math.hypot(position[0], position[1]), 0.0, vertical_resolution_time_s

    # Otherwise ask the existing straight-line lateral solver only about the period
    # before vertical resolution. If vertical never resolves, this becomes the full
    # projection horizon.
    lateral_horizon_s = _combined_projection_horizon_s(projection_time_s, vertical_resolution_time_s)
    lateral_is_separated, min_distance_m, closest_time_s = catch_up_projection_interval(
        a_lat=a_lat,
        a_lon=a_lon,
        a_heading=a_heading,
        a_speed_kt=a_speed_kt,
        b_lat=b_lat,
        b_lon=b_lon,
        b_heading=b_heading,
        b_speed_kt=b_speed_kt,
        separation_threshold_m=separation_threshold_m,
        speed_diff_kt=speed_diff_kt,
        projection_time_s=lateral_horizon_s,
    )
    return lateral_is_separated, min_distance_m, closest_time_s, vertical_resolution_time_s


@numba.njit(cache=True, fastmath=True)
def catch_up_projection_interval_with_turns_and_vertical(
    a_lat: float,
    a_lon: float,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    a_current_fl: float,
    a_selected_fl: float,
    b_lat: float,
    b_lon: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    b_current_fl: float,
    b_selected_fl: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    vertical_rate_fpm: float = DEFAULT_VERTICAL_RATE_FPM,
    required_vertical_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    vertical_rounding_fl: int = DEFAULT_VERTICAL_ROUNDING_FL,
    use_interval_local_lipschitz: bool = TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
    turn_speed_schedule_uncertainty_kt: float = TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
) -> tuple[bool, float, float, float]:
    """
    Evaluate turn-aware lateral safety with an independent vertical overlap check.

    The same vertical shortcut is used as in
    `catch_up_projection_interval_with_vertical`, but the lateral interval before
    vertical resolution is certified with the turn-aware lateral solver.

    Parameters
    ----------
    a_lat, a_lon : float
        Aircraft A current position.
    a_heading0_deg : float
        Aircraft A current heading (start of turn, or straight heading).
    a_target_heading_deg : float
        Aircraft A heading after turn (== heading0 if straight).
    a_speed_kt : float
        Aircraft A nominal speed in knots.
    a_turn_rate_deg_sec : float
        Aircraft A rate of turn in deg/s; 0.0 if straight.
    a_current_fl : float
        Aircraft A current flight level.
    a_selected_fl : float
        Aircraft A selected flight level.
    b_lat, b_lon : float
        Aircraft B current position.
    b_heading0_deg : float
        Aircraft B current heading.
    b_target_heading_deg : float
        Aircraft B heading after turn.
    b_speed_kt : float
        Aircraft B nominal speed in knots.
    b_turn_rate_deg_sec : float
        Aircraft B rate of turn in deg/s; 0.0 if straight.
    b_current_fl : float
        Aircraft B current flight level.
    b_selected_fl : float
        Aircraft B selected flight level.
    separation_threshold_m : float
        Required lateral spacing in metres.
    speed_diff_kt : float
        Symmetric lateral speed uncertainty in knots.
    projection_time_s : float
        Total projection horizon in seconds.
    vertical_rate_fpm : float, optional
        Shared climb/descent rate magnitude in feet per minute, used for both
        aircraft. Negative values are treated as magnitudes.
    required_vertical_gap_fl : float, optional
        Required gap between the two remaining vertical bands, in flight levels.
    vertical_rounding_fl : int, optional
        Flight-level rounding step used before measuring the vertical gap. The lower
        compared level is rounded upward and the higher compared level downward.
        Values less than or equal to zero use unrounded floating-point band edges.
    use_interval_local_lipschitz : bool, optional
        If True, use the tighter interval-local Lipschitz bound in the turn-aware
        lateral solver. If False, use its simpler global bound.
    turn_speed_schedule_uncertainty_kt : float, optional
        Optional robustness assumption passed through to the turn-aware lateral solver.

    Returns
    -------
    tuple[bool, float, float, float]
        `(is_safe, lateral_min_distance_m, lateral_closest_time_s,
        vertical_resolution_time_s)`.

        `vertical_resolution_time_s` estimates the first time at which the vertical bands are
        resolved, or `VERTICAL_OVERLAP_NEVER_RESOLVES_S` if they never resolve. The
        lateral distance/time pair is reported from the shortened turn-aware lateral
        horizon, or from `t=0` when the vertical bands are already resolved.

    Notes
    -----
    The vertical timing estimate has the rounding tolerance and floating-point
    limitations described in `catch_up_projection_interval_with_vertical`. It does
    not independently verify the unrounded gap. Use
    `certify_heading_separation_with_vertical` for that verification.

    The turn-aware solver checks the resulting lateral horizon. Both aircraft must
    meet or exceed the minimum vertical rate while progressing immediately towards
    their selected levels, without overshoot, and remain there on arrival. Lateral
    speed bounds must hold throughout climb/descent.
    """
    _validate_heading_and_rate(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec)
    _validate_heading_and_rate(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec)
    _validate_deviation_envelope(0.0, 0.0, turn_speed_schedule_uncertainty_kt, 0.0, 0.0)
    if not _lateral_parameters_within_numerical_range(
        a_speed_kt, b_speed_kt, speed_diff_kt, projection_time_s, separation_threshold_m
    ):
        return False, math.nan, math.nan, math.nan
    vertical_resolution_time_s = time_to_vertical_overlap_resolution(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        required_gap_fl=required_vertical_gap_fl,
        rounded=vertical_rounding_fl,
    )

    # A changed target with zero turn rate is allowed when no lateral check is needed.
    if vertical_resolution_time_s == 0.0:
        position = latlon_to_local_xy(b_lat, b_lon, a_lat, a_lon)
        if not _geometry_within_numerical_domain(position, 0.0, separation_threshold_m):
            return False, math.nan, math.nan, vertical_resolution_time_s
        return True, math.hypot(position[0], position[1]), 0.0, vertical_resolution_time_s

    # Certify turn-aware lateral separation only until vertical resolution. The turn
    # solver remains conservative in time, so this preserves its existing safety meaning.
    lateral_horizon_s = _combined_projection_horizon_s(projection_time_s, vertical_resolution_time_s)
    lateral_is_separated, min_distance_m, closest_time_s = catch_up_projection_interval_with_turns(
        a_lat=a_lat,
        a_lon=a_lon,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_speed_kt=a_speed_kt,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        b_lat=b_lat,
        b_lon=b_lon,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_speed_kt=b_speed_kt,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        separation_threshold_m=separation_threshold_m,
        speed_diff_kt=speed_diff_kt,
        projection_time_s=lateral_horizon_s,
        use_interval_local_lipschitz=use_interval_local_lipschitz,
        turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
    )
    return lateral_is_separated, min_distance_m, closest_time_s, vertical_resolution_time_s
