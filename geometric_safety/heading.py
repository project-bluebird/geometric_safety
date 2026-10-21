"""Assurance for a prescribed heading law with speed varying over the whole horizon.

The nominal law is one immediate, fixed-rate shortest turn, followed by its target
heading. Speeds may be any measurable profiles inside the declared intervals.
Coordinates are Cartesian east/north metres, headings are clockwise from north.

At each time, replace each speed profile by its mean over [0, t]. The resulting
constant-speed position belongs to the existing fixed-time reachable hull. With
rho = U - L, angular rate magnitude omega in radians/s, and elapsed shortest-turn
sweep theta in radians, its position error is at most
4 * rho / omega * (1 - cos(theta / 2)). Indeed, subtracting the final
direction makes the integrand vanish after the turn, and integrating the direction
chord bound gives this expression. The mean can change with t; inclusion is
pointwise. This module freezes each envelope at its horizon maximum and adds
explicit position, velocity and heading-deviation bounds by triangle inequality.

The executed relative velocity must admit the decomposition
s_A(t) u(psi_A(t) + delta_A(t)) - s_B(t) u(psi_B(t) + delta_B(t)) + e(t).
Here psi is the prescribed law, each s is in its declared interval, each
circular direction deviation delta is bounded by heading_deviation_max_deg,
and ||e(t)|| is bounded by relative_velocity_error_mps. The initial relative
position error is bounded by relative_position_error_m. Bounds hold pointwise.

This is an exact-real enclosure argument, using the existing numerical kernel.
Its empirical floating-point guard is not a proved error bound for all inputs.
A certified_safe result inherits that numerical limitation and assumes the
caller's motion declarations and deviation bounds hold throughout the horizon.
"""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from geometric_safety.relevant_aircraft import (
    DIAGNOSTIC_INTERVAL_BOUND,
    DIAGNOSTIC_SAMPLED_MINIMUM,
    DIAGNOSTIC_STRAIGHT_MINIMUM,
    DIAGNOSTIC_UNAVAILABLE,
    NUMERICAL_SCALAR_LIMIT,
    TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
    TURN_TIME_CERT_MIN_INTERVAL_S,
    TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
    _catch_up_projection_interval_with_turns_local,
    _geometry_within_numerical_domain,
    _speed_rectangle_corner_points,
    _straight_hull_projection_interval,
)
from geometric_safety.util import (
    KT_TO_MPS,
    heading_diff,
    heading_to_unit_vector,
    shortest_turn_parameters,
    small_convex_hull,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HeadingMotion:
    """Declare one aircraft's horizontal motion, in degrees, seconds and m/s.

    speed_min_mps <= s(t) <= speed_max_mps must hold for the complete horizon,
    including any climb/descent and the maintained-heading continuation. No
    acceleration law is required. turn_rate_deg_s is a nonnegative magnitude;
    the shortest signed heading difference selects direction, with -180 at a tie.

    heading_deviation_max_deg bounds the circular direction difference of each
    speed-bounded velocity component from the prescribed law at every instant, not a
    target-heading or turn-rate tolerance. A separate relative_velocity_error_mps
    can bound additive relative velocity disturbances, including residual wind.
    These declarations do not establish that an arbitrary wind/heading model
    conforms. Directed turns, delays and speed-dependent laws are unsupported.
    """

    heading0_deg: float
    target_heading_deg: float
    turn_rate_deg_s: float
    speed_min_mps: float
    speed_max_mps: float
    turn_direction: Literal["shortest", "left", "right"] = "shortest"
    heading_law: Literal["fixed_rate", "speed_dependent", "route_turn"] = "fixed_rate"
    turn_delay_s: float = 0.0
    heading_deviation_max_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("heading0_deg", "target_heading_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "turn_rate_deg_s",
            "speed_min_mps",
            "speed_max_mps",
            "turn_delay_s",
            "heading_deviation_max_deg",
        ):
            _require_nonnegative_finite(name, getattr(self, name))
        if self.speed_max_mps < self.speed_min_mps:
            raise ValueError("speed_max_mps must be at least speed_min_mps")
        if self.heading_deviation_max_deg > 180.0:
            raise ValueError("heading_deviation_max_deg must be at most 180")


@dataclass(frozen=True, slots=True)
class HeadingAssuranceResult:
    """An acceptance decision and surrogate constant-speed hull diagnostics.

    unresolved does not assert a conflict. unsupported means no certificate was
    attempted for the declared motion class; its inflation_m is zero.

    diagnostic_kind identifies how base_min_distance_m was calculated. A
    straight_minimum comes from the straight hull projection. A sampled_minimum
    is the smallest sampled turn-hull distance, not a continuous-time minimum.
    An interval_bound is a lower bound over base_bound_interval_s and has no
    base_closest_time_s. Otherwise base_closest_time_s is the model time for the
    straight minimum or the sample time. These diagnostics describe the
    surrogate constant-speed hull, not an executed profile or a minimum after
    subtracting the envelope. Queries with no diagnostics have kind None.
    """

    status: Literal["certified_safe", "unresolved", "unsupported"]
    reason: str
    inflation_m: float
    base_min_distance_m: float | None = None
    base_closest_time_s: float | None = None
    diagnostic_kind: Literal["straight_minimum", "sampled_minimum", "interval_bound"] | None = None
    base_bound_interval_s: tuple[float, float] | None = None


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_heading_query(
    relative_position_m: tuple[float, float],
    separation_threshold_m: float,
    horizon_s: float,
    relative_position_error_m: float,
    relative_velocity_error_mps: float,
) -> None:
    """Validate query values before either a lateral or vertical acceptance."""
    if len(relative_position_m) != 2 or not all(math.isfinite(value) for value in relative_position_m):
        raise ValueError("relative_position_m must contain two finite coordinates")
    for name, value in (
        ("separation_threshold_m", separation_threshold_m),
        ("horizon_s", horizon_s),
        ("relative_position_error_m", relative_position_error_m),
        ("relative_velocity_error_mps", relative_velocity_error_mps),
    ):
        _require_nonnegative_finite(name, value)


def _turn_angle_deg(motion: HeadingMotion) -> float:
    # Passing abs_diff explicitly avoids repeated Numba dispatch after JIT warm-up.
    # heading_diff preserves the exact input-difference side at wrap boundaries.
    return float(heading_diff(motion.heading0_deg, motion.target_heading_deg, False))


def _unsupported_reason(motion: HeadingMotion) -> str | None:
    if motion.turn_direction != "shortest":
        return "directed turns are unsupported; declare a shortest turn"
    if motion.heading_law != "fixed_rate":
        return "only a prescribed fixed-rate heading law is supported"
    if motion.turn_delay_s != 0.0:
        return "heading-response delay is unsupported"
    if _turn_angle_deg(motion) != 0.0 and motion.turn_rate_deg_s == 0.0:
        return "a changed heading with zero turn rate is unsupported"
    return None


def _mean_speed_envelope_m(motion: HeadingMotion, horizon_s: float) -> float:
    """Horizon maximum of the mean-speed position envelope, stable near zero."""
    angle_deg, signed_rate_deg_s, turn_duration_s = shortest_turn_parameters(
        motion.heading0_deg, motion.target_heading_deg, motion.turn_rate_deg_s
    )
    rho = motion.speed_max_mps - motion.speed_min_mps
    if angle_deg == 0.0 or rho == 0.0 or horizon_s == 0.0:
        return 0.0
    active_time = min(horizon_s, turn_duration_s)
    theta = abs(signed_rate_deg_s) * active_time * (math.pi / 180.0)
    quarter_theta = 0.25 * theta
    sinc = math.sin(quarter_theta) / quarter_theta if quarter_theta != 0.0 else 1.0
    # Equivalent to 8*rho/omega*sin(theta/4)^2, without dividing by tiny omega.
    return (rho * active_time) * (0.5 * theta * sinc * sinc)


def _inflation_m(
    a: HeadingMotion,
    b: HeadingMotion,
    horizon_s: float,
    relative_position_error_m: float,
    relative_velocity_error_mps: float,
) -> float:
    terms = [
        relative_position_error_m,
        relative_velocity_error_mps * horizon_s,
        _mean_speed_envelope_m(a, horizon_s),
        _mean_speed_envelope_m(b, horizon_s),
    ]
    for motion in (a, b):
        chord = 2.0 * math.sin(math.radians(motion.heading_deviation_max_deg) * 0.5)
        terms.append((motion.speed_max_mps * horizon_s) * chord)
    return sum(terms)


def _enclosing_common_speed_bounds_kt(a: HeadingMotion, b: HeadingMotion) -> tuple[float, float, float] | None:
    """Enclose both independent m/s intervals in the kernel's shared half-width.

    Repeated outward rounding is checked against the exact floating operations
    used by the kernel to reconstruct the m/s endpoints. The narrower interval
    may be enlarged substantially; this affects acceptance, not coverage.
    """
    motions = (a, b)
    centres = tuple((motion.speed_min_mps * 0.5 + motion.speed_max_mps * 0.5) / KT_TO_MPS for motion in motions)
    lower_kt = tuple(
        math.nextafter(motion.speed_min_mps / KT_TO_MPS, -math.inf) if motion.speed_min_mps > 0.0 else 0.0
        for motion in motions
    )
    upper_kt = tuple(
        math.nextafter(motion.speed_max_mps / KT_TO_MPS, math.inf) if motion.speed_max_mps > 0.0 else 0.0
        for motion in motions
    )
    half_width = max(max(centre - lower, upper - centre) for centre, lower, upper in zip(centres, lower_kt, upper_kt))
    for _ in range(64):
        if all(
            max(centre - half_width, 0.0) * KT_TO_MPS <= motion.speed_min_mps
            and (centre + half_width) * KT_TO_MPS >= motion.speed_max_mps
            for centre, motion in zip(centres, motions)
        ):
            return centres[0], centres[1], half_width
        half_width = math.nextafter(half_width, math.inf)
    return None


def _straight_query(
    relative_position: NDArray[np.float64],
    a: HeadingMotion,
    b: HeadingMotion,
    threshold_m: float,
    horizon_s: float,
) -> tuple[bool, float, float]:
    # Unlike the turn kernel, this path retains independent, asymmetric intervals.
    a_basis = heading_to_unit_vector(math.fmod(a.target_heading_deg, 360.0)) * horizon_s
    b_basis = heading_to_unit_vector(math.fmod(b.target_heading_deg, 360.0)) * horizon_s
    corners = _speed_rectangle_corner_points(
        np.zeros(2),
        a_basis,
        a.speed_min_mps,
        a.speed_max_mps,
        b_basis,
        b.speed_min_mps,
        b.speed_max_mps,
    )
    hull = small_convex_hull(np.vstack((corners, np.zeros((1, 2)))))
    return _straight_hull_projection_interval(relative_position, hull, threshold_m, horizon_s)


def certify_heading_separation(
    relative_position_m: tuple[float, float],
    a: HeadingMotion,
    b: HeadingMotion,
    *,
    separation_threshold_m: float,
    horizon_s: float,
    relative_position_error_m: float = 0.0,
    relative_velocity_error_mps: float = 0.0,
) -> HeadingAssuranceResult:
    """Certify separation over [0, horizon_s] for the declared horizontal class.

    relative_position_m is A minus B in one Cartesian east/north frame. The
    separation threshold and initial relative-position error are metres; the
    additive relative-velocity error is m/s. Common additive drift cancels in
    relative motion; otherwise the caller must supply a valid disturbance bound.

    No vertical-resolution shortcut is applied. Positive numerical acceptance
    means the reused kernel certifies its enlarged constant-speed hull against
    the threshold plus the maximum deviation envelope. Straight motion retains
    exact-real bounded-speed geometry. Turning uses a conservative enclosure and
    the kernel's continuous-time sufficient test. A failed test or geometry
    outside the kernel's numerical domain is unresolved. The domain ensures
    resolution of the empirical guard, not a proved floating-point error bound.
    """
    _validate_heading_query(
        relative_position_m,
        separation_threshold_m,
        horizon_s,
        relative_position_error_m,
        relative_velocity_error_mps,
    )
    for label, motion in (("a", a), ("b", b)):
        reason = _unsupported_reason(motion)
        if reason is not None:
            return HeadingAssuranceResult("unsupported", f"{label}: {reason}", 0.0)
        if _turn_angle_deg(motion) != 0.0 and motion.turn_rate_deg_s * (math.pi / 180.0) == 0.0:
            return HeadingAssuranceResult("unresolved", f"{label}: angular rate is below numerical range", 0.0)

    inflation = _inflation_m(a, b, horizon_s, relative_position_error_m, relative_velocity_error_mps)
    threshold = separation_threshold_m + inflation
    scales = (
        a.speed_max_mps,
        b.speed_max_mps,
        horizon_s,
    )
    relative_position = np.asarray(relative_position_m, dtype=np.float64)
    travel_m = (a.speed_max_mps + b.speed_max_mps) * horizon_s
    if not all(value <= NUMERICAL_SCALAR_LIMIT for value in scales) or not _geometry_within_numerical_domain(
        relative_position, travel_m, threshold
    ):
        return HeadingAssuranceResult("unresolved", "geometry or envelope exceeds numerical range", inflation)

    diagnostic_kind: Literal["straight_minimum", "sampled_minimum", "interval_bound"] = "straight_minimum"
    bound_interval: tuple[float, float] | None = None
    base_closest_time: float | None = None
    stationary_geometry = horizon_s == 0.0 or a.speed_max_mps == b.speed_max_mps == 0.0
    if stationary_geometry or (_turn_angle_deg(a) == 0.0 and _turn_angle_deg(b) == 0.0):
        safe, distance, closest_time = _straight_query(relative_position, a, b, threshold, horizon_s)
        base_closest_time = float(closest_time)
    else:
        speed_bounds = _enclosing_common_speed_bounds_kt(a, b)
        if speed_bounds is None:
            return HeadingAssuranceResult(
                "unresolved", "speed conversion could not enclose the declared bounds", inflation
            )
        a_speed_kt, b_speed_kt, half_width_kt = speed_bounds
        a_max_speed_mps = (a_speed_kt + half_width_kt) * KT_TO_MPS
        b_max_speed_mps = (b_speed_kt + half_width_kt) * KT_TO_MPS
        if (
            a_max_speed_mps > NUMERICAL_SCALAR_LIMIT
            or b_max_speed_mps > NUMERICAL_SCALAR_LIMIT
            or not _geometry_within_numerical_domain(
                relative_position, (a_max_speed_mps + b_max_speed_mps) * horizon_s, threshold
            )
        ):
            return HeadingAssuranceResult("unresolved", "enlarged speed hull exceeds numerical range", inflation)
        diagnostic_out = np.array((DIAGNOSTIC_UNAVAILABLE, math.nan, math.nan), dtype=np.float64)
        # Explicit defaults keep Numba on its normal dispatch path for repeated queries.
        safe, distance, closest_time = _catch_up_projection_interval_with_turns_local(
            rel_pos0=relative_position,
            a_heading0_deg=math.fmod(a.heading0_deg, 360.0),
            a_target_heading_deg=math.fmod(a.target_heading_deg, 360.0),
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=a.turn_rate_deg_s,
            b_heading0_deg=math.fmod(b.heading0_deg, 360.0),
            b_target_heading_deg=math.fmod(b.target_heading_deg, 360.0),
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=b.turn_rate_deg_s,
            separation_threshold_m=threshold,
            speed_diff_kt=half_width_kt,
            projection_time_s=horizon_s,
            use_interval_local_lipschitz=TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
            turn_speed_schedule_uncertainty_kt=TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
            min_cert_interval_s=TURN_TIME_CERT_MIN_INTERVAL_S,
            eta_const_m=0.0,
            eta_rate_mps=0.0,
            turn_delay_max_s=0.0,
            turn_rate_error_max=0.0,
            directional_mode=2,
            diagnostic_out=diagnostic_out,
        )
        if diagnostic_out[0] == DIAGNOSTIC_INTERVAL_BOUND:
            diagnostic_kind = "interval_bound"
            bound_interval = (float(diagnostic_out[1]), float(diagnostic_out[2]))
        elif diagnostic_out[0] == DIAGNOSTIC_SAMPLED_MINIMUM:
            diagnostic_kind = "sampled_minimum"
            base_closest_time = float(closest_time)
        elif diagnostic_out[0] == DIAGNOSTIC_STRAIGHT_MINIMUM:
            base_closest_time = float(closest_time)
        else:
            return HeadingAssuranceResult("unresolved", "base kernel returned unavailable diagnostics", inflation)
    if not math.isfinite(distance) or (bound_interval is None and not math.isfinite(closest_time)):
        return HeadingAssuranceResult("unresolved", "base kernel returned nonfinite diagnostics", inflation)
    if bound_interval is not None and (
        not all(math.isfinite(value) for value in bound_interval)
        or not 0.0 <= bound_interval[0] <= bound_interval[1] <= horizon_s
    ):
        return HeadingAssuranceResult("unresolved", "base kernel returned invalid diagnostic interval", inflation)
    status: Literal["certified_safe", "unresolved"] = "certified_safe" if safe else "unresolved"
    reason = "declared motion enclosure certified" if safe else "base hull and envelope did not certify separation"
    return HeadingAssuranceResult(
        status,
        reason,
        inflation,
        float(distance),
        base_closest_time,
        diagnostic_kind=diagnostic_kind,
        base_bound_interval_s=bound_interval,
    )
