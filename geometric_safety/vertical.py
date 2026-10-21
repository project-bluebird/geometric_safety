"""Vertical cleared-band timing for aircraft level changes."""

from fractions import Fraction
import math

import numba

# Levels are represented directly as flight levels throughout this module. The only
# physical unit conversion needed is the user-facing climb/descent rate in feet/minute.
FT_PER_FL = 100.0
SECONDS_PER_MINUTE = 60.0
DEFAULT_VERTICAL_RATE_FPM = 1000.0
DEFAULT_VERTICAL_SEPARATION_FL = 10.0
DEFAULT_VERTICAL_ROUNDING_FL = 10
# Grid comparisons snap edges within this tolerance towards the boundary.
# That can enlarge a reported gap by up to twice this value in exact arithmetic;
# it is not a bound on errors in the floating-point motion calculation.
_ROUNDING_GRID_EPSILON_FL = 1.0e-9

# A real resolution time is always non-negative, so a negative sentinel can represent
# "never resolves" without relying on infinity semantics under Numba fast-math.
VERTICAL_OVERLAP_NEVER_RESOLVES_S = -1.0


@numba.njit(cache=True, fastmath={"contract"})
def _validate_vertical_motion(current_fl: float, selected_fl: float, vertical_rate_fpm: float, t_s: float) -> None:
    """Reject missing motion declarations before level comparisons or grid conversion."""
    if not math.isfinite(current_fl) or not math.isfinite(selected_fl) or not math.isfinite(vertical_rate_fpm):
        raise ValueError("Flight levels and vertical rate must be finite")
    if not math.isfinite(t_s):
        raise ValueError("Elapsed time must be finite")


@numba.njit(cache=True, fastmath={"contract"})
def _validate_vertical_options(required_gap_fl: float, rounded: int) -> None:
    """Check the gap and grid without changing legacy nonpositive rounding semantics."""
    if not math.isfinite(required_gap_fl):
        raise ValueError("Required vertical gap must be finite")
    if not math.isfinite(rounded):
        raise ValueError("Vertical rounding must be finite")


@numba.njit(cache=True, fastmath=False)
def _rate_fl_per_second(vertical_rate_fpm: float) -> float:
    """Convert a climb/descent rate magnitude in feet per minute to flight levels per second."""
    rate_fpm = abs(vertical_rate_fpm)
    if rate_fpm == 0.0:
        return 0.0
    return rate_fpm / FT_PER_FL / SECONDS_PER_MINUTE


@numba.njit(cache=True, fastmath=False)
def level_at_time_fl(current_fl: float, selected_fl: float, vertical_rate_fpm: float, t_s: float) -> float:
    """
    Return the aircraft level after flying toward its selected level for `t_s`.

    Levels are expressed as flight levels, so FL350 is passed as `350.0`. The vertical
    rate is expressed in feet per minute and is applied with the sign required to move
    from the current level to the selected level. Once selected level is reached, the
    aircraft remains there.

    Parameters
    ----------
    current_fl : float
        Aircraft current flight level.
    selected_fl : float
        Aircraft selected flight level.
    vertical_rate_fpm : float
        Climb/descent rate magnitude in feet per minute. Negative values are
        treated as magnitudes; direction comes from current and selected levels.
    t_s : float
        Elapsed time in seconds.

    Returns
    -------
    float
        Aircraft flight level at `t_s`.
    """
    _validate_vertical_motion(current_fl, selected_fl, vertical_rate_fpm, t_s)
    # The vertical model is deliberately simple: fly at constant vertical rate toward
    # selected level, then clamp there once the selected level has been reached.
    rate_flps = _rate_fl_per_second(vertical_rate_fpm)
    if rate_flps <= 0.0 or t_s <= 0.0 or current_fl == selected_fl:
        return current_fl

    delta_fl = selected_fl - current_fl
    if t_s >= _time_to_selected_s(current_fl, selected_fl, vertical_rate_fpm):
        return selected_fl
    max_step_fl = rate_flps * t_s
    if abs(delta_fl) <= max_step_fl:
        return selected_fl
    if delta_fl > 0.0:
        return current_fl + max_step_fl
    return current_fl - max_step_fl


@numba.njit(cache=True, fastmath=True)
def _vertical_band_at_time_fl(
    current_fl: float, selected_fl: float, vertical_rate_fpm: float, t_s: float
) -> tuple[float, float]:
    """Return the remaining cleared-to-selected level band at time `t_s`."""
    # At time t, the aircraft still occupies the controller-relevant interval from its
    # instantaneous level to its selected level. The order depends on climb vs descent.
    level_fl = level_at_time_fl(current_fl, selected_fl, vertical_rate_fpm, t_s)
    if level_fl <= selected_fl:
        return level_fl, selected_fl
    return selected_fl, level_fl


@numba.njit(cache=True, fastmath=True)
def _directional_compared_levels_fl(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float,
    t_s: float,
    b_above_a: bool,
) -> tuple[float, float]:
    """Return compared lower and higher band edges for one vertical ordering."""
    # Pick the upper edge of the lower band and the lower edge of the higher band for
    # the tested ordering. Rounding, if needed, is applied by the caller.
    a_low_fl, a_high_fl = _vertical_band_at_time_fl(a_current_fl, a_selected_fl, vertical_rate_fpm, t_s)
    b_low_fl, b_high_fl = _vertical_band_at_time_fl(b_current_fl, b_selected_fl, vertical_rate_fpm, t_s)
    if b_above_a:
        return a_high_fl, b_low_fl
    return b_high_fl, a_low_fl


@numba.njit(cache=True, fastmath=True)
def _rounded_lower_index(level_fl: float, rounded: int) -> int:
    """Round the lower compared edge upward, with the stated grid tolerance."""
    return math.ceil((level_fl - _ROUNDING_GRID_EPSILON_FL) / rounded)


@numba.njit(cache=True, fastmath=True)
def _rounded_higher_index(level_fl: float, rounded: int) -> int:
    """Round the higher compared edge downward, with the stated grid tolerance."""
    return math.floor((level_fl + _ROUNDING_GRID_EPSILON_FL) / rounded)


@numba.njit(cache=True, fastmath=True)
def _directional_band_gap_fl(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float,
    t_s: float,
    b_above_a: bool,
    rounded: int,
) -> float:
    """Return one rounded directional gap between the remaining vertical bands."""
    lower_level_fl, higher_level_fl = _directional_compared_levels_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=t_s,
        b_above_a=b_above_a,
    )

    # Close the gap to the grid, allowing each edge the documented 1e-9 FL
    # boundary tolerance. The guarded certificate separately checks the raw gap.
    if rounded > 0:
        lower_level_fl = _rounded_lower_index(lower_level_fl, rounded) * rounded
        higher_level_fl = _rounded_higher_index(higher_level_fl, rounded) * rounded
    return higher_level_fl - lower_level_fl


@numba.njit(cache=True, fastmath=True)
def vertical_band_gap_fl(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float = DEFAULT_VERTICAL_RATE_FPM,
    t_s: float = 0.0,
    rounded: int = DEFAULT_VERTICAL_ROUNDING_FL,
) -> float:
    """
    Return the signed gap between the two remaining vertical bands.

    A non-negative value means the bands no longer overlap. A value of `10.0`, for
    example, means the bands are separated by 10 flight levels, i.e. 1000 ft. When the
    bands overlap, the returned value is negative by the amount of overlap. Compared
    band edges are rounded toward each other to multiples of `rounded` before the gap is
    measured; pass `rounded <= 0` to use unrounded floating-point band edges.
    With rounding enabled, a 1e-9 FL tolerance per edge can snap an edge to a
    nearby grid boundary and enlarge the reported gap by up to 2e-9 FL, apart
    from arithmetic error. This helper does not independently bound that error.

    Parameters
    ----------
    a_current_fl : float
        Aircraft A current flight level.
    a_selected_fl : float
        Aircraft A selected flight level.
    b_current_fl : float
        Aircraft B current flight level.
    b_selected_fl : float
        Aircraft B selected flight level.
    vertical_rate_fpm : float, optional
        Climb/descent rate magnitude in feet per minute, used for both
        aircraft. Negative values are treated as magnitudes.
    t_s : float, optional
        Elapsed time in seconds.
    rounded : int, optional
        Flight-level rounding step used before measuring the gap. The lower compared
        level is rounded upward and the higher compared level is rounded downward.
        Values less than or equal to zero use unrounded floating-point band edges.

    Returns
    -------
    float
        Signed gap between the remaining vertical bands, in flight levels.
    """
    _validate_vertical_options(0.0, rounded)
    # Either B can be above A, or A can be above B. The signed gap is whichever of those
    # two directional gaps is larger; negative values mean both directions still overlap.
    b_above_a_gap_fl = _directional_band_gap_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=t_s,
        b_above_a=True,
        rounded=rounded,
    )
    a_above_b_gap_fl = _directional_band_gap_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=t_s,
        b_above_a=False,
        rounded=rounded,
    )
    return max(b_above_a_gap_fl, a_above_b_gap_fl)


@numba.njit(cache=True, fastmath=True)
def vertical_bands_are_resolved(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float = DEFAULT_VERTICAL_RATE_FPM,
    required_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    t_s: float = 0.0,
    rounded: int = DEFAULT_VERTICAL_ROUNDING_FL,
) -> bool:
    """
    Return whether the computed band gap reaches `required_gap_fl`.

    This comparison inherits vertical_band_gap_fl's 1e-9 FL tolerance per
    rounded edge and floating-point arithmetic. It is not an independent
    numerical verification of the underlying unrounded separation.

    Parameters
    ----------
    a_current_fl : float
        Aircraft A current flight level.
    a_selected_fl : float
        Aircraft A selected flight level.
    b_current_fl : float
        Aircraft B current flight level.
    b_selected_fl : float
        Aircraft B selected flight level.
    vertical_rate_fpm : float, optional
        Climb/descent rate magnitude in feet per minute, used for both
        aircraft. Negative values are treated as magnitudes.
    required_gap_fl : float, optional
        Required gap between the two remaining vertical bands, in flight levels.
    t_s : float, optional
        Elapsed time in seconds.
    rounded : int, optional
        Flight-level rounding step used before measuring the gap. The lower compared
        level is rounded upward and the higher compared level is rounded downward.
        Values less than or equal to zero use unrounded floating-point band edges.

    Returns
    -------
    bool
        True if the computed gap, including the grid tolerance, reaches
        `required_gap_fl`.
    """
    _validate_vertical_options(required_gap_fl, rounded)
    if required_gap_fl < 0.0:
        required_gap_fl = 0.0
    gap_fl = vertical_band_gap_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=t_s,
        rounded=rounded,
    )
    return gap_fl >= required_gap_fl


@numba.njit(cache=True, fastmath=False)
def _time_to_selected_s(current_fl: float, selected_fl: float, vertical_rate_fpm: float) -> float:
    """Return how long the aircraft takes to reach selected level."""
    # A zero rate is only valid if there is no level change to make. Otherwise the
    # aircraft never reaches selected level under this model.
    rate_flps = _rate_fl_per_second(vertical_rate_fpm)
    if rate_flps <= 0.0:
        if current_fl == selected_fl:
            return 0.0
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S
    return abs(selected_fl - current_fl) / rate_flps


@numba.njit(cache=True, fastmath=False)
def _first_directional_gap_crossing_in_interval_s(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float,
    required_gap_fl: float,
    start_s: float,
    end_s: float,
    b_above_a: bool,
    rounded: int,
) -> float:
    """Return when one directional band gap first reaches `required_gap_fl`."""
    # Between two adjacent "aircraft reaches selected level" events, each band edge is
    # linear in time. Without rounding that leaves one linear gap; with rounding, the
    # directional gap with grid rounding changes only at grid boundary events.
    start_lower_level_fl, start_higher_level_fl = _directional_compared_levels_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=start_s,
        b_above_a=b_above_a,
    )

    # Express the starting rounded gap as a difference in grid indices. This is the
    # same grid rounding used by `vertical_band_gap_fl`, just kept as indices
    # so the rounded branch can count how many boundary events are still needed.
    if rounded > 0:
        start_higher_index = _rounded_higher_index(start_higher_level_fl, rounded)
        start_lower_index = _rounded_lower_index(start_lower_level_fl, rounded)
        start_gap_fl = (start_higher_index - start_lower_index) * rounded
    else:
        start_higher_index = 0
        start_lower_index = 0
        start_gap_fl = start_higher_level_fl - start_lower_level_fl
    if start_gap_fl >= required_gap_fl:
        return start_s

    if end_s <= start_s:
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S

    end_lower_level_fl, end_higher_level_fl = _directional_compared_levels_fl(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        t_s=end_s,
        b_above_a=b_above_a,
    )

    # The endpoint tells us whether this interval can possibly contain the crossing.
    # If the rounded or exact gap has not improved enough by `end_s`, later intervals
    # or final selected levels must handle it instead.
    if rounded > 0:
        end_gap_fl = (
            _rounded_higher_index(end_higher_level_fl, rounded) - _rounded_lower_index(end_lower_level_fl, rounded)
        ) * rounded
    else:
        end_gap_fl = end_higher_level_fl - end_lower_level_fl
    if end_gap_fl < required_gap_fl or end_gap_fl <= start_gap_fl:
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S

    # Exact edges still have the closed-form crossing, since the directional gap is a
    # single line over this interval.
    if rounded <= 0:
        return start_s + (required_gap_fl - start_gap_fl) * (end_s - start_s) / (end_gap_fl - start_gap_fl)

    # With rounding, each upward crossing by the higher compared edge or downward
    # crossing by the lower compared edge increases the rounded gap by one grid step.
    # Both moving streams have the same period because the model uses one shared rate.
    required_gap_steps = math.ceil(required_gap_fl / rounded)
    needed_gap_steps = required_gap_steps - (start_higher_index - start_lower_index)
    # The phase endpoints are level-off events. Determine motion from that
    # phase, not from a rounded difference that can turn a fixed edge into a
    # spurious second event stream.
    if b_above_a:
        lower_current, lower_selected = a_current_fl, a_selected_fl
        higher_current, higher_selected = b_current_fl, b_selected_fl
    else:
        lower_current, lower_selected = b_current_fl, b_selected_fl
        higher_current, higher_selected = a_current_fl, a_selected_fl
    rate_flps = _rate_fl_per_second(vertical_rate_fpm)
    higher_moves = higher_current < higher_selected and start_s < _time_to_selected_s(
        higher_current, higher_selected, vertical_rate_fpm
    )
    lower_moves = lower_current > lower_selected and start_s < _time_to_selected_s(
        lower_current, lower_selected, vertical_rate_fpm
    )
    higher_slope_flps = rate_flps if higher_moves else 0.0
    lower_slope_flps = -rate_flps if lower_moves else 0.0

    # Find the first helpful grid crossing made by each compared edge. A higher edge
    # moving upward or a lower edge moving downward each opens the rounded gap by one
    # `rounded` step; stationary or wrong-way edges do not contribute in this interval.
    first_higher_crossing_s = VERTICAL_OVERLAP_NEVER_RESOLVES_S
    first_lower_crossing_s = VERTICAL_OVERLAP_NEVER_RESOLVES_S
    period_s = 0.0
    if higher_slope_flps > 0.0:
        next_higher_boundary_fl = (start_higher_index + 1) * rounded
        first_higher_crossing_s = start_s + (next_higher_boundary_fl - start_higher_level_fl) / higher_slope_flps
        period_s = rounded / higher_slope_flps
    if lower_slope_flps < 0.0:
        next_lower_boundary_fl = (start_lower_index - 1) * rounded
        first_lower_crossing_s = start_s + (next_lower_boundary_fl - start_lower_level_fl) / lower_slope_flps
        period_s = rounded / -lower_slope_flps

    # Select the Nth helpful boundary event. With one active edge this is one arithmetic
    # progression; with two active edges, the two progressions have the same period and
    # either coincide exactly or alternate with a fixed offset.
    if (
        first_higher_crossing_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S
        and first_lower_crossing_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S
    ):
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S
    if first_higher_crossing_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S:
        crossing_s = first_lower_crossing_s + (needed_gap_steps - 1) * period_s
    elif first_lower_crossing_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S:
        crossing_s = first_higher_crossing_s + (needed_gap_steps - 1) * period_s
    else:
        # Two active event streams either coincide on every period, or interleave with
        # fixed offsets. Count simultaneous events as two grid steps.
        if abs(first_higher_crossing_s - first_lower_crossing_s) <= 1e-9:
            crossing_s = first_higher_crossing_s + ((needed_gap_steps + 1) // 2 - 1) * period_s
        else:
            first_crossing_s = min(first_higher_crossing_s, first_lower_crossing_s)
            second_crossing_s = max(first_higher_crossing_s, first_lower_crossing_s)
            if needed_gap_steps % 2 == 1:
                crossing_s = first_crossing_s + (needed_gap_steps // 2) * period_s
            else:
                crossing_s = second_crossing_s + ((needed_gap_steps - 2) // 2) * period_s

    # The endpoint check above should normally keep the selected event inside this
    # interval. Keep a small tolerance for floating-point boundary arithmetic.
    if crossing_s > end_s:
        if crossing_s - end_s <= 1e-9:
            return end_s
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S
    return crossing_s


@numba.njit(cache=True, fastmath=True)
def time_to_vertical_overlap_resolution(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float = DEFAULT_VERTICAL_RATE_FPM,
    required_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    rounded: int = DEFAULT_VERTICAL_ROUNDING_FL,
) -> float:
    """
    Estimate the first time at which the remaining vertical bands are resolved.

    Each aircraft is assumed to fly from current level toward selected level at
    `vertical_rate_fpm`, stopping once selected level is reached. At any time, its
    vertical band is the interval between its current level at that time and its
    selected level. The returned time estimates when those two bands are
    separated by at least `required_gap_fl`, after rounding compared band edges toward
    each other to multiples of `rounded`.

    Returns `0.0` if the bands are already resolved, and
    `VERTICAL_OVERLAP_NEVER_RESOLVES_S` if they never resolve under the selected levels
    and climb/descent rate.

    Parameters
    ----------
    a_current_fl : float
        Aircraft A current flight level.
    a_selected_fl : float
        Aircraft A selected flight level.
    b_current_fl : float
        Aircraft B current flight level.
    b_selected_fl : float
        Aircraft B selected flight level.
    vertical_rate_fpm : float, optional
        Shared climb/descent rate magnitude in feet per minute, used for both
        aircraft. Negative values are treated as magnitudes; direction comes
        from each aircraft's current and selected levels.
    required_gap_fl : float, optional
        Required gap between the two remaining vertical bands, in flight levels.
    rounded : int, optional
        Flight-level rounding step used before measuring the gap. The lower compared
        level is rounded upward and the higher compared level is rounded downward.
        Pass `rounded <= 0` to use unrounded floating-point band edges.

    Returns
    -------
    float
        Estimated first time in seconds at which the remaining vertical bands are separated by
        `required_gap_fl`; `0.0` if already resolved; or
        `VERTICAL_OVERLAP_NEVER_RESOLVES_S` if they never resolve.

    Notes
    -----
    Band edges are piecewise linear in time, and their slopes can only change when one
    aircraft reaches its selected level. The function therefore checks at most two
    intervals and both possible vertical orderings. Unrounded checks use direct linear
    interpolation; rounded checks count the rounding-grid boundary events needed to
    satisfy the required gap.
    Passing a shared minimum rate encloses independent, faster rates when both
    aircraft progress immediately towards their selected levels without overshoot
    and remain there on arrival. The remaining bands shrink as time increases.
    Rounded comparisons use a 1e-9 FL tolerance per edge, as documented by
    vertical_band_gap_fl. The returned boundary is a floating-point estimate.
    """
    _validate_vertical_options(required_gap_fl, rounded)
    if required_gap_fl < 0.0:
        required_gap_fl = 0.0

    if vertical_bands_are_resolved(
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        vertical_rate_fpm=vertical_rate_fpm,
        required_gap_fl=required_gap_fl,
        rounded=rounded,
        t_s=0.0,
    ):
        return 0.0

    a_done_s = _time_to_selected_s(a_current_fl, a_selected_fl, vertical_rate_fpm)
    b_done_s = _time_to_selected_s(b_current_fl, b_selected_fl, vertical_rate_fpm)
    if a_done_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S or b_done_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S:
        return VERTICAL_OVERLAP_NEVER_RESOLVES_S

    # The band geometry can only change slope when an aircraft reaches selected level.
    # That leaves at most two linear intervals to test: before the first aircraft levels,
    # then before the second one levels.
    first_done_s = min(a_done_s, b_done_s)
    second_done_s = max(a_done_s, b_done_s)

    for interval_index in range(2):
        if interval_index == 0:
            start_s = 0.0
            end_s = first_done_s
        else:
            start_s = first_done_s
            end_s = second_done_s

        # Check both possible final orderings separately. This matters for crossing
        # climb/descent cases because max(B-above-A, A-above-B) is not itself one line.
        b_above_crossing_s = _first_directional_gap_crossing_in_interval_s(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=vertical_rate_fpm,
            required_gap_fl=required_gap_fl,
            start_s=start_s,
            end_s=end_s,
            b_above_a=True,
            rounded=rounded,
        )
        a_above_crossing_s = _first_directional_gap_crossing_in_interval_s(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=vertical_rate_fpm,
            required_gap_fl=required_gap_fl,
            start_s=start_s,
            end_s=end_s,
            b_above_a=False,
            rounded=rounded,
        )

        if b_above_crossing_s != VERTICAL_OVERLAP_NEVER_RESOLVES_S and (
            a_above_crossing_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S or b_above_crossing_s <= a_above_crossing_s
        ):
            return b_above_crossing_s
        if a_above_crossing_s != VERTICAL_OVERLAP_NEVER_RESOLVES_S:
            return a_above_crossing_s

    # If neither linear interval crossed the required gap, both aircraft are level by
    # `second_done_s` and the selected-level bands remain fixed from then on.
    return VERTICAL_OVERLAP_NEVER_RESOLVES_S


def _exact_gap_deficit(
    levels: tuple[int, int, int, int],
    required_gap: int,
    level_denominator: int,
    rate_ratio: tuple[int, int],
    time_ratio: tuple[int, int],
) -> int:
    """Return an exact deficit numerator; positive means insufficient separation.

    Levels and required_gap share level_denominator. The returned numerator
    has denominator level_denominator * progress_den, defined below.
    """
    lower_current, lower_selected, higher_current, higher_selected = levels
    rate_num, rate_den = rate_ratio
    time_num, time_den = time_ratio
    conversion = int(FT_PER_FL * SECONDS_PER_MINUTE)
    progress_num = rate_num * time_num * level_denominator
    progress_den = conversion * rate_den * time_den
    lower_top = max(lower_selected * progress_den, lower_current * progress_den - progress_num)
    higher_bottom = min(higher_selected * progress_den, higher_current * progress_den + progress_num)
    return required_gap * progress_den - (higher_bottom - lower_top)


def _conservative_vertical_resolution_time_s(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    minimum_vertical_rate_fpm: float,
    required_vertical_gap_fl: float,
    vertical_rounding_fl: int,
) -> float | None:
    """Verify a proposed vertical shortcut, or return None to retain lateral checking.

    Inputs have already passed the heading interface's numeric validation.
    The floating-point timing kernel proposes a time using its rounding grid.
    Exact integer arithmetic on the supplied binary64 values then checks the
    unrounded gap, without the kernel's grid tolerance. If necessary,
    the time is increased and verified again. This can return a later boundary
    than the first mathematical resolution time, including with rounding off.
    """
    levels = (a_current_fl, a_selected_fl, b_current_fl, b_selected_fl)
    # Keep the proposal's grid-index products within binary64/int64 range.
    # This bound protects arithmetic range; exact verification below protects
    # acceptance from loss of precision within that range.
    if max(*(abs(level) for level in levels), required_vertical_gap_fl, vertical_rounding_fl) > 2**52:
        return None
    if vertical_bands_are_resolved(
        a_current_fl,
        a_selected_fl,
        b_current_fl,
        b_selected_fl,
        0.0,
        required_vertical_gap_fl,
        0.0,
        vertical_rounding_fl,
    ):
        resolution_s = 0.0
    else:
        rate_flps = minimum_vertical_rate_fpm / FT_PER_FL / SECONDS_PER_MINUTE
        if rate_flps == 0.0:
            return None
        for current_fl, selected_fl in ((a_current_fl, a_selected_fl), (b_current_fl, b_selected_fl)):
            if current_fl != selected_fl:
                completion_s = abs(selected_fl - current_fl) / rate_flps
                if not math.isfinite(completion_s) or completion_s == 0.0:
                    return None
        resolution_s = float(
            time_to_vertical_overlap_resolution(
                a_current_fl,
                a_selected_fl,
                b_current_fl,
                b_selected_fl,
                minimum_vertical_rate_fpm,
                required_vertical_gap_fl,
                vertical_rounding_fl,
            )
        )
        if not math.isfinite(resolution_s) or resolution_s < 0.0:
            return None
        if vertical_rounding_fl > 0 and not vertical_bands_are_resolved(
            a_current_fl,
            a_selected_fl,
            b_current_fl,
            b_selected_fl,
            minimum_vertical_rate_fpm,
            required_vertical_gap_fl,
            resolution_s,
            vertical_rounding_fl,
        ):
            return None

    # Selected levels determine the only possible separated final ordering.
    # When they coincide and the required gap is zero, current levels choose
    # the ordering that can become disjoint first.
    if (a_selected_fl, a_current_fl) <= (b_selected_fl, b_current_fl):
        ordered_levels = (a_current_fl, a_selected_fl, b_current_fl, b_selected_fl)
    else:
        ordered_levels = (b_current_fl, b_selected_fl, a_current_fl, a_selected_fl)

    # Binary64 denominators are powers of two, so the largest is a common
    # denominator. Scaling to it preserves each supplied value exactly.
    ratios = tuple(float(value).as_integer_ratio() for value in (*ordered_levels, required_vertical_gap_fl))
    level_denominator = max(denominator for _, denominator in ratios)
    lower_current, lower_selected, higher_current, higher_selected, required_gap = (
        numerator * (level_denominator // denominator) for numerator, denominator in ratios
    )
    scaled_levels = (lower_current, lower_selected, higher_current, higher_selected)
    if higher_selected - lower_selected < required_gap:
        return None
    rate_ratio = float(minimum_vertical_rate_fpm).as_integer_ratio()
    time_ratio = resolution_s.as_integer_ratio()
    deficit = _exact_gap_deficit(scaled_levels, required_gap, level_denominator, rate_ratio, time_ratio)
    if deficit <= 0:
        return resolution_s
    rate_num, _ = rate_ratio
    if rate_num == 0:
        return None

    # Until the required gap is reached, at least one compared edge must keep
    # moving at the minimum rate: otherwise the separated selected levels
    # could never be reached. Dividing the deficit by one aircraft's rate
    # therefore gives a conservative delay, also when both edges are moving.
    # Dividing the deficit by the rate cancels the unit conversion and the
    # rate denominator, leaving level_denominator * time_den * rate_num.
    time_num, time_den = time_ratio
    corrected_num = time_num * level_denominator * rate_num + deficit
    corrected_den = level_denominator * time_den * rate_num
    try:
        resolution_s = float(Fraction(corrected_num, corrected_den))
    except OverflowError:
        return None
    if not math.isfinite(resolution_s):
        return None
    final_time_num, final_time_den = resolution_s.as_integer_ratio()
    if final_time_num * corrected_den < corrected_num * final_time_den:
        resolution_s = math.nextafter(resolution_s, math.inf)
    if not math.isfinite(resolution_s):
        return None
    if (
        _exact_gap_deficit(scaled_levels, required_gap, level_denominator, rate_ratio, resolution_s.as_integer_ratio())
        > 0
    ):
        return None
    if vertical_rounding_fl > 0 and not vertical_bands_are_resolved(
        a_current_fl,
        a_selected_fl,
        b_current_fl,
        b_selected_fl,
        minimum_vertical_rate_fpm,
        required_vertical_gap_fl,
        resolution_s,
        vertical_rounding_fl,
    ):
        return None
    return resolution_s
