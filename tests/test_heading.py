"""Independent prescribed-profile controls for the generic heading contract."""

from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

from geometric_safety.heading import (
    HeadingMotion,
    _enclosing_common_speed_bounds_kt,
    _mean_speed_envelope_m,
    certify_heading_separation,
)
from geometric_safety.relevant_aircraft import _catch_up_projection_interval_with_turns_local
from geometric_safety.util import KT_TO_MPS


def _motion(**changes: float | str) -> HeadingMotion:
    return replace(
        HeadingMotion(
            heading0_deg=0.0,
            target_heading_deg=90.0,
            turn_rate_deg_s=1.5,
            speed_min_mps=20.0,
            speed_max_mps=80.0,
        ),
        **changes,
    )


def _stationary() -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=0.0,
        target_heading_deg=0.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=0.0,
        speed_max_mps=0.0,
    )


def _profile(low: float, high: float, switch_s: float) -> Callable[[float], float]:
    return lambda time_s: low if time_s < switch_s else high


def _displacement(
    motion: HeadingMotion,
    speed: Callable[[float], float],
    time_s: float,
    speed_breaks_s: tuple[float, ...] = (),
) -> np.ndarray:
    """Integrate the prescribed velocity without calling any library motion basis."""
    sweep = (motion.target_heading_deg - motion.heading0_deg + 180.0) % 360.0 - 180.0
    direction = -1.0 if sweep < 0.0 else 1.0
    turn_end = abs(sweep) / motion.turn_rate_deg_s if motion.turn_rate_deg_s > 0.0 else math.inf
    breaks = tuple(sorted({0.0, time_s, *(point for point in (*speed_breaks_s, turn_end) if 0.0 < point < time_s)}))

    def heading(time: float) -> float:
        progress = min(motion.turn_rate_deg_s * time, abs(sweep))
        return math.radians(motion.heading0_deg + direction * progress)

    east = 0.0
    north = 0.0
    for start, end in pairwise(breaks):
        east += quad(lambda time: speed(time) * math.sin(heading(time)), start, end, epsabs=1e-8, epsrel=1e-12)[0]
        north += quad(lambda time: speed(time) * math.cos(heading(time)), start, end, epsabs=1e-8, epsrel=1e-12)[0]
    return np.array([east, north])


@pytest.mark.parametrize("switch_s", [15.0, 30.0, 100.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_full_horizon_profiles_are_inside_mean_speed_envelope(switch_s: float, reverse: bool) -> None:
    motion = _motion()
    low, high = (80.0, 20.0) if reverse else (20.0, 80.0)
    profile = _profile(low, high, switch_s)
    for time_s in (1.0, 30.0, 60.0, 120.0, 600.0):
        low_time = min(time_s, switch_s)
        mean = (low * low_time + high * (time_s - low_time)) / time_s
        executed = _displacement(motion, profile, time_s, (switch_s,))
        surrogate = _displacement(motion, lambda _time, value=mean: value, time_s)
        error = float(np.linalg.norm(executed - surrogate))
        assert error <= _mean_speed_envelope_m(motion, time_s) + 1e-7
    assert _mean_speed_envelope_m(motion, 600.0) == _mean_speed_envelope_m(motion, 60.0)


@pytest.mark.parametrize(
    ("rate", "horizon"),
    [(1.5, 600.0), (1e-8, 600.0), (1e-12, 600.0), (1e-100, 1.0)],
)
def test_envelope_matches_high_precision_without_small_angle_cancellation(rate: float, horizon: float) -> None:
    motion = _motion(turn_rate_deg_s=rate)
    with mp.workdps(250):
        omega = mp.mpf(rate) * mp.pi / 180
        theta = min(omega * mp.mpf(horizon), mp.pi / 2)
        expected = 4 * mp.mpf(60) / omega * (1 - mp.cos(theta / 2))
    assert _mean_speed_envelope_m(motion, horizon) == pytest.approx(float(expected), rel=2e-14, abs=0.0)


def test_continued_high_speed_has_a_collision_witness_and_is_not_accepted() -> None:
    motion = _motion()
    profile = _profile(20.0, 80.0, 30.0)
    encounter_time = 100.0  # Forty seconds after the turn ends.
    encounter_position = _displacement(motion, profile, encounter_time, (30.0,))
    relative_position = (-float(encounter_position[0]), -float(encounter_position[1]))
    assert np.linalg.norm(np.asarray(relative_position) + encounter_position) < 1e-8
    b = replace(_stationary(), heading0_deg=90.0, target_heading_deg=90.0)
    # Use exactly the same enlarged speed intervals as the wrapper. The base
    # constant-speed turn law certifies this pair, so rejection is due to the
    # new full-horizon profile envelope rather than artificial B uncertainty.
    converted = _enclosing_common_speed_bounds_kt(motion, b)
    assert converted is not None
    a_speed_kt, b_speed_kt, half_width_kt = converted
    base_safe, base_distance, _ = _catch_up_projection_interval_with_turns_local(
        rel_pos0=np.asarray(relative_position),
        a_heading0_deg=motion.heading0_deg,
        a_target_heading_deg=motion.target_heading_deg,
        a_speed_kt=a_speed_kt,
        a_turn_rate_deg_sec=motion.turn_rate_deg_s,
        b_heading0_deg=b.heading0_deg,
        b_target_heading_deg=b.target_heading_deg,
        b_speed_kt=b_speed_kt,
        b_turn_rate_deg_sec=b.turn_rate_deg_s,
        separation_threshold_m=1.0,
        speed_diff_kt=half_width_kt,
        projection_time_s=120.0,
    )
    assert base_safe
    assert base_distance > 1.0
    result = certify_heading_separation(relative_position, motion, b, separation_threshold_m=1.0, horizon_s=120.0)
    assert result.status == "unresolved"
    assert result.inflation_m == pytest.approx(_mean_speed_envelope_m(motion, 120.0))
    assert result.base_min_distance_m is not None


def test_separated_turning_profiles_are_certified_and_independently_checked() -> None:
    a = _motion()
    b = _motion(
        heading0_deg=180.0, target_heading_deg=120.0, turn_rate_deg_s=0.75, speed_min_mps=10.0, speed_max_mps=40.0
    )
    relative_position = (20000.0, 20000.0)
    result = certify_heading_separation(relative_position, a, b, separation_threshold_m=1000.0, horizon_s=120.0)
    assert result.status == "certified_safe"
    assert result.diagnostic_kind == "sampled_minimum"
    assert result.base_closest_time_s is not None
    assert result.base_bound_interval_s is None
    for switch_s in (15.0, 90.0):
        a_profile = _profile(20.0, 80.0, switch_s)
        b_profile = _profile(40.0, 10.0, switch_s)
        for time_s in np.linspace(0.0, 120.0, 17):
            rel = (
                np.asarray(relative_position)
                + _displacement(a, a_profile, float(time_s), (switch_s,))
                - _displacement(b, b_profile, float(time_s), (switch_s,))
            )
            assert np.linalg.norm(rel) >= 1000.0


def test_straight_geometry_retains_independent_intervals_and_exact_minimum() -> None:
    a = _motion(heading0_deg=90.0, target_heading_deg=90.0, turn_rate_deg_s=0.0, speed_min_mps=10.0, speed_max_mps=20.0)
    # A shared uncertainty width would incorrectly enlarge B's zero-speed interval.
    result = certify_heading_separation((-100.0, 5.0), a, _stationary(), separation_threshold_m=4.0, horizon_s=10.0)
    assert result.status == "certified_safe"
    assert result.inflation_m == 0.0
    assert result.diagnostic_kind == "straight_minimum"
    assert result.base_bound_interval_s is None
    assert result.base_min_distance_m == pytest.approx(5.0)
    assert result.base_closest_time_s is not None
    assert 5.0 <= result.base_closest_time_s <= 10.0
    rejected = certify_heading_separation((-100.0, 5.0), a, _stationary(), separation_threshold_m=6.0, horizon_s=10.0)
    assert rejected.status == "unresolved"
    # A nonconstant profile reaches the same 5 m minimum at t=7.5.
    rel = np.array([-100.0, 5.0]) + _displacement(a, _profile(10.0, 20.0, 5.0), 7.5, (5.0,))
    assert np.linalg.norm(rel) == pytest.approx(5.0)


def test_declared_deviations_can_reject_an_otherwise_separated_straight_pair() -> None:
    a = _motion(heading0_deg=90.0, target_heading_deg=90.0, speed_min_mps=10.0, speed_max_mps=10.0)
    pair = ((-100.0, 10.0), a, _stationary())
    assert certify_heading_separation(*pair, separation_threshold_m=9.0, horizon_s=20.0).status == "certified_safe"
    position = certify_heading_separation(
        *pair, separation_threshold_m=9.0, horizon_s=20.0, relative_position_error_m=2.0
    )
    velocity = certify_heading_separation(
        *pair, separation_threshold_m=9.0, horizon_s=20.0, relative_velocity_error_mps=0.2
    )
    direction = certify_heading_separation(
        pair[0],
        replace(a, heading_deviation_max_deg=1.0),
        pair[2],
        separation_threshold_m=9.0,
        horizon_s=20.0,
    )
    assert position.status == velocity.status == direction.status == "unresolved"
    assert position.inflation_m == 2.0
    assert velocity.inflation_m == 4.0
    assert direction.inflation_m == pytest.approx(400.0 * math.sin(math.radians(0.5)))
    # A permitted 1 degree direction deviation gives a real encounter below 9 m.
    crossing_time = 100.0 / (10.0 * math.cos(math.radians(1.0)))
    witness_north = 10.0 - 10.0 * math.sin(math.radians(1.0)) * crossing_time
    assert witness_north < 9.0


def test_unresolved_circle_reports_interval_bound_without_a_sample_time() -> None:
    a = _motion(
        target_heading_deg=180.0,
        turn_rate_deg_s=3.0,
        speed_min_mps=100.0,
        speed_max_mps=100.0,
    )
    radius = 100.0 / math.radians(3.0)
    threshold = radius - 0.5
    # The trajectory has exact radius 6000/pi and centre (radius-6000/pi, 0).
    # Its norm exceeds the threshold throughout the horizon, even though a
    # coarse interval bound is zero.
    with mp.workdps(80):
        exact_radius = mp.mpf(6000) / mp.pi
        minimum_distance = exact_radius - abs(mp.mpf(radius) - exact_radius)
        assert minimum_distance > mp.mpf(threshold)
    result = certify_heading_separation(
        (radius, 0.0), a, _stationary(), separation_threshold_m=threshold, horizon_s=60.0
    )
    assert result.status == "unresolved"
    assert result.inflation_m == 0.0
    assert result.diagnostic_kind == "interval_bound"
    assert result.base_min_distance_m == 0.0
    assert result.base_closest_time_s is None
    assert result.base_bound_interval_s == (0.0, 60.0)


def test_turning_collision_reports_sample_time_without_a_bound_interval() -> None:
    a = _motion(turn_rate_deg_s=3.0, speed_min_mps=100.0, speed_max_mps=100.0)
    radius = 100.0 / math.radians(3.0)
    result = certify_heading_separation(
        (-radius, -radius), a, _stationary(), separation_threshold_m=1.0, horizon_s=60.0
    )
    assert result.status == "unresolved"
    assert result.diagnostic_kind == "sampled_minimum"
    assert result.base_min_distance_m is not None
    assert result.base_min_distance_m < 1e-8
    assert result.base_closest_time_s == 30.0
    assert result.base_bound_interval_s is None


@pytest.mark.parametrize("target_heading_deg", [0.0, 90.0])
def test_finite_geometry_outside_guard_resolution_domain_is_unresolved(target_heading_deg: float) -> None:
    a = _motion(
        target_heading_deg=target_heading_deg,
        speed_min_mps=100.0,
        speed_max_mps=100.0,
    )
    result = certify_heading_separation((1e8, 1e8), a, _stationary(), separation_threshold_m=1.0, horizon_s=60.0)
    assert result.status == "unresolved"
    assert "numerical range" in result.reason
    assert result.diagnostic_kind is None
    assert result.base_min_distance_m is None
    assert result.base_closest_time_s is None
    assert result.base_bound_interval_s is None


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"turn_direction": "left"}, "directed turns"),
        ({"turn_direction": "right"}, "directed turns"),
        ({"heading_law": "speed_dependent"}, "fixed-rate heading"),
        ({"heading_law": "route_turn"}, "fixed-rate heading"),
        ({"turn_delay_s": 1.0}, "delay"),
        ({"turn_rate_deg_s": 0.0}, "zero turn rate"),
    ],
)
def test_unsupported_declarations_do_not_silently_change_the_motion(
    changes: dict[str, float | str], reason: str
) -> None:
    result = certify_heading_separation(
        (100000.0, 100000.0), _motion(**changes), _stationary(), separation_threshold_m=1.0, horizon_s=120.0
    )
    assert result.status == "unsupported"
    assert reason in result.reason
    assert result.diagnostic_kind is None
    assert result.base_min_distance_m is None
    assert result.base_closest_time_s is None
    assert result.base_bound_interval_s is None


@pytest.mark.parametrize("speed", [0.0, 0.1, 1.0, 13.7, 250.123456789, 1000.0])
def test_zero_width_speed_conversion_encloses_original_value(speed: float) -> None:
    a = _motion(speed_min_mps=speed, speed_max_mps=speed)
    converted = _enclosing_common_speed_bounds_kt(a, _stationary())
    assert converted is not None
    a_centre, b_centre, half_width = converted
    assert max(a_centre - half_width, 0.0) * KT_TO_MPS <= speed
    assert (a_centre + half_width) * KT_TO_MPS >= speed
    assert max(b_centre - half_width, 0.0) * KT_TO_MPS == 0.0


def test_inexact_roundtrip_constant_speed_turn_is_not_rejected_by_conversion() -> None:
    # Derive a value with an inexact roundtrip rather than assuming a platform bit pattern.
    speed = next(value / 10.0 for value in range(1, 1000) if (value / 10.0 / KT_TO_MPS) * KT_TO_MPS != value / 10.0)
    a = _motion(speed_min_mps=speed, speed_max_mps=speed)
    result = certify_heading_separation(
        (100000.0, 100000.0), a, _stationary(), separation_threshold_m=1.0, horizon_s=120.0
    )
    assert result.status == "certified_safe"


def test_asymmetric_speed_conversion_encloses_both_intervals() -> None:
    a = _motion(speed_min_mps=0.1, speed_max_mps=13.7)
    b = _motion(speed_min_mps=250.0, speed_max_mps=250.123456789)
    converted = _enclosing_common_speed_bounds_kt(a, b)
    assert converted is not None
    a_centre, b_centre, half_width = converted
    for centre, motion in ((a_centre, a), (b_centre, b)):
        assert max(centre - half_width, 0.0) * KT_TO_MPS <= motion.speed_min_mps
        assert (centre + half_width) * KT_TO_MPS >= motion.speed_max_mps


@pytest.mark.parametrize(
    ("target", "rate", "horizon", "reject_threshold", "accept_threshold"),
    [(91.0, 1e-8, 600.0, 9260.0, 9250.0), (89.5, 3.0, 60.0, 9250.0, 9000.0)],
)
def test_near_parallel_small_sweeps_have_accepting_and_rejecting_controls(
    target: float, rate: float, horizon: float, reject_threshold: float, accept_threshold: float
) -> None:
    a = _motion(
        heading0_deg=89.0,
        target_heading_deg=target,
        turn_rate_deg_s=rate,
        speed_min_mps=250.0,
        speed_max_mps=250.0,
    )
    b = replace(a, target_heading_deg=89.0, turn_rate_deg_s=0.0)
    relative_position = (0.0, 9260.0)
    rel_end = (
        np.array(relative_position)
        + _displacement(a, lambda _time: 250.0, horizon)
        - _displacement(b, lambda _time: 250.0, horizon)
    )
    assert np.linalg.norm(rel_end) < reject_threshold - 1e-5
    result = certify_heading_separation(
        relative_position, a, b, separation_threshold_m=reject_threshold, horizon_s=horizon
    )
    assert result.status == "unresolved"
    positive = certify_heading_separation(
        relative_position, a, b, separation_threshold_m=accept_threshold, horizon_s=horizon
    )
    assert positive.status == "certified_safe"


def test_shortest_half_turn_uses_negative_180_tie() -> None:
    a = _motion(target_heading_deg=180.0, turn_rate_deg_s=3.0, speed_min_mps=100.0, speed_max_mps=100.0)
    position = _displacement(a, lambda _time: 100.0, 30.0)
    assert position[0] < 0.0
    result = certify_heading_separation(
        (-float(position[0]), -float(position[1])), a, _stationary(), separation_threshold_m=1.0, horizon_s=60.0
    )
    assert result.status == "unresolved"


@pytest.mark.parametrize("horizon_s", [0.0, 600.0])
def test_zero_speed_motion_is_stationary_even_with_a_heading_turn(horizon_s: float) -> None:
    a = _motion(speed_min_mps=0.0, speed_max_mps=0.0)
    result = certify_heading_separation((3.0, 4.0), a, _stationary(), separation_threshold_m=4.999, horizon_s=horizon_s)
    assert result.status == "certified_safe"
    assert result.inflation_m == 0.0
    assert result.base_min_distance_m == pytest.approx(5.0)
    rejected = certify_heading_separation((3.0, 4.0), a, _stationary(), separation_threshold_m=5.0, horizon_s=horizon_s)
    assert rejected.status == "unresolved"


@pytest.mark.parametrize(
    "changes",
    [
        {"heading0_deg": math.nan},
        {"target_heading_deg": math.inf},
        {"turn_rate_deg_s": -1.0},
        {"turn_delay_s": math.inf},
        {"speed_min_mps": -1.0},
        {"speed_min_mps": 81.0},
        {"heading_deviation_max_deg": 181.0},
    ],
)
def test_invalid_motion_parameters_raise(changes: dict[str, float | str]) -> None:
    with pytest.raises(ValueError, match="must"):
        _motion(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon_s": -1.0},
        {"horizon_s": math.inf},
        {"separation_threshold_m": math.nan},
        {"relative_position_error_m": -1.0},
        {"relative_velocity_error_mps": math.inf},
    ],
)
def test_invalid_query_parameters_raise(changes: dict[str, float]) -> None:
    parameters = {"separation_threshold_m": 1.0, "horizon_s": 120.0} | changes
    with pytest.raises(ValueError, match="must"):
        certify_heading_separation((100000.0, 100000.0), _motion(), _stationary(), **parameters)


def test_nonfinite_position_is_invalid_and_overflowing_finite_motion_is_unresolved() -> None:
    with pytest.raises(ValueError, match="relative_position_m"):
        certify_heading_separation(
            (math.nan, 0.0), _motion(), _stationary(), separation_threshold_m=1.0, horizon_s=120.0
        )
    result = certify_heading_separation(
        (1.0, 1.0),
        _motion(speed_min_mps=1e200, speed_max_mps=2e200),
        _stationary(),
        separation_threshold_m=1.0,
        horizon_s=1e200,
    )
    assert result.status == "unresolved"
    assert "numerical range" in result.reason


def test_unrepresentable_angular_rate_is_unresolved() -> None:
    result = certify_heading_separation(
        (100000.0, 100000.0),
        _motion(turn_rate_deg_s=math.ulp(0.0)),
        _stationary(),
        separation_threshold_m=1.0,
        horizon_s=120.0,
    )
    assert result.status == "unresolved"
    assert "angular rate" in result.reason
