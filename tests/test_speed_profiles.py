"""Independent integral controls and public contracts for profile refinement."""

from collections.abc import Callable
from dataclasses import replace
import math

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import quad

from geometric_safety import (
    HeadingMotion,
    certify_heading_separation,
    certify_heading_separation_from_latlon,
    certify_heading_separation_with_vertical,
)
from geometric_safety._speed_profiles import (
    Parameters,
    certify_speed_profiles,
    directional_interval,
    minimum_projection,
    prepare_motion,
    projection_integrals,
)
from geometric_safety.util import EARTH_RADIUS_IN_METERS


def _stationary() -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=0.0, target_heading_deg=0.0, turn_rate_deg_s=0.0, speed_min_mps=0.0, speed_max_mps=0.0
    )


def _parameters(motion: HeadingMotion) -> Parameters:
    return prepare_motion(
        motion.heading0_deg,
        motion.target_heading_deg,
        motion.turn_rate_deg_s,
        motion.speed_min_mps,
        motion.speed_max_mps,
    )


def _direction(motion: HeadingMotion, time: float) -> tuple[float, float]:
    sweep = (motion.target_heading_deg - motion.heading0_deg + 180.0) % 360.0 - 180.0
    angle = math.radians(
        math.fmod(motion.heading0_deg, 360.0) + math.copysign(min(abs(sweep), motion.turn_rate_deg_s * time), sweep)
    )
    return math.sin(angle), math.cos(angle)


def _integrate(motion: HeadingMotion, function: Callable[[float], float], time: float) -> float:
    # Fixed subintervals and the turn endpoint guide quadrature independently of
    # the production antiderivative and its projection-zero calculation.
    sweep = (motion.target_heading_deg - motion.heading0_deg + 180.0) % 360.0 - 180.0
    finish = abs(sweep) / motion.turn_rate_deg_s if motion.turn_rate_deg_s else 0.0
    points = sorted({*np.linspace(0.0, time, 33).tolist(), min(finish, time)})
    return float(quad(function, 0.0, time, points=points, epsabs=1e-7, epsrel=1e-11, limit=200)[0])


@pytest.mark.parametrize("headings", [(0.0, 170.0), (170.0, 0.0), (350.0, 10.0), (90.0, 270.0), (720.0, 810.0)])
@pytest.mark.parametrize("time", [0.0, 17.0, 120.0, 600.0])
def test_projection_integrals_match_independent_quadrature(headings: tuple[float, float], time: float) -> None:
    motion = replace(_stationary(), heading0_deg=headings[0], target_heading_deg=headings[1], turn_rate_deg_s=1.5)
    normal = np.array([0.6, -0.8])

    def projected(t: float) -> float:
        east, north = _direction(motion, t)
        return 0.6 * east - 0.8 * north

    signed, absolute = projection_integrals(_parameters(motion), normal, time)
    assert signed == pytest.approx(_integrate(motion, projected, time), abs=1e-7)
    assert absolute == pytest.approx(_integrate(motion, lambda t: abs(projected(t)), time), abs=1e-7)


def test_projection_minimum_selects_each_aircrafts_speed_independently() -> None:
    a = HeadingMotion(
        heading0_deg=0.0, target_heading_deg=170.0, turn_rate_deg_s=1.5, speed_min_mps=20.0, speed_max_mps=80.0
    )
    b = HeadingMotion(
        heading0_deg=200.0, target_heading_deg=70.0, turn_rate_deg_s=2.0, speed_min_mps=10.0, speed_max_mps=50.0
    )
    normal, position = np.array([0.6, -0.8]), np.array([3000.0, -4000.0])

    def component(motion: HeadingMotion, t: float, minimise: bool) -> float:
        east, north = _direction(motion, t)
        projected = 0.6 * east - 0.8 * north
        values = motion.speed_min_mps * projected, motion.speed_max_mps * projected
        return min(values) if minimise else max(values)

    pa, pb = _parameters(a), _parameters(b)
    for time in (25.0, 100.0, 300.0):
        expected = 5000.0 + _integrate(a, lambda t: component(a, t, True), time)
        expected -= _integrate(b, lambda t: component(b, t, False), time)
        assert minimum_projection(position, pa, pb, normal, time) == pytest.approx(expected, abs=1e-6)
    bound = directional_interval(position, pa, pb, normal, 20.0, 130.0)
    for time in np.linspace(20.0, 130.0, 23):
        expected = 5000.0 + _integrate(a, lambda t: component(a, t, True), float(time))
        expected -= _integrate(b, lambda t: component(b, t, False), float(time))
        assert bound <= expected + 1e-6


def test_interval_bound_covers_projection_dip_between_safe_endpoints() -> None:
    a = replace(_stationary(), target_heading_deg=180.0, turn_rate_deg_s=90.0, speed_min_mps=1.0, speed_max_mps=1.0)
    pa, pb = _parameters(a), _parameters(_stationary())
    position = np.array([0.0, -100.0])
    bound = directional_interval(position, pa, pb, np.array([0.0, -1.0]), 0.0, 2.0)
    assert bound <= 100.0 - 2.0 / math.pi
    # The position at 1 s is (-2/pi, -100+2/pi), inside the required radius.
    assert math.hypot(2.0 / math.pi, 100.0 - 2.0 / math.pi) < 99.5
    assert not certify_speed_profiles(position, pa, pb, 2.0, 99.5, 3.0, 4096)[0]


@pytest.mark.parametrize("heading", [1e16, 1e18, 1e20, 1e22, 1e30, 1e100])
def test_large_equivalent_headings_do_not_certify_interior_collision(heading: float) -> None:
    moving = replace(
        _stationary(), heading0_deg=heading, target_heading_deg=heading, speed_min_mps=1.0, speed_max_mps=1.0
    )
    position = -50.0 * np.array(_direction(moving, 0.0))
    result = certify_speed_profiles(position, _parameters(moving), _parameters(_stationary()), 100.0, 26.0, 3.0, 4096)
    assert not result[0]
    assert all(math.isnan(value) for value in result[1:])


def test_finite_slow_turn_does_not_evaluate_infinite_turn_endpoint() -> None:
    motion = replace(
        _stationary(), target_heading_deg=90.0, turn_rate_deg_s=1e-308, speed_min_mps=1.0, speed_max_mps=1.0
    )
    parameters = _parameters(motion)
    assert math.isinf(parameters.duration)
    assert projection_integrals(parameters, np.array([0.0, 1.0]), 100.0) == pytest.approx((100.0, 100.0))
    accepted, bound, left, right = certify_speed_profiles(
        np.array([1000.0, 0.0]), parameters, _parameters(_stationary()), 100.0, 925.0, 3.0, 4096
    )
    assert accepted
    assert bound >= 925.0
    assert (left, right) == (0.0, 100.0)


def _overtaking() -> tuple[HeadingMotion, HeadingMotion]:
    knot = 1852.0 / 3600.0
    a = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=0.0,
        turn_rate_deg_s=1.5,
        speed_min_mps=225.0 * knot,
        speed_max_mps=275.0 * knot,
    )
    b = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=120.0,
        turn_rate_deg_s=1.5,
        speed_min_mps=175.0 * knot,
        speed_max_mps=225.0 * knot,
    )
    return a, b


def test_profile_certificate_recovers_turning_case_with_distinct_diagnostics() -> None:
    a, b = _overtaking()
    result = certify_heading_separation((-11112.0, -3704.0), a, b, separation_threshold_m=9260.0, horizon_s=180.0)
    assert result.status == "certified_safe"
    assert result.profile_min_distance_lower_bound_m is not None
    assert result.profile_min_distance_lower_bound_m >= 9285.0
    assert result.profile_bound_interval_s is not None
    left, right = result.profile_bound_interval_s
    assert 0.0 <= left <= right <= 180.0
    assert result.base_min_distance_m is not None
    assert result.base_min_distance_m < 9260.0 + result.inflation_m


@pytest.mark.parametrize("error", ["position", "velocity", "heading_a", "heading_b"])
def test_profile_refinement_does_not_drop_additional_error_allowances(error: str) -> None:
    a, b = _overtaking()
    if error == "heading_a":
        a = replace(a, heading_deviation_max_deg=180.0)
    if error == "heading_b":
        b = replace(b, heading_deviation_max_deg=180.0)
    result = certify_heading_separation(
        (-11112.0, -3704.0),
        a,
        b,
        separation_threshold_m=9260.0,
        horizon_s=180.0,
        relative_position_error_m=12000.0 if error == "position" else 0.0,
        relative_velocity_error_mps=100.0 if error == "velocity" else 0.0,
    )
    assert result.status == "unresolved"
    assert result.profile_min_distance_lower_bound_m is None
    assert result.profile_bound_interval_s is None


def test_exact_straight_acceptance_is_preserved_below_profile_margin() -> None:
    moving = replace(_stationary(), heading0_deg=90.0, target_heading_deg=90.0, speed_min_mps=1.0, speed_max_mps=1.0)
    result = certify_heading_separation(
        (-1.0, 26.01), moving, _stationary(), separation_threshold_m=26.0, horizon_s=3.0
    )
    assert result.status == "certified_safe"
    assert result.base_min_distance_m == pytest.approx(26.01)
    assert result.profile_min_distance_lower_bound_m is None


def test_accepted_profile_bound_subtracts_all_declared_errors() -> None:
    a = replace(
        _stationary(), target_heading_deg=90.0, turn_rate_deg_s=1.5, speed_max_mps=100.0, heading_deviation_max_deg=1.0
    )
    result = certify_heading_separation(
        (1000.0, 1000.0),
        a,
        _stationary(),
        separation_threshold_m=1250.0,
        horizon_s=60.0,
        relative_position_error_m=10.0,
        relative_velocity_error_mps=0.2,
    )
    # Every nominal displacement is into the first quadrant. Speed zero attains
    # the initial point, so its projection on the initial normal stays minimal.
    errors = 10.0 + 0.2 * 60.0 + 2.0 * 100.0 * 60.0 * math.sin(math.radians(0.5))
    assert result.status == "certified_safe"
    assert result.profile_bound_interval_s == (0.0, 60.0)
    assert result.profile_min_distance_lower_bound_m == pytest.approx(math.hypot(1000.0, 1000.0) - errors, abs=1e-8)


def test_exhaustion_returns_no_partial_profile_bound() -> None:
    a, b = _overtaking()
    accepted, bound, left, right = certify_speed_profiles(
        np.array([-11112.0, -3704.0]), _parameters(a), _parameters(b), 180.0, 9285.0, 3.0, 0
    )
    assert not accepted
    assert all(math.isnan(value) for value in (bound, left, right))


def test_accepted_subinterval_cannot_hide_an_unchecked_collision_interval() -> None:
    moving = replace(_stationary(), heading0_deg=90.0, target_heading_deg=90.0, speed_min_mps=1.0, speed_max_mps=1.0)
    position = np.array([-25.0, 0.0])
    a, b = _parameters(moving), _parameters(_stationary())
    # The stack visits the separated second half before the first half, which
    # contains an attained collision at 25 s. Two cells exhaust the budget there.
    assert directional_interval(position, a, b, np.array([1.0, 0.0]), 50.0, 100.0) >= 5.0
    accepted, bound, left, right = certify_speed_profiles(position, a, b, 100.0, 5.0, 3.0, 2)
    assert not accepted
    assert all(math.isnan(value) for value in (bound, left, right))


def test_refinement_keeps_unsupported_and_numerical_domain_gates() -> None:
    a, b = _overtaking()
    unsupported = certify_heading_separation(
        (-11112.0, -3704.0), replace(a, turn_delay_s=1.0), b, separation_threshold_m=9260.0, horizon_s=180.0
    )
    outside = certify_heading_separation((1e20, 0.0), a, b, separation_threshold_m=9260.0, horizon_s=180.0)
    assert unsupported.status == "unsupported"
    assert outside.status == "unresolved"
    assert unsupported.profile_min_distance_lower_bound_m is outside.profile_min_distance_lower_bound_m is None


def test_geographic_and_shortened_vertical_queries_use_profile_certificate() -> None:
    a, b = _overtaking()
    geographic = certify_heading_separation_from_latlon(
        a,
        b,
        a_lat=0.0,
        a_lon=0.0,
        b_lat=math.degrees(3704.0 / EARTH_RADIUS_IN_METERS),
        b_lon=math.degrees(11112.0 / EARTH_RADIUS_IN_METERS),
        separation_threshold_m=9260.0,
        horizon_s=180.0,
    )
    assert geographic.status == "certified_safe"
    assert geographic.profile_min_distance_lower_bound_m is not None
    combined = certify_heading_separation_with_vertical(
        (-11112.0, -3704.0),
        a,
        b,
        a_current_fl=100.0,
        a_selected_fl=100.0,
        b_current_fl=100.0,
        b_selected_fl=120.0,
        minimum_vertical_rate_fpm=1000.0 / 3.0,
        vertical_rounding_fl=0,
        separation_threshold_m=9260.0,
        horizon_s=300.0,
    )
    assert combined.status == "certified_safe"
    assert combined.lateral_horizon_s == pytest.approx(180.0)
    assert combined.lateral_result is not None
    assert combined.lateral_result.profile_min_distance_lower_bound_m is not None
    assert combined.lateral_result.profile_bound_interval_s is not None
    assert combined.lateral_result.profile_bound_interval_s[1] <= combined.lateral_horizon_s


@pytest.mark.parametrize("turn", [-120.0, -45.0, 30.0, 100.0, 180.0])
def test_constructed_variable_speed_collision_cannot_be_accepted(turn: float) -> None:
    a = HeadingMotion(
        heading0_deg=15.0, target_heading_deg=15.0 + turn, turn_rate_deg_s=1.5, speed_min_mps=40.0, speed_max_mps=140.0
    )
    b = HeadingMotion(
        heading0_deg=200.0, target_heading_deg=260.0, turn_rate_deg_s=2.0, speed_min_mps=30.0, speed_max_mps=120.0
    )

    def displacement(motion: HeadingMotion, reverse: bool) -> NDArray[np.float64]:
        def speed(t: float) -> float:
            return motion.speed_min_mps if (t < 64.0) != reverse else motion.speed_max_mps

        return np.array(
            [
                _integrate(motion, lambda t: speed(t) * _direction(motion, t)[0], 128.0),
                _integrate(motion, lambda t: speed(t) * _direction(motion, t)[1], 128.0),
            ]
        )

    position = displacement(b, True) - displacement(a, False)
    result = certify_heading_separation(
        (float(position[0]), float(position[1])), a, b, separation_threshold_m=50.0, horizon_s=256.0
    )
    assert result.status == "unresolved"
    assert result.profile_min_distance_lower_bound_m is None


@pytest.mark.parametrize(
    ("final_heading", "sweep", "rate"), [(17.0, 25.0, 1.25), (83.0, 65.0, 2.0), (175.0, 110.0, 3.0)]
)
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_final_track_direction_certifies_analytical_clearance(
    final_heading: float, sweep: float, rate: float, mirror: bool, swap: bool
) -> None:
    if mirror:
        sweep = -sweep
    initial, final = math.radians(final_heading - sweep), math.radians(final_heading)
    omega = math.copysign(math.radians(rate), sweep)
    duration = abs(sweep) / rate
    track = np.array([math.sin(final), math.cos(final)])
    normal = math.copysign(1.0, sweep) * np.array([math.cos(final), -math.sin(final)])
    turn_displacement = np.array([math.cos(initial) - math.cos(final), math.sin(final) - math.sin(initial)]) / omega
    moving = HeadingMotion(
        heading0_deg=final_heading - sweep,
        target_heading_deg=final_heading,
        turn_rate_deg_s=rate,
        speed_min_mps=100.0,
        speed_max_mps=150.0,
    )
    stationary = replace(_stationary(), heading0_deg=final_heading + 90.0, target_heading_deg=final_heading + 90.0)
    clearance = 10000.0
    position = clearance * normal - 150.0 * (turn_displacement + 80.0 * track)
    # Projection decreases during the turn and is constant afterwards. Maximum
    # speed attains this bound at duration + 80 s, so it is the exact minimum.
    attained = position + 150.0 * (turn_displacement + 80.0 * track)
    assert np.linalg.norm(attained) == pytest.approx(clearance, abs=1e-8)
    a, b = (stationary, moving) if swap else (moving, stationary)
    if swap:
        position = -position
    horizon = duration + 170.0
    accepted, bound, left, right = certify_speed_profiles(
        position, _parameters(a), _parameters(b), horizon, clearance - 5.0, 3.0, 1
    )
    assert accepted
    assert bound == pytest.approx(clearance, abs=1e-8)
    assert (left, right) == (0.0, horizon)
    result = certify_heading_separation(
        (float(position[0]), float(position[1])), a, b, separation_threshold_m=clearance - 30.0, horizon_s=horizon
    )
    assert result.status == "certified_safe"
    assert result.profile_min_distance_lower_bound_m == pytest.approx(clearance, abs=1e-8)
    # Moving the threshold beyond an attained distance must prevent acceptance,
    # even though the chosen normal remains perpendicular to the final track.
    rejected = certify_heading_separation(
        (float(position[0]), float(position[1])), a, b, separation_threshold_m=clearance + 1.0, horizon_s=horizon
    )
    assert rejected.status == "unresolved"
    assert rejected.profile_min_distance_lower_bound_m is None


@pytest.mark.parametrize("horizon", [180.0, 300.0])
def test_opposing_turns_preserve_clearance_through_straight_continuation(horizon: float) -> None:
    knot = 1852.0 / 3600.0
    a = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=60.0,
        turn_rate_deg_s=1.5,
        speed_min_mps=240.0 * knot,
        speed_max_mps=260.0 * knot,
    )
    b = replace(a, heading0_deg=270.0, target_heading_deg=240.0)
    position = np.array([-27780.0, -3704.0])
    omega, angle = math.pi / 120.0, math.pi / 6.0
    turn_end = position + 2.0 * a.speed_max_mps / omega * np.array([math.sin(angle), 1.0 - math.cos(angle)])
    normal, track = np.array([-0.5, math.sqrt(3.0) / 2.0]), np.array([math.sqrt(3.0) / 2.0, 0.5])
    exact_minimum = float(normal @ turn_end)
    closest_time = 20.0 - float(track @ turn_end) / (2.0 * a.speed_max_mps)
    assert 20.0 < closest_time < horizon
    attained = turn_end + 2.0 * a.speed_max_mps * (closest_time - 20.0) * track
    assert np.linalg.norm(attained) == pytest.approx(exact_minimum, abs=1e-8)
    result = certify_heading_separation(
        (float(position[0]), float(position[1])), a, b, separation_threshold_m=9260.0, horizon_s=horizon
    )
    assert result.status == "certified_safe"
    assert result.profile_min_distance_lower_bound_m == pytest.approx(exact_minimum, abs=1e-8)
    assert result.profile_bound_interval_s == (0.0, horizon)
