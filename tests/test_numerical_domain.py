"""Adversarial controls for the declared numerical certification domain."""

import math

import mpmath as mp
import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra
from geometric_safety.heading import HeadingMotion, certify_heading_separation


def _stationary() -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=0.0,
        target_heading_deg=0.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=0.0,
        speed_max_mps=0.0,
    )


def test_distinct_subpicosecond_interval_has_nonzero_lipschitz_bound() -> None:
    assert ra._interval_local_lipschitz_mps(
        0.0, 90.0, 9e14, 1e12, 1e12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-13, 1e12
    ) == pytest.approx(1e12)


@pytest.mark.parametrize("interface", [False, True])
def test_subpicosecond_midpoint_collision_is_not_certified(interface: bool) -> None:
    position = (-0.01864616142890283, -0.045015815807855304)
    speed, rate, horizon = 1e12, 9e14, 1e-13
    with mp.workdps(80):
        omega = mp.mpf(rate) * mp.pi / 180
        middle = mp.mpf(horizon) / 2
        x = mp.mpf(position[0]) + speed * (1 - mp.cos(omega * middle)) / omega
        y = mp.mpf(position[1]) + speed * mp.sin(omega * middle) / omega
        assert mp.hypot(x, y) < mp.mpf("1e-17")
    if interface:
        a = HeadingMotion(
            heading0_deg=0.0,
            target_heading_deg=90.0,
            turn_rate_deg_s=rate,
            speed_min_mps=speed,
            speed_max_mps=speed,
        )
        assert (
            certify_heading_separation(
                position, a, _stationary(), separation_threshold_m=0.001, horizon_s=horizon
            ).status
            == "unresolved"
        )
    else:
        assert not ra._catch_up_projection_interval_with_turns_local(
            np.array(position),
            0.0,
            90.0,
            speed / ra.KT_TO_MPS,
            rate,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001,
            0.0,
            horizon,
        )[0]


def test_directional_guard_loss_is_outside_supported_domain() -> None:
    position = (884050573712.5953, 1921686834429.4648)
    threshold = 2115283835919.2146
    with mp.workdps(80):
        exact_distance = mp.hypot(*(mp.mpf(value) for value in position))
        assert mp.mpf(threshold) - exact_distance > mp.mpf("9e-5")
    assert threshold + ra.TURN_CERT_FP_GUARD_M == threshold
    a = HeadingMotion(
        heading0_deg=-65.2957332885886,
        target_heading_deg=24.704266711411403,
        turn_rate_deg_s=1.0,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    result = certify_heading_separation(position, a, _stationary(), separation_threshold_m=threshold, horizon_s=90.0)
    assert result.status == "unresolved"
    assert math.isnan(
        ra._catch_up_projection_interval_with_turns_local(
            np.array(position),
            a.heading0_deg,
            a.target_heading_deg,
            300.0 / ra.KT_TO_MPS,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            threshold,
            0.0,
            90.0,
        )[1]
    )


def test_numerical_domain_retains_guard_resolution() -> None:
    limit = ra.NUMERICAL_GEOMETRY_LIMIT_M
    assert limit + ra.TURN_CERT_FP_GUARD_M > limit
    assert math.ulp(limit) * ra.TURN_CERT_GUARD_RESOLUTION_FACTOR <= ra.TURN_CERT_FP_GUARD_M
    assert ra._geometry_within_numerical_domain(np.array([limit, 0.0]), 0.0, limit)
    outside = math.nextafter(limit, math.inf)
    assert not ra._geometry_within_numerical_domain(np.array([outside, 0.0]), 0.0, limit)
    assert not ra._geometry_within_numerical_domain(np.zeros(2), 0.0, outside)
    assert not ra._geometry_within_numerical_domain(np.array([0.75 * limit, 0.0]), 0.5 * limit, 1.0)


@pytest.mark.parametrize("outside", [False, True])
def test_stationary_query_at_numerical_domain_boundary(outside: bool) -> None:
    distance = ra.NUMERICAL_GEOMETRY_LIMIT_M
    if outside:
        distance = math.nextafter(distance, math.inf)
    result = certify_heading_separation(
        (distance, 0.0), _stationary(), _stationary(), separation_threshold_m=9260.0, horizon_s=1.0
    )
    assert result.status == ("unresolved" if outside else "certified_safe")
    assert (result.base_min_distance_m is None) == outside


@pytest.mark.parametrize("turning", [False, True])
@pytest.mark.parametrize("clearance_m", [4e-6, 0.001])
def test_accepted_query_near_domain_edge_has_high_precision_margin(turning: bool, clearance_m: float) -> None:
    # The minimum is at zero: heading starts tangent and turns away from B.
    horizon = 90.0
    travel = 300.0 * horizon
    position = ra.NUMERICAL_GEOMETRY_LIMIT_M - travel - 1.0
    a = HeadingMotion(
        heading0_deg=0.0,
        target_heading_deg=90.0 if turning else 0.0,
        turn_rate_deg_s=1.0 if turning else 0.0,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    threshold = position - clearance_m
    result = certify_heading_separation(
        (position, 0.0), a, _stationary(), separation_threshold_m=threshold, horizon_s=horizon
    )
    assert result.status == "certified_safe"
    with mp.workdps(80):
        exact_position = mp.mpf(position)
        assert exact_position - mp.mpf(threshold) > mp.mpf(clearance_m) * mp.mpf("0.99")
        for time in (0, 22.5, 45, 90):
            exact_time = mp.mpf(time)
            if turning:
                omega = mp.pi / 180
                x = exact_position + 300 * (1 - mp.cos(omega * exact_time)) / omega
                y = 300 * mp.sin(omega * exact_time) / omega
            else:
                x, y = exact_position, 300 * exact_time
            assert mp.hypot(x, y) >= exact_position


@pytest.mark.parametrize("envelope", [False, True])
def test_core_domain_refusal_has_unavailable_diagnostics(envelope: bool) -> None:
    diagnostic = np.empty(3)
    result = ra._catch_up_projection_interval_with_turns_local(
        np.array([math.nextafter(ra.NUMERICAL_GEOMETRY_LIMIT_M, math.inf), 0.0]),
        0.0,
        1.0,
        300.0 / ra.KT_TO_MPS,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9260.0,
        0.0,
        1.0,
        eta_const_m=1.0 if envelope else 0.0,
        diagnostic_out=diagnostic,
    )
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])
    assert diagnostic[0] == ra.DIAGNOSTIC_UNAVAILABLE
    assert np.isnan(diagnostic[1:]).all()


@pytest.mark.parametrize("minimum_interval_s", [0.0, 3.0, math.ulp(0.0)])
def test_nonprogressing_timestamp_split_fails_closed(minimum_interval_s: float) -> None:
    # The stationary pair is separated throughout. Only the numerical guard
    # prevents acceptance near the horizon, where timestamp spacing is 128 s.
    # Refinement must stop even when the requested stopping width is smaller.
    diagnostic = np.empty(3)
    query = {
        "rel_pos0": np.array([1.0, 0.0]),
        "a_heading0_deg": 0.0,
        "a_target_heading_deg": 0.0,
        "a_speed_kt": 0.0,
        "a_turn_rate_deg_sec": 0.0,
        "b_heading0_deg": 0.0,
        "b_target_heading_deg": 0.0,
        "b_speed_kt": 0.0,
        "b_turn_rate_deg_sec": 0.0,
        "separation_threshold_m": 0.5,
        "speed_diff_kt": 0.0,
        "eta_rate_mps": 2.0**-61,
        "min_cert_interval_s": minimum_interval_s,
    }
    assert ra._catch_up_projection_interval_with_turns_local(**query, projection_time_s=1.0)[0]
    separated, lower_bound, time_s = ra._catch_up_projection_interval_with_turns_local(
        **query, projection_time_s=2.0**60, diagnostic_out=diagnostic
    )
    assert not separated
    assert lower_bound == 1.0
    assert math.isnan(time_s)
    assert diagnostic[0] == ra.DIAGNOSTIC_INTERVAL_BOUND
    assert 0.0 <= diagnostic[1] < diagnostic[2] <= 2.0**60


def test_heading_preserves_interval_bound_when_timestamp_split_stops() -> None:
    horizon_s = 2.0**60
    speed_mps = 1e-18
    a = HeadingMotion(
        heading0_deg=89.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=1.0,
        speed_min_mps=speed_mps,
        speed_max_mps=speed_mps,
    )
    # The one-second turn is negligible at this speed. The later eastbound pass
    # stays about one metre away, inside the numerical guard above the threshold.
    result = certify_heading_separation(
        (-0.75 * horizon_s * speed_mps, 1.0),
        a,
        _stationary(),
        separation_threshold_m=1.0 - 5e-7,
        horizon_s=horizon_s,
    )
    assert result.status == "unresolved"
    assert result.diagnostic_kind == "interval_bound"
    assert result.base_min_distance_m is not None
    assert result.base_closest_time_s is None
    assert result.base_bound_interval_s is not None
