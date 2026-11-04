"""Regression checks against independently integrated instructed motion."""

import math

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.integrate import quad

from geometric_safety import relevant_aircraft as ra


def encounter(
    *,
    rel_pos0: NDArray[np.float64] | None = None,
    a_heading0_deg: float = -0.5,
    a_turn_rate_deg_sec: float = 1.5,
) -> tuple[bool, float, float]:
    return ra._catch_up_projection_interval_with_turns_local(
        rel_pos0=np.array([9260.1, 0.0]) if rel_pos0 is None else rel_pos0,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=0.0,
        a_speed_kt=300.0,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        b_heading0_deg=0.0,
        b_target_heading_deg=0.0,
        b_speed_kt=300.0,
        b_turn_rate_deg_sec=0.0,
        separation_threshold_m=9260.0,
        speed_diff_kt=0.0,
        projection_time_s=60.0,
    )


def test_small_turn_must_not_certify_a_real_violation() -> None:
    assert not encounter()[0]
    assert encounter(rel_pos0=np.array([9262.0, 0.0]))[0]


@pytest.mark.parametrize(
    ("heading", "angle", "rate", "time"),
    [
        (-0.5, 0.5, 1.5, 0.2),
        (0.5, -0.5, 1.5, 60.0),
        (359.0, 2.0, 1.00001e-6, 600.0),
        (89.0, 2.0, 1e-8, 600.0),
        (180.0, -0.01, 0.01, 20.0),
        (30.0, 60.0, 1.5, 60.0),
    ],
)
def test_basis_matches_independent_quadrature(heading: float, angle: float, rate: float, time: float) -> None:
    duration = abs(angle) / rate
    signed_rate = math.copysign(rate, angle)
    reference = []
    for trig in (math.sin, math.cos):
        turning = quad(
            lambda t, trig=trig: trig(math.radians(heading + signed_rate * t)), 0.0, min(time, duration), epsabs=1e-11
        )[0]
        reference.append(turning + max(time - duration, 0) * trig(math.radians(heading + angle)))
    actual = ra._turn_displacement_basis(heading, heading + angle, rate, time)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-12)


def test_small_turn_heading_and_envelopes_remain_active() -> None:
    assert ra._unwrapped_heading_deg_at_time(-0.5, 0.0, 1.5, 0.1) == pytest.approx(-0.35)
    assert ra._turn_execution_envelope_m(-0.5, 0.0, 1.5, 1.0, 160.0, 1.0, 0.1) > 0
    assert ra._turn_speed_schedule_envelope_m(-0.5, 0.0, 1.5, 1.0, 10.0, 140.0, 160.0) > 0
    assert ra._turn_speed_schedule_envelope_m(89.0, 91.0, 1e-8, 600.0, 10.0, 140.0, 160.0) > 0


def test_zero_rate_with_changed_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero turn rate"):
        encounter(a_turn_rate_deg_sec=0.0)


def test_exactly_straight_motion_still_delegates() -> None:
    safe, distance, time = encounter(a_heading0_deg=0.0, a_turn_rate_deg_sec=0.0)
    assert safe
    assert distance == pytest.approx(9260.1)
    assert time == 0.0


def test_nearly_parallel_relative_speed_does_not_cancel() -> None:
    angle_deg = 1e-8
    expected = 300.0 * math.sin(math.radians(angle_deg) / 2)
    assert ra._relative_speed_mps(150.0, 150.0, angle_deg) == pytest.approx(expected, rel=1e-12)
