"""Heading arithmetic controls against exact binary64 input differences."""

from fractions import Fraction
import math
import sys

import numba
import numpy as np
import pytest
from scipy.integrate import quad

from geometric_safety.heading import HeadingMotion, _turn_angle_deg, certify_heading_separation
from geometric_safety.relevant_aircraft import _catch_up_projection_interval_with_turns_local, _turn_displacement_basis
from geometric_safety.util import KT_TO_MPS, heading_diff


def _exact_shortest_difference(h1: float, h2: float) -> Fraction:
    difference = (Fraction.from_float(h2) - Fraction.from_float(h1)) % 360
    return difference - 360 if difference >= 180 else difference


def _boundary_cases() -> list[tuple[float, float]]:
    cases = []
    for origin in (0.0, 0.1, -0.1, 89.9, 359.9, -359.9, 1e12, -1e12):
        for offset in (-360.0, -180.0, 0.0, 180.0, 360.0):
            target = origin + offset
            cases.extend(
                (origin, adjacent)
                for adjacent in (math.nextafter(target, -math.inf), target, math.nextafter(target, math.inf))
            )
    cases.extend(
        [
            (0.0, 1e-300),
            (1e-300, 0.0),
            (0.0, -1e-300),
            (-1e-300, 0.0),
            (1e-300, 360.0),
            (360.0, 1e-300),
            (-1e-300, -360.0),
            (-360.0, -1e-300),
            (0.0, math.ulp(0.0)),
            (math.ulp(0.0), 0.0),
            (sys.float_info.max, -sys.float_info.max),
            (-sys.float_info.max, sys.float_info.max),
            (1e308, -1e308),
            (-1e308, 1e308),
        ]
    )
    return list(dict.fromkeys(cases))


@numba.njit(fastmath=True)
def _fastmath_caller(h1: float, h2: float, abs_diff: bool = False) -> float:
    return heading_diff(h1, h2, abs_diff)


@pytest.mark.parametrize(("h1", "h2"), _boundary_cases())
def test_shortest_difference_preserves_exact_binary64_side_and_fastmath_parity(h1: float, h2: float) -> None:
    exact = _exact_shortest_difference(h1, h2)
    result = heading_diff(h1, h2)
    assert -180.0 <= result < 180.0
    if exact == 0:
        assert result == 0.0
    elif exact > 0:
        assert result > 0.0
    else:
        assert result < 0.0
    assert abs(Fraction.from_float(result) - exact) <= Fraction.from_float(math.ulp(float(exact)))
    assert _fastmath_caller(h1, h2) == result
    assert heading_diff(h1, h2, True) == abs(result)
    assert _fastmath_caller(h1, h2, True) == abs(result)
    motion = HeadingMotion(
        heading0_deg=h1,
        target_heading_deg=h2,
        turn_rate_deg_s=3.0,
        speed_min_mps=100.0,
        speed_max_mps=100.0,
    )
    assert _turn_angle_deg(motion) == result


@pytest.mark.parametrize(
    ("h1", "h2"),
    [
        (0.0, math.nextafter(180.0, 0.0)),
        (0.0, math.nextafter(180.0, math.inf)),
        (0.1, 180.1),
        (0.1, math.nextafter(180.1, math.inf)),
        (0.0, 180.0),
        (0.0, -180.0),
    ],
)
def test_half_turn_sides_and_ties_are_explicit(h1: float, h2: float) -> None:
    exact = _exact_shortest_difference(h1, h2)
    result = heading_diff(h1, h2)
    if abs(exact) == 180:
        assert result == -180.0
    elif exact > 0:
        assert result > 0.0
    else:
        assert result < 0.0
    if h1 == 0.1 and h2 == 180.1:
        # Rounded subtraction alone gives 180 despite a strictly smaller exact difference.
        assert h2 - h1 == 180.0
        assert 0 < exact < 180
        assert result == math.nextafter(180.0, 0.0)


def _independent_displacement(h1: float, h2: float, time_s: float) -> np.ndarray:
    exact = _exact_shortest_difference(h1, h2)
    signed_rate = 3.0 if exact > 0 else -3.0
    total_sweep = float(abs(exact))

    def angle(time: float) -> float:
        progress = min(3.0 * time, total_sweep)
        return math.radians(h1 + math.copysign(progress, signed_rate))

    east = quad(lambda time: 100.0 * math.sin(angle(time)), 0.0, time_s, epsabs=1e-9, epsrel=1e-12)[0]
    north = quad(lambda time: 100.0 * math.cos(angle(time)), 0.0, time_s, epsabs=1e-9, epsrel=1e-12)[0]
    return np.array([east, north])


@pytest.mark.parametrize(
    ("h1", "h2"),
    [
        (0.0, math.nextafter(180.0, 0.0)),
        (0.0, math.nextafter(180.0, math.inf)),
        (0.1, 180.1),
        (0.1, math.nextafter(180.1, math.inf)),
    ],
)
def test_actual_turn_basis_preserves_residual_sensitive_direction(h1: float, h2: float) -> None:
    for time_s in (30.0, 60.0):
        reference = _independent_displacement(h1, h2, time_s)
        basis_position = 100.0 * _turn_displacement_basis(h1, h2, 3.0, time_s)
        np.testing.assert_allclose(basis_position, reference, rtol=1e-13, atol=1e-8)


@pytest.mark.parametrize(
    ("h1", "h2"),
    [
        (0.0, math.nextafter(180.0, 0.0)),
        (0.1, 180.1),
        (0.0, math.nextafter(180.0, math.inf)),
    ],
)
def test_half_turn_collision_is_not_certified_by_core_or_heading_adapter(h1: float, h2: float) -> None:
    witness = _independent_displacement(h1, h2, 60.0)
    relative_position = -witness
    safe, distance, _ = _catch_up_projection_interval_with_turns_local(
        rel_pos0=relative_position,
        a_heading0_deg=h1,
        a_target_heading_deg=h2,
        a_speed_kt=100.0 / KT_TO_MPS,
        a_turn_rate_deg_sec=3.0,
        b_heading0_deg=0.0,
        b_target_heading_deg=0.0,
        b_speed_kt=0.0,
        b_turn_rate_deg_sec=0.0,
        separation_threshold_m=1000.0,
        speed_diff_kt=0.0,
        projection_time_s=60.0,
    )
    assert not safe
    assert distance < 1e-6
    a = HeadingMotion(
        heading0_deg=h1,
        target_heading_deg=h2,
        turn_rate_deg_s=3.0,
        speed_min_mps=100.0,
        speed_max_mps=100.0,
    )
    b = HeadingMotion(
        heading0_deg=0.0,
        target_heading_deg=0.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=0.0,
        speed_max_mps=0.0,
    )
    result = certify_heading_separation(
        (float(relative_position[0]), float(relative_position[1])),
        a,
        b,
        separation_threshold_m=1000.0,
        horizon_s=60.0,
    )
    assert result.status == "unresolved"
    assert result.base_min_distance_m is not None
    assert result.base_min_distance_m < 1e-6


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_heading_difference_returns_nan(value: float) -> None:
    assert math.isnan(heading_diff(value, 0.0))
    assert math.isnan(heading_diff(0.0, value))
