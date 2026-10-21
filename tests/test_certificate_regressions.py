"""End-to-end regressions for degenerate geometry and vertical shortcuts."""

from fractions import Fraction
import math
from typing import Literal

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
import pytest

from geometric_safety import (
    HeadingMotion,
    catch_up_projection_interval,
    catch_up_projection_interval_with_turns,
    catch_up_projection_interval_with_turns_and_vertical,
    catch_up_projection_interval_with_vertical,
    certify_heading_separation,
    certify_heading_separation_with_vertical,
)
from geometric_safety.relevant_aircraft import _turn_displacement_basis
from geometric_safety.util import EARTH_RADIUS_IN_METERS, KT_TO_MPS
from geometric_safety.vertical import time_to_vertical_overlap_resolution
from paper.turn_numerical_reference import basis


def _motion(heading: float, lower: float, upper: float, target: float | None = None) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=heading,
        target_heading_deg=heading if target is None else target,
        turn_rate_deg_s=0.0 if target is None else 3.0,
        speed_min_mps=lower,
        speed_max_mps=upper,
    )


@pytest.mark.parametrize("heading", [0.0, 31.0, 45.0, 90.0, 98.65741990093858, 180.0, 270.0, 359.0])
def test_in_trail_collision_is_not_certified(heading: float) -> None:
    direction = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
    relative = -200_000.0 * direction
    # Constant speeds 250 and 160 m/s are admissible and collide within the horizon.
    witness = relative + (250.0 - 160.0) * (200_000.0 / 90.0) * direction
    assert np.linalg.norm(witness) < 1e-8
    result = certify_heading_separation(
        (float(relative[0]), float(relative[1])),
        _motion(heading, 250.0, 250.0),
        _motion(heading, 160.0, 210.0),
        separation_threshold_m=9260.0,
        horizon_s=3600.0,
    )
    assert result.status == "unresolved"


def test_stationary_partner_collision_is_not_certified() -> None:
    heading = 273.74352459141346
    relative = np.array([41361.43767611355, -1022.0754562165887])
    velocity = 200.0 * np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
    time = -float(relative @ velocity) / float(velocity @ velocity)
    assert 0 < time < 600
    assert np.linalg.norm(relative + time * velocity) < 1700.0
    result = certify_heading_separation(
        (float(relative[0]), float(relative[1])),
        _motion(heading, 180.06897589739222, 220.06897589739222),
        _motion(0.0, 0.0, 0.0),
        separation_threshold_m=9260.0,
        horizon_s=600.0,
    )
    assert result.status == "unresolved"


def test_identical_turn_conflict_is_not_certified() -> None:
    heading, target = 166.9767696300122, 164.75187810332784
    initial, final = math.radians(heading), math.radians(target)
    relative = -9546.637684281503 * np.array([math.sin(initial), math.cos(initial)])
    omega = -math.radians(3.0)
    duration = (final - initial) / omega
    basis = np.array(
        [
            (math.cos(initial) - math.cos(final)) / omega + (600.0 - duration) * math.sin(final),
            (math.sin(final) - math.sin(initial)) / omega + (600.0 - duration) * math.cos(final),
        ]
    )
    speed_difference = -float(relative @ basis) / float(basis @ basis)
    assert 0.0 < speed_difference < 205.50164437959685 - 184.92386660181907
    assert np.linalg.norm(relative + speed_difference * basis) < 400.0
    aircraft = _motion(heading, 184.92386660181907, 205.50164437959685, target)
    result = certify_heading_separation(
        (float(relative[0]), float(relative[1])),
        aircraft,
        aircraft,
        separation_threshold_m=1000.0,
        horizon_s=600.0,
    )
    assert result.status == "unresolved"


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("aircraft", ["a", "b"])
def test_legacy_straight_rejects_nonfinite_headings(invalid: float, aircraft: str) -> None:
    result = catch_up_projection_interval(
        51.0,
        -1.0,
        invalid if aircraft == "a" else 0.0,
        400.0,
        51.0,
        -1.0,
        invalid if aircraft == "b" else 180.0,
        400.0,
        9260.0,
        20.0,
        300.0,
        rel_pos_override=np.array([0.0, -18520.0]),
    )
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


@pytest.mark.parametrize("latitude", [0.0, 51.0, 85.0])
@pytest.mark.parametrize("longitude", [179.95, -179.95])
def test_legacy_date_line_uses_local_distance(latitude: float, longitude: float) -> None:
    # At t=0 the result must use the wrapped 0.1-degree displacement.
    expected = EARTH_RADIUS_IN_METERS * math.cos(math.radians(latitude)) * math.radians(0.1)
    result = catch_up_projection_interval(
        latitude,
        longitude,
        90.0,
        400.0,
        latitude,
        -longitude,
        270.0,
        400.0,
        9260.0,
        20.0,
        0.0,
    )
    assert result[1] == pytest.approx(expected, abs=1e-6)
    assert result[0] == (expected > 9260.0)


def test_coheading_endpoint_projection_keeps_finite_diagnostics() -> None:
    result = catch_up_projection_interval(
        51.0,
        -1.0,
        0.0,
        380.0,
        51.0,
        -1.0,
        0.0,
        400.0,
        9260.0,
        25.0,
        300.0,
        rel_pos_override=np.array([-6.0 * 1852.0, -20.0 * 1852.0]),
    )
    expected = math.hypot(6.0 * 1852.0, 20.0 * 1852.0 - 30.0 * KT_TO_MPS * 300.0)
    assert result[0]
    assert result[1] == pytest.approx(expected, abs=1e-8)
    assert result[2] == 300.0


def test_straight_boundary_uses_the_numerical_guard() -> None:
    result = certify_heading_separation(
        (9260.0 - 1e-10, 1.0e6),
        _motion(180.0, 1000.0, 1000.0),
        _motion(0.0, 0.0, 0.0),
        separation_threshold_m=9260.0,
        horizon_s=2000.0,
    )
    assert result.status == "unresolved"


type EnvelopeField = Literal[
    "eta_const_m", "eta_rate_mps", "turn_speed_schedule_uncertainty_kt", "turn_delay_max_s", "turn_rate_error_max"
]


@pytest.mark.parametrize(
    "field",
    ["eta_const_m", "eta_rate_mps", "turn_speed_schedule_uncertainty_kt", "turn_delay_max_s", "turn_rate_error_max"],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_legacy_turn_envelopes_require_finite_nonnegative_bounds(field: EnvelopeField, value: float) -> None:
    with pytest.raises(ValueError, match="envelope"):
        catch_up_projection_interval_with_turns(
            51.0,
            -1.0,
            0.0,
            90.0,
            250.0,
            3.0,
            51.1,
            -1.0,
            90.0,
            90.0,
            250.0,
            0.0,
            5556.0,
            10.0,
            120.0,
            eta_const_m=value if field == "eta_const_m" else 0.0,
            eta_rate_mps=value if field == "eta_rate_mps" else 0.0,
            turn_speed_schedule_uncertainty_kt=value if field == "turn_speed_schedule_uncertainty_kt" else 0.0,
            turn_delay_max_s=value if field == "turn_delay_max_s" else 0.0,
            turn_rate_error_max=value if field == "turn_rate_error_max" else 0.0,
        )


@pytest.mark.parametrize("error", [1.0, 1.5, 10.0])
def test_legacy_turn_rate_error_must_be_below_one(error: float) -> None:
    with pytest.raises(ValueError, match="envelope"):
        catch_up_projection_interval_with_turns(
            51.0,
            -1.0,
            0.0,
            90.0,
            250.0,
            3.0,
            51.1,
            -1.0,
            90.0,
            90.0,
            250.0,
            0.0,
            5556.0,
            10.0,
            120.0,
            turn_rate_error_max=error,
        )


def _exact_gap(levels: tuple[float, float, float, float], rate: float, time: float) -> Fraction:
    progress = Fraction(rate) * Fraction(time) / 6000
    bands: list[tuple[Fraction, Fraction]] = []
    for current, selected in ((levels[0], levels[1]), (levels[2], levels[3])):
        start, end = Fraction(current), Fraction(selected)
        step = min(abs(end - start), progress)
        level = start + step if end >= start else start - step
        bands.append((min(level, end), max(level, end)))
    return max(bands[1][0] - bands[0][1], bands[0][0] - bands[1][1])


@pytest.mark.parametrize(
    ("levels", "rate", "gap", "expected"),
    [
        ((60.0, -42.5, -9.5, 15.5), 999.0, 20.0, 420.42042042042044),
        ((40.0, 89.24237309867549, 158.0022223624343, 56.08867843926494), 999.0, 20.0, 588.6019361107166),
        ((31.57641402215304, 139.5, 349.5, 64.5), 500.0, 5.0, 2754.0),
    ],
)
def test_vertical_leveloff_does_not_create_a_spurious_event_stream(
    levels: tuple[float, float, float, float],
    rate: float,
    gap: float,
    expected: float,
) -> None:
    time = time_to_vertical_overlap_resolution(*levels, rate, gap, 10)
    assert time == pytest.approx(expected, abs=1e-7)
    assert _exact_gap(levels, rate, time) >= Fraction(gap)


@pytest.mark.parametrize("turning", [False, True])
def test_legacy_vertical_shortcut_cannot_hide_a_head_on_conflict(turning: bool) -> None:
    longitude = -1.0 + math.degrees(101852.0 / (EARTH_RADIUS_IN_METERS * math.cos(math.radians(51.0))))
    if turning:
        result = catch_up_projection_interval_with_turns_and_vertical(
            51.0,
            -1.0,
            90.0,
            90.0,
            250.0,
            0.0,
            60.0,
            -42.5,
            51.0,
            longitude,
            270.0,
            270.0,
            250.0,
            0.0,
            -9.5,
            15.5,
            9260.0,
            0.0,
            1000.0,
            vertical_rate_fpm=999.0,
            required_vertical_gap_fl=20.0,
        )
    else:
        result = catch_up_projection_interval_with_vertical(
            51.0,
            -1.0,
            90.0,
            250.0,
            60.0,
            -42.5,
            51.0,
            longitude,
            270.0,
            250.0,
            -9.5,
            15.5,
            9260.0,
            0.0,
            1000.0,
            vertical_rate_fpm=999.0,
            required_vertical_gap_fl=20.0,
        )
    assert not result[0]
    assert result[3] > 420.0


def test_already_separated_legacy_bands_do_not_evaluate_an_unused_turn() -> None:
    result = catch_up_projection_interval_with_turns_and_vertical(
        51.0,
        -1.0,
        0.0,
        90.0,
        300.0,
        0.0,
        300.0,
        300.0,
        51.0,
        -0.99,
        270.0,
        270.0,
        300.0,
        0.0,
        350.0,
        350.0,
        9260.0,
        20.0,
        600.0,
    )
    assert result[0]
    assert result[1] == pytest.approx(700.56, abs=0.1)
    assert result[2:] == (0.0, 0.0)


def test_unrounded_vertical_resolution_reaches_the_exact_verifier() -> None:
    levels = (110.0, 106.37, 127.85968150464853, 140.0)
    result = certify_heading_separation_with_vertical(
        (2000.0, 0.0),
        _motion(270.0, 50.0, 50.0),
        _motion(90.0, 50.0, 50.0),
        a_current_fl=levels[0],
        a_selected_fl=levels[1],
        b_current_fl=levels[2],
        b_selected_fl=levels[3],
        minimum_vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=20.0,
        vertical_rounding_fl=0,
        separation_threshold_m=1000.0,
        horizon_s=60.0,
    )
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s is not None
    assert result.vertical_resolution_time_s == pytest.approx(6.420955486054396)
    assert _exact_gap(levels, 1000.0, result.vertical_resolution_time_s) >= 20


def _reference_displacement(heading: float, target: float, speed: float, time: float) -> NDArray[np.float64]:
    """Use the independent antiderivative reference at 80 decimal digits."""
    with mp.workdps(80):
        displacement = basis(mp.mpf(heading), mp.mpf(target), mp.mpf(3), mp.mpf(time))
        return np.array([float(mp.mpf(speed) * coordinate) for coordinate in displacement])


@pytest.mark.parametrize(
    "family", ["parallel", "reciprocal", "stationary", "identical_turn", "turn_straight", "two_turns"]
)
def test_constructed_collision_families_are_never_certified(family: str) -> None:
    rng = np.random.default_rng(202610203)
    for _ in range(200):
        heading = float(rng.uniform(0.0, 360.0))
        angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 150.0))
        target = heading + angle if family in ("identical_turn", "turn_straight", "two_turns") else heading
        b_heading = heading + (180.0 if family == "reciprocal" else 0.0)
        if family == "two_turns":
            b_heading += 70.0
        b_target = b_heading + angle if family in ("identical_turn", "two_turns") else b_heading
        a_speed, b_speed = 250.0, 150.0
        if family == "stationary":
            b_speed = 0.0
        a = _motion(heading, a_speed, a_speed, target if target != heading else None)
        b = _motion(
            b_heading,
            b_speed,
            b_speed,
            b_target if b_target != b_heading else None,
        )
        time = float(rng.uniform(0.1, 599.9))
        a_displacement = _reference_displacement(heading, target, a_speed, time)
        b_displacement = _reference_displacement(b_heading, b_target, b_speed, time)
        relative = b_displacement - a_displacement
        actual_a = a_speed * _turn_displacement_basis(heading, target, 3.0, time)
        actual_b = b_speed * _turn_displacement_basis(b_heading, b_target, 3.0, time)
        np.testing.assert_allclose(actual_a, a_displacement, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(actual_b, b_displacement, rtol=0.0, atol=1e-7)
        assert np.linalg.norm(relative + actual_a - actual_b) < 1e-7
        result = certify_heading_separation(
            (float(relative[0]), float(relative[1])),
            a,
            b,
            separation_threshold_m=1.0,
            horizon_s=600.0,
        )
        assert result.status == "unresolved", (family, heading, target, b_heading, b_target, time, result)


def test_parallel_safe_controls_certify_at_every_integer_heading() -> None:
    for heading in range(360):
        radians = math.radians(heading)
        direction = np.array([math.sin(radians), math.cos(radians)])
        normal = np.array([direction[1], -direction[0]])
        relative = 12_000.0 * normal - 40_000.0 * direction
        result = certify_heading_separation(
            (float(relative[0]), float(relative[1])),
            _motion(float(heading), 200.0, 250.0),
            _motion(float(heading), 170.0, 220.0),
            separation_threshold_m=9260.0,
            horizon_s=600.0,
        )
        # The normal component is constant for every admissible speed profile.
        assert result.status == "certified_safe", (heading, result)
        assert result.base_min_distance_m is not None
        assert result.base_min_distance_m >= 12_000.0 - 1e-7
