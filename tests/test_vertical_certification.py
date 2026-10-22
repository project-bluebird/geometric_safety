"""Vertical shortcuts must preserve the requested physical gap."""

from decimal import Decimal, localcontext
from fractions import Fraction
import random

import pytest

from geometric_safety import HeadingMotion, certify_heading_separation_with_vertical


def _straight(heading: float, speed: float = 0.5) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=heading,
        target_heading_deg=heading,
        turn_rate_deg_s=0.0,
        speed_min_mps=speed,
        speed_max_mps=speed,
    )


def test_large_absolute_levels_cannot_hide_a_joint_separation_loss() -> None:
    offset = float(2**51)
    early_time = 841.2857142857143
    initial_distance = 1000.0 + early_time + 0.01
    result = certify_heading_separation_with_vertical(
        (initial_distance, 0.0),
        _straight(270.0),
        _straight(90.0),
        a_current_fl=offset + 100.0,
        a_selected_fl=offset + 120.0,
        b_current_fl=offset - 10.25,
        b_selected_fl=offset + 150.0,
        minimum_vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=0,
        separation_threshold_m=1000.0,
        horizon_s=early_time + 1.0,
    )
    # A has levelled at FL120 relative to the common offset. B reaches FL130
    # at 841.5 s. Before then both declared separation conditions can fail.
    witness_time = Fraction.from_float(early_time + 0.1)
    assert Fraction.from_float(initial_distance) - witness_time < 1000
    assert Fraction(-41, 4) + witness_time / 6 - 120 < 10
    assert result.status == "unresolved"
    if result.vertical_resolution_time_s is not None:
        assert Fraction.from_float(result.vertical_resolution_time_s) >= Fraction(1683, 2)


def test_grid_tolerance_cannot_replace_the_requested_vertical_gap() -> None:
    result = certify_heading_separation_with_vertical(
        (0.0, 0.0),
        _straight(0.0, 0.0),
        _straight(0.0, 0.0),
        a_current_fl=100.0 + 5e-10,
        a_selected_fl=100.0 + 5e-10,
        b_current_fl=110.0,
        b_selected_fl=110.0,
        minimum_vertical_rate_fpm=0.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=10,
        separation_threshold_m=1000.0,
        horizon_s=60.0,
    )
    assert result.status == "unresolved"
    assert result.vertical_resolution_time_s is None
    assert result.lateral_horizon_s == 60.0


@pytest.mark.parametrize("rate", [999.0, 999.125, 1000.0, 1001.0, 1800.0])
@pytest.mark.parametrize("required_gap", [10.0, 10.125])
def test_nonrepresentable_boundary_is_rounded_up_without_losing_the_shortcut(rate: float, required_gap: float) -> None:
    result = certify_heading_separation_with_vertical(
        (60_000.0, 0.0),
        _straight(270.0),
        _straight(90.0),
        a_current_fl=100.0,
        a_selected_fl=100.0,
        b_current_fl=100.0,
        b_selected_fl=120.0,
        minimum_vertical_rate_fpm=rate,
        required_vertical_gap_fl=required_gap,
        vertical_rounding_fl=0,
        separation_threshold_m=1000.0,
        horizon_s=600.0,
    )
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s is not None
    exact_time = Fraction(required_gap) * 6000 / Fraction(rate)
    assert Fraction(result.vertical_resolution_time_s) >= exact_time
    assert result.vertical_resolution_time_s == pytest.approx(float(exact_time))
    assert result.lateral_horizon_s < 600.0


@pytest.mark.parametrize("offset", [0.0, float(2**20), float(2**50), float(2**51)])
@pytest.mark.parametrize("swapped", [False, True])
def test_safe_resolution_survives_translation_and_aircraft_reordering(offset: float, swapped: bool) -> None:
    a_levels = (offset + 100.0, offset + 120.0)
    b_levels = (offset - 10.25, offset + 150.0)
    if swapped:
        a_levels, b_levels = b_levels, a_levels
    result = certify_heading_separation_with_vertical(
        (60_000.0, 0.0),
        _straight(0.0),
        _straight(0.0),
        a_current_fl=a_levels[0],
        a_selected_fl=a_levels[1],
        b_current_fl=b_levels[0],
        b_selected_fl=b_levels[1],
        minimum_vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=0,
        separation_threshold_m=1000.0,
        horizon_s=1000.0,
    )
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s is not None
    assert Fraction(result.vertical_resolution_time_s) >= Fraction(1683, 2)
    assert result.vertical_resolution_time_s == pytest.approx(841.5)


def test_grid_tolerance_at_both_moving_edges_still_requires_initial_lateral_separation() -> None:
    a_start, b_start = 100.0 + 5e-10, 110.0 - 5e-10
    result = certify_heading_separation_with_vertical(
        (0.0, 0.0),
        _straight(0.0, 0.0),
        _straight(0.0, 0.0),
        a_current_fl=a_start,
        a_selected_fl=80.0,
        b_current_fl=b_start,
        b_selected_fl=130.0,
        minimum_vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=10,
        separation_threshold_m=1000.0,
        horizon_s=60.0,
    )
    exact_first_time = (10 - (Fraction(b_start) - Fraction(a_start))) * 3
    assert result.status == "unresolved"
    assert result.vertical_resolution_time_s is not None
    assert Fraction(result.vertical_resolution_time_s) >= exact_first_time > 0
    assert result.lateral_result is not None


def test_returned_boundaries_enclose_an_independent_high_precision_model() -> None:
    rng = random.Random(20261015)
    verified = 0
    motion = _straight(0.0, 0.0)
    for _ in range(500):
        offset = rng.choice([0.0, float(2**20), float(2**50), float(2**51)])
        levels = tuple(offset + rng.uniform(-400.0, 400.0) for _ in range(4))
        rate = rng.choice([1e-200, 999.0, 1000.0, 1001.0, 1e200])
        required = rng.choice([0.0, 0.1, 0.5, 10.0, 10.125, 15.0])
        result = certify_heading_separation_with_vertical(
            (60_000.0, 0.0),
            motion,
            motion,
            a_current_fl=levels[0],
            a_selected_fl=levels[1],
            b_current_fl=levels[2],
            b_selected_fl=levels[3],
            minimum_vertical_rate_fpm=rate,
            required_vertical_gap_fl=required,
            vertical_rounding_fl=rng.choice([0, 1, 10]),
            separation_threshold_m=1000.0,
            horizon_s=0.0,
        )
        resolution = result.vertical_resolution_time_s
        if resolution is None:
            continue
        with localcontext() as context:
            context.prec = 1200
            distance = Decimal.from_float(rate) * Decimal.from_float(resolution) / 6000
            bands: list[tuple[Decimal, Decimal]] = []
            for current_fl, selected_fl in (levels[:2], levels[2:]):
                current, selected = Decimal.from_float(current_fl), Decimal.from_float(selected_fl)
                displacement = min(abs(selected - current), distance)
                level = current + displacement if selected >= current else current - displacement
                bands.append((min(level, selected), max(level, selected)))
            gap = max(bands[0][0] - bands[1][1], bands[1][0] - bands[0][1])
            assert gap >= Decimal.from_float(required), (levels, rate, required, resolution, gap)
        verified += 1
    assert verified > 100
