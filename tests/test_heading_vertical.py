"""Combined certificates, with independent vertical profiles and collision witnesses."""

from dataclasses import replace
import math
from typing import Literal, TypedDict

import numpy as np
import pytest

from geometric_safety import (
    CombinedHeadingAssuranceResult,
    HeadingMotion,
    certify_heading_separation,
    certify_heading_separation_with_vertical,
)


class _Query(TypedDict):
    a_current_fl: float
    a_selected_fl: float
    b_current_fl: float
    b_selected_fl: float
    minimum_vertical_rate_fpm: float
    separation_threshold_m: float
    horizon_s: float
    required_vertical_gap_fl: float
    vertical_rounding_fl: int


def _query() -> _Query:
    return {
        "a_current_fl": 100.0,
        "a_selected_fl": 100.0,
        "b_current_fl": 100.0,
        "b_selected_fl": 120.0,
        "minimum_vertical_rate_fpm": 1000.0,
        "separation_threshold_m": 5.0 * 1852.0,
        "horizon_s": 600.0,
        "required_vertical_gap_fl": 10.0,
        "vertical_rounding_fl": 10,
    }


def _straight(heading: float) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=heading,
        target_heading_deg=heading,
        turn_rate_deg_s=0.0,
        speed_min_mps=95.0,
        speed_max_mps=105.0,
    )


def test_vertical_resolution_removes_a_later_horizontal_crossing() -> None:
    a, b = _straight(270.0), _straight(90.0)
    query = _query()
    full = certify_heading_separation(
        (60_000.0, 0.0), a, b, separation_threshold_m=query["separation_threshold_m"], horizon_s=query["horizon_s"]
    )
    assert full.status == "unresolved"
    result = certify_heading_separation_with_vertical((60_000.0, 0.0), a, b, **query)
    assert isinstance(result, CombinedHeadingAssuranceResult)
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s == pytest.approx(60.0)
    assert result.lateral_horizon_s == pytest.approx(60.0)
    assert result.lateral_result is not None
    assert result.lateral_result.status == "certified_safe"
    assert result.lateral_result.base_min_distance_m == pytest.approx(47_400.0)


def test_lateral_loss_before_vertical_resolution_is_not_accepted() -> None:
    result = certify_heading_separation_with_vertical((10_000.0, 0.0), _straight(270.0), _straight(90.0), **_query())
    assert result.status == "unresolved"
    assert result.vertical_resolution_time_s == pytest.approx(60.0)
    assert result.lateral_result is not None
    assert result.lateral_result.status == "unresolved"


@pytest.mark.parametrize("rate", [0.0, math.ulp(0.0), 1000.0, 1e308])
def test_already_separated_bands_need_no_supported_horizontal_law(rate: float) -> None:
    query = _query()
    query.update(b_current_fl=120.0, minimum_vertical_rate_fpm=rate)
    a = replace(_straight(270.0), heading_law="route_turn", turn_direction="right", turn_delay_s=10.0)
    result = certify_heading_separation_with_vertical((0.0, 0.0), a, _straight(90.0), **query)
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s == result.lateral_horizon_s == 0.0
    assert result.lateral_result is None


def test_unsupported_horizontal_law_is_reported_when_it_is_needed() -> None:
    a = replace(_straight(270.0), heading_law="route_turn")
    result = certify_heading_separation_with_vertical((60_000.0, 0.0), a, _straight(90.0), **_query())
    assert result.status == "unsupported"
    assert result.lateral_horizon_s == pytest.approx(60.0)
    assert result.lateral_result is not None
    assert result.lateral_result.status == "unsupported"


@pytest.mark.parametrize("horizon", [0.0, 30.0, 60.0, 120.0])
def test_lateral_horizon_is_capped_and_envelopes_use_that_horizon(horizon: float) -> None:
    query = _query()
    query["horizon_s"] = horizon
    a = replace(_straight(270.0), target_heading_deg=300.0, turn_rate_deg_s=1.0)
    b = _straight(90.0)
    result = certify_heading_separation_with_vertical(
        (60_000.0, 20_000.0), a, b, relative_position_error_m=20.0, relative_velocity_error_mps=2.0, **query
    )
    expected_horizon = min(horizon, 60.0)
    assert result.lateral_horizon_s == pytest.approx(expected_horizon)
    expected = certify_heading_separation(
        (60_000.0, 20_000.0),
        a,
        b,
        separation_threshold_m=query["separation_threshold_m"],
        horizon_s=result.lateral_horizon_s,
        relative_position_error_m=20.0,
        relative_velocity_error_mps=2.0,
    )
    assert result.lateral_result == expected
    assert result.status == expected.status


@pytest.mark.parametrize("reason", ["same_target", "zero_rate", "tiny_rate", "large_levels", "large_grid"])
def test_unusable_vertical_resolution_retains_the_full_lateral_check(reason: str) -> None:
    query = _query()
    if reason == "same_target":
        query["b_selected_fl"] = 100.0
    elif reason == "zero_rate":
        query["minimum_vertical_rate_fpm"] = 0.0
    elif reason == "tiny_rate":
        query["minimum_vertical_rate_fpm"] = math.ulp(0.0)
    elif reason == "large_levels":
        query.update(a_current_fl=1e100, a_selected_fl=1e100)
    else:
        query["vertical_rounding_fl"] = 2**62
    a, b = _straight(270.0), _straight(90.0)
    result = certify_heading_separation_with_vertical((60_000.0, 0.0), a, b, **query)
    assert result.vertical_resolution_time_s is None
    assert result.lateral_horizon_s == query["horizon_s"]
    expected = certify_heading_separation(
        (60_000.0, 0.0), a, b, separation_threshold_m=query["separation_threshold_m"], horizon_s=query["horizon_s"]
    )
    assert result.lateral_result == expected
    assert result.status == expected.status == "unresolved"


def test_rounding_and_fractional_flight_levels_are_preserved() -> None:
    query = _query()
    query.update(a_current_fl=127.0, a_selected_fl=127.0, b_current_fl=127.0, b_selected_fl=147.0)
    rounded = certify_heading_separation_with_vertical((60_000.0, 0.0), _straight(270.0), _straight(90.0), **query)
    query["vertical_rounding_fl"] = 0
    exact = certify_heading_separation_with_vertical((60_000.0, 0.0), _straight(270.0), _straight(90.0), **query)
    assert rounded.vertical_resolution_time_s == pytest.approx(78.0)
    assert exact.vertical_resolution_time_s == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("levels", "expected_s"),
    [
        ((100.0, 100.0, 100.0, 120.0), 60.0),
        ((120.0, 120.0, 120.0, 100.0), 60.0),
        ((100.0, 140.0, 100.0, 60.0), 30.0),
        ((100.0, 140.0, 130.0, 160.0), 120.0),
        ((160.0, 130.0, 140.0, 100.0), 120.0),
        ((100.0, 200.0, 250.0, 150.0), 480.0),
    ],
)
def test_faster_independent_varying_vertical_rates_remain_separated(
    levels: tuple[float, float, float, float], expected_s: float
) -> None:
    query = _query()
    query.update(
        a_current_fl=levels[0],
        a_selected_fl=levels[1],
        b_current_fl=levels[2],
        b_selected_fl=levels[3],
        vertical_rounding_fl=0,
    )
    result = certify_heading_separation_with_vertical((200_000.0, 0.0), _straight(270.0), _straight(90.0), **query)
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s == pytest.approx(expected_s)

    def executed_level(current: float, selected: float, time: float, first: float, second: float) -> float:
        # Integrate an independent two-rate profile in ft/min, then level off.
        switch = 17.0
        distance_fl = (first * min(time, switch) + second * max(time - switch, 0.0)) / 6000.0
        return current + math.copysign(min(distance_fl, abs(selected - current)), selected - current)

    for a_rates, b_rates in (
        ((1000.0, 1000.0), (1000.0, 1000.0)),
        ((1000.0, 2300.0), (3500.0, 1100.0)),
        ((4200.0, 1700.0), (1000.0, 5000.0)),
    ):
        for time in np.linspace(expected_s, max(600.0, expected_s), 41):
            a_fl = executed_level(levels[0], levels[1], float(time), *a_rates)
            b_fl = executed_level(levels[2], levels[3], float(time), *b_rates)
            assert abs(a_fl - b_fl) >= 10.0 - 1e-12


def test_equal_minimum_rate_trajectories_are_not_used_as_vertical_separation() -> None:
    query = _query()
    query.update(a_selected_fl=140.0, b_current_fl=130.0, b_selected_fl=160.0)
    result = certify_heading_separation_with_vertical((20_000.0, 0.0), _straight(270.0), _straight(90.0), **query)
    assert result.status == "unresolved"
    assert result.vertical_resolution_time_s == pytest.approx(120.0)
    # At 60 s, A climbing at 6000 ft/min has levelled at FL140, while B at
    # 1000 ft/min reaches FL140. At 100 m/s each, their lateral gap is 8 km.
    a_fl = min(140.0, 100.0 + 6000.0 * 60.0 / 6000.0)
    b_fl = min(160.0, 130.0 + 1000.0 * 60.0 / 6000.0)
    assert a_fl == b_fl == 140.0
    lateral_gap_m = 20_000.0 - 2.0 * 100.0 * 60.0
    assert lateral_gap_m < query["separation_threshold_m"]


@pytest.mark.parametrize(("rate", "expected_s"), [(500.0, 120.0), (1000.0, 60.0), (2000.0, 30.0)])
def test_minimum_rate_controls_the_resolution_time(rate: float, expected_s: float) -> None:
    query = _query()
    query["minimum_vertical_rate_fpm"] = rate
    result = certify_heading_separation_with_vertical((60_000.0, 0.0), _straight(270.0), _straight(90.0), **query)
    assert result.vertical_resolution_time_s == pytest.approx(expected_s)
    assert result.lateral_horizon_s == pytest.approx(expected_s)


@pytest.mark.parametrize("field", ["a_current_fl", "a_selected_fl", "b_current_fl", "b_selected_fl"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_levels_raise_before_vertical_acceptance(
    field: Literal["a_current_fl", "a_selected_fl", "b_current_fl", "b_selected_fl"], invalid: float
) -> None:
    query = _query()
    query["b_current_fl"] = 120.0
    query[field] = invalid
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        certify_heading_separation_with_vertical((0.0, 0.0), _straight(270.0), _straight(90.0), **query)


@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf])
def test_invalid_minimum_rate_is_not_silently_made_positive(invalid: float) -> None:
    query = _query()
    query.update(b_current_fl=120.0, minimum_vertical_rate_fpm=invalid)
    with pytest.raises(ValueError, match="minimum_vertical_rate_fpm"):
        certify_heading_separation_with_vertical((0.0, 0.0), _straight(270.0), _straight(90.0), **query)


@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf])
def test_invalid_horizon_raises_before_vertical_acceptance(invalid: float) -> None:
    query = _query()
    query.update(b_current_fl=120.0, horizon_s=invalid)
    with pytest.raises(ValueError, match="horizon_s"):
        certify_heading_separation_with_vertical((0.0, 0.0), _straight(270.0), _straight(90.0), **query)


def test_invalid_relative_position_raises_before_vertical_acceptance() -> None:
    query = _query()
    query["b_current_fl"] = 120.0
    with pytest.raises(ValueError, match="relative_position_m"):
        certify_heading_separation_with_vertical((math.nan, 0.0), _straight(270.0), _straight(90.0), **query)


def test_negative_gap_and_rounding_are_rejected() -> None:
    query = _query()
    query["required_vertical_gap_fl"] = -1.0
    with pytest.raises(ValueError, match="required_vertical_gap_fl"):
        certify_heading_separation_with_vertical((0.0, 0.0), _straight(270.0), _straight(90.0), **query)
    query.update(required_vertical_gap_fl=10.0, vertical_rounding_fl=-1)
    with pytest.raises(ValueError, match="vertical_rounding_fl"):
        certify_heading_separation_with_vertical((0.0, 0.0), _straight(270.0), _straight(90.0), **query)
