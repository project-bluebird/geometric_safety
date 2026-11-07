"""Execute the guide examples and protect their application decisions."""

from collections.abc import Callable
from dataclasses import replace
import math
from pathlib import Path
import re

import pytest

from examples import heading_filter, uncertainty_bounds, vertical_progress
from geometric_safety import CombinedHeadingAssuranceResult, HeadingAssuranceResult, HeadingMotion


@pytest.mark.parametrize("run", [heading_filter.main, uncertainty_bounds.main, vertical_progress.main])
def test_documented_output_comes_from_real_example(run: Callable[[], None], capsys: pytest.CaptureFixture[str]) -> None:
    run()
    output = capsys.readouterr()
    guide = (Path(__file__).resolve().parents[1] / "docs" / "usage-guide.md").read_text(encoding="utf-8")
    assert output.out
    assert f"```text\n{output.out}```" in guide
    assert output.err == ""


@pytest.mark.parametrize("section_number", [1, 2, 3])
def test_guide_code_produces_documented_results(section_number: int, capsys: pytest.CaptureFixture[str]) -> None:
    guide = (Path(__file__).resolve().parents[1] / "docs" / "usage-guide.md").read_text(encoding="utf-8")
    section = next(
        section for section in re.split(r"^## ", guide, flags=re.MULTILINE) if section.startswith(f"{section_number}. ")
    )
    blocks = re.findall(r"^```python\n(.*?)^```$", section, flags=re.MULTILINE | re.DOTALL)
    assert blocks
    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    for index, block in enumerate(blocks, start=1):
        exec(compile(block, f"usage-guide-example-{section_number}-block-{index}", "exec"), namespace)
    output = capsys.readouterr()
    assert output.out
    assert f"```text\n{output.out}```" in section
    assert output.err == ""
    if section_number == 3:
        combined = namespace["combined"]
        assert isinstance(combined, CombinedHeadingAssuranceResult)
        assert combined.lateral_result is not None
        # At 210 kt each, the pair closes by 12,964 m over the 60-second lateral horizon.
        assert combined.lateral_result.base_min_distance_m == pytest.approx(47036.0)


def test_candidate_filter_requires_acceptance_from_every_supplied_pair() -> None:
    assessments = heading_filter.run_example()
    expected_headings = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    assert tuple(assessments) == expected_headings
    assert {
        heading: {pair: result.status for pair, result in pairs.items()} for heading, pairs in assessments.items()
    } == {
        heading: {
            "opposing": "certified_safe" if heading <= 40 else "unresolved",
            "parallel": "certified_safe",
        }
        for heading in expected_headings
    }
    assert heading_filter.passing_candidates(assessments) == (0, 10, 20, 30, 40)
    assert heading_filter.passing_candidates({0.0: {}}) == ()


def test_custom_headings_are_assessed_as_absolute_targets() -> None:
    ownship, traffic = heading_filter.example_snapshot()
    assessments = heading_filter.assess_candidates(
        ownship,
        traffic,
        target_headings_deg=(65.0, 35.5, 90.0),
        separation_threshold_m=9260.0,
        horizon_s=180.0,
    )
    assert tuple(assessments) == (65.0, 35.5, 90.0)
    assert heading_filter.passing_candidates(assessments) == (35.5,)


def test_second_traffic_aircraft_can_withhold_a_candidate() -> None:
    ownship, traffic = heading_filter.example_snapshot()
    ownship = replace(ownship, motion=replace(ownship.motion, target_heading_deg=0.0))
    traffic["parallel"] = replace(
        traffic["parallel"], latitude_deg=ownship.latitude_deg, longitude_deg=ownship.longitude_deg
    )
    pairs = heading_filter.assess_traffic(ownship, traffic, separation_threshold_m=9260.0, horizon_s=180.0)
    assert pairs["opposing"].status == "certified_safe"
    assert pairs["parallel"].status == "unresolved"
    assert heading_filter.passing_candidates({0.0: pairs}) == ()


def test_turn_path_is_checked_before_reaching_the_target_heading() -> None:
    speed_mps = 200.0 * 1852.0 / 3600.0
    turning = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=0.0,
        turn_rate_deg_s=3.0,
        speed_min_mps=speed_mps,
        speed_max_mps=speed_mps,
    )
    stationary = HeadingMotion(
        heading0_deg=0.0, target_heading_deg=0.0, turn_rate_deg_s=0.0, speed_min_mps=0.0, speed_max_mps=0.0
    )
    ownship = heading_filter.AircraftSnapshot(0.0, 51.0, -1.0, turning)
    traffic = {"near_turn": heading_filter.AircraftSnapshot(0.0, 51.0054, -0.97855, stationary)}
    # Independent circular-arc position at 15 s, in the geographic API's plane.
    omega = math.radians(3.0)
    east = speed_mps / omega * math.sin(omega * 15.0)
    north = speed_mps / omega * (1.0 - math.cos(omega * 15.0))
    traffic_east = 6378137.0 * math.radians(0.02145) * math.cos(math.radians(51.0))
    traffic_north = 6378137.0 * math.radians(0.0054)
    assert math.hypot(east - traffic_east, north - traffic_north) < 500.0
    candidates = heading_filter.assess_candidates(
        ownship, traffic, target_headings_deg=(0.0,), separation_threshold_m=500.0, horizon_s=60.0
    )
    assert candidates[0.0]["near_turn"].status == "unresolved"
    assert 0.0 not in heading_filter.passing_candidates(candidates)
    instantaneous = replace(ownship, motion=replace(turning, heading0_deg=0.0))
    other_model = heading_filter.assess_traffic(instantaneous, traffic, separation_threshold_m=500.0, horizon_s=60.0)
    assert other_model["near_turn"].status == "certified_safe"


@pytest.mark.parametrize("timestamp_s", [999.0, float("nan"), float("inf")])
def test_unaligned_traffic_is_not_silently_reused(timestamp_s: float) -> None:
    ownship, traffic = heading_filter.example_snapshot()
    traffic["opposing"] = replace(traffic["opposing"], timestamp_s=timestamp_s)
    with pytest.raises(ValueError, match="same finite starting time"):
        heading_filter.assess_traffic(ownship, traffic, separation_threshold_m=9260.0, horizon_s=180.0)


@pytest.mark.parametrize("horizon_s", [-1.0, float("nan")])
def test_invalid_query_is_reported_to_the_caller(horizon_s: float) -> None:
    ownship, traffic = heading_filter.example_snapshot()
    with pytest.raises(ValueError, match="horizon_s"):
        heading_filter.assess_traffic(ownship, traffic, separation_threshold_m=9260.0, horizon_s=horizon_s)


def test_relative_error_budget_changes_the_real_separation_decision() -> None:
    results = uncertainty_bounds.run_example()
    bounded = results["bounded_errors"]
    wider = results["wider_position_bounds"]
    assert bounded.status == "certified_safe"
    assert wider.status == "unresolved"
    assert bounded.inflation_m == pytest.approx(323.4666666666667)
    assert wider.inflation_m == pytest.approx(923.4666666666667)
    assert bounded.base_min_distance_m == pytest.approx(10000.0)
    # Both aircraft can hold 200 kt, keeping the 10 km gap, so only the error allowance decides.
    assert 10000.0 - wider.inflation_m < 5.0 * 1852.0 < 10000.0 - bounded.inflation_m


def test_vertical_shortening_and_lateral_fallback_have_different_requirements() -> None:
    combined = vertical_progress.assess_encounter(1000.0)
    assert isinstance(combined, CombinedHeadingAssuranceResult)
    assert combined.status == "certified_safe"
    assert combined.vertical_resolution_time_s == pytest.approx(60.0)
    assert combined.lateral_horizon_s == pytest.approx(60.0)
    assert combined.lateral_result is not None
    # Even the fastest permitted pair remains over 47 km apart at vertical resolution.
    maximum_speed_mps = 210.0 * 1852.0 / 3600.0
    assert combined.lateral_result.base_min_distance_m == pytest.approx(60000.0 - 2.0 * maximum_speed_mps * 60.0)
    lateral = vertical_progress.assess_encounter(None)
    assert isinstance(lateral, HeadingAssuranceResult)
    assert lateral.status == "unresolved"
    # Some constant-speed pair within the bounds meets inside the full horizon.
    assert lateral.base_min_distance_m == pytest.approx(0.0, abs=1e-6)
    assert lateral.base_closest_time_s is not None
    assert 0.0 < lateral.base_closest_time_s < vertical_progress.HORIZON_S
    zero_rate = vertical_progress.assess_encounter(0.0)
    assert isinstance(zero_rate, CombinedHeadingAssuranceResult)
    assert zero_rate.vertical_resolution_time_s is None
    assert zero_rate.lateral_horizon_s == 600.0
    assert zero_rate.status == "unresolved"
