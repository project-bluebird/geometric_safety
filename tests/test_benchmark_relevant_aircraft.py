"""Exercise benchmark instrumentation through real solver paths."""

from pathlib import Path
import subprocess
import sys

import pytest

from geometric_safety import relevant_aircraft as ra
from geometric_safety._instrumentation import python_function
from scripts.benchmark_relevant_aircraft import (
    build_isolated_scenarios,
    count_fixed_time_evals,
    local_xy_to_latlon,
    run_turn_aware,
)


@pytest.mark.parametrize(
    ("case", "expected_count"),
    [("straight", 0), ("initial_overlap", 0), ("endpoint", 1), ("midpoint_collision", 2)],
)
def test_benchmark_counts_fixed_time_samples(case: str, expected_count: int) -> None:
    kwargs = build_isolated_scenarios()[1].kwargs.copy()
    if case == "straight":
        kwargs["a_target_heading_deg"] = kwargs["a_heading0_deg"]
    elif case == "initial_overlap":
        kwargs["b_lat"], kwargs["b_lon"] = kwargs["a_lat"], kwargs["a_lon"]
    elif case == "endpoint":
        kwargs["projection_time_s"] = 1.0
    else:
        # Both endpoints are clear, but the aircraft meet near the midpoint.
        # The small turn keeps this on the adaptive path rather than straight delegation.
        kwargs["b_lat"], kwargs["b_lon"] = local_xy_to_latlon(0.0, 10000.0, 51.0, -1.0)
        kwargs.update(
            a_target_heading_deg=0.1,
            a_turn_rate_deg_sec=1.0,
            a_speed_kt=400.0,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=400.0,
            speed_diff_kt=0.0,
            projection_time_s=50.0,
        )

    production_globals = python_function(ra._catch_up_projection_interval_with_turns_local).__globals__
    sample = production_globals["_sample_relative_hull_at_time"]
    core = production_globals["_catch_up_projection_interval_with_turns_local"]
    before = run_turn_aware(kwargs)
    count, result = count_fixed_time_evals(kwargs)
    repeated_count, repeated_result = count_fixed_time_evals(kwargs)
    after = run_turn_aware(kwargs)

    assert production_globals["_sample_relative_hull_at_time"] is sample
    assert production_globals["_catch_up_projection_interval_with_turns_local"] is core
    assert count == repeated_count == expected_count
    assert result[0] == repeated_result[0] == before[0] == after[0]
    assert result[1:] == pytest.approx(before[1:], abs=1e-9)
    assert repeated_result == result
    assert after == before


def test_benchmark_script_entrypoint(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_relevant_aircraft.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--random-cases" in result.stdout
