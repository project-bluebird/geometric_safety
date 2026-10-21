"""Analytic checks for projected-motion interval certificates."""

import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra


def test_projected_range_includes_interior_extrema_and_turn_completion() -> None:
    # Clockwise through east, ending at south, followed by straight motion.
    assert ra._projected_heading_range(45.0, 180.0, 3.0, np.array([1.0, 0.0]), 0.0, 80.0) == pytest.approx((0, 1))
    # Counterclockwise through west, including heading wrap.
    assert ra._projected_heading_range(315.0, 180.0, -3.0, np.array([1.0, 0.0]), 0.0, 80.0) == pytest.approx((-1, 0))
    assert ra._projected_heading_range(45.0, 180.0, 3.0, np.array([1.0, 0.0]), 50.0, 80.0) == pytest.approx((0, 0))


def test_projected_affine_envelope_minimum() -> None:
    assert ra._projected_motion_lower_bound(10.0, 10.0, -2.0, 2.0, 4.0) == 6.0
    assert ra._projected_motion_lower_bound(10.0, 18.0, 2.0, 2.0, 4.0) == 10.0


def test_combined_hull_direction_recovers_straight_interval_minimum() -> None:
    args = (np.array([-10.0, 5.0]), 90.0, 90.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.inf)
    assert ra._interval_directional_lower_bound(*args, False) == pytest.approx(0.0)
    assert ra._interval_directional_lower_bound(*args, True) == pytest.approx(5.0)


def test_bound_covers_interior_speeds_not_just_corner_distances() -> None:
    # Every corner is ten metres away, but the admissible speed segment crosses zero.
    args = (np.array([-10.0, 0.0]), 90.0, 90.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, math.inf)
    assert ra._interval_directional_lower_bound(*args) == 0.0


def test_inflation_is_applied_to_directional_acceptance() -> None:
    # A fast eastbound aircraft passes a stationary aircraft with exactly 5 m clearance.
    # Constant inflation forces the adaptive path even though headings are fixed.
    case = {
        "rel_pos0": np.array([-10.0, 5.0]),
        "a_heading0_deg": 90.0,
        "a_target_heading_deg": 90.0,
        "a_turn_rate_deg_sec": 0.0,
        "a_speed_kt": 20.0 / ra.KT_TO_MPS,
        "b_heading0_deg": 0.0,
        "b_target_heading_deg": 0.0,
        "b_turn_rate_deg_sec": 0.0,
        "b_speed_kt": 0.0,
        "separation_threshold_m": 3.0,
        "speed_diff_kt": 0.0,
        "projection_time_s": 1.0,
        "eta_const_m": 1.0,
    }
    assert not ra._catch_up_projection_interval_with_turns_local(**case, directional_mode=0)[0]
    assert ra._catch_up_projection_interval_with_turns_local(**case)[0]
    case["eta_const_m"] = 2.0
    assert not ra._catch_up_projection_interval_with_turns_local(**case)[0]
    case["eta_const_m"] = 1.0
    case["eta_rate_mps"] = 2.0
    assert not ra._catch_up_projection_interval_with_turns_local(**case)[0]


def test_directional_bounds_do_not_depend_on_first_compilation_caller(tmp_path: Path) -> None:
    script = """
import json, math, sys
import numpy as np
from geometric_safety import relevant_aircraft as ra
from paper.run_turn_numerics import cases
from paper.validate_directional_production import interval_args
inputs = list(cases(48, 2026092031))
if sys.argv[1] == "helpers":
    ra._projected_heading_range(45.0, 180.0, 3.0, np.array([1.0, 0.0]), 0.0, 80.0)
    ra._projected_motion_lower_bound(10.0, 10.0, -2.0, 2.0, 4.0)
    arguments = (np.array([-10.0, 5.0]), 90.0, 90.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.inf)
    for include_hull in (False, True):
        ra._interval_directional_lower_bound(*arguments, include_hull)
else:
    ra._catch_up_projection_interval_with_turns_local(**inputs[0][1])
records = []
for _, case in inputs:
    bound = float(ra._interval_directional_lower_bound(*interval_args(case, 0.0, case["projection_time_s"])))
    records.append(bound.hex())
    query = case.copy()
    for threshold in (case["separation_threshold_m"], max(1.0, bound-ra.TURN_CERT_FP_GUARD_M)):
        query["separation_threshold_m"] = threshold
        result = ra._catch_up_projection_interval_with_turns_local(**query)
        records.append([bool(result[0]), float(result[1]).hex(), float(result[2]).hex()])
print(json.dumps(records))
"""
    outputs = []
    for mode in ("helpers", "api"):
        result = subprocess.run(
            [sys.executable, "-c", script, mode],
            env=dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / mode), PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
