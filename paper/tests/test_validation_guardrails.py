"""Failures in numerical validation must remain fatal under Python optimisation."""

import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra
from paper import run_turn_numerics as scalar, validate_directional_production as directional
from paper.numerical_validation import TurnCase, python_function, require_solver_parity


def test_missing_directional_hook_is_rejected() -> None:
    source = inspect.getsource(python_function(ra._catch_up_projection_interval_with_turns_local))
    renamed = source.replace("directional_lower_bound_m", "renamed_directional_bound")
    with pytest.raises(RuntimeError, match=r"directional.*hook|hook.*directional"):
        directional.instrument_solver(renamed)


def test_duplicate_directional_hook_is_rejected() -> None:
    source = inspect.getsource(python_function(ra._catch_up_projection_interval_with_turns_local))

    class DuplicateHook(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
            if any(
                isinstance(target, ast.Name) and target.id == "directional_lower_bound_m" for target in node.targets
            ):
                return [node, copy.deepcopy(node)]
            return node

    duplicated = ast.unparse(DuplicateHook().visit(ast.parse(source)))
    with pytest.raises(RuntimeError, match=r"directional.*hook|hook.*directional"):
        directional.instrument_solver(duplicated)


@pytest.mark.parametrize("copies", [0, 2])
def test_directional_comparison_hook_count_is_checked(copies: int) -> None:
    source = inspect.getsource(python_function(ra._catch_up_projection_interval_with_turns_local))

    class ChangeComparisonCount(ast.NodeTransformer):
        def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
            self.generic_visit(node)
            if (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "directional_lower_bound_m"
            ):
                return [copy.deepcopy(node) for _ in range(copies)]
            return node

    changed = ast.unparse(ChangeComparisonCount().visit(ast.parse(source)))
    with pytest.raises(RuntimeError, match="directional comparison hook structure changed"):
        directional.instrument_solver(changed)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            TurnCase(
                rel_pos0=np.array([-16275.427595662642, -1869.405801333638]),
                a_heading0_deg=333.10591142974397,
                a_target_heading_deg=468.17322508293364,
                a_turn_rate_deg_sec=4.840129415919734,
                a_speed_kt=43.324606601140104,
                b_heading0_deg=180.67285395037473,
                b_target_heading_deg=315.62898407457567,
                b_turn_rate_deg_sec=4.170835218813311,
                b_speed_kt=65.32498085256013,
                speed_diff_kt=21.705951616144652,
                projection_time_s=353.1763872205616,
                separation_threshold_m=819.157998033264,
                eta_const_m=50.29671091288135,
                eta_rate_mps=7.178747868370947,
                turn_speed_schedule_uncertainty_kt=0.0,
                turn_delay_max_s=9.84478429322914,
                turn_rate_error_max=0.17319354504124035,
            ),
            id="wide-speed-intervals",
        ),
        pytest.param(
            TurnCase(
                rel_pos0=np.array([-65093.71340190715, -113592.95473134817]),
                a_heading0_deg=155.55803313141513,
                a_target_heading_deg=235.95618554301575,
                a_turn_rate_deg_sec=0.6795549553793536,
                a_speed_kt=305.85338379295905,
                b_heading0_deg=328.4452576778782,
                b_target_heading_deg=464.92411909936015,
                b_turn_rate_deg_sec=5.984993695081028,
                b_speed_kt=599.1660002263113,
                speed_diff_kt=1.3427246376336852e-06,
                projection_time_s=933.187713557014,
                separation_threshold_m=107801.9311396253,
                eta_const_m=112.68555033338264,
                eta_rate_mps=5.382912010396175,
                turn_speed_schedule_uncertainty_kt=0.0,
                turn_delay_max_s=9.851790945341058,
                turn_rate_error_max=0.10947221953661,
            ),
            id="narrow-speed-intervals",
        ),
    ],
)
def test_directional_trace_records_compiled_decision(case: TurnCase) -> None:
    solver = directional.traced_solver()
    trace = np.full((2048, 5), np.nan)
    branches: set[bool] = set()
    saw_reconstruction_disagreement = False

    def check_threshold(threshold: float) -> bool:
        nonlocal saw_reconstruction_disagreement
        query = case.copy()
        query["separation_threshold_m"] = threshold
        trace.fill(np.nan)
        result = solver(**query, study_trace=trace)
        assert result == ra._catch_up_projection_interval_with_turns_local(**query)
        assert result[0]
        rows = trace[~np.isnan(trace[:, 0])]
        assert len(rows) > 0
        assert rows[0, 0] == 0.0
        assert rows[0, 1] == case["projection_time_s"]
        assert rows[0, 4] in (0.0, 1.0)
        recorded = bool(rows[0, 4])
        # A rejected full interval produces a comparison on a child. Observe
        # that control flow independently of the recorded acceptance flag.
        accepted = len(rows) == 1
        assert recorded == accepted
        if not accepted:
            assert rows[1, 1] - rows[1, 0] < case["projection_time_s"]
        branches.add(accepted)
        reconstructed = float(rows[0, 2]) >= threshold + float(rows[0, 3]) + ra.TURN_CERT_FP_GUARD_M
        saw_reconstruction_disagreement |= reconstructed != recorded
        return accepted

    check_threshold(case["separation_threshold_m"])
    lower, inflation = float(trace[0, 2]), float(trace[0, 3])
    centre = math.fsum((lower, -inflation, -ra.TURN_CERT_FP_GUARD_M))
    step = math.ulp(max(abs(lower), abs(inflation), abs(centre)))
    # Locate the compiled transition on this platform. The rounded trace only
    # supplies an initial estimate; subdivision determines each search step.
    for _ in range(8):
        low, high = centre - step, centre + step
        low_accepted, high_accepted = check_threshold(low), check_threshold(high)
        if low_accepted and not high_accepted:
            break
        step *= 2.0
    else:
        pytest.fail("Could not bracket the compiled directional comparison")
    while math.nextafter(low, math.inf) < high:
        midpoint = low + (high - low) / 2.0
        assert low < midpoint < high
        if check_threshold(midpoint):
            low = midpoint
        else:
            high = midpoint
    assert branches == {False, True}
    assert saw_reconstruction_disagreement
    with pytest.raises(RuntimeError, match="trace capacity exceeded"):
        solver(**case, study_trace=np.empty((0, 5)))


@pytest.fixture(scope="module")
def numerical_result() -> directional.NumericalResult:
    # Cover turning encounters and thresholds near their computed bounds.
    return directional.numerical_checks(33, 2026092031)


def test_directional_audit_uses_real_production_and_reference(numerical_result: directional.NumericalResult) -> None:
    assert numerical_result["comparisons"] > 0
    assert numerical_result["accepted"] > 0
    assert numerical_result["precision_rechecks"] > 0
    assert numerical_result["negative_reference_acceptances"] == 0
    assert numerical_result["scalar_acceptances_lost"] == 0


def test_directional_audit_keeps_compiled_acceptance(numerical_result: directional.NumericalResult) -> None:
    encounters = {
        tuple(case["rel_pos0"]): case
        for _, case in scalar.cases(numerical_result["encounters"], numerical_result["seed"])
    }
    solver = directional.traced_solver()
    trace = np.empty((16384, 5))
    accepted_count = 0
    saw_reconstruction_disagreement = False
    for record in numerical_result["records"]:
        query = encounters[tuple(record["case"]["rel_pos0"])].copy()
        query["separation_threshold_m"] = record["case"]["separation_threshold_m"]
        assert scalar.serialisable(query) == record["case"]
        trace.fill(np.nan)
        result = solver(**query, study_trace=trace)
        assert result == ra._catch_up_projection_interval_with_turns_local(**query)
        rows = trace[(trace[:, 0] == record["left_s"]) & (trace[:, 1] == record["right_s"])]
        assert rows.shape == (1, 5)
        assert rows[0, 2] == record["lower_m"]
        assert rows[0, 3] == record["eta_m"]
        accepted = bool(rows[0, 4])
        assert record["accepted"] == accepted
        accepted_count += int(accepted)
        if accepted:
            assert record["reference_residual_m"] >= 0.0
        reconstructed = record["lower_m"] >= query["separation_threshold_m"] + record["eta_m"] + ra.TURN_CERT_FP_GUARD_M
        saw_reconstruction_disagreement |= reconstructed != accepted
    assert 0 < accepted_count < len(numerical_result["records"])
    assert numerical_result["accepted"] == accepted_count
    assert numerical_result["comparisons"] == len(numerical_result["records"])
    assert saw_reconstruction_disagreement


def test_scalar_audit_uses_real_production_and_reference(tmp_path: Path) -> None:
    solver, _ = scalar.traced_solver()
    result = scalar.run_split(solver, 2, 42, tmp_path, "regression", ra.TURN_CERT_FP_GUARD_M)
    assert result["comparisons"] > 0
    assert result["reference_intervals"] > 0
    assert result["precision_rechecks"] > 0
    assert result["instrumentation_mismatches"] == 0
    assert result["candidates"][str(ra.TURN_CERT_FP_GUARD_M)]["accepted_negative_reference_comparisons"] == 0


@pytest.mark.parametrize("optimisation", [0, 1])
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("empty", "recorded no directional comparisons"),
        ("negative", "accepted a negative reference residual"),
        ("scalar", "lost a scalar acceptance"),
        ("error", "exceeded the empirical guard allowance"),
    ],
)
def test_numerical_failures_remain_fatal(
    optimisation: int, failure: str, message: str, numerical_result: directional.NumericalResult, tmp_path: Path
) -> None:
    invalid = numerical_result.copy()
    if failure == "empty":
        invalid["comparisons"] = 0
        invalid["records"] = []
    elif failure == "negative":
        invalid["negative_reference_acceptances"] = 1
    elif failure == "scalar":
        invalid["scalar_acceptances_lost"] = 1
    else:
        invalid["max_optimistic_error_m"] = ra.TURN_CERT_FP_GUARD_M
    report = tmp_path / "invalid_audit.json"
    report.write_text(json.dumps(invalid))
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from paper.validate_directional_production import NumericalResult, validate_numerical_result\n"
        "result: NumericalResult = json.loads(Path(sys.argv[1]).read_text())\n"
        "validate_numerical_result(result)\n"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONOPTIMIZE", None)
    completed = subprocess.run(
        [sys.executable, *(["-O"] if optimisation else []), "-c", code, str(report)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode != 0, completed.stdout
    assert message in completed.stderr


@pytest.mark.parametrize("optimisation", [0, 1])
def test_solver_mismatch_remains_fatal(optimisation: int) -> None:
    _, case = next(scalar.cases(1, 42))
    expected = ra._catch_up_projection_interval_with_turns_local(**case)
    # A decision change is invalid even when its numerical diagnostics agree.
    different = (not expected[0], expected[1], expected[2])
    require_solver_parity(expected, expected, "production control")
    code = (
        "from paper.numerical_validation import require_solver_parity\n"
        f"require_solver_parity({different!r}, {expected!r}, 'production control')\n"
    )
    completed = subprocess.run(
        [sys.executable, *(["-O"] if optimisation else []), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, completed.stdout
    assert "Instrumented solver differs from production" in completed.stderr


def test_scalar_hook_check_survives_optimisation() -> None:
    code = textwrap.dedent(
        """
        import inspect
        from paper import run_turn_numerics as v
        source = inspect.getsource(v.python_function(v.ra._catch_up_projection_interval_with_turns_local))
        source = source.replace('TURN_CERT_FP_GUARD_M', 'renamed_numerical_guard')
        v.instrument_solver(source)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert completed.returncode != 0, completed.stdout
    assert "scalar comparison hook structure changed" in completed.stderr


@pytest.mark.parametrize("encounters", [0, -1])
def test_directional_audit_requires_requested_encounters(encounters: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        directional.numerical_checks(encounters, 1)


def test_scalar_trace_capacity_is_checked_before_writing() -> None:
    solver, _ = scalar.traced_solver()
    _, case = next(scalar.cases(1, 42))
    with pytest.raises(RuntimeError, match="trace capacity exceeded"):
        solver(**case, study_trace=np.empty((0, 6)))
