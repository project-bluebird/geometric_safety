"""Failures in numerical validation must remain fatal under Python optimisation."""

import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra
from paper import run_turn_numerics as scalar, validate_directional_production as directional
from paper.numerical_validation import require_solver_parity


def test_missing_directional_hook_is_rejected() -> None:
    source = inspect.getsource(ra._catch_up_projection_interval_with_turns_local.py_func)
    renamed = source.replace("directional_lower_bound_m", "renamed_directional_bound")
    with pytest.raises(RuntimeError, match=r"directional.*hook|hook.*directional"):
        directional.instrument_solver(renamed)


def test_duplicate_directional_hook_is_rejected() -> None:
    source = inspect.getsource(ra._catch_up_projection_interval_with_turns_local.py_func)

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


@pytest.fixture(scope="module")
def numerical_result() -> directional.NumericalResult:
    return directional.numerical_checks(8, 42)


def test_directional_audit_uses_real_production_and_reference(numerical_result: directional.NumericalResult) -> None:
    assert numerical_result["comparisons"] > 0
    assert numerical_result["accepted"] > 0
    assert numerical_result["precision_rechecks"] > 0
    assert numerical_result["negative_reference_acceptances"] == 0
    assert numerical_result["scalar_acceptances_lost"] == 0


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
        source = inspect.getsource(v.ra._catch_up_projection_interval_with_turns_local.py_func)
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
