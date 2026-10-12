"""Failures in numerical validation must remain fatal under Python optimisation."""

import ast
import copy
import inspect
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra
from paper import run_turn_numerics as scalar, validate_directional_production as directional


def test_missing_directional_hook_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    source = inspect.getsource(ra._catch_up_projection_interval_with_turns_local.py_func)
    renamed = source.replace("directional_lower_bound_m", "renamed_directional_bound")
    monkeypatch.setattr(directional.inspect, "getsource", lambda _function: renamed)
    with pytest.raises(RuntimeError, match=r"directional.*hook|hook.*directional"):
        directional.traced_solver()


def test_duplicate_directional_hook_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    source = inspect.getsource(ra._catch_up_projection_interval_with_turns_local.py_func)

    class DuplicateHook(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
            if any(
                isinstance(target, ast.Name) and target.id == "directional_lower_bound_m" for target in node.targets
            ):
                return [node, copy.deepcopy(node)]
            return node

    duplicated = ast.unparse(DuplicateHook().visit(ast.parse(source)))
    monkeypatch.setattr(directional.inspect, "getsource", lambda _function: duplicated)
    with pytest.raises(RuntimeError, match=r"directional.*hook|hook.*directional"):
        directional.traced_solver()


@pytest.mark.parametrize("optimisation", [0, 1])
@pytest.mark.parametrize("failure", ["empty", "mismatch", "negative"])
def test_numerical_failures_remain_fatal(optimisation: int, failure: str) -> None:
    code = textwrap.dedent(
        f"""
        import mpmath as mp
        from paper import validate_directional_production as v
        failure = {failure!r}
        case = next(v.cases(1, 1))[1]
        case.update(projection_time_s=1.0, separation_threshold_m=1.0)
        def traced(**kwargs):
            trace = kwargs['study_trace']
            if failure != 'empty':
                trace[0] = [0.0, 1.0, 200.0, 0.0]
            return (False, 10.0, 0.0)
        v.traced_solver = lambda: traced
        v.cases = lambda n, seed: [('mock', case)]
        v.interval_args = lambda *args: ()
        v.certificate = lambda *args, **kwargs: {{'eta': mp.mpf(0)}}
        v.directional_reference = lambda *args, **kwargs: mp.mpf(-1)
        v.ra._interval_directional_lower_bound = lambda *args: 0.0
        v.ra._catch_up_projection_interval_with_turns_local = lambda **kwargs: (failure == 'mismatch', 10.0, 0.0)
        try:
            v.numerical_checks(1, 1)
        except RuntimeError as error:
            print('REJECTED', error)
        else:
            raise RuntimeError('Numerical validation silently accepted ' + failure)
        """
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONOPTIMIZE", None)
    completed = subprocess.run(
        [sys.executable, *(["-O"] if optimisation else []), "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REJECTED" in completed.stdout


def test_scalar_hook_check_survives_optimisation() -> None:
    code = textwrap.dedent(
        """
        import inspect
        from paper import run_turn_numerics as v
        source = inspect.getsource(v.ra._catch_up_projection_interval_with_turns_local.py_func)
        source = source.replace('TURN_CERT_FP_GUARD_M', 'renamed_numerical_guard')
        v.inspect.getsource = lambda function: source
        try:
            v.traced_solver()
        except RuntimeError as error:
            print('REJECTED', error)
        else:
            raise RuntimeError('Scalar validation silently accepted missing comparison hooks')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REJECTED" in completed.stdout


@pytest.mark.parametrize("encounters", [0, -1])
def test_directional_audit_requires_requested_encounters(encounters: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        directional.numerical_checks(encounters, 1)


def test_scalar_trace_capacity_is_checked_before_writing() -> None:
    solver, _ = scalar.traced_solver()
    _, case = next(scalar.cases(1, 42))
    with pytest.raises(RuntimeError, match="trace capacity exceeded"):
        solver(**case, study_trace=np.empty((0, 6)))
