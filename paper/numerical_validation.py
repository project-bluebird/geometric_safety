"""Shared contracts and fail-closed instrumentation for numerical studies.

These helpers inspect and record production code. They contain no geometry or
certificate calculation, so the arbitrary-precision reference stays independent.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import inspect
import math
from types import FunctionType
from typing import Protocol, TypedDict, Unpack, cast

import mpmath as mp
import numba
import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type SolverResult = tuple[bool, float, float]


class MotionParameters(TypedDict):
    a_heading0_deg: float
    a_target_heading_deg: float
    a_turn_rate_deg_sec: float
    a_speed_kt: float
    b_heading0_deg: float
    b_target_heading_deg: float
    b_turn_rate_deg_sec: float
    b_speed_kt: float
    speed_diff_kt: float
    projection_time_s: float
    separation_threshold_m: float
    eta_const_m: float
    eta_rate_mps: float
    turn_speed_schedule_uncertainty_kt: float
    turn_delay_max_s: float
    turn_rate_error_max: float


class TurnCase(MotionParameters):
    rel_pos0: FloatArray


class SerialisedTurnCase(MotionParameters):
    rel_pos0: list[float]


class StudyArguments(TurnCase, total=False):
    study_trace: FloatArray
    study_guard: float


class StudySolver(Protocol):
    def __call__(self, **kwargs: Unpack[StudyArguments]) -> SolverResult: ...


class SolverInputs(TurnCase, total=False):
    use_interval_local_lipschitz: bool
    min_cert_interval_s: float


class ProductionArguments(SolverInputs, total=False):
    directional_mode: int


class ProductionSolver(Protocol):
    def __call__(self, **kwargs: Unpack[ProductionArguments]) -> SolverResult: ...


class ReferenceCertificate(TypedDict):
    lower: mp.mpf
    eta: mp.mpf
    residual: mp.mpf
    left_distance: mp.mpf
    right_distance: mp.mpf
    lipschitz: mp.mpf


def python_function(dispatcher: object) -> FunctionType:
    """Recover the Python function for source and globals instrumentation."""
    if not callable(dispatcher):
        raise TypeError("Numerical instrumentation requires a callable")
    function = inspect.unwrap(dispatcher)
    if not isinstance(function, FunctionType):
        raise TypeError("Numerical instrumentation requires a Python function")
    return function


def solver_ast(source: str, name: str, guard: float | None = None) -> tuple[ast.Module, ast.FunctionDef]:
    """Extract the single solver and add study arguments without changing its body."""
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Production solver structure changed; expected one function")
    function = tree.body[0]
    function.decorator_list = []
    function.name = name
    extras: tuple[tuple[str, float | None], ...] = (("study_trace", None),)
    if guard is not None:
        extras += (("study_guard", guard),)
    existing = {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
    for argument_name, default in extras:
        if argument_name in existing:
            raise RuntimeError(f"Production solver already defines {argument_name}")
        function.args.args.append(ast.arg(arg=argument_name))
        function.args.defaults.append(ast.Constant(value=default))
    first = function.body[0] if function.body else None
    has_docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    function.body.insert(int(has_docstring), ast.parse("study_count = 0").body[0])
    return tree, function


def require_hooks(actual: int, expected: int, study: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Production {study} hook structure changed: expected {expected}, found {actual}")


def trace_writes(expressions: tuple[str, ...]) -> list[ast.stmt]:
    """Write one comparison, rejecting an exhausted buffer before any write."""
    assignments = "\n".join(
        f"study_trace[study_count, {index}] = {expression}" for index, expression in enumerate(expressions)
    )
    return ast.parse(
        "if study_count >= study_trace.shape[0]:\n"
        '    raise RuntimeError("Numerical validation trace capacity exceeded")\n' + assignments + "\nstudy_count += 1"
    ).body


def compile_solver(
    tree: ast.Module, function: ast.FunctionDef, globals_: Mapping[str, object], filename: str
) -> StudySolver:
    ast.fix_missing_locations(tree)
    namespace = dict(globals_)
    exec(compile(tree, filename, "exec"), namespace)
    compiled = namespace[function.name]
    if not isinstance(compiled, FunctionType):
        raise RuntimeError("Instrumented solver did not compile to a function")
    # AST-generated keyword signatures and Numba dispatchers cannot be expressed
    # by the compiler's dynamic return type. The study contract is checked above.
    return cast("StudySolver", numba.njit(fastmath=True)(compiled))


def require_positive_encounters(n: int) -> None:
    if n <= 0:
        raise ValueError("Numerical validation requires a positive encounter count")


def require_coverage(comparisons: int, study: str) -> None:
    if comparisons == 0:
        raise RuntimeError(f"Numerical validation recorded no {study} comparisons")


def require_solver_parity(actual: SolverResult, expected: SolverResult, context: str) -> None:
    same_diagnostics = all(
        a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(actual[1:], expected[1:], strict=True)
    )
    if actual[0] != expected[0] or not same_diagnostics:
        raise RuntimeError(f"Instrumented solver differs from production: {context}; {actual} != {expected}")


def boundary_thresholds(lower: float, envelope: float, guard: float) -> tuple[tuple[str, float], ...]:
    """Thresholds immediately around this guarded certificate's acceptance boundary."""
    cutoff = math.fsum((lower, -envelope, -guard))
    if cutoff <= 100.0:
        return ()
    return (("below", math.nextafter(cutoff, -math.inf)), ("at", cutoff), ("above", math.nextafter(cutoff, math.inf)))
