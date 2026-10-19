"""Validate and time the integrated directional solver on the experiment server.

The high-precision calculation checks valid certificates along the actual
binary64 candidate directions. It does not rely on their being optimal.
Results are empirical numerical checks, not directed-rounding proofs.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
from pathlib import Path
import time
from typing import TypedDict, cast

import mpmath as mp
import numba
import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety.util import closest_point_on_convex_polygon, convex_hull
from paper.benchmark_library_versions import list_value, number, object_json, object_value
from paper.numerical_validation import (
    FloatArray,
    ProductionArguments,
    ProductionSolver,
    SerialisedTurnCase,
    SolverResult,
    StudySolver,
    TurnCase,
    compile_solver,
    require_coverage,
    require_hooks,
    require_positive_encounters,
    require_solver_parity,
    solver_ast,
    trace_writes,
)
from paper.run_turn_numerics import cases, serialisable
from paper.turn_numerical_reference import basis, certificate, turn

type IntervalArguments = tuple[
    FloatArray, float, float, float, float, float, float, float, float, float, float, float, float, float
]
type DirectionalBound = Callable[
    [FloatArray, float, float, float, float, float, float, float, float, float, float, float, float, float], float
]
type ReferenceMotion = tuple[mp.mpf, mp.mpf, mp.mpf, tuple[mp.mpf, mp.mpf]]


class NumericalRecord(TypedDict):
    family: str
    case: SerialisedTurnCase
    left_s: float
    right_s: float
    lower_m: float
    eta_m: float
    accepted: bool
    reference_residual_m: float
    optimistic_error_m: float


class NumericalResult(TypedDict):
    encounters: int
    seed: int
    comparisons: int
    accepted: int
    max_optimistic_error_m: float
    negative_reference_acceptances: int
    scalar_acceptances_lost: int
    precision_rechecks: int
    max_precision_change_m: float
    records: list[NumericalRecord]


class BenchmarkRow(TypedDict):
    suite: str
    reference: str
    safe: list[bool]


class BenchmarkGroup(TypedDict):
    mean_us: list[float]
    packed_mean_us: list[float]
    safe_counts: list[int]


class BenchmarkResult(TypedDict):
    modes: list[str]
    cases: list[BenchmarkRow]
    packed_modes: list[str]
    baseline_sha256: str | None
    raw_us: list[list[list[float]]]
    mean_us: list[float]
    packed_raw_us: list[list[list[float]]]
    packed_mean_us: list[float]
    groups: dict[str, BenchmarkGroup]
    inputs_sha256: str


def interval_args(case: TurnCase, left: float, right: float) -> IntervalArguments:
    width = case["speed_diff_kt"]
    return (
        case["rel_pos0"],
        case["a_heading0_deg"],
        case["a_target_heading_deg"],
        case["a_turn_rate_deg_sec"],
        max(case["a_speed_kt"] - width, 0.0) * ra.KT_TO_MPS,
        (case["a_speed_kt"] + width) * ra.KT_TO_MPS,
        case["b_heading0_deg"],
        case["b_target_heading_deg"],
        case["b_turn_rate_deg_sec"],
        max(case["b_speed_kt"] - width, 0.0) * ra.KT_TO_MPS,
        (case["b_speed_kt"] + width) * ra.KT_TO_MPS,
        left,
        right,
        math.inf,
    )


def directional_reference(case: TurnCase, left: float, right: float, dps: int = 80) -> mp.mpf:
    args = interval_args(case, left, right)
    # Selection is allowed to use approximate geometry. All subsequent bounds
    # are recomputed from the original inputs, with independent integration.
    float_points = []
    for t in (left, right):
        aa = ra._turn_displacement_basis(args[1], args[2], args[3], t)
        bb = ra._turn_displacement_basis(args[6], args[7], args[8], t)
        float_points.append(ra._speed_rectangle_corner_points(args[0], aa, args[4], args[5], bb, args[9], args[10]))
    normals = []
    for points in (*float_points, np.concatenate(float_points)):
        closest, _ = closest_point_on_convex_polygon(np.zeros(2), convex_hull(points))
        length = math.hypot(*closest)
        if length:
            normals.append(closest / length)
    with mp.workdps(dps):
        aircraft: list[ReferenceMotion] = []
        for prefix in ("a", "b"):
            parameters = {key: mp.mpf(value) for key, value in case.items() if isinstance(value, float | int)}
            h, target, rate = (
                parameters[f"{prefix}_{key}"] for key in ("heading0_deg", "target_heading_deg", "turn_rate_deg_sec")
            )
            speed, width = parameters[f"{prefix}_speed_kt"], parameters["speed_diff_kt"]
            speeds = (max(speed - width, 0) * mp.mpf(1852) / 3600, (speed + width) * mp.mpf(1852) / 3600)
            aircraft.append((h, target, rate, speeds))
        a, b = aircraft
        lft, rgt = mp.mpf(left), mp.mpf(right)
        duration = rgt - lft
        ba = [basis(a[0], a[1], a[2], t) for t in (lft, rgt)]
        bb = [basis(b[0], b[1], b[2], t) for t in (lft, rgt)]
        position = [mp.mpf(float(x)) for x in case["rel_pos0"]]
        best = mp.mpf(0)
        for normal in normals:
            n = [mp.mpf(float(x)) for x in normal]
            norm = mp.hypot(*n)
            n = [x / norm for x in n]
            ranges = []
            for motion in (a, b):
                h, target, rate = motion[0], motion[1], motion[2]
                angle, end = turn(h, target, rate)
                headings = [
                    mp.radians(h + mp.sign(angle) * abs(rate) * min(t, end) if angle else target) for t in (lft, rgt)
                ]
                lo, hi = sorted(headings)
                phase = mp.atan2(n[0], n[1])
                first, last = int(mp.ceil((lo - phase) / mp.pi)), int(mp.floor((hi - phase) / mp.pi))
                values = [
                    n[0] * mp.sin(hh) + n[1] * mp.cos(hh)
                    for hh in [lo, hi, *[phase + k * mp.pi for k in range(first, last + 1)]]
                ]
                ranges.append((min(values), max(values)))
            corner_bounds = []
            for sa, sb in itertools.product(a[3], b[3]):
                q = [sum(n[k] * (position[k] + sa * ba[j][k] - sb * bb[j][k]) for k in range(2)) for j in range(2)]
                lower = sa * ranges[0][0] - sb * ranges[1][1]
                upper = sa * ranges[0][1] - sb * ranges[1][0]
                times = [mp.mpf(0), duration]
                if upper > lower:
                    crossing = (q[0] - q[1] + upper * duration) / (upper - lower)
                    if 0 < crossing < duration:
                        times.append(crossing)
                corner_bounds.append(min(max(q[0] + lower * t, q[1] - upper * (duration - t)) for t in times))
            best = max(best, min(corner_bounds))
        return +best


def instrument_solver(source: str) -> tuple[ast.Module, ast.FunctionDef]:
    """Add directional recording to a validated production source snapshot."""
    tree, function = solver_ast(source, "traced_directional_solver")

    class Recorder(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
            if any(
                isinstance(target, ast.Name) and target.id == "directional_lower_bound_m" for target in node.targets
            ):
                self.count += 1
                return [node, *trace_writes(("left_t_s", "right_t_s", "directional_lower_bound_m", "interval_eta_m"))]
            return node

    recorder = Recorder()
    recorder.visit(tree)
    require_hooks(recorder.count, 1, "directional comparison")
    return tree, function


def traced_solver() -> StudySolver:
    tree, function = instrument_solver(inspect.getsource(ra._catch_up_projection_interval_with_turns_local.py_func))
    return compile_solver(tree, function, vars(ra), "<directional-validation>")


def numerical_checks(n: int, seed: int) -> NumericalResult:
    require_positive_encounters(n)
    # Numba's ParamSpec dispatcher stubs do not preserve TypedDict/tuple unpacking in ty.
    production_solver = cast("ProductionSolver", ra._catch_up_projection_interval_with_turns_local)
    directional_bound = cast(
        "DirectionalBound",
        ra._interval_directional_lower_bound,
    )
    solver = traced_solver()
    trace = np.empty((16384, 4))
    result: NumericalResult = {
        "encounters": n,
        "seed": seed,
        "comparisons": 0,
        "accepted": 0,
        "max_optimistic_error_m": 0.0,
        "negative_reference_acceptances": 0,
        "scalar_acceptances_lost": 0,
        "precision_rechecks": 0,
        "max_precision_change_m": 0.0,
        "records": [],
    }
    for family, case in cases(n, seed):
        probes = [case]
        horizon = case["projection_time_s"]
        full = directional_bound(*interval_args(case, 0.0, horizon))
        eta = float(certificate(case, 0.0, horizon)["eta"])
        cutoff = full - eta - ra.TURN_CERT_FP_GUARD_M
        if cutoff > 100:
            for threshold in (np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf)):
                probe = case.copy()
                probe["separation_threshold_m"] = float(threshold)
                probes.append(probe)
        cache: dict[tuple[float, float], tuple[mp.mpf, mp.mpf]] = {}
        for probe in probes:
            trace.fill(np.nan)
            actual = solver(**probe, study_trace=trace)
            require_solver_parity(actual, production_solver(**probe), family)
            scalar = production_solver(**probe, directional_mode=0)
            result["scalar_acceptances_lost"] += int(scalar[0] and not actual[0])
            for left, right, lower, inflation in trace[~np.isnan(trace[:, 0])]:
                left, right, lower, inflation = map(float, (left, right, lower, inflation))
                key = (left, right)
                if key not in cache:
                    ref = directional_reference(case, left, right)
                    ref_eta = certificate(case, left, right)["eta"]
                    cache[key] = (ref, ref_eta)
                    if len(result["records"]) % 37 == 0:
                        high = directional_reference(case, left, right, dps=120)
                        result["precision_rechecks"] += 1
                        result["max_precision_change_m"] = max(result["max_precision_change_m"], float(abs(high - ref)))
                ref, ref_eta = cache[key]
                accepted = lower >= probe["separation_threshold_m"] + inflation + ra.TURN_CERT_FP_GUARD_M
                with mp.workdps(80):
                    error = float((mp.mpf(lower) - mp.mpf(inflation)) - (ref - ref_eta))
                    residual = float(ref - ref_eta - mp.mpf(probe["separation_threshold_m"]))
                result["comparisons"] += 1
                result["accepted"] += int(accepted)
                result["max_optimistic_error_m"] = max(result["max_optimistic_error_m"], error)
                result["negative_reference_acceptances"] += int(accepted and residual < 0)
                result["records"].append(
                    {
                        "family": family,
                        "case": serialisable(probe),
                        "left_s": left,
                        "right_s": right,
                        "lower_m": lower,
                        "eta_m": inflation,
                        "accepted": bool(accepted),
                        "reference_residual_m": residual,
                        "optimistic_error_m": error,
                    }
                )
    validate_numerical_result(result)
    return result


def validate_numerical_result(result: NumericalResult) -> None:
    """Reject empty coverage or a failed check before publishing an audit result."""
    require_coverage(result["comparisons"], "directional")
    if result["negative_reference_acceptances"]:
        raise RuntimeError("Directional validation accepted a negative reference residual")
    if result["scalar_acceptances_lost"]:
        raise RuntimeError("Directional validation lost a scalar acceptance")
    if result["max_optimistic_error_m"] * 100 >= ra.TURN_CERT_FP_GUARD_M:
        raise RuntimeError("Directional optimistic error exceeded the empirical guard allowance")


@numba.njit(cache=True)
def packed_call(p: FloatArray, mode: int) -> bool:
    return ra._catch_up_projection_interval_with_turns_local(
        p[:2],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        p[10],
        p[11],
        p[12],
        directional_mode=mode,
    )[0]


def benchmark(inputs: Path, baseline: Path | None = None) -> BenchmarkResult:
    # Keep the native dispatcher: only its ParamSpec type boundary needs narrowing.
    production_solver = cast("ProductionSolver", ra._catch_up_projection_interval_with_turns_local)
    original_packed: Callable[[FloatArray], bool] | None = None
    if baseline is not None:
        spec = importlib.util.spec_from_file_location("geometric_safety.before_directional", baseline)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load baseline solver from {baseline}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_solver: object = module._catch_up_projection_interval_with_turns_local
        if not callable(loaded_solver):
            raise RuntimeError("Baseline solver is not callable")
        # This archived module is the one unavoidable dynamic solver boundary.
        original_solver = cast("Callable[..., SolverResult]", loaded_solver)

        @numba.njit
        def original_packed(p: FloatArray) -> bool:
            return original_solver(
                p[:2],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
                p[8],
                p[9],
                p[10],
                p[11],
                p[12],
            )[0]

    rows = [object_value(row) for row in list_value(object_json(inputs.read_text())["cases"])]
    if not rows:
        raise ValueError("Directional benchmark requires at least one case")
    calls: list[ProductionArguments] = []
    results: list[BenchmarkRow] = []
    packed: list[FloatArray] = []
    for row in rows:
        encounter = object_value(row["encounter"])
        case: ProductionArguments = {
            "rel_pos0": np.array([number(encounter["rel_east_m"]), number(encounter["rel_north_m"])]),
            "a_heading0_deg": number(encounter["a_heading0_deg"]),
            "a_target_heading_deg": number(encounter["a_target_heading_deg"]),
            "a_speed_kt": number(encounter["a_speed_kt"]),
            "a_turn_rate_deg_sec": number(encounter["a_turn_rate_deg_sec"]),
            "b_heading0_deg": number(encounter["b_heading0_deg"]),
            "b_target_heading_deg": number(encounter["b_target_heading_deg"]),
            "b_speed_kt": number(encounter["b_speed_kt"]),
            "b_turn_rate_deg_sec": number(encounter["b_turn_rate_deg_sec"]),
            "separation_threshold_m": number(encounter["separation_threshold_m"]),
            "speed_diff_kt": number(encounter["speed_diff_kt"]),
            "projection_time_s": number(encounter["projection_time_s"]),
            "eta_const_m": number(encounter.get("eta_const_m", 0.0)),
            "eta_rate_mps": number(encounter.get("eta_rate_mps", 0.0)),
            "turn_speed_schedule_uncertainty_kt": number(encounter.get("turn_speed_schedule_uncertainty_kt", 0.0)),
            "turn_delay_max_s": number(encounter.get("turn_delay_max_s", 0.0)),
            "turn_rate_error_max": number(encounter.get("turn_rate_error_max", 0.0)),
        }
        if "use_interval_local_lipschitz" in encounter:
            local_lipschitz = encounter["use_interval_local_lipschitz"]
            if not isinstance(local_lipschitz, bool):
                raise ValueError("Expected a boolean use_interval_local_lipschitz")
            case["use_interval_local_lipschitz"] = local_lipschitz
        if "min_cert_interval_s" in encounter:
            case["min_cert_interval_s"] = number(encounter["min_cert_interval_s"])
        suite = row["suite"]
        classification = object_value(row["reference"])["classification"]
        if not isinstance(suite, str) or not isinstance(classification, str):
            raise ValueError("Expected string suite and reference classification")
        outcomes = [bool(production_solver(**case, directional_mode=m)[0]) for m in range(3)]
        if outcomes[0] and not all(outcomes):
            raise RuntimeError("Directional benchmark lost a scalar acceptance")
        if classification == "unsafe" and any(outcomes):
            raise RuntimeError("Directional benchmark accepted a reference-unsafe case")
        calls.append(case)
        p = np.array(
            [
                *case["rel_pos0"],
                *[
                    case[k]
                    for k in (
                        "a_heading0_deg",
                        "a_target_heading_deg",
                        "a_speed_kt",
                        "a_turn_rate_deg_sec",
                        "b_heading0_deg",
                        "b_target_heading_deg",
                        "b_speed_kt",
                        "b_turn_rate_deg_sec",
                        "separation_threshold_m",
                        "speed_diff_kt",
                        "projection_time_s",
                    )
                ],
            ]
        )
        if [bool(packed_call(p, mode)) for mode in range(3)] != outcomes:
            raise RuntimeError("Packed directional benchmark differs from the keyword solver")
        if original_packed is not None and bool(original_packed(p)) != outcomes[0]:
            raise RuntimeError("Scalar directional benchmark differs from the baseline")
        packed.append(p)
        results.append({"suite": suite, "reference": classification, "safe": outcomes})
    durations = np.empty((len(rows), 3, 5))
    for round_index in range(5):
        for index, case in enumerate(calls):
            for shifted in range(3):
                mode = (round_index + shifted) % 3
                start = time.perf_counter_ns()
                for _ in range(30):
                    production_solver(**case, directional_mode=mode)
                durations[index, mode, round_index] = (time.perf_counter_ns() - start) / 30 / 1000
    per_case = np.median(durations, axis=2)
    packed_modes = 3 if original_packed is None else 4
    packed_durations = np.empty((len(rows), packed_modes, 5))
    for round_index in range(5):
        for index, p in enumerate(packed):
            for shifted in range(packed_modes):
                mode = (round_index + shifted) % packed_modes
                start = time.perf_counter_ns()
                for _ in range(30):
                    if mode == 3:
                        if original_packed is None:
                            raise RuntimeError("Missing baseline packed solver")
                        original_packed(p)
                    else:
                        packed_call(p, mode)
                packed_durations[index, mode, round_index] = (time.perf_counter_ns() - start) / 30 / 1000
    packed_per_case = np.median(packed_durations, axis=2)
    return {
        "modes": ["scalar", "endpoint directions", "endpoint and hull directions"],
        "cases": results,
        "packed_modes": ["scalar", "endpoint directions", "endpoint and hull directions"]
        + (["original solver"] if baseline else []),
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline else None,
        "raw_us": durations.tolist(),
        "mean_us": per_case.mean(axis=0).tolist(),
        "packed_raw_us": packed_durations.tolist(),
        "packed_mean_us": packed_per_case.mean(axis=0).tolist(),
        "groups": {
            suite: {
                "mean_us": per_case[[r["suite"] == suite for r in rows]].mean(axis=0).tolist(),
                "packed_mean_us": packed_per_case[[r["suite"] == suite for r in rows]].mean(axis=0).tolist(),
                "safe_counts": np.array([r["safe"] for r in results if r["suite"] == suite]).sum(axis=0).tolist(),
            }
            for suite in ("broad", "targeted")
        },
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output: dict[str, str | NumericalResult | BenchmarkResult] = {
        "source_sha256": hashlib.sha256(Path(ra.__file__).read_bytes()).hexdigest(),
        "validation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "instrumentation_source_sha256": hashlib.sha256(
            Path(__file__).with_name("numerical_validation.py").read_bytes()
        ).hexdigest(),
        "reference_source_sha256": hashlib.sha256(Path(inspect.getfile(certificate)).read_bytes()).hexdigest(),
    }
    for split, n, seed in (("calibration", 128, 2026092030), ("validation", 512, 2026092031)):
        split_result = numerical_checks(n, seed)
        output[split] = split_result
        print(json.dumps({k: v for k, v in split_result.items() if k != "records"}), flush=True)
    benchmark_result = benchmark(args.inputs, args.baseline)
    output["benchmark"] = benchmark_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(benchmark_result["groups"]), flush=True)
