"""Calibrate and independently validate the scalar turn comparison tolerance.

Run on the experiment server. Instrumentation is generated from the current
production function and only records its certification comparisons. Every
default-guard run is checked against the uninstrumented compiled function.
No production source is modified. Results concern local planar inputs only.
The directional certificate is disabled to preserve this study's reference model.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator
import csv
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import TypedDict, cast

import mpmath as mp
import numba
import numpy as np

from geometric_safety import relevant_aircraft as ra
from paper.isolated_worker import run_isolated
from paper.numerical_validation import (
    ProductionSolver,
    ReferenceCertificate,
    SerialisedTurnCase,
    StudySolver,
    TurnCase,
    boundary_thresholds,
    compile_solver,
    python_function,
    require_coverage,
    require_hooks,
    require_positive_encounters,
    require_solver_parity,
    solver_ast,
    trace_writes,
)
from paper.turn_numerical_reference import certificate

FAMILIES = ("broad", "small_turn", "slow_turn", "parallel", "thin", "turn_end", "antiparallel", "short_horizon")
GUARDS = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)


class CandidateStats(TypedDict):
    solver_runs: int
    safe_returns: int
    accepted_comparisons: int
    accepted_negative_reference_comparisons: int


class FamilyStats(TypedDict):
    encounters: int
    comparisons: int
    positive_error_max_m: float


class WorstObservation(TypedDict):
    case: SerialisedTurnCase
    family: str
    index: int
    probe: str
    guard_m: float
    left_s: float
    right_s: float
    signed_error_m: float
    lower_m: float
    rhs_m: float


class SplitStats(TypedDict):
    encounters: int
    solver_runs: int
    comparisons: int
    reference_intervals: int
    positive_error_max_m: float
    absolute_error_max_m: float
    precision_rechecks: int
    precision_recheck_families: dict[str, int]
    precision_change_max_m: float
    production_parity_checks: int
    accepted_negative_reference_comparisons: int
    families: dict[str, FamilyStats]
    candidates: dict[str, CandidateStats]
    worst_positive_error: WorstObservation | None


def instrument_solver(source: str) -> tuple[ast.Module, ast.FunctionDef]:
    """Add scalar comparison recording to a validated production source snapshot."""
    tree, function = solver_ast(
        source,
        "traced_turn_solver",
        ra.TURN_CERT_FP_GUARD_M,
    )

    class Recorder(ast.NodeTransformer):
        count = 0

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == "TURN_CERT_FP_GUARD_M":
                node.id = "study_guard"
            return node

        def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
            if isinstance(node.test, ast.Name) and node.test.id == "directional_mode":
                return []
            self.generic_visit(node)
            names = {part.id for part in ast.walk(node.test) if isinstance(part, ast.Name)}
            if "study_guard" not in names:
                return node
            # Endpoint refusal is not a lower-bound acceptance certificate.
            if (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "end_dist_m"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Lt)
            ):
                return node
            if not isinstance(node.test, ast.Compare) or len(node.test.comparators) != 1:
                raise RuntimeError("Production scalar comparison hook changed; expected one comparison")
            self.count += 1
            initial = "initial_lower_bound_m" in names
            lower = "initial_lower_bound_m" if initial else "interval_lower_bound_m"
            if lower not in names:
                raise RuntimeError("Production scalar comparison hook changed; missing lower bound")
            left, right = ("0.0", "projection_time_s") if initial else ("left_t_s", "right_t_s")
            eta = "eta_horizon_m" if initial else "interval_eta_m"
            values = (left, right, lower, eta, ast.unparse(node.test.comparators[0]), ast.unparse(node.test))
            return [*trace_writes(values), node]

    recorder = Recorder()
    recorder.visit(tree)
    require_hooks(recorder.count, 2, "scalar comparison")
    return tree, function


def traced_solver() -> tuple[StudySolver, str]:
    tree, function = instrument_solver(
        inspect.getsource(python_function(ra._catch_up_projection_interval_with_turns_local))
    )
    solver = compile_solver(tree, function, vars(ra), "<turn-numerical-study>")
    return solver, ast.unparse(tree)


def cases(n: int, seed: int) -> Iterator[tuple[str, TurnCase]]:
    rng = np.random.default_rng(seed)
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        h = rng.uniform(0, 360, 2)
        angle = rng.uniform(-170, 170, 2)
        rate = rng.uniform(0.2, 6, 2)
        speed = rng.uniform(0, 600, 2)
        width = rng.uniform(0, 50)
        horizon = rng.uniform(3, 1200)
        if family == "small_turn":
            angle = rng.choice((-1.0, 1.0), 2) * 10.0 ** rng.uniform(-12, -1, 2)
        elif family == "slow_turn":
            rate = 10.0 ** rng.uniform(-10, -3, 2)
        elif family == "parallel":
            h[1] = h[0] + 10.0 ** rng.uniform(-12, -3)
            angle[1], rate[1], speed[1] = angle[0], rate[0], speed[0]
        elif family == "thin":
            width = float(rng.choice((0.0, 10.0 ** rng.uniform(-12, -3))))
        elif family == "turn_end":
            horizon = abs(angle[0]) / rate[0]
            horizon = float(np.nextafter(horizon, rng.choice((0.0, np.inf))))
        elif family == "antiparallel":
            h[1] = h[0] + 180 + rng.choice((-1.0, 0.0, 1.0)) * 10.0 ** rng.uniform(-12, -3)
        elif family == "short_horizon":
            horizon = 10.0 ** rng.uniform(-8, math.log10(3))
        if i % 7 == 0:
            angle[1] = 0.0
        target = h + angle
        # Every study encounter retains a nonzero turn. Straight delegation is
        # outside the scope of the turn comparison tolerance.
        theta = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(9260, 185200)
        case: TurnCase = {
            "rel_pos0": np.array([radius * np.cos(theta), radius * np.sin(theta)]),
            "a_heading0_deg": float(h[0]),
            "a_target_heading_deg": float(target[0]),
            "a_turn_rate_deg_sec": float(rate[0]),
            "a_speed_kt": float(speed[0]),
            "b_heading0_deg": float(h[1]),
            "b_target_heading_deg": float(target[1]),
            "b_turn_rate_deg_sec": float(rate[1]),
            "b_speed_kt": float(speed[1]),
            "speed_diff_kt": float(width),
            "projection_time_s": float(horizon),
            "separation_threshold_m": 9260.0,
            "eta_const_m": 0.0,
            "eta_rate_mps": 0.0,
            "turn_speed_schedule_uncertainty_kt": 0.0,
            "turn_delay_max_s": 0.0,
            "turn_rate_error_max": 0.0,
        }
        # Do not combine in-turn speed variation with the execution classes.
        mode = (i // len(FAMILIES)) % 3
        if mode == 1:
            case.update(
                eta_const_m=float(rng.uniform(0, 200)),
                eta_rate_mps=float(rng.uniform(0, 20)),
                turn_delay_max_s=float(rng.uniform(0, 10)),
                turn_rate_error_max=float(rng.uniform(0, 0.2)),
            )
        elif mode == 2:
            case["turn_speed_schedule_uncertainty_kt"] = float(rng.uniform(0, 20))
        yield family, case


def serialisable(case: TurnCase) -> SerialisedTurnCase:
    # Dict expansion loses TypedDict literal keys. Every motion field is retained;
    # only the known float array is replaced by a list of Python floats.
    return cast("SerialisedTurnCase", {**case, "rel_pos0": [float(value) for value in case["rel_pos0"]]})


def run_split(solver: StudySolver, n: int, seed: int, output: Path, split: str, selected: float | None) -> SplitStats:
    require_positive_encounters(n)
    # Numba's ParamSpec dispatcher stubs do not preserve TypedDict unpacking in ty.
    production_solver = cast("ProductionSolver", ra._catch_up_projection_interval_with_turns_local)
    stats: SplitStats = {
        "worst_positive_error": None,
        "encounters": n,
        "solver_runs": 0,
        "comparisons": 0,
        "reference_intervals": 0,
        "positive_error_max_m": 0.0,
        "absolute_error_max_m": 0.0,
        "precision_rechecks": 0,
        "precision_recheck_families": {},
        "precision_change_max_m": 0.0,
        "production_parity_checks": 0,
        "accepted_negative_reference_comparisons": 0,
        "families": {},
        "candidates": {},
    }
    guards = sorted(set(GUARDS + ((selected,) if selected is not None else ())))
    for g in guards:
        stats["candidates"][str(g)] = {
            "solver_runs": 0,
            "safe_returns": 0,
            "accepted_comparisons": 0,
            "accepted_negative_reference_comparisons": 0,
        }
    trace = np.empty((2048, 6))
    worst: WorstObservation | None = None
    with (
        gzip.open(output / f"{split}_comparisons.csv.gz", "wt", newline="") as stream,
        gzip.open(output / f"{split}_cases.jsonl.gz", "wt") as case_stream,
    ):
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "family",
                "probe",
                "guard_m",
                "left_s",
                "right_s",
                "lower_m",
                "eta_m",
                "rhs_m",
                "accepted",
                "reference_residual_m",
                "signed_error_m",
            )
        )
        for index, (family, original) in enumerate(cases(n, seed)):
            case_stream.write(json.dumps({"index": index, "family": family, **serialisable(original)}) + "\n")
            family_stats = stats["families"].setdefault(
                family, {"encounters": 0, "comparisons": 0, "positive_error_max_m": 0.0}
            )
            family_stats["encounters"] += 1
            cache: dict[tuple[float, float], ReferenceCertificate] = {}

            def reference(
                case: TurnCase,
                t0: float,
                t1: float,
                cache: dict[tuple[float, float], ReferenceCertificate] = cache,
                original: TurnCase = original,
                index: int = index,
                family: str = family,
            ) -> mp.mpf:
                key = (t0, t1)
                if key not in cache:
                    ref = certificate(original, t0, t1)
                    stats["reference_intervals"] += 1
                    # Regular checks plus the worst observations are recomputed at
                    # 120 digits. The inputs remain the original binary64 values.
                    if index % 37 == 0:
                        high = certificate(original, t0, t1, dps=120)
                        stats["precision_rechecks"] += 1
                        counts = stats["precision_recheck_families"]
                        counts[family] = counts.get(family, 0) + 1
                        change = float(abs(high["residual"] - ref["residual"]))
                        stats["precision_change_max_m"] = max(stats["precision_change_max_m"], change)
                    cache[key] = ref
                ref = cache[key]
                return (
                    ref["residual"]
                    + mp.mpf(original["separation_threshold_m"])
                    - mp.mpf(case["separation_threshold_m"])
                )

            trace.fill(np.nan)
            base = solver(**original, study_trace=trace)
            production = production_solver(**original, directional_mode=0)
            require_solver_parity(base, production, f"{split}/{index}/{family}")
            stats["production_parity_checks"] += 1
            base_rows = trace[~np.isnan(trace[:, 0])].copy()
            for guard in guards:
                probes = [("original", original)]
                # The reference geometry is unchanged when only D changes.
                if len(base_rows):
                    for label, threshold in boundary_thresholds(float(base_rows[0, 2]), float(base_rows[0, 3]), guard):
                        threshold_case = original.copy()
                        threshold_case["separation_threshold_m"] = threshold
                        probes.append((label, threshold_case))
                for probe, case in probes:
                    trace.fill(np.nan)
                    answer = solver(**case, study_trace=trace, study_guard=guard)
                    stats["solver_runs"] += 1
                    candidate = stats["candidates"][str(guard)]
                    candidate["solver_runs"] += 1
                    candidate["safe_returns"] += bool(answer[0])
                    if guard == ra.TURN_CERT_FP_GUARD_M:
                        actual = production_solver(**case, directional_mode=0)
                        require_solver_parity(answer, actual, f"{split}/{index}/{probe}")
                        stats["production_parity_checks"] += 1
                    for t0, t1, lower, eta, rhs, accepted in trace[~np.isnan(trace[:, 0])]:
                        residual = reference(case, float(t0), float(t1))
                        error = mp.mpf(float(lower)) - mp.mpf(float(rhs)) - (residual - mp.mpf(guard))
                        signed = float(error)
                        stats["comparisons"] += 1
                        family_stats["comparisons"] += 1
                        stats["absolute_error_max_m"] = max(stats["absolute_error_max_m"], abs(signed))
                        family_stats["positive_error_max_m"] = max(family_stats["positive_error_max_m"], signed)
                        if signed > stats["positive_error_max_m"]:
                            stats["positive_error_max_m"] = signed
                            worst = {
                                "case": serialisable(case),
                                "family": family,
                                "index": index,
                                "probe": probe,
                                "guard_m": guard,
                                "left_s": float(t0),
                                "right_s": float(t1),
                                "signed_error_m": signed,
                                "lower_m": float(lower),
                                "rhs_m": float(rhs),
                            }
                        candidate["accepted_comparisons"] += int(accepted)
                        violation = bool(accepted and residual < 0)
                        candidate["accepted_negative_reference_comparisons"] += violation
                        stats["accepted_negative_reference_comparisons"] += violation
                        writer.writerow(
                            (
                                index,
                                family,
                                probe,
                                guard,
                                t0,
                                t1,
                                lower,
                                eta,
                                rhs,
                                int(accepted),
                                mp.nstr(residual, 30),
                                repr(signed),
                            )
                        )
            if (index + 1) % 100 == 0:
                print(
                    f"{split}: {index + 1}/{n}; max optimistic error {stats['positive_error_max_m']:.4g} m", flush=True
                )
    require_coverage(stats["comparisons"], "scalar")
    if worst:
        case = worst["case"]
        ref80 = certificate(case, worst["left_s"], worst["right_s"], dps=80)
        ref120 = certificate(case, worst["left_s"], worst["right_s"], dps=120)
        stats["precision_rechecks"] += 1
        stats["precision_change_max_m"] = max(
            stats["precision_change_max_m"], float(abs(ref120["residual"] - ref80["residual"]))
        )
    stats["worst_positive_error"] = worst
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=600)
    parser.add_argument("--validation", type=int, default=1800)
    parser.add_argument("--audit-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.audit_worker:
        run_isolated(
            "paper.run_turn_numerics", [*sys.argv[1:], "--audit-worker"], Path(ra.__file__).resolve().parents[1]
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 80
    solver, instrumented_source = traced_solver()
    (args.output_dir / "instrumented_solver.py").write_text(instrumented_source + "\n")
    start = time.monotonic()
    calibration = run_split(solver, args.calibration, 2026092001, args.output_dir, "calibration", None)
    # Fixed before validation: require >=100 times the calibration maximum,
    # round upwards to a decimal power, and retain the existing 1 um guard if
    # it is larger. This is an empirical allowance, not an error theorem.
    error = calibration["positive_error_max_m"]
    selected = max(ra.TURN_CERT_FP_GUARD_M, 10.0 ** math.ceil(math.log10(max(100 * error, 1e-15))))
    (args.output_dir / "calibration.json").write_text(
        json.dumps({"results": calibration, "selected_guard_m": selected}, indent=2)
    )
    print(f"Frozen guard before validation: {selected:g} m", flush=True)
    validation = run_split(solver, args.validation, 2026092002, args.output_dir, "validation", selected)
    sources = [
        Path(__file__),
        Path(inspect.getfile(certificate)),
        Path(__file__).with_name("numerical_validation.py"),
        *sorted(Path(ra.__file__).parent.glob("*.py")),
    ]
    report = {
        "calibration": calibration,
        "validation": validation,
        "selected_guard_m": selected,
        "existing_guard_m": ra.TURN_CERT_FP_GUARD_M,
        "selection_rule": "max(existing guard, decimal power ceiling of 100 times calibration optimistic error)",
        "seeds": [2026092001, 2026092002],
        "reference_decimal_digits": [80, 120],
        "scope": (
            "Local planar positive-horizon turn certification; excludes straight delegation, "
            "coordinate projection and measurement error"
        ),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "mpmath": mp.__version__,
        },
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "elapsed_s": time.monotonic() - start,
    }
    report["validation_passed"] = (
        validation["positive_error_max_m"] * 100 <= selected
        and validation["candidates"][str(selected)]["accepted_negative_reference_comparisons"] == 0
        and max(calibration["precision_change_max_m"], validation["precision_change_max_m"]) < 1e-30
    )
    report["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir() if p.is_file()
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("selected_guard_m", "validation_passed", "elapsed_s")}), flush=True)
    if not report["validation_passed"]:
        raise RuntimeError("Held-out validation failed; do not claim support for the selected guard")


if __name__ == "__main__":
    main()
