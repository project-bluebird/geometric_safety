"""Compare complete isolated packages on fixed, seeded and boundary inputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import TYPE_CHECKING, Literal, TypedDict

from geometric_safety import heading, relevant_aircraft as ra
from paper.benchmark_contracts import (
    Json,
    boolean,
    digest,
    finite_number,
    integer,
    list_value,
    module_hashes,
    number,
    object_json,
    object_value,
    parse_list,
    parse_mapping,
    string,
)
from paper.benchmark_library_versions import KT, Case, read_cases
from paper.isolated_worker import worker_command

if TYPE_CHECKING:
    from paper.numerical_validation import SolverResult


class ParityInterface(TypedDict):
    status: Literal["certified_safe", "unresolved", "unsupported"]
    reason: str
    inflation_m: float
    base_min_distance_m: float | None
    base_closest_time_s: float | None
    diagnostic_kind: Literal["straight_minimum", "sampled_minimum", "interval_bound"] | None
    base_bound_interval_s: tuple[float, float] | None


class ParityRecord(TypedDict):
    id: str
    suite: str
    core: SolverResult
    interface: ParityInterface


class ParityReport(TypedDict):
    seed: int
    cases_sha256: str
    package_root: str
    source_sha256: dict[str, str]
    corpus: Literal["standard", "saved_only"]
    guard_m: float
    records: list[ParityRecord]


class ParityComparison(TypedDict):
    cases: int
    maximum_distance_change_m: dict[str, float]
    diagnostic_time_changes: dict[str, int]


class ParitySummary(ParityComparison):
    queries: int
    decision_changes: int
    seed: int
    corpus: Literal["standard", "saved_only"]
    guard_m: float
    source_sha256: dict[str, dict[str, str]]
    script_sha256: str
    cases_sha256: str
    raw_sha256: dict[str, str]
    scope: str


def corpus(path: Path) -> list[Case]:
    saved = read_cases(path)
    result = saved.copy()
    rng = random.Random(2026101206)
    for i in range(20000):
        ah, bh = rng.uniform(-720, 720), rng.uniform(-720, 720)
        da, db = rng.uniform(-180, 180), rng.uniform(-180, 180)
        if i % 8 == 0:
            da = rng.choice([0.0, 1e-12, -1e-12, 180.0, math.nextafter(180.0, 0.0)])
        if i % 8 == 1:
            db = da = 0.0
        radius = rng.uniform(0.0, 100000.0)
        angle = rng.uniform(-math.pi, math.pi)
        result.append(
            Case(
                f"seeded-{i}",
                "seeded",
                radius * math.cos(angle),
                radius * math.sin(angle),
                ah,
                ah + da,
                rng.uniform(100, 650),
                rng.choice([0.03, 0.3, 1.5, 3.0]),
                bh,
                bh + db,
                rng.uniform(100, 650),
                rng.choice([0.03, 0.3, 1.5, 3.0]),
                rng.choice([0.0, 1e-12, 5.0, 40.0]),
                rng.choice([0.0, 1.0, 60.0, 600.0, 3600.0]),
                rng.choice([100.0, 9260.0, 18520.0]),
            )
        )
    # Sweep the numerical acceptance guard on the analytically known circular case.
    base = Case(
        id="guard-circle",
        suite="guard_boundary",
        x=0.0,
        y=9260.0,
        ah=0.0,
        at=170.0,
        av=300.0 / KT,
        ar=0.03,
        bh=0.0,
        bt=170.0,
        bv=300.0 / KT,
        br=0.03,
        width=0.0,
        horizon=170.0 / 0.03,
        threshold=9260.0,
    )
    for delta in (-2e-6, -1e-6, -1e-7, 0.0, 1e-7, 0.9e-6, 1e-6, 1.1e-6, 2e-6, 1e-5, 0.01):
        radius = 9260.0 + delta
        alpha = math.degrees(math.asin(radius * math.radians(base.ar) / 600.0))
        for factor in (0.5, 1.0, 2.0):
            result.append(
                replace(
                    base,
                    id=f"boundary-{delta}-{factor}",
                    suite="guard_boundary",
                    y=radius,
                    ah=alpha,
                    at=alpha + 170.0,
                    bh=-alpha,
                    bt=-alpha + 170.0,
                    horizon=factor * 170.0 / base.ar,
                    width=0.0,
                )
            )
    return result


def evaluate_cases(cases: list[Case]) -> list[ParityRecord]:
    """Record both production interfaces on the same ordered encounter inputs."""
    rows: list[ParityRecord] = []
    for i, case in enumerate(cases):
        p = case.packed()
        # Cover all private certificate modes and the legacy envelope paths.
        style = i % 7 if case.suite == "seeded" else 0
        core = ra._catch_up_projection_interval_with_turns_local(
            p[:2],
            *[float(value) for value in p[2:]],
            directional_mode=i % 3 if style else 2,
            use_interval_local_lipschitz=style != 1,
            eta_const_m=50.0 if style == 2 else 0.0,
            eta_rate_mps=0.5 if style == 3 else 0.0,
            turn_delay_max_s=2.0 if style == 4 else 0.0,
            turn_rate_error_max=0.1 if style == 5 else 0.0,
            turn_speed_schedule_uncertainty_kt=3.0 if style == 6 else 0.0,
        )
        a = heading.HeadingMotion(
            heading0_deg=case.ah,
            target_heading_deg=case.at,
            turn_rate_deg_s=case.ar,
            speed_min_mps=max(0, case.av - case.width) * KT,
            speed_max_mps=(case.av + case.width) * KT,
            heading_deviation_max_deg=0.1 if style == 2 else 0.0,
        )
        b = heading.HeadingMotion(
            heading0_deg=case.bh,
            target_heading_deg=case.bt,
            turn_rate_deg_s=case.br,
            speed_min_mps=max(0, case.bv - case.width / 2) * KT,
            speed_max_mps=(case.bv + case.width / 2) * KT,
        )
        interface = heading.certify_heading_separation(
            (case.x, case.y),
            a,
            b,
            separation_threshold_m=case.threshold,
            horizon_s=case.horizon,
            relative_position_error_m=50.0 if style == 3 else 0.0,
            relative_velocity_error_mps=0.5 if style == 4 else 0.0,
        )
        rows.append(
            {
                "id": case.id,
                "suite": case.suite,
                "core": (bool(core[0]), float(core[1]), float(core[2])),
                "interface": {
                    "status": interface.status,
                    "reason": interface.reason,
                    "inflation_m": interface.inflation_m,
                    "base_min_distance_m": interface.base_min_distance_m,
                    "base_closest_time_s": interface.base_closest_time_s,
                    "diagnostic_kind": interface.diagnostic_kind,
                    "base_bound_interval_s": interface.base_bound_interval_s,
                },
            }
        )
    return rows


def worker(cases_path: Path, output: Path, *, saved_only: bool = False) -> None:
    cases = read_cases(cases_path) if saved_only else corpus(cases_path)
    if not cases:
        raise ValueError("Parity comparison has no cases")
    rows = evaluate_cases(cases)
    package = Path(ra.__file__).parent
    report: ParityReport = {
        "seed": 2026101206,
        "cases_sha256": digest(cases_path),
        "package_root": str(package.resolve()),
        "source_sha256": module_hashes(package),
        "corpus": "saved_only" if saved_only else "standard",
        "guard_m": ra.TURN_CERT_FP_GUARD_M,
        "records": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")


def _diagnostic_number(value: Json) -> float:
    result = number(value)
    if math.isinf(result):
        raise ValueError("A distance or time diagnostic cannot be infinite")
    return result


def _unavailable(value: float | None) -> bool:
    return value is None or math.isnan(value)


def parse_record(value: Json) -> ParityRecord:
    row = object_value(value)
    core, interface = list_value(row["core"]), object_value(row["interface"])
    if len(core) != 3:
        raise ValueError("Parity core result must have three entries")
    status: Literal["certified_safe", "unresolved", "unsupported"]
    if interface["status"] == "certified_safe":
        status = "certified_safe"
    elif interface["status"] == "unresolved":
        status = "unresolved"
    elif interface["status"] == "unsupported":
        status = "unsupported"
    else:
        raise ValueError("Unknown parity interface status")
    distance, time_s = interface["base_min_distance_m"], interface["base_closest_time_s"]
    kind: Literal["straight_minimum", "sampled_minimum", "interval_bound"] | None
    if interface["diagnostic_kind"] == "straight_minimum":
        kind = "straight_minimum"
    elif interface["diagnostic_kind"] == "sampled_minimum":
        kind = "sampled_minimum"
    elif interface["diagnostic_kind"] == "interval_bound":
        kind = "interval_bound"
    elif interface["diagnostic_kind"] is None:
        kind = None
    else:
        raise ValueError("Unknown parity diagnostic kind")
    interval_value = interface["base_bound_interval_s"]
    interval = None
    if interval_value is not None:
        bounds = list_value(interval_value)
        if len(bounds) != 2:
            raise ValueError("A bound interval must have two endpoints")
        interval = (finite_number(bounds[0]), finite_number(bounds[1]))
        if interval[0] > interval[1]:
            raise ValueError("A bound interval must be ordered")
    parsed_core = (boolean(core[0]), _diagnostic_number(core[1]), _diagnostic_number(core[2]))
    if parsed_core[0] and any(math.isnan(value) for value in parsed_core[1:]):
        raise ValueError("An accepted core result requires finite diagnostics")

    return ParityRecord(
        id=string(row["id"]),
        suite=string(row["suite"]),
        core=parsed_core,
        interface=ParityInterface(
            status=status,
            reason=string(interface["reason"]),
            inflation_m=finite_number(interface["inflation_m"]),
            base_min_distance_m=None if distance is None else finite_number(distance),
            base_closest_time_s=None if time_s is None else finite_number(time_s),
            diagnostic_kind=kind,
            base_bound_interval_s=interval,
        ),
    )


def read_parity_report(path: Path) -> ParityReport:
    row = object_json(path.read_text())
    selected_corpus: Literal["standard", "saved_only"]
    if row["corpus"] == "standard":
        selected_corpus = "standard"
    elif row["corpus"] == "saved_only":
        selected_corpus = "saved_only"
    else:
        raise ValueError("Unknown parity corpus")
    records = parse_list(row["records"], parse_record)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Parity report contains duplicate case IDs")
    return ParityReport(
        seed=integer(row["seed"]),
        cases_sha256=string(row["cases_sha256"]),
        package_root=string(row["package_root"]),
        source_sha256=parse_mapping(row["source_sha256"], string),
        corpus=selected_corpus,
        guard_m=finite_number(row["guard_m"]),
        records=records,
    )


def run_worker(source: Path, cases_path: Path, output: Path, cache: Path, *, saved_only: bool = False) -> ParityReport:
    """Run the current harness against precisely the selected library package."""
    package = source.resolve() / "geometric_safety"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Selected library has no geometric_safety package: {source}")
    expected = module_hashes(package)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache.resolve()), PYTHONDONTWRITEBYTECODE="1")
    arguments = ["--worker", "--cases", str(cases_path.resolve()), "--output", str(output.resolve())]
    if saved_only:
        arguments.append("--saved-only")
    subprocess.run(worker_command("paper.validate_performance_parity", arguments, source), env=env, check=True)
    report = read_parity_report(output)
    if Path(report["package_root"]) != package or report["source_sha256"] != expected:
        raise RuntimeError("Parity worker library identity differs from the selected source")
    if report["cases_sha256"] != digest(cases_path):
        raise RuntimeError("Parity worker evaluated a different case file")
    expected_corpus = "saved_only" if saved_only else "standard"
    expected_cases = read_cases(cases_path) if saved_only else corpus(cases_path)
    if report["corpus"] != expected_corpus or [record["id"] for record in report["records"]] != [
        case.id for case in expected_cases
    ]:
        raise RuntimeError("Parity worker did not evaluate the requested corpus")
    return report


def compare_reports(baseline: ParityReport, candidate: ParityReport) -> ParityComparison:
    """Validate saved decisions and distances, and count diagnostic time changes."""
    if baseline["corpus"] != candidate["corpus"] or baseline["cases_sha256"] != candidate["cases_sha256"]:
        raise ValueError("Parity reports describe different input corpora")
    before, after = baseline["records"], candidate["records"]
    if not before or not after:
        raise ValueError("Parity comparison has no records")
    if len(before) != len(after):
        raise ValueError("Parity record counts differ")
    distance_delta = {"core": 0.0, "interface": 0.0}
    time_changes = {"core": 0, "interface": 0}
    for a, b in zip(before, after, strict=True):
        if a["id"] != b["id"]:
            raise ValueError(f"Parity case IDs differ: {a['id']} / {b['id']}")
        ac, bc = a["core"], b["core"]
        ai, bi = a["interface"], b["interface"]
        if ac[0] != bc[0]:
            raise ValueError(f"Core decisions differ for {a['id']}: {ac} / {bc}")
        for field in ("status", "reason", "inflation_m", "diagnostic_kind"):
            if ai[field] != bi[field]:
                raise ValueError(f"Interface {field} differs for {a['id']}: {ai[field]} / {bi[field]}")
        for method, d0, d1, t0, t1 in (
            ("core", ac[1], bc[1], ac[2], bc[2]),
            (
                "interface",
                ai["base_min_distance_m"],
                bi["base_min_distance_m"],
                ai["base_closest_time_s"],
                bi["base_closest_time_s"],
            ),
        ):
            if _unavailable(d0) != _unavailable(d1):
                raise ValueError(f"{method} distance availability differs for {a['id']}")
            if d0 is not None and d1 is not None and not math.isnan(d0):
                if not math.isclose(d0, d1, rel_tol=2e-13, abs_tol=1e-7):
                    raise ValueError(f"{method} distances differ for {a['id']}: {d0} / {d1}")
                distance_delta[method] = max(distance_delta[method], abs(d1 - d0))
            time_changed = _unavailable(t0) != _unavailable(t1) or (not _unavailable(t0) and t0 != t1)
            if method == "interface":
                time_changed |= ai["base_bound_interval_s"] != bi["base_bound_interval_s"]
            time_changes[method] += time_changed
    return {
        "cases": len(before),
        "maximum_distance_change_m": distance_delta,
        "diagnostic_time_changes": time_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--saved-only",
        action="store_true",
        help="Compare only the supplied cases; omit the standard seeded and boundary corpus",
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.cases, args.output, saved_only=args.saved_only)
        return
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required for a comparison")
    args.output.mkdir(exist_ok=False)
    reports: list[ParityReport] = []
    for role, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        path = args.output / (role + ".json")
        reports.append(
            run_worker(source, args.cases, path, args.output / (role + "_cache"), saved_only=args.saved_only)
        )
    comparison = compare_reports(reports[0], reports[1])
    summary: ParitySummary = {
        "cases": comparison["cases"],
        "maximum_distance_change_m": comparison["maximum_distance_change_m"],
        "diagnostic_time_changes": comparison["diagnostic_time_changes"],
        "queries": 2 * comparison["cases"],
        "decision_changes": 0,
        "seed": 2026101206,
        "corpus": reports[0]["corpus"],
        "guard_m": reports[1]["guard_m"],
        "source_sha256": {
            role: report["source_sha256"] for role, report in zip(("baseline", "candidate"), reports, strict=True)
        },
        "script_sha256": digest(Path(__file__)),
        "cases_sha256": digest(args.cases),
        "raw_sha256": {p.name: digest(p) for p in args.output.glob("*.json")},
        "scope": (
            "Empirical parity on the supplied cases only; seeded and boundary cases were omitted."
            if args.saved_only
            else "Empirical parity, including envelope and boundary cases; not a universal numerical proof."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
