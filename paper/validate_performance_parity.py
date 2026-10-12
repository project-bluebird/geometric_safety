"""Compare complete isolated packages on fixed, seeded and boundary inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from paper.benchmark_library_versions import KT, Case, digest, list_value, number, object_json, object_value, read_cases


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


def worker(cases_path: Path, output: Path) -> None:
    from geometric_safety import heading, relevant_aircraft as ra

    rows = []
    cases = corpus(cases_path)
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
                "core": [bool(core[0]), float(core[1]), float(core[2])],
                "interface": asdict(interface),
            }
        )
    package = Path(ra.__file__).parent
    output.write_text(
        json.dumps(
            {
                "seed": 2026101206,
                "cases_sha256": digest(cases_path),
                "source_sha256": {p.name: digest(p) for p in sorted(package.glob("*.py"))},
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.cases, args.output)
        return
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required for a comparison")
    args.output.mkdir(exist_ok=False)
    reports = []
    for role, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        env = dict(
            os.environ,
            PYTHONPATH=str(source.resolve()),
            NUMBA_CACHE_DIR=str((args.output / (role + "_cache")).resolve()),
        )
        path = args.output / (role + ".json")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--cases",
                str(args.cases.resolve()),
                "--output",
                str(path.resolve()),
            ],
            env=env,
            check=True,
        )
        reports.append(object_json(path.read_text()))
    before, after = [list_value(report["records"]) for report in reports]
    if not before or not after:
        raise ValueError("Parity comparison has no records")
    if len(before) != len(after):
        raise ValueError("Parity record counts differ")
    distance_delta = {"core": 0.0, "interface": 0.0}
    time_changes = {"core": 0, "interface": 0}
    for left, right in zip(before, after, strict=True):
        a, b = object_value(left), object_value(right)
        if a["id"] != b["id"]:
            raise ValueError(f"Parity case IDs differ: {a['id']} / {b['id']}")
        ac, bc = list_value(a["core"]), list_value(b["core"])
        ai, bi = object_value(a["interface"]), object_value(b["interface"])
        if ac[0] != bc[0]:
            raise ValueError(f"Core decisions differ for {a['id']}: {ac} / {bc}")
        for field in ("status", "reason", "inflation_m"):
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
            if d0 is None or d1 is None:
                raise ValueError(f"Missing {method} distance for {a['id']}")
            if not math.isclose(number(d0), number(d1), rel_tol=2e-13, abs_tol=1e-7):
                raise ValueError(f"{method} distances differ for {a['id']}: {d0} / {d1}")
            distance_delta[method] = max(distance_delta[method], abs(number(d1) - number(d0)))
            time_changes[method] += t0 != t1
    summary = {
        "cases": len(before),
        "queries": 2 * len(before),
        "decision_changes": 0,
        "maximum_distance_change_m": distance_delta,
        "diagnostic_time_changes": time_changes,
        "seed": 2026101206,
        "guard_m": 1e-6,
        "source_sha256": {
            role: report["source_sha256"] for role, report in zip(("baseline", "candidate"), reports, strict=True)
        },
        "script_sha256": digest(Path(__file__)),
        "cases_sha256": digest(args.cases),
        "raw_sha256": {p.name: digest(p) for p in args.output.glob("*.json")},
        "scope": "Empirical parity, including envelope and boundary cases; not a universal numerical proof.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
