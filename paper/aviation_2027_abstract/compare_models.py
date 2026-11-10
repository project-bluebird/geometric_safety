"""Compare speed models on the saved, distinct ten-minute paper geometries."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import TypedDict

import mpmath as mp

from geometric_safety import HeadingMotion, certify_heading_separation
from paper.aviation_2027_abstract.study_data import CaseInput, ComparedRow, LibraryDecision, QueryInput
from paper.heading_study_reference import Classification, Model, Motion, Query, classify

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPOSITORY = ROOT.parents[1]
type Geometry = tuple[str, int, int]


class Reference(TypedDict):
    status: Classification
    reason: str
    lower_m: str
    upper_m: str
    attaining_time_s: str | None
    constant_speeds_mps: list[str] | None
    precision_agrees: bool


class Comparison(TypedDict):
    case: CaseInput
    variable_reference: str
    library: LibraryDecision
    constant: Reference
    corners: Classification
    corner_checks: list[Reference]
    corner_basis: str


class Wider(TypedDict):
    case: CaseInput
    reference: Reference
    library: LibraryDecision


def geometry(case: CaseInput) -> Geometry:
    if case["family"] == "overtaking":
        return case["family"], case["da_nm"] - case["db_nm"], 0
    return case["family"], case["da_nm"], case["db_nm"]


def query_from(data: QueryInput) -> Query:
    return Query(
        (data["relative_position_m"][0], data["relative_position_m"][1]),
        Motion(**data["a"]),
        Motion(**data["b"]),
        data["horizon_s"],
        data["threshold_m"],
    )


def checked(query: Query, model: Model) -> Reference:
    first = classify(query, model, digits=80)
    second = classify(query, model, digits=120)
    if first.classification != second.classification:
        raise ArithmeticError(f"Precision disagreement for {query}")
    speeds = None
    if model == "C" and second.witness is not None and second.witness.time > 0:
        with mp.workdps(120):
            speeds = [
                mp.nstr(
                    sum(
                        weight * getattr(vertex, aircraft)[0].speed
                        for weight, vertex in zip(second.witness.weights, second.witness.vertices, strict=True)
                    ),
                    60,
                )
                for aircraft in ("a_segments", "b_segments")
            ]
    return {
        "status": second.classification,
        "reason": second.reason,
        "lower_m": mp.nstr(second.lower_m, 60),
        "upper_m": mp.nstr(second.upper_m, 60),
        "attaining_time_s": None if second.witness is None else mp.nstr(second.witness.time, 60),
        "constant_speeds_mps": speeds,
        "precision_agrees": True,
    }


def compare(row: ComparedRow) -> Comparison:
    query = query_from(row["case"]["query"])
    constant = checked(query, "C")
    corners: Classification = "separated"
    corner_checks: list[Reference] = []
    basis = "All constant speeds are separated, which includes the four corners"
    if query.a.lower_mps == query.a.upper_mps and query.b.lower_mps == query.b.upper_mps:
        corners = constant["status"]
        basis = "Zero-width bounds give one constant-speed trajectory"
    elif constant["status"] != "separated":
        basis = "Individual corner trajectories; stop after a demonstrated loss"
        for speed_a, speed_b in sorted(
            set(
                product(
                    (query.a.lower_mps, query.a.upper_mps),
                    (query.b.lower_mps, query.b.upper_mps),
                )
            )
        ):
            corner = replace(
                query,
                a=replace(query.a, lower_mps=speed_a, upper_mps=speed_a),
                b=replace(query.b, lower_mps=speed_b, upper_mps=speed_b),
            )
            result = checked(corner, "C")
            corner_checks.append(result)
            if result["status"] == "conflict":
                corners = "conflict"
                break
            if result["status"] == "unresolved":
                corners = "unresolved"
    return {
        "case": row["case"],
        "variable_reference": row["reference"],
        "library": row["library"],
        "constant": constant,
        "corners": corners,
        "corner_checks": corner_checks,
        "corner_basis": basis,
    }


def heading(motion: Motion) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=motion.heading_deg,
        target_heading_deg=motion.target_deg,
        turn_rate_deg_s=motion.rate_deg_s,
        speed_min_mps=motion.lower_mps,
        speed_max_mps=motion.upper_mps,
    )


def wider(case: CaseInput) -> Wider:
    query = query_from(case["query"])
    reference = checked(query, "V")
    result = certify_heading_separation(
        query.relative_position_m,
        heading(query.a),
        heading(query.b),
        separation_threshold_m=query.threshold_m,
        horizon_s=query.horizon_s,
    )
    if result.status == "certified_safe" and reference["status"] == "conflict":
        raise ArithmeticError(f"Library/reference disagreement: {case}")
    return {
        "case": case,
        "reference": reference,
        "library": {
            "status": result.status,
            "reason": result.reason,
            "profile_min_distance_lower_bound_m": result.profile_min_distance_lower_bound_m,
        },
    }


def source_hashes() -> dict[str, str]:
    paths = [
        *REPOSITORY.joinpath("geometric_safety").glob("*.py"),
        REPOSITORY / "paper/heading_study_reference.py",
        Path(__file__).resolve(),
        DATA / "library-results.jsonl",
    ]
    return {p.relative_to(REPOSITORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> None:
    before = source_hashes()
    # The two centre speeds meet at t=100 s. Every corner trajectory has
    # at least 10,000*|v_B-v_A/2|/hypot(v_A,v_B) metres of clearance (>399 m).
    control = Query(
        (-10000.0, 5000.0),
        Motion(90, 90, 0, 90, 110),
        Motion(0, 0, 0, 40, 60),
        200,
        100,
    )
    assert checked(control, "C")["status"] == "conflict"
    for a_speed, b_speed in product((90, 110), (40, 60)):
        corner = replace(
            control,
            a=replace(control.a, lower_mps=a_speed, upper_mps=a_speed),
            b=replace(control.b, lower_mps=b_speed, upper_mps=b_speed),
        )
        assert checked(corner, "C")["status"] == "separated"
    rows: list[ComparedRow] = [json.loads(line) for line in (DATA / "library-results.jsonl").read_text().splitlines()]
    unique: dict[tuple[Geometry, int, int], ComparedRow] = {}
    for row in rows:
        case = row["case"]
        if case["horizon_s"] != 600:
            continue
        identity = geometry(case), case["scale"], case["turn_deg"]
        if identity in unique:
            previous = unique[identity]
            assert (row["reference"], row["library"]["status"]) == (
                previous["reference"],
                previous["library"]["status"],
            )
        else:
            unique[identity] = row
    assert len(unique) == 621
    with (
        ProcessPoolExecutor(max_workers=4) as pool,
        (DATA / "model-comparison.jsonl").open("w") as output,
    ):
        futures = [pool.submit(compare, row) for row in unique.values()]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
            if count % 20 == 0 or count == len(futures):
                print(f"Constant/corner comparison: {count}/{len(futures)}", flush=True)
    geometries = {identity[0] for identity in unique}
    empty = {
        g
        for g in geometries
        if not any(
            row["library"]["status"] == "certified_safe"
            for (shape, scale, _), row in unique.items()
            if shape == g and scale == 2
        )
    }
    assert len(empty) == 7
    extensions: list[CaseInput] = []
    for g in sorted(empty):
        original = unique[(g, 2, 0)]["case"]
        for turn in (*range(-90, -20, 5), *range(25, 91, 5)):
            case = deepcopy(original)
            case["turn_deg"] = turn
            case["query"]["a"]["target_deg"] = case["query"]["a"]["heading_deg"] + turn
            extensions.append(case)
    with (
        ProcessPoolExecutor(max_workers=4) as pool,
        (DATA / "doubled-wider.jsonl").open("w") as output,
    ):
        futures = [pool.submit(wider, case) for case in extensions]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
            if count % 20 == 0 or count == len(futures):
                print(f"Doubled wider turns: {count}/{len(futures)}", flush=True)
    assert source_hashes() == before
    (DATA / "comparison-manifest.json").write_text(json.dumps(before, indent=2) + "\n")
    compared: list[Comparison] = [
        json.loads(line) for line in (DATA / "model-comparison.jsonl").read_text().splitlines()
    ]
    counts = Counter((r["corners"], r["constant"]["status"], r["variable_reference"]) for r in compared)
    print(json.dumps({str(k): v for k, v in counts.items()}, indent=2))


if __name__ == "__main__":
    main()
