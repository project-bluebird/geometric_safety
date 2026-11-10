"""Evaluate a frozen heading-menu protocol against independent speed-profile sets."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import time

import mpmath as mp
import numpy as np

import geometric_safety
from geometric_safety import HeadingMotion, certify_heading_separation
from geometric_safety.heading import PROFILE_REFINEMENT_MARGIN_M
from geometric_safety.relevant_aircraft import _catch_up_projection_interval_with_turns_local
from paper.benchmark_contracts import (
    Json,
    digest,
    finite_number,
    integer,
    module_hashes,
    object_json,
    object_value,
    parse_list,
    string,
)
from paper.heading_study_reference import Classification, Model, Motion, Query, ReferenceResult, SpeedSegment, classify


@dataclass(frozen=True)
class StudyCase:
    identifier: str
    scene: str
    target_deg: float
    width_kt: float
    a_nominal_kt: float
    b_nominal_kt: float
    query: Query


@dataclass(frozen=True)
class Settings:
    digits: int
    recheck_digits: int
    margin_m: float
    max_support: int
    max_cells: int


def load_protocol(path: Path) -> tuple[list[StudyCase], Settings]:
    protocol = object_json(path.read_text())
    errors = object_value(protocol["additional_errors"])
    if any(finite_number(value) != 0 for value in errors.values()):
        raise ValueError("The independent study reference covers zero additional errors")
    if protocol["turn_direction"] != "shortest" or finite_number(protocol["turn_delay_s"]) != 0:
        raise ValueError("The protocol requires immediate shortest turns")
    frame, own = object_value(protocol["frame"]), object_value(protocol["ownship"])
    unit = finite_number(frame["nm_to_m"])
    if frame["relative_position"] != "A minus B" or frame["kt_to_mps"] != "1852/3600" or unit != 1852:
        raise ValueError("Unexpected frame or unit contract")
    knot = 1852 / 3600
    own_position = parse_list(own["position_nm"], finite_number)
    if len(own_position) != 2:
        raise ValueError("Expected two ownship coordinates")
    own_heading, own_speed = finite_number(own["initial_heading_deg"]), finite_number(own["nominal_speed_kt"])
    rate = finite_number(protocol["turn_rate_deg_s"])
    threshold = unit * finite_number(protocol["separation_threshold_nm"])
    cases: list[StudyCase] = []
    for scene in parse_list(protocol["pair_scenes"], object_value):
        name = string(scene["id"])
        position = parse_list(scene["traffic_position_nm"], finite_number)
        if len(position) != 2:
            raise ValueError("Expected two traffic coordinates")
        relative = ((own_position[0] - position[0]) * unit, (own_position[1] - position[1]) * unit)
        traffic_speed = finite_number(scene["traffic_nominal_speed_kt"])
        for horizon in parse_list(protocol["horizon_s"], finite_number):
            for width in parse_list(protocol["speed_half_widths_kt"], finite_number):
                for target in parse_list(protocol["target_headings_deg"], finite_number):
                    a = Motion(own_heading, target, rate, (own_speed - width) * knot, (own_speed + width) * knot)
                    b = Motion(
                        finite_number(scene["traffic_initial_heading_deg"]),
                        finite_number(scene["traffic_target_heading_deg"]),
                        rate,
                        (traffic_speed - width) * knot,
                        (traffic_speed + width) * knot,
                    )
                    identifier = f"{name}_t{horizon:g}_w{width:g}_h{target:g}"
                    cases.append(
                        StudyCase(
                            identifier,
                            name,
                            target,
                            width,
                            own_speed,
                            traffic_speed,
                            Query(relative, a, b, horizon, threshold),
                        )
                    )
    if len(cases) != integer(protocol["query_count_per_model"]) or len({case.identifier for case in cases}) != len(
        cases
    ):
        raise ValueError("Case enumeration differs from the frozen protocol")
    reference = object_value(protocol["reference"])
    settings = Settings(
        integer(reference["precision_decimal_digits"]),
        integer(reference["decisive_result_recheck_digits"]),
        finite_number(reference["classification_exclusion_band_m"]),
        integer(reference["max_support_evaluations_per_query"]),
        integer(reference["max_time_cells_per_query"]),
    )
    return cases, settings


def heading_motion(motion: Motion) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=motion.heading_deg,
        target_heading_deg=motion.target_deg,
        turn_rate_deg_s=motion.rate_deg_s,
        speed_min_mps=motion.lower_mps,
        speed_max_mps=motion.upper_mps,
    )


def core_accepts(case: StudyCase, width: float) -> bool:
    query = case.query
    return bool(
        _catch_up_projection_interval_with_turns_local(
            rel_pos0=np.array(query.relative_position_m, dtype=np.float64),
            a_heading0_deg=query.a.heading_deg,
            a_target_heading_deg=query.a.target_deg,
            a_speed_kt=case.a_nominal_kt,
            a_turn_rate_deg_sec=query.a.rate_deg_s,
            b_heading0_deg=query.b.heading_deg,
            b_target_heading_deg=query.b.target_deg,
            b_speed_kt=case.b_nominal_kt,
            b_turn_rate_deg_sec=query.b.rate_deg_s,
            separation_threshold_m=query.threshold_m,
            speed_diff_kt=width,
            projection_time_s=query.horizon_s,
            turn_speed_schedule_uncertainty_kt=0.0,
        )[0]
    )


def decimal(value: mp.mpf) -> str:
    return mp.nstr(value, n=130)


def saved_segments(segments: tuple[SpeedSegment, ...]) -> list[Json]:
    return [[decimal(segment.left), decimal(segment.right), decimal(segment.speed)] for segment in segments]


@dataclass(frozen=True)
class SavedReference:
    classification: Classification
    reason: str
    lower_m: str
    upper_m: str
    support_evaluations: int
    time_cells: int
    witness: dict[str, Json] | None


def save_reference(result: ReferenceResult) -> SavedReference:
    witness = result.witness
    payload: dict[str, Json] | None = None
    if witness is not None:
        payload = {
            "time_s": decimal(witness.time),
            "weights": [decimal(weight) for weight in witness.weights],
            "vertices": [
                {
                    "point_m": [decimal(vertex.point.x), decimal(vertex.point.y)],
                    "a_segments": saved_segments(vertex.a_segments),
                    "b_segments": saved_segments(vertex.b_segments),
                }
                for vertex in witness.vertices
            ],
        }
    return SavedReference(
        result.classification,
        result.reason,
        decimal(result.lower_m),
        decimal(result.upper_m),
        result.support_evaluations,
        result.time_cells,
        payload,
    )


@dataclass(frozen=True)
class ComparedReference:
    classification: Classification
    primary: SavedReference
    recheck: SavedReference | None


def checked_reference(query: Query, model: Model, settings: Settings) -> ComparedReference:
    first = classify(
        query,
        model,
        digits=settings.digits,
        margin_m=settings.margin_m,
        max_support=settings.max_support,
        max_cells=settings.max_cells,
    )
    second = None
    classification = first.classification
    if classification != "unresolved":
        second = classify(
            query,
            model,
            digits=settings.recheck_digits,
            margin_m=settings.margin_m,
            max_support=settings.max_support,
            max_cells=settings.max_cells,
        )
        if second.classification != classification:
            classification = "unresolved"
    return ComparedReference(classification, save_reference(first), None if second is None else save_reference(second))


@dataclass(frozen=True)
class StudyResult:
    case: StudyCase
    input_hex: dict[str, str]
    nominal_accepted: bool
    constant_accepted: bool
    variable_status: str
    variable_inflation_m: float
    constant_reference: ComparedReference
    variable_reference: ComparedReference
    evaluation_seconds: float


def evaluate(case: StudyCase, settings: Settings) -> StudyResult:
    start = time.perf_counter()
    query = case.query
    nominal, constant = core_accepts(case, 0.0), core_accepts(case, case.width_kt)
    variable = certify_heading_separation(
        query.relative_position_m,
        heading_motion(query.a),
        heading_motion(query.b),
        separation_threshold_m=query.threshold_m,
        horizon_s=query.horizon_s,
    )
    if variable.status == "unsupported":
        raise AssertionError(f"Supported protocol query was rejected as unsupported: {case.identifier}")
    reference_c = checked_reference(query, "C", settings)
    reference_v = checked_reference(query, "V", settings)
    if constant and reference_c.classification == "conflict":
        raise AssertionError(f"Constant-speed acceptance has a conflicting witness: {case.identifier}")
    if variable.status == "certified_safe" and reference_v.classification == "conflict":
        raise AssertionError(f"Varying-speed acceptance has a conflicting witness: {case.identifier}")
    values = {
        "relative_east": query.relative_position_m[0],
        "relative_north": query.relative_position_m[1],
        "horizon": query.horizon_s,
        "threshold": query.threshold_m,
    }
    for name, motion in (("a", query.a), ("b", query.b)):
        values.update(
            {
                f"{name}_heading": motion.heading_deg,
                f"{name}_target": motion.target_deg,
                f"{name}_rate": motion.rate_deg_s,
                f"{name}_lower": motion.lower_mps,
                f"{name}_upper": motion.upper_mps,
            }
        )
    return StudyResult(
        case,
        {name: float(value).hex() for name, value in values.items()},
        nominal,
        constant,
        variable.status,
        variable.inflation_m,
        reference_c,
        reference_v,
        time.perf_counter() - start,
    )


def run(protocol: Path, output: Path, workers: int) -> None:
    if workers < 1:
        raise ValueError("At least one worker is required")
    cases, settings = load_protocol(protocol)
    if settings.margin_m != PROFILE_REFINEMENT_MARGIN_M:
        raise ValueError("The reference exclusion band must match the library's profile refinement margin")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol.read_bytes())
    (output / "inputs.json").write_text(json.dumps([asdict(case) for case in cases], indent=2) + "\n")
    package = Path(geometric_safety.__file__).resolve().parent
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("heading_study_reference.py"),
        Path(__file__).with_name("benchmark_contracts.py"),
    ]
    source_hashes = {path.name: digest(path) for path in source_paths}
    package_before = module_hashes(package)
    manifest = {
        "protocol_sha256": digest(protocol),
        "inputs_sha256": digest(output / "inputs.json"),
        "source_sha256": source_hashes,
        "package_sha256": package_before,
        "library_version": version("geometric_safety"),
        "lock_sha256": digest(Path(__file__).resolve().parents[1] / "uv.lock"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": workers,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    pending: dict[Future[StudyResult], str] = {}
    remaining = iter(cases)
    count = 0
    try:
        with (output / "results.jsonl").open("x") as stream:
            while True:
                while len(pending) < workers * 2:
                    case = next(remaining, None)
                    if case is None:
                        break
                    pending[executor.submit(evaluate, case, settings)] = case.identifier
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    del pending[future]
                    record = future.result()
                    stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")
                    stream.flush()
                    count += 1
                    print(
                        f"{count}/{len(cases)} {record.case.identifier} "
                        f"C={record.constant_reference.classification} V={record.variable_reference.classification} "
                        f"({record.evaluation_seconds:.2f}s)",
                        flush=True,
                    )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if package_before != module_hashes(package) or source_hashes != {path.name: digest(path) for path in source_paths}:
        raise RuntimeError("Source identity changed during the study")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "count": count,
                "seconds": time.perf_counter() - start,
                "results_sha256": digest(output / "results.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.protocol, args.output, args.workers)


if __name__ == "__main__":
    main()
