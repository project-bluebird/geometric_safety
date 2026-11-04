"""Typed benchmark records and validation at JSON and worker boundaries."""

from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Timings = dict[str, list[list[float]]]
type Outcomes = dict[str, list[bool]]
type WorkerRole = Literal["old", "before", "current"]


class Statistics(TypedDict):
    mean: float
    median: float
    p95: float
    p99: float
    max: float


class MethodStatistics(Statistics):
    accepted: int
    slowest_ids: list[int]


class SummaryGroup(TypedDict):
    count: int
    methods: dict[str, MethodStatistics]


class DecisionDifference(TypedDict):
    id: str
    suite: str
    width_kt: float
    reference: str
    old: bool
    core: bool
    interface: bool


class DecisionChange(TypedDict):
    id: str
    before: bool
    after: bool


class BenchmarkChecks(TypedDict):
    known_collision_repairs_verified: bool
    known_separated_controls_verified: bool
    no_current_acceptance_of_saved_unsafe_cases: bool
    interface_matches_effective_core: bool
    decision_differences: list[DecisionDifference]
    previous_false_rejections_recovered: NotRequired[list[str]]
    repair_changes: NotRequired[dict[str, list[DecisionChange]]]
    before_after_decision_parity: NotRequired[list[str]]


class WorkerHeader(TypedDict):
    ready: bool
    role: WorkerRole
    preparation_seconds: float
    source_sha256: dict[str, str]
    outcomes: Outcomes
    inflation_m: list[float]
    python: str
    numpy: str
    numba: str
    affinity: list[int]


class StopRequest(TypedDict):
    op: Literal["stop"]


class TimingRequest(TypedDict):
    op: Literal["time"]
    method: str
    indices: list[int]
    repeats: int


class TimingResponse(TypedDict):
    wall_us: list[float]
    cpu_us: list[float]


class BenchmarkProtocol(TypedDict):
    host: str
    started_at_utc: str
    old_revision: str
    current_wheel_sha256: str
    before_manifest_sha256: str | None
    expected_recovered_false_rejections: int | None
    require_before_decision_parity: bool
    current_manifest_sha256: str
    cases: int
    suites: dict[str, int]
    rounds: int
    calls_per_batch: int
    tail_rounds: int
    tail_calls_per_measurement: int
    cpu: int
    cpu_model: str
    host_load_average_start: list[float] | tuple[float, float, float]
    host_load_average_end: list[float] | tuple[float, float, float]
    single_thread_environment: dict[str, str | None]
    population_statistics: str
    tail_statistics: str
    timed_boundary: str
    semantics: str
    limitations: str


class BenchmarkReport(TypedDict):
    protocol: BenchmarkProtocol
    source_sha256: str
    inputs_sha256: str
    cases_sha256: str
    old_archive_sha256: str
    workers: dict[str, WorkerHeader]
    checks: BenchmarkChecks
    summary: dict[str, SummaryGroup]
    raw_wall_us: Timings
    raw_cpu_us: Timings
    slow_indices: list[int]
    slow_case_ids: list[str]
    slow_raw_wall_us: Timings
    slow_raw_cpu_us: Timings
    slow_summary: dict[str, Statistics]


class ModuleProvenance(TypedDict):
    module_sha256: dict[str, str]


class PackageManifest(TypedDict):
    wheel_sha256: str
    provenance: ModuleProvenance


def object_json(text: str) -> dict[str, Json]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    # json.loads has an untyped return. A JSON object has string keys and JSON
    # values; known application schemas are validated by the parsers below.
    return cast("dict[str, Json]", value)


def object_value(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value


def list_value(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise ValueError("Expected a list")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Expected a number")
    return float(value)


def finite_number(value: Json) -> float:
    result = number(value)
    if not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result


def integer(value: Json) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer")
    return value


def string(value: Json) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def boolean(value: Json) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Expected a boolean")
    return value


def optional_string(value: Json) -> str | None:
    return None if value is None else string(value)


def parse_list[T](value: Json, parse: Callable[[Json], T]) -> list[T]:
    return [parse(item) for item in list_value(value)]


def parse_mapping[T](value: Json, parse: Callable[[Json], T]) -> dict[str, T]:
    return {key: parse(item) for key, item in object_value(value).items()}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_hashes(package: Path) -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(package.glob("*.py"))}


def parse_worker_header(value: Json) -> WorkerHeader:
    row = object_value(value)
    role: WorkerRole
    if row["role"] == "old":
        role = "old"
    elif row["role"] == "before":
        role = "before"
    elif row["role"] == "current":
        role = "current"
    else:
        raise ValueError("Unknown benchmark worker role")
    if row["ready"] is not True:
        raise ValueError("Benchmark worker is not ready")
    outcomes = parse_mapping(row["outcomes"], lambda values: parse_list(values, boolean))
    if not outcomes:
        raise ValueError("Benchmark worker has no outcomes")
    return WorkerHeader(
        ready=True,
        role=role,
        preparation_seconds=finite_number(row["preparation_seconds"]),
        source_sha256=parse_mapping(row["source_sha256"], string),
        outcomes=outcomes,
        inflation_m=parse_list(row["inflation_m"], finite_number),
        python=string(row["python"]),
        numpy=string(row["numpy"]),
        numba=string(row["numba"]),
        affinity=parse_list(row["affinity"], integer),
    )


def parse_request(text: str) -> TimingRequest | StopRequest:
    row = object_json(text)
    if row["op"] == "stop":
        return StopRequest(op="stop")
    if row["op"] != "time":
        raise ValueError("Unknown benchmark operation")
    indices = parse_list(row["indices"], integer)
    repeats = integer(row["repeats"])
    if not indices or any(index < 0 for index in indices) or repeats <= 0:
        raise ValueError("Timing requests require nonnegative indices and positive repeats")
    return TimingRequest(op="time", method=string(row["method"]), indices=indices, repeats=repeats)


def parse_response(text: str, count: int) -> TimingResponse:
    row = object_json(text)
    result = TimingResponse(
        wall_us=parse_list(row["wall_us"], finite_number),
        cpu_us=parse_list(row["cpu_us"], finite_number),
    )
    if any(
        len(values) != count or any(value < 0 for value in values) for values in (result["wall_us"], result["cpu_us"])
    ):
        raise ValueError("Timing response has invalid sample counts or durations")
    return result


def parse_manifest(path: Path) -> PackageManifest:
    row = object_json(path.read_text())
    modules = parse_mapping(object_value(row["provenance"])["module_sha256"], string)
    if not modules:
        raise ValueError("Package manifest has no module identities")
    return PackageManifest(wheel_sha256=string(row["wheel_sha256"]), provenance=ModuleProvenance(module_sha256=modules))


def parse_statistics(value: Json) -> Statistics:
    row = object_value(value)
    return Statistics(
        mean=finite_number(row["mean"]),
        median=finite_number(row["median"]),
        p95=finite_number(row["p95"]),
        p99=finite_number(row["p99"]),
        max=finite_number(row["max"]),
    )


def parse_method_statistics(value: Json) -> MethodStatistics:
    row = object_value(value)
    return MethodStatistics(
        **parse_statistics(value),
        accepted=integer(row["accepted"]),
        slowest_ids=parse_list(row["slowest_ids"], integer),
    )


def parse_summary_group(value: Json) -> SummaryGroup:
    row = object_value(value)
    return SummaryGroup(count=integer(row["count"]), methods=parse_mapping(row["methods"], parse_method_statistics))


def parse_difference(value: Json) -> DecisionDifference:
    row = object_value(value)
    return DecisionDifference(
        id=string(row["id"]),
        suite=string(row["suite"]),
        width_kt=finite_number(row["width_kt"]),
        reference=string(row["reference"]),
        old=boolean(row["old"]),
        core=boolean(row["core"]),
        interface=boolean(row["interface"]),
    )


def parse_change(value: Json) -> DecisionChange:
    row = object_value(value)
    return DecisionChange(id=string(row["id"]), before=boolean(row["before"]), after=boolean(row["after"]))


def parse_checks(value: Json) -> BenchmarkChecks:
    row = object_value(value)
    result = BenchmarkChecks(
        known_collision_repairs_verified=boolean(row["known_collision_repairs_verified"]),
        known_separated_controls_verified=boolean(row["known_separated_controls_verified"]),
        no_current_acceptance_of_saved_unsafe_cases=boolean(row["no_current_acceptance_of_saved_unsafe_cases"]),
        interface_matches_effective_core=boolean(row["interface_matches_effective_core"]),
        decision_differences=parse_list(row["decision_differences"], parse_difference),
    )
    if "previous_false_rejections_recovered" in row:
        result["previous_false_rejections_recovered"] = parse_list(row["previous_false_rejections_recovered"], string)
    if "repair_changes" in row:
        result["repair_changes"] = parse_mapping(
            row["repair_changes"], lambda changes: parse_list(changes, parse_change)
        )
    if "before_after_decision_parity" in row:
        result["before_after_decision_parity"] = parse_list(row["before_after_decision_parity"], string)
    return result


def parse_protocol(value: Json) -> BenchmarkProtocol:
    row = object_value(value)
    recovered = row["expected_recovered_false_rejections"]
    return BenchmarkProtocol(
        host=string(row["host"]),
        started_at_utc=string(row["started_at_utc"]),
        old_revision=string(row["old_revision"]),
        current_wheel_sha256=string(row["current_wheel_sha256"]),
        before_manifest_sha256=optional_string(row["before_manifest_sha256"]),
        expected_recovered_false_rejections=None if recovered is None else integer(recovered),
        require_before_decision_parity=boolean(row["require_before_decision_parity"]),
        current_manifest_sha256=string(row["current_manifest_sha256"]),
        cases=integer(row["cases"]),
        suites=parse_mapping(row["suites"], integer),
        rounds=integer(row["rounds"]),
        calls_per_batch=integer(row["calls_per_batch"]),
        tail_rounds=integer(row["tail_rounds"]),
        tail_calls_per_measurement=integer(row["tail_calls_per_measurement"]),
        cpu=integer(row["cpu"]),
        cpu_model=string(row["cpu_model"]),
        host_load_average_start=parse_list(row["host_load_average_start"], finite_number),
        host_load_average_end=parse_list(row["host_load_average_end"], finite_number),
        single_thread_environment=parse_mapping(row["single_thread_environment"], optional_string),
        population_statistics=string(row["population_statistics"]),
        tail_statistics=string(row["tail_statistics"]),
        timed_boundary=string(row["timed_boundary"]),
        semantics=string(row["semantics"]),
        limitations=string(row["limitations"]),
    )


def parse_timings(value: Json) -> Timings:
    return parse_mapping(value, lambda rows: parse_list(rows, lambda row: parse_list(row, finite_number)))


def read_benchmark_report(path: Path) -> BenchmarkReport:
    row = object_json(path.read_text())
    report = BenchmarkReport(
        protocol=parse_protocol(row["protocol"]),
        source_sha256=string(row["source_sha256"]),
        inputs_sha256=string(row["inputs_sha256"]),
        cases_sha256=string(row["cases_sha256"]),
        old_archive_sha256=string(row["old_archive_sha256"]),
        workers=parse_mapping(row["workers"], parse_worker_header),
        checks=parse_checks(row["checks"]),
        summary=parse_mapping(row["summary"], parse_summary_group),
        raw_wall_us=parse_timings(row["raw_wall_us"]),
        raw_cpu_us=parse_timings(row["raw_cpu_us"]),
        slow_indices=parse_list(row["slow_indices"], integer),
        slow_case_ids=parse_list(row["slow_case_ids"], string),
        slow_raw_wall_us=parse_timings(row["slow_raw_wall_us"]),
        slow_raw_cpu_us=parse_timings(row["slow_raw_cpu_us"]),
        slow_summary=parse_mapping(row["slow_summary"], parse_statistics),
    )
    protocol = report["protocol"]
    count = protocol["cases"]
    outcomes = {
        method: values for header in report["workers"].values() for method, values in header["outcomes"].items()
    }
    methods = set(outcomes)
    if not methods or count <= 0 or any(len(values) != count for values in outcomes.values()):
        raise ValueError("Benchmark outcome counts do not match its cases")
    slow = report["slow_indices"]
    if not slow or len(set(slow)) != len(slow) or any(index < 0 or index >= count for index in slow):
        raise ValueError("Benchmark slow indices are invalid")
    for timings, rounds, columns in (
        (report["raw_wall_us"], protocol["rounds"], count),
        (report["raw_cpu_us"], protocol["rounds"], count),
        (report["slow_raw_wall_us"], protocol["tail_rounds"], len(slow)),
        (report["slow_raw_cpu_us"], protocol["tail_rounds"], len(slow)),
    ):
        if set(timings) != methods or rounds <= 0:
            raise ValueError("Benchmark timing methods or rounds do not match its outcomes")
        if any(
            len(rows) != rounds or any(len(values) != columns or any(value < 0 for value in values) for values in rows)
            for rows in timings.values()
        ):
            raise ValueError("Benchmark timing sample counts or durations are invalid")
    if set(report["slow_summary"]) != methods or len(report["slow_case_ids"]) != len(slow):
        raise ValueError("Benchmark slow summary does not match its samples")
    return report
