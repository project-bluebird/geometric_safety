"""Compare preserved and installed geometry implementations on one remote CPU.

The persistent workers import complete, separate package versions. IPC,
input generation, object preparation and compilation are outside timed regions.
The construction-inclusive interface measurement is labelled separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import functools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING, TextIO

import numba
import numpy as np
from numpy.typing import NDArray

from geometric_safety import relevant_aircraft as ra
from paper.benchmark_contracts import (
    BenchmarkChecks,
    BenchmarkReport,
    DecisionDifference,
    MethodStatistics,
    Outcomes,
    Statistics,
    SummaryGroup,
    TimingRequest,
    TimingResponse,
    Timings,
    WorkerHeader,
    WorkerRole,
    digest,
    list_value,
    module_hashes,
    number,
    object_json,
    object_value,
    parse_manifest,
    parse_request,
    parse_response,
    parse_worker_header,
)
from paper.isolated_worker import worker_command

if TYPE_CHECKING:
    from geometric_safety.heading import HeadingMotion

type FloatArray = NDArray[np.float64]

OLD_REVISION = "9b7869d363286e69aa45471c11d13b83861320e9"
KT = 1852.0 / 3600.0
METHODS = (
    "old_python",
    "core_python",
    "interface_prebuilt",
    "interface_construct",
    "interface_core_only",
    "old_compiled",
    "core_compiled",
)


@dataclass(frozen=True)
class BenchmarkOptions:
    inputs: Path
    old_source: Path
    old_archive: Path
    manifest: Path
    output: Path
    cpu: int = 6
    rounds: int = 7
    repeats: int = 16
    tail_rounds: int = 31
    before_source: Path | None = None
    before_manifest: Path | None = None
    expected_recovered: int = 36
    require_before_decision_parity: bool = False


@dataclass(frozen=True)
class Case:
    id: str
    suite: str
    x: float
    y: float
    ah: float
    at: float
    av: float
    ar: float
    bh: float
    bt: float
    bv: float
    br: float
    width: float
    horizon: float
    threshold: float
    reference: str = "unclassified"

    def packed(self) -> FloatArray:
        return np.array(
            [
                self.x,
                self.y,
                self.ah,
                self.at,
                self.av,
                self.ar,
                self.bh,
                self.bt,
                self.bv,
                self.br,
                self.threshold,
                self.width,
                self.horizon,
            ],
            dtype=np.float64,
        )


def read_cases(path: Path) -> list[Case]:
    rows = list_value(object_json(path.read_text())["cases"])
    result: list[Case] = []
    for value in rows:
        row = object_value(value)
        result.append(
            Case(
                id=str(row["id"]),
                suite=str(row["suite"]),
                **{
                    k: number(row[k])
                    for k in (
                        "x",
                        "y",
                        "ah",
                        "at",
                        "av",
                        "ar",
                        "bh",
                        "bt",
                        "bv",
                        "br",
                        "width",
                        "horizon",
                        "threshold",
                    )
                },
                reference=str(row["reference"]),
            )
        )
    return result


def generate_cases(source: Path) -> list[Case]:
    cases: list[Case] = []
    for value in list_value(object_json(source.read_text())["cases"]):
        row = object_value(value)
        encounter = object_value(row["encounter"])
        names = {
            "x": "rel_east_m",
            "y": "rel_north_m",
            "ah": "a_heading0_deg",
            "at": "a_target_heading_deg",
            "av": "a_speed_kt",
            "ar": "a_turn_rate_deg_sec",
            "bh": "b_heading0_deg",
            "bt": "b_target_heading_deg",
            "bv": "b_speed_kt",
            "br": "b_turn_rate_deg_sec",
            "width": "speed_diff_kt",
            "horizon": "projection_time_s",
            "threshold": "separation_threshold_m",
        }
        cases.append(
            Case(
                id=str(row["id"]),
                suite=str(row["suite"]),
                **{key: number(encounter[name]) for key, name in names.items()},
                reference="saved constant-speed " + str(object_value(row["reference"])["classification"]),
            )
        )
    rng = random.Random(2026101201)
    for i in range(128):
        ah, bh = rng.uniform(0, 360), rng.uniform(0, 360)
        if i % 4 == 0:
            bh = ah
        cases.append(
            Case(
                f"straight-{i}",
                "straight",
                rng.uniform(-60000, 60000),
                rng.uniform(-60000, 60000),
                ah,
                ah,
                rng.uniform(200, 500),
                0.0,
                bh,
                bh,
                rng.uniform(200, 500),
                0.0,
                rng.choice([0.0, 10.0, 40.0]),
                rng.choice([60.0, 600.0, 1800.0]),
                9260.0,
            )
        )
    # Equal-rate circular relative motion has constant separation R until turn
    # completion, then tangent straight continuation increases the separation.
    for slack in (0.0, 0.0001, 0.01, 0.1, 1.0, 10.0, 100.0):
        radius = 9260.0 + slack
        for rate in (0.03, 0.3, 1.5):
            omega = math.radians(rate)
            alpha = math.degrees(math.asin(radius * omega / 600.0))
            for sweep in (60.0, 170.0):
                for horizon_factor in (0.5, 2.0):
                    for width in (0.0, 5.0):
                        cases.append(
                            Case(
                                f"circle-{len(cases)}",
                                "circular_clearance",
                                0.0,
                                radius,
                                alpha,
                                alpha + sweep,
                                300.0 / KT,
                                rate,
                                -alpha,
                                -alpha + sweep,
                                300.0 / KT,
                                rate,
                                width,
                                horizon_factor * sweep / rate,
                                9260.0,
                                "analytic nominal clearance " + repr(slack) + " m" if width == 0 else "unclassified",
                            )
                        )
    small = Case(
        "small-turn-collision",
        "regression",
        9260.1,
        0.0,
        -0.5,
        0.0,
        300.0,
        1.5,
        0.0,
        0.0,
        300.0,
        0.0,
        0.0,
        60.0,
        9260.0,
        "known small-turn collision",
    )
    cases.extend(
        [
            small,
            replace(small, id="small-turn-safe", x=9262.0, reference="small-turn separated control"),
            Case(
                "half-turn-collision",
                "regression",
                -3819.718634205488,
                0.0,
                0.0,
                math.nextafter(180.0, 0.0),
                100.0 / KT,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                60.0,
                1000.0,
                "known half-turn collision",
            ),
            Case(
                "half-turn-exact-control",
                "regression",
                -3819.718634205488,
                0.0,
                0.0,
                180.0,
                100.0 / KT,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                60.0,
                1000.0,
                "known separated exact-half-turn control",
            ),
            replace(small, id="slow-turn", at=5.0, ar=1e-7, x=10000.0, horizon=1200.0, reference="unclassified"),
            replace(small, id="tiny-turn", ah=-1e-12, x=9260.0001, reference="unclassified"),
        ]
    )
    if not (len({c.id for c in cases}) == len(cases)):
        raise RuntimeError("Benchmark validation failed: duplicate case IDs")
    return cases


@numba.njit(cache=True)
def compiled_call(p: FloatArray) -> bool:
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
    )[0]


def direct_call(p: FloatArray) -> bool:
    return bool(
        ra._catch_up_projection_interval_with_turns_local(
            rel_pos0=p[:2],
            a_heading0_deg=float(p[2]),
            a_target_heading_deg=float(p[3]),
            a_speed_kt=float(p[4]),
            a_turn_rate_deg_sec=float(p[5]),
            b_heading0_deg=float(p[6]),
            b_target_heading_deg=float(p[7]),
            b_speed_kt=float(p[8]),
            b_turn_rate_deg_sec=float(p[9]),
            separation_threshold_m=float(p[10]),
            speed_diff_kt=float(p[11]),
            projection_time_s=float(p[12]),
        )[0]
    )


def api_methods(case: Case) -> tuple[dict[str, Callable[[], bool]], float]:
    # Only the current-version worker imports the newly introduced interface.
    from geometric_safety import heading

    def make_motion(first: bool) -> HeadingMotion:
        speed = case.av if first else case.bv
        return heading.HeadingMotion(
            heading0_deg=case.ah if first else case.bh,
            target_heading_deg=case.at if first else case.bt,
            turn_rate_deg_s=case.ar if first else case.br,
            speed_min_mps=max(0.0, speed - case.width) * KT,
            speed_max_mps=(speed + case.width) * KT,
        )

    a, b = make_motion(True), make_motion(False)
    position = (case.x, case.y)

    def prebuilt() -> bool:
        return (
            heading.certify_heading_separation(
                position,
                a,
                b,
                separation_threshold_m=case.threshold,
                horizon_s=case.horizon,
            ).status
            == "certified_safe"
        )

    def constructed() -> bool:
        return (
            heading.certify_heading_separation(
                position,
                make_motion(True),
                make_motion(False),
                separation_threshold_m=case.threshold,
                horizon_s=case.horizon,
            ).status
            == "certified_safe"
        )

    inflation = heading._inflation_m(a, b, case.horizon, 0.0, 0.0)
    if case.ah == case.at and case.bh == case.bt:
        relative = np.array(position)

        def core_only() -> bool:
            return bool(heading._straight_query(relative, a, b, case.threshold + inflation, case.horizon)[0])
    else:
        bounds = heading._enclosing_common_speed_bounds_kt(a, b)
        if bounds is None:
            raise AssertionError("Benchmark intervals cannot be enclosed")
        p = case.packed()
        p[2], p[3], p[6], p[7] = (math.fmod(value, 360.0) for value in (case.ah, case.at, case.bh, case.bt))
        p[4], p[8], p[11] = bounds
        p[10] += inflation
        core_only = functools.partial(direct_call, p)
    return {
        "interface_prebuilt": prebuilt,
        "interface_construct": constructed,
        "interface_core_only": core_only,
    }, inflation


def worker(cases_path: Path, role: WorkerRole) -> None:
    started = time.perf_counter()
    cases = read_cases(cases_path)
    calls: dict[str, list[Callable[[], bool]]] = {}
    outcomes: Outcomes = {}
    inflations: list[float] = []
    for case in cases:
        p = case.packed()
        prefix = "old" if role == "old" else "before_core" if role == "before" else "core"
        prepared: dict[str, Callable[[], bool]] = {
            prefix + "_python": functools.partial(direct_call, p),
            prefix + "_compiled": functools.partial(compiled_call, p),
        }
        if role in ("current", "before"):
            api, inflation = api_methods(case)
            prepared.update({("before_" + name if role == "before" else name): call for name, call in api.items()})
            inflations.append(inflation)
        for method, call in prepared.items():
            verdict = bool(call())
            if not (bool(call()) == verdict):
                raise RuntimeError(f"Benchmark validation failed: unstable warm-up decision for {method}")
            calls.setdefault(method, []).append(call)
            outcomes.setdefault(method, []).append(verdict)
    if not (outcomes[prefix + "_python"] == outcomes[prefix + "_compiled"]):
        raise RuntimeError("Benchmark validation failed: Python and compiled decisions differ")
    if role in ("current", "before"):
        api_prefix = "before_" if role == "before" else ""
        if not (outcomes[api_prefix + "interface_prebuilt"] == outcomes[api_prefix + "interface_construct"]):
            raise RuntimeError("Benchmark validation failed: prepared and constructed interface decisions differ")
        if not (outcomes[api_prefix + "interface_prebuilt"] == outcomes[api_prefix + "interface_core_only"]):
            raise RuntimeError("Benchmark validation failed: interface and effective core decisions differ")
    package = Path(ra.__file__).parent
    header: WorkerHeader = {
        "ready": True,
        "role": role,
        "preparation_seconds": time.perf_counter() - started,
        "source_sha256": module_hashes(package),
        "outcomes": outcomes,
        "inflation_m": inflations,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    print(json.dumps(header), flush=True)
    for line in sys.stdin:
        request = parse_request(line)
        if request["op"] == "stop":
            return
        method = request["method"]
        indices = request["indices"]
        repeats = request["repeats"]
        wall, cpu = [], []
        for index in indices:
            call = calls[method][index]
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            for _ in range(repeats):
                verdict = call()
            cpu_elapsed = time.thread_time_ns() - cpu_start
            elapsed = time.perf_counter_ns() - start
            if not (bool(verdict) == outcomes[method][index]):
                raise RuntimeError(f"Benchmark validation failed: timed decision changed for {method}, case {index}")
            wall.append(elapsed / repeats / 1000.0)
            cpu.append(cpu_elapsed / repeats / 1000.0)
        response: TimingResponse = {"wall_us": wall, "cpu_us": cpu}
        print(json.dumps(response), flush=True)


@dataclass
class Worker:
    process: subprocess.Popen[str]
    log: TextIO
    header: WorkerHeader

    def request(self, method: str, indices: list[int], repeats: int) -> TimingResponse:
        if not (self.process.stdin is not None):
            raise RuntimeError("Benchmark validation failed: self.process.stdin is not None")
        if not (self.process.stdout is not None):
            raise RuntimeError("Benchmark validation failed: self.process.stdout is not None")
        request = TimingRequest(op="time", method=method, indices=indices, repeats=repeats)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"Worker ended with {self.process.poll()}; inspect its log")
        return parse_response(response, len(indices))

    def close(self, *, grace_seconds: float = 30.0, termination_seconds: float = 5.0) -> None:
        shutdown_process(self.process, self.log, grace_seconds=grace_seconds, termination_seconds=termination_seconds)


def shutdown_process(
    process: subprocess.Popen[str], log: TextIO, *, grace_seconds: float = 30.0, termination_seconds: float = 5.0
) -> None:
    """Stop, escalate if needed, reap, then close every owned stream."""
    try:
        if process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"op":"stop"}\n')
                    process.stdin.flush()
                except (OSError, ValueError):
                    # A worker may have closed its input while exiting or failing.
                    pass
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    process.wait(timeout=termination_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        else:
            process.wait()
    finally:
        errors: list[Exception] = []
        for stream in (process.stdin, process.stdout, log):
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Benchmark stream cleanup failed", errors)


def close_workers(workers: Iterable[Worker]) -> None:
    errors: list[Exception] = []
    for worker_info in workers:
        try:
            worker_info.close()
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Benchmark worker cleanup failed", errors)


def start_worker(role: WorkerRole, cases_path: Path, output: Path, source: Path | None = None) -> Worker:
    """Start the current harness against one complete, verified package."""
    if role != "current" and source is None:
        raise ValueError(f"A baseline source is required for the {role} worker")
    if source is not None and not (source / "geometric_safety" / "__init__.py").is_file():
        raise ValueError("Source must be a directory containing a complete geometric_safety package")
    expected_source = module_hashes(source / "geometric_safety") if source is not None else None
    env = os.environ.copy()
    env.update({"NUMBA_CACHE_DIR": str(output / (role + "_cache")), "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("PYTHONPATH", None)
    log = (output / (role + "_worker.log")).open("w")
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            worker_command(
                "paper.benchmark_library_versions", ["--worker", role, "--cases", str(cases_path.resolve())], source
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
        )
        if process.stdout is None:
            raise RuntimeError("Benchmark worker has no output stream")
        response = process.stdout.readline()
        if not response:
            raise RuntimeError(f"{role} worker failed during preparation; inspect its log")
        header = parse_worker_header(object_json(response))
        if header["role"] != role:
            raise ValueError("Benchmark worker reported a different role")
        if expected_source is not None and header["source_sha256"] != expected_source:
            raise RuntimeError("Benchmark worker source identity differs from the selected package")
        return Worker(process, log, header)
    except BaseException as error:
        try:
            if process is None:
                log.close()
            else:
                shutdown_process(process, log)
        except Exception as cleanup_error:
            error.add_note(f"Worker startup cleanup also failed: {cleanup_error}")
        raise


def statistics_us(values: list[float]) -> Statistics:
    data = np.array(values)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
    }


def measure(
    workers: dict[str, Worker],
    cases: list[Case],
    indices: list[int],
    rounds: int,
    repeats: int,
    seed: int,
    label: str,
) -> tuple[Timings, Timings]:
    owners = {method: role for role, worker_info in workers.items() for method in worker_info.header["outcomes"]}
    methods = (*METHODS, *(method for method in owners if method not in METHODS))
    wall: Timings = {method: [[0.0] * len(indices) for _ in range(rounds)] for method in methods}
    cpu: Timings = {method: [[0.0] * len(indices) for _ in range(rounds)] for method in methods}
    positions = {index: i for i, index in enumerate(indices)}
    rng = random.Random(seed)
    for round_index in range(rounds):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for chunk_index, start in enumerate(range(0, len(shuffled), 24)):
            chunk = shuffled[start : start + 24]
            shift = (round_index + chunk_index) % len(methods)
            order = methods[shift:] + methods[:shift]
            for method in order:
                response = workers[owners[method]].request(method, chunk, repeats)
                for values, destination in ((response["wall_us"], wall), (response["cpu_us"], cpu)):
                    for index, value in zip(chunk, values, strict=True):
                        destination[method][round_index][positions[index]] = value
        print(f"{label}: round {round_index + 1}/{rounds}, {len(indices)} inputs", flush=True)
    if not (all(0 <= i < len(cases) for i in indices)):
        raise RuntimeError("Benchmark validation failed: all((0 <= i < len(cases) for i in indices))")
    return wall, cpu


def summary(cases: list[Case], raw: Timings, outcomes: Outcomes) -> dict[str, SummaryGroup]:
    groups = [
        "all",
        *sorted({case.suite for case in cases}),
        "zero_width",
        "positive_width",
        "all_three_accept",
        "core_interface_same_decision",
    ]
    result: dict[str, SummaryGroup] = {}
    for group in groups:
        indices = [
            i
            for i, c in enumerate(cases)
            if group == "all"
            or c.suite == group
            or (group == "zero_width" and c.width == 0.0)
            or (group == "positive_width" and c.width > 0.0)
            or (
                group == "all_three_accept"
                and all(outcomes[m][i] for m in ("old_python", "core_python", "interface_prebuilt"))
            )
            or (
                group == "core_interface_same_decision"
                and outcomes["core_python"][i] == outcomes["interface_prebuilt"][i]
            )
        ]
        result[group] = SummaryGroup(
            count=len(indices),
            methods={
                method: MethodStatistics(
                    **statistics_us([statistics.median(round_values[i] for round_values in values) for i in indices]),
                    accepted=sum(outcomes[method][i] for i in indices),
                    slowest_ids=sorted(indices, key=lambda i: statistics.median(v[i] for v in values), reverse=True)[
                        :10
                    ],
                )
                for method, values in raw.items()
            },
        )
    return result


def checks(cases: list[Case], outcomes: Outcomes) -> BenchmarkChecks:
    positions = {case.id: i for i, case in enumerate(cases)}
    for name in ("small-turn-collision", "half-turn-collision"):
        index = positions[name]
        if not (outcomes["old_python"][index]):
            raise RuntimeError("Benchmark validation failed: outcomes['old_python'][index]")
        if outcomes["core_python"][index]:
            raise RuntimeError("Benchmark validation failed: not outcomes['core_python'][index]")
        if outcomes["interface_prebuilt"][index]:
            raise RuntimeError("Benchmark validation failed: not outcomes['interface_prebuilt'][index]")
    for name in ("small-turn-safe", "half-turn-exact-control"):
        index = positions[name]
        if not (outcomes["core_python"][index]):
            raise RuntimeError("Benchmark validation failed: outcomes['core_python'][index]")
        if not (outcomes["interface_prebuilt"][index]):
            raise RuntimeError("Benchmark validation failed: outcomes['interface_prebuilt'][index]")
    if not (
        all(
            (
                not outcomes["core_python"][i] and (not outcomes["interface_prebuilt"][i])
                for i, case in enumerate(cases)
                if case.reference == "saved constant-speed unsafe"
            )
        )
    ):
        raise RuntimeError("Benchmark validation failed: a saved unsafe case was accepted")
    disagreements: list[DecisionDifference] = [
        {
            "id": case.id,
            "suite": case.suite,
            "width_kt": case.width,
            "reference": case.reference,
            "old": outcomes["old_python"][i],
            "core": outcomes["core_python"][i],
            "interface": outcomes["interface_prebuilt"][i],
        }
        for i, case in enumerate(cases)
        if len({outcomes[m][i] for m in ("old_python", "core_python", "interface_prebuilt")}) > 1
    ]
    return {
        "known_collision_repairs_verified": True,
        "known_separated_controls_verified": True,
        "no_current_acceptance_of_saved_unsafe_cases": True,
        "decision_differences": disagreements,
        "interface_matches_effective_core": True,
    }


def run(args: BenchmarkOptions) -> None:
    started_at_utc = datetime.now(UTC).isoformat()
    args.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    starting_load = os.getloadavg()
    cases = generate_cases(args.inputs)
    cases_path = (args.output / "cases.json").resolve()
    cases_path.write_text(
        json.dumps({"input_sha256": digest(args.inputs), "cases": [asdict(c) for c in cases]}, indent=2) + "\n"
    )
    workers: dict[str, Worker] = {}
    try:
        roles: tuple[WorkerRole, ...] = ("old", "before", "current") if args.before_source else ("old", "current")
        for role in roles:
            print(f"Preparing isolated {role} worker", flush=True)
            source = args.old_source if role == "old" else args.before_source if role == "before" else None
            workers[role] = start_worker(role, cases_path, args.output, source)
        outcomes = {
            name: values for worker_info in workers.values() for name, values in worker_info.header["outcomes"].items()
        }
        checked = checks(cases, outcomes)
        manifest = parse_manifest(args.manifest)
        expected = manifest["provenance"]["module_sha256"]
        actual = workers["current"].header["source_sha256"]
        if not (all((actual[Path(name).name] == value for name, value in expected.items()))):
            raise RuntimeError("Benchmark validation failed: installed candidate does not match its manifest")
        if not (workers["old"].header["numpy"] == workers["current"].header["numpy"]):
            raise RuntimeError("Benchmark validation failed: original and candidate NumPy versions differ")
        if not (workers["old"].header["numba"] == workers["current"].header["numba"]):
            raise RuntimeError("Benchmark validation failed: original and candidate Numba versions differ")
        if args.before_source:
            if args.before_manifest is None:
                raise ValueError("A baseline manifest is required with --before-source")
            before_manifest = parse_manifest(args.before_manifest)
            before_expected = before_manifest["provenance"]["module_sha256"]
            before_actual = workers["before"].header["source_sha256"]
            if not (all((before_actual[Path(name).name] == value for name, value in before_expected.items()))):
                raise RuntimeError("Benchmark validation failed: baseline does not match its manifest")
            if not (workers["before"].header["python"] == workers["current"].header["python"]):
                raise RuntimeError("Benchmark validation failed: baseline and candidate Python versions differ")
            if not (workers["before"].header["numpy"] == workers["current"].header["numpy"]):
                raise RuntimeError("Benchmark validation failed: baseline and candidate NumPy versions differ")
            if not (workers["before"].header["numba"] == workers["current"].header["numba"]):
                raise RuntimeError("Benchmark validation failed: baseline and candidate Numba versions differ")
            affected = [
                i
                for i, case in enumerate(cases)
                if case.suite == "circular_clearance"
                and case.width == 0.0
                and outcomes["before_core_python"][i]
                and not outcomes["before_interface_prebuilt"][i]
            ]
            if not (len(affected) == args.expected_recovered):
                raise RuntimeError("Benchmark validation failed: false-rejection control count differs")
            if not (all(outcomes["interface_prebuilt"][i] for i in affected)):
                raise RuntimeError("Benchmark validation failed: a false-rejection control was not recovered")
            checked["previous_false_rejections_recovered"] = [cases[i].id for i in affected]
            checked["repair_changes"] = {
                method: [
                    {"id": case.id, "before": outcomes["before_" + method][i], "after": outcomes[method][i]}
                    for i, case in enumerate(cases)
                    if outcomes["before_" + method][i] != outcomes[method][i]
                ]
                for method in ("core_python", "interface_prebuilt")
            }
            if args.require_before_decision_parity:
                parity_methods = [method for method in outcomes if "before_" + method in outcomes]
                if not (len(parity_methods) == 5):
                    raise RuntimeError("Benchmark validation failed: len(parity_methods) == 5")
                for method in parity_methods:
                    if not (outcomes[method] == outcomes["before_" + method]):
                        raise RuntimeError(
                            "Benchmark validation failed: outcomes[method] == outcomes['before_' + method]"
                            ": " + str(method)
                        )
                checked["before_after_decision_parity"] = parity_methods
        print(
            f"Preflight passed: {len(cases)} cases, identical direct/compiled decisions, matched interface enclosure",
            flush=True,
        )
        raw, cpu = measure(workers, cases, list(range(len(cases))), args.rounds, args.repeats, 2026101202, "population")
        # Select the union of the ten slowest inputs for every method, then measure
        # it independently. Selection and repetition remain explicit in the record.
        slow: set[int] = set()
        for values in raw.values():
            slow.update(
                sorted(range(len(cases)), key=lambda i: statistics.median(v[i] for v in values), reverse=True)[:10]
            )
        slow_indices = sorted(slow)
        tail, tail_cpu = measure(workers, cases, slow_indices, args.tail_rounds, 1, 2026101203, "slow inputs")
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
        report: BenchmarkReport = {
            "protocol": {
                "host": platform.node(),
                "started_at_utc": started_at_utc,
                "old_revision": OLD_REVISION,
                "current_wheel_sha256": manifest["wheel_sha256"],
                "before_manifest_sha256": digest(args.before_manifest) if args.before_manifest is not None else None,
                "expected_recovered_false_rejections": args.expected_recovered if args.before_source else None,
                "require_before_decision_parity": args.require_before_decision_parity,
                "current_manifest_sha256": digest(args.manifest),
                "cases": len(cases),
                "suites": dict(Counter(c.suite for c in cases)),
                "rounds": args.rounds,
                "calls_per_batch": args.repeats,
                "tail_rounds": args.tail_rounds,
                "tail_calls_per_measurement": 1,
                "cpu": args.cpu,
                "cpu_model": cpu_model,
                "host_load_average_start": starting_load,
                "host_load_average_end": os.getloadavg(),
                "single_thread_environment": {
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
                },
                "population_statistics": "Statistics across per-input median batch means, microseconds per call.",
                "tail_statistics": (
                    "Individual calls on the union of ten slowest inputs per method; "
                    "selection uses the preceding population measurements."
                ),
                "timed_boundary": (
                    "Python call into each prepared method. IPC, compilation and input construction excluded, "
                    "except interface_construct includes both HeadingMotion constructors."
                ),
                "semantics": (
                    "Old and repaired cores assume horizon-constant speeds through turns. "
                    "The interface covers arbitrary bounded speed profiles. Zero-width cases remove that distinction. "
                    "interface_core_only uses the interface's exact enlarged intervals and inflated threshold "
                    "to isolate wrapper work."
                ),
                "limitations": (
                    "Synthetic inputs, one shared host and one pinned CPU. No real-time bound or operational traffic "
                    "claim. CPU affinity does not reserve the CPU. Compiled wrappers are a separate call path, "
                    "not the complete Python interface."
                ),
            },
            "source_sha256": digest(Path(__file__)),
            "inputs_sha256": digest(args.inputs),
            "cases_sha256": digest(cases_path),
            "old_archive_sha256": digest(args.old_archive),
            "workers": {name: worker_info.header for name, worker_info in workers.items()},
            "checks": checked,
            "summary": summary(cases, raw, outcomes),
            "raw_wall_us": raw,
            "raw_cpu_us": cpu,
            "slow_indices": slow_indices,
            "slow_case_ids": [cases[i].id for i in slow_indices],
            "slow_raw_wall_us": tail,
            "slow_raw_cpu_us": tail_cpu,
            "slow_summary": {
                m: statistics_us([v for round_values in values for v in round_values]) for m, values in tail.items()
            },
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"summary": report["summary"], "slow_summary": report["slow_summary"]}), flush=True)
    finally:
        close_workers(workers.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("old", "current", "before"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--old-source", type=Path)
    parser.add_argument("--old-archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--before-source", type=Path)
    parser.add_argument("--before-manifest", type=Path)
    parser.add_argument("--expected-recovered", type=int, default=36)
    parser.add_argument("--require-before-decision-parity", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--tail-rounds", type=int, default=31)
    args = parser.parse_args()
    if bool(args.before_source) != bool(args.before_manifest):
        parser.error("--before-source and --before-manifest must be supplied together")
    if args.require_before_decision_parity and not args.before_source:
        parser.error("--require-before-decision-parity requires --before-source")
    if args.worker:
        if args.cases is None:
            parser.error("--cases is required for a worker")
        worker(args.cases, args.worker)
    else:
        for name in ("inputs", "old_source", "old_archive", "manifest", "output"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for a benchmark")
        if min(args.rounds, args.repeats, args.tail_rounds) <= 0:
            parser.error("--rounds, --repeats and --tail-rounds must be positive")
        run(
            BenchmarkOptions(
                inputs=args.inputs,
                old_source=args.old_source,
                old_archive=args.old_archive,
                manifest=args.manifest,
                output=args.output,
                cpu=args.cpu,
                rounds=args.rounds,
                repeats=args.repeats,
                tail_rounds=args.tail_rounds,
                before_source=args.before_source,
                before_manifest=args.before_manifest,
                expected_recovered=args.expected_recovered,
                require_before_decision_parity=args.require_before_decision_parity,
            )
        )


if __name__ == "__main__":
    main()
