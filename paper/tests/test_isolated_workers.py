"""Real package and process regressions for version isolation and cleanup."""

import ast
from collections.abc import Iterator
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import TextIO

import pytest

from geometric_safety import relevant_aircraft as ra
from paper.benchmark_contracts import TimingRequest, WorkerRole, list_value, module_hashes, object_json, object_value
from paper.benchmark_library_versions import (
    Case,
    Worker,
    close_workers,
    shutdown_process,
    start_worker,
)
from paper.validate_directional_production import benchmark
from paper.validate_performance_parity import compare_reports, run_worker


def _snapshot(destination: Path, *, archived_paper: bool = False, without_heading: bool = False) -> Path:
    package = destination / "geometric_safety"
    package.mkdir(parents=True)
    for path in Path(ra.__file__).parent.glob("*.py"):
        shutil.copy2(path, package / path.name)
    if without_heading:
        # The legacy benchmark only requires the original lateral interfaces.
        (package / "__init__.py").write_text(
            "from .relevant_aircraft import (\n"
            "    catch_up_projection_interval, catch_up_projection_interval_with_turns,\n"
            ")\n"
        )
        (package / "heading.py").unlink()
    if archived_paper:
        paper = destination / "paper"
        paper.mkdir()
        (paper / "__init__.py").write_text('raise RuntimeError("Archived harness must not be imported")\n')
    return destination


def _case_file(path: Path) -> Path:
    # Identical motion keeps the separation fixed throughout the turn. The
    # near-threshold case distinguishes different numerical guard versions.
    cases = [
        Case(
            f"translation-{distance}",
            "regression",
            0.0,
            distance,
            0.0,
            90.0,
            300.0,
            3.0,
            0.0,
            90.0,
            300.0,
            3.0,
            0.0,
            60.0,
            9260.0,
        )
        for distance in (10000.0, 9260.1, 1000.0)
    ]
    path.write_text(json.dumps({"cases": [asdict(case) for case in cases]}))
    return path


def _directional_inputs(path: Path) -> Path:
    cases = [
        Case(
            "half-turn-collision",
            "broad",
            -3819.718634205488,
            0.0,
            0.0,
            math.nextafter(180.0, 0.0),
            100.0 / ra.KT_TO_MPS,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            60.0,
            1000.0,
            "unsafe",
        ),
        Case(
            "separated",
            "targeted",
            0.0,
            10000.0,
            0.0,
            90.0,
            300.0,
            3.0,
            0.0,
            90.0,
            300.0,
            3.0,
            0.0,
            60.0,
            9260.0,
            "safe",
        ),
    ]
    rows = [
        {
            "suite": case.suite,
            "reference": {"classification": case.reference},
            "encounter": {
                "rel_east_m": case.x,
                "rel_north_m": case.y,
                "a_heading0_deg": case.ah,
                "a_target_heading_deg": case.at,
                "a_speed_kt": case.av,
                "a_turn_rate_deg_sec": case.ar,
                "b_heading0_deg": case.bh,
                "b_target_heading_deg": case.bt,
                "b_speed_kt": case.bv,
                "b_turn_rate_deg_sec": case.br,
                "speed_diff_kt": case.width,
                "projection_time_s": case.horizon,
                "separation_threshold_m": case.threshold,
            },
        }
        for case in cases
    ]
    path.write_text(json.dumps({"cases": rows}))
    return path


@pytest.mark.parametrize("role", ["old", "before", "current"])
@pytest.mark.parametrize("optimised", [False, True])
def test_benchmark_worker_rejects_empty_inputs(tmp_path: Path, role: WorkerRole, optimised: bool) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text('{"cases": []}')
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimised else []),
            "-m",
            "paper.benchmark_library_versions",
            "--worker",
            role,
            "--cases",
            str(cases),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ValueError: Benchmark worker requires at least one case" in result.stderr
    assert not result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="Benchmark workers record Linux CPU affinity")
def test_benchmark_worker_times_single_and_repeated_calls(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "library")
    cases = _case_file(tmp_path / "cases.json")
    worker = start_worker("current", cases, tmp_path, source)
    try:
        assert worker.header["outcomes"]["core_python"] == [True, True, False]
        for repeats in (1, 3):
            for method in ("core_python", "core_compiled", "interface_prebuilt"):
                response = worker.request(method, [0, 2], repeats)
                for timings in (response["wall_us"], response["cpu_us"]):
                    assert len(timings) == 2
                    assert all(math.isfinite(timing) and timing > 0.0 for timing in timings)
    finally:
        worker.close()
    assert worker.process.returncode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Benchmark workers record Linux CPU affinity")
def test_directional_benchmark_uses_a_complete_package_and_current_harness(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "baseline", archived_paper=True, without_heading=True)
    inputs = _directional_inputs(tmp_path / "inputs.json")
    result = benchmark(inputs, source)
    header = result["baseline"]
    assert header is not None
    assert header["source_sha256"] == module_hashes(source / "geometric_safety")
    assert header["outcomes"]["old_compiled"] == header["outcomes"]["old_python"] == [False, True]
    assert [row["safe"][0] for row in result["cases"]] == [False, True]
    assert result["packed_modes"][-1] == "baseline package"
    assert all(len(row) == 4 and all(len(mode) == 5 for mode in row) for row in result["packed_raw_us"])
    assert all(math.isfinite(value) and value > 0.0 for value in result["packed_mean_us"])


@pytest.mark.skipif(sys.platform != "linux", reason="Benchmark workers record Linux CPU affinity")
def test_directional_benchmark_detects_a_real_helper_version_difference(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "baseline", archived_paper=True, without_heading=True)
    utility = source / "geometric_safety" / "util.py"
    tree = ast.parse(utility.read_text())
    heading_difference = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "heading_diff"
    )
    # Restore the heading arithmetic from 9b7869d in an otherwise identical
    # package. The real solver then takes the other side of the half-turn tie.
    heading_difference.body = ast.parse(
        "diff = (h2 - h1 + 180.0) % 360.0 - 180.0\nif abs_diff:\n    return abs(diff)\nreturn diff\n"
    ).body
    utility.write_text(ast.unparse(tree) + "\n")
    expected = module_hashes(source / "geometric_safety")
    current = module_hashes(Path(ra.__file__).parent)
    assert expected["relevant_aircraft.py"] == current["relevant_aircraft.py"]
    assert expected["util.py"] != current["util.py"]
    inputs = _directional_inputs(tmp_path / "inputs.json")
    with pytest.raises(RuntimeError, match="Scalar directional benchmark differs from the baseline package"):
        benchmark(inputs, source)


def test_directional_benchmark_without_a_baseline(tmp_path: Path) -> None:
    result = benchmark(_directional_inputs(tmp_path / "inputs.json"))
    assert result["baseline"] is None
    assert len(result["packed_modes"]) == 3
    assert [row["safe"] for row in result["cases"]] == [[False] * 3, [True] * 3]


def test_directional_benchmark_rejects_a_single_module_baseline(tmp_path: Path) -> None:
    inputs = _directional_inputs(tmp_path / "inputs.json")
    with pytest.raises(ValueError, match="directory containing a complete geometric_safety package"):
        benchmark(inputs, Path(ra.__file__))


@pytest.mark.parametrize("archived_paper", [False, True])
def test_parity_uses_selected_packages_and_current_harness(tmp_path: Path, archived_paper: bool) -> None:
    source = _snapshot(tmp_path / "selected", archived_paper=archived_paper)
    cases = _case_file(tmp_path / "cases.json")
    report = run_worker(source, cases, tmp_path / "worker.json", tmp_path / "cache", saved_only=True)
    assert report["package_root"] == str((source / "geometric_safety").resolve())
    assert report["source_sha256"] == module_hashes(source / "geometric_safety")
    assert report["corpus"] == "saved_only"
    assert len(report["records"]) == 3
    assert [row["core"][0] for row in report["records"]] == [True, True, False]
    assert [row["interface"]["status"] for row in report["records"]] == [
        "certified_safe",
        "certified_safe",
        "unresolved",
    ]


def test_parity_detects_a_real_difference_between_selected_versions(tmp_path: Path) -> None:
    before = _snapshot(tmp_path / "before", archived_paper=True)
    after = _snapshot(tmp_path / "after")
    module = before / "geometric_safety" / "relevant_aircraft.py"
    source = module.read_text()
    guard = "TURN_CERT_FP_GUARD_M = 1e-6"
    assert source.count(guard) == 1
    module.write_text(source.replace(guard, "TURN_CERT_FP_GUARD_M = 1.0"))
    cases = _case_file(tmp_path / "cases.json")
    baseline = run_worker(before, cases, tmp_path / "before.json", tmp_path / "before_cache", saved_only=True)
    candidate = run_worker(after, cases, tmp_path / "after.json", tmp_path / "after_cache", saved_only=True)
    assert baseline["source_sha256"] != candidate["source_sha256"]
    assert not baseline["records"][1]["core"][0]
    assert candidate["records"][1]["core"][0]
    with pytest.raises(ValueError, match="Core decisions differ"):
        compare_reports(baseline, candidate)


def test_parity_never_falls_back_from_an_invalid_selected_package(tmp_path: Path) -> None:
    cases = _case_file(tmp_path / "cases.json")
    with pytest.raises(ValueError, match="no geometric_safety package"):
        run_worker(tmp_path / "missing", cases, tmp_path / "missing.json", tmp_path / "cache", saved_only=True)
    broken = _snapshot(tmp_path / "broken")
    (broken / "geometric_safety" / "__init__.py").write_text(
        'raise RuntimeError("Selected library cannot be imported")\n'
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_worker(broken, cases, tmp_path / "broken.json", tmp_path / "cache", saved_only=True)
    assert not (tmp_path / "broken.json").exists()


@pytest.fixture(scope="module")
def prepared_worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Worker]:
    if sys.platform != "linux":
        pytest.skip("Library benchmark workers require Linux CPU affinity")
    root = tmp_path_factory.mktemp("benchmark_worker")
    source = _snapshot(root / "library", without_heading=True)
    cases = _case_file(root / "cases.json")
    worker = start_worker("old", cases, root, source)
    try:
        yield worker
    finally:
        worker.close()


def test_shutdown_reaps_a_real_benchmark_worker_during_an_outstanding_batch(prepared_worker: Worker) -> None:
    worker = prepared_worker
    assert worker.process.stdin is not None
    request = TimingRequest(op="time", method="old_python", indices=[0], repeats=2**60)
    worker.process.stdin.write(json.dumps(request) + "\n")
    worker.process.stdin.flush()
    worker.close(grace_seconds=0.05, termination_seconds=0.05)
    assert worker.process.poll() is not None
    assert worker.process.stdin.closed
    assert worker.process.stdout is not None
    assert worker.process.stdout.closed
    assert worker.log.closed


def _process(tmp_path: Path, name: str, code: str) -> tuple[subprocess.Popen[str], TextIO]:
    log = (tmp_path / f"{name}.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        return process, log
    except BaseException:
        shutdown_process(process, log, grace_seconds=0.05, termination_seconds=0.05)
        raise


@pytest.mark.skipif(os.name != "posix", reason="The kill escalation control uses POSIX SIGTERM")
def test_shutdown_kills_and_reaps_a_process_that_ignores_termination(tmp_path: Path) -> None:
    process, log = _process(
        tmp_path,
        "ignore_term",
        (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n"
            "time.sleep(600)\n"
        ),
    )
    try:
        shutdown_process(process, log, grace_seconds=0.05, termination_seconds=0.05)
        assert process.returncode == -signal.SIGKILL
        assert log.closed
        assert process.stdin is not None
        assert process.stdin.closed
        assert process.stdout is not None
        assert process.stdout.closed
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cleanup_continues_after_a_real_stream_close_error(tmp_path: Path, prepared_worker: Worker) -> None:
    workers: list[Worker] = []
    for name in ("first", "second"):
        process, log = _process(tmp_path, name, "import sys\nprint('ready', flush=True)\nsys.stdin.readline()\n")
        workers.append(Worker(process, log, prepared_worker.header))
    # Closing the descriptor independently produces a real OSError on close.
    os.close(workers[0].log.fileno())
    try:
        with pytest.raises(ExceptionGroup, match="worker cleanup failed"):
            close_workers(workers)
        assert all(worker.process.poll() is not None and worker.log.closed for worker in workers)
        assert all(worker.process.stdout is not None and worker.process.stdout.closed for worker in workers)
    finally:
        for worker in workers:
            if worker.process.poll() is None:
                shutdown_process(worker.process, worker.log, grace_seconds=0.05, termination_seconds=0.05)


@pytest.mark.skipif(
    sys.platform != "linux", reason="The startup cleanup control uses Linux CPU affinity and /proc/self/fd"
)
def test_failed_startup_cleans_up_before_worker_ownership_transfers(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "library")
    pid_path = tmp_path / "pid.txt"
    initialiser = source / "geometric_safety" / "__init__.py"
    initialiser.write_text(
        initialiser.read_text()
        + (
            "\nimport os\n"
            "from pathlib import Path\n"
            f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
            "print('{}', flush=True)\n"
        )
    )
    cases = _case_file(tmp_path / "cases.json")
    with pytest.raises(KeyError, match="role"):
        start_worker("old", cases, tmp_path, source)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = descriptor.readlink()
        except FileNotFoundError:
            continue
        assert target != tmp_path / "old_worker.log"


def test_parity_does_not_fill_a_missing_submodule_from_editable_install(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "missing_heading", without_heading=True)
    cases = _case_file(tmp_path / "cases.json")
    output = tmp_path / "result.json"
    with pytest.raises(subprocess.CalledProcessError):
        run_worker(source, cases, output, tmp_path / "cache", saved_only=True)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eta_const_m", 1.0),
        ("eta_rate_mps", 1.0),
        ("turn_speed_schedule_uncertainty_kt", 1.0),
        ("turn_delay_max_s", 1.0),
        ("turn_rate_error_max", 0.1),
        ("use_interval_local_lipschitz", False),
        ("min_cert_interval_s", 0.1),
    ],
)
def test_directional_timing_rejects_models_not_representable_by_packed_cases(
    tmp_path: Path, field: str, value: float | bool
) -> None:
    inputs = _directional_inputs(tmp_path / "inputs.json")
    document = object_json(inputs.read_text())
    for row in list_value(document["cases"]):
        object_value(object_value(row)["encounter"])[field] = value
    inputs.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=field):
        benchmark(inputs)


def test_directional_candidate_recompiles_after_a_helper_edit(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "candidate")
    inputs = _directional_inputs(tmp_path / "inputs.json")
    harness = Path(__file__).resolve().parents[2]
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import geometric_safety
sys.path.insert(0, sys.argv[2])
from paper.validate_directional_production import benchmark
benchmark(Path(sys.argv[3]))
"""
    environment = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "shared_cache"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-I", "-c", code, str(source), str(harness), str(inputs)]
    control = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert control.returncode == 0, control.stderr
    utility = source / "geometric_safety/util.py"
    tree = ast.parse(utility.read_text())
    heading_difference = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "heading_diff"
    )
    heading_difference.body = ast.parse(
        "diff = (h2 - h1 + 180.0) % 360.0 - 180.0\nif abs_diff:\n    return abs(diff)\nreturn diff\n"
    ).body
    utility.write_text(ast.unparse(tree) + "\n")
    changed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert changed.returncode != 0, changed.stdout
    assert "accepted a reference-unsafe case" in changed.stderr
