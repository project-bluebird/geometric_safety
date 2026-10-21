"""Failure controls for reusable benchmark validation and reporting."""

import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import pytest

from geometric_safety import relevant_aircraft as ra
from paper.benchmark_contracts import (
    BenchmarkReport,
    WorkerHeader,
    digest,
    list_value,
    module_hashes,
    object_json,
    object_value,
    read_benchmark_report,
)
from paper.benchmark_library_versions import Case, statistics_us, summary
from paper.summarise_performance import summarise
from paper.validate_performance_parity import ParityReport, compare_reports, corpus, evaluate_cases, read_parity_report


@pytest.fixture(scope="module")
def parity_report() -> ParityReport:
    cases = [
        Case(
            f"rigid-translation-{distance}",
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
        for distance in (10000.0, 1000.0)
    ]
    records = evaluate_cases(cases)
    assert records[0]["core"][0]
    assert records[0]["interface"]["status"] == "certified_safe"
    assert not records[1]["core"][0]
    assert records[1]["interface"]["status"] == "unresolved"
    package = Path(ra.__file__).parent
    return ParityReport(
        seed=2026101206,
        cases_sha256="fixture",
        package_root=str(package),
        source_sha256=module_hashes(package),
        corpus="saved_only",
        guard_m=ra.TURN_CERT_FP_GUARD_M,
        records=records,
    )


def _parity_process(
    tmp_path: Path, report: ParityReport, *, optimised: bool, empty: bool
) -> subprocess.CompletedProcess[str]:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report))
    changed = copy.deepcopy(report)
    if empty:
        changed["records"] = []
    else:
        first = changed["records"][0]
        first["core"] = (False, first["core"][1], first["core"][2])
    candidate.write_text(json.dumps(changed))
    probe = """
import sys
from pathlib import Path
from paper.validate_performance_parity import compare_reports, read_parity_report
before, after = (read_parity_report(Path(path)) for path in sys.argv[1:])
compare_reports(before, after)
"""
    command = [sys.executable, *(["-O"] if optimised else []), "-c", probe, str(baseline), str(candidate)]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_optimised_parity_comparison_rejects_a_changed_decision(tmp_path: Path, parity_report: ParityReport) -> None:
    result = _parity_process(tmp_path, parity_report, optimised=True, empty=False)
    assert result.returncode != 0, result.stdout
    assert "decisions differ" in result.stderr


def test_parity_comparison_rejects_empty_evidence(tmp_path: Path, parity_report: ParityReport) -> None:
    result = _parity_process(tmp_path, parity_report, optimised=False, empty=True)
    assert result.returncode != 0, result.stdout
    assert "no records" in result.stderr


def test_parity_comparison_accepts_real_production_records(parity_report: ParityReport) -> None:
    result = compare_reports(parity_report, copy.deepcopy(parity_report))
    assert result["cases"] == 2
    assert result["maximum_distance_change_m"] == {"core": 0.0, "interface": 0.0}
    assert result["diagnostic_time_changes"] == {"core": 0, "interface": 0}


def test_guard_corpus_does_not_depend_on_saved_case_order(tmp_path: Path) -> None:
    # The guard family has its own analytic motion, even without a saved corpus.
    path = tmp_path / "cases.json"
    path.write_text('{"cases": []}')
    cases = [case for case in corpus(path) if case.suite == "guard_boundary"]
    assert len(cases) == 33
    assert all(case.x == 0.0 and case.av == case.bv and case.ar == case.br and case.width == 0.0 for case in cases)
    assert all(case.threshold == 9260.0 for case in cases)


def _report_fixture(tmp_path: Path) -> tuple[Path, Path]:
    cases = [
        Case(
            f"case-{i}",
            suite,
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
            float(i % 2) * 5.0,
            60.0,
            9260.0,
        )
        for i, suite in enumerate(("broad", "targeted", "straight", "circular_clearance"))
    ]
    methods = ["old_python", "old_compiled"] + [
        prefix + name
        for prefix in ("", "before_")
        for name in ("core_python", "core_compiled", "interface_prebuilt", "interface_construct", "interface_core_only")
    ]
    records = evaluate_cases(cases)
    outcomes = {
        method: [
            record["interface"]["status"] == "certified_safe" if "interface" in method else record["core"][0]
            for record in records
        ]
        for method in methods
    }
    # Deliberately failed controls exercise truthful reporting of an invalid run.
    outcomes["core_python"][0] = False
    outcomes["interface_prebuilt"][0] = False
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [asdict(case) for case in cases]}))
    raw = {method: [[1.0] * len(cases)] for method in methods}
    header: WorkerHeader = {
        "ready": True,
        "role": "current",
        "preparation_seconds": 0.0,
        "source_sha256": module_hashes(Path(ra.__file__).parent),
        "outcomes": outcomes,
        "inflation_m": [0.0] * len(cases),
        "python": platform.python_version(),
        "numpy": "fixture",
        "numba": "fixture",
        "affinity": [0],
    }
    report: BenchmarkReport = {
        "protocol": {
            "host": "review-test-host",
            "started_at_utc": "2026-10-12T12:00:00+00:00",
            "cpu_model": "test CPU",
            "rounds": 1,
            "calls_per_batch": 1,
            "tail_rounds": 1,
            "old_revision": "fixture",
            "current_wheel_sha256": "fixture",
            "before_manifest_sha256": None,
            "expected_recovered_false_rejections": 0,
            "require_before_decision_parity": False,
            "current_manifest_sha256": "fixture",
            "cases": len(cases),
            "suites": {case.suite: 1 for case in cases},
            "tail_calls_per_measurement": 1,
            "cpu": 0,
            "host_load_average_start": [0.0, 0.0, 0.0],
            "host_load_average_end": [0.0, 0.0, 0.0],
            "single_thread_environment": {},
            "population_statistics": "fixture",
            "tail_statistics": "fixture",
            "timed_boundary": "fixture",
            "semantics": "fixture",
            "limitations": "fixture",
        },
        "source_sha256": "fixture",
        "inputs_sha256": "fixture",
        "old_archive_sha256": "fixture",
        "raw_wall_us": raw,
        "raw_cpu_us": raw,
        "workers": {"current": header},
        "summary": summary(cases, raw, outcomes),
        "slow_indices": [0],
        "slow_case_ids": [cases[0].id],
        "slow_raw_wall_us": {method: [[1.0]] for method in methods},
        "slow_raw_cpu_us": {method: [[1.0]] for method in methods},
        "slow_summary": {method: statistics_us([1.0]) for method in methods},
        "checks": {
            "before_after_decision_parity": [],
            "known_collision_repairs_verified": False,
            "known_separated_controls_verified": True,
            "no_current_acceptance_of_saved_unsafe_cases": True,
            "interface_matches_effective_core": True,
            "decision_differences": [],
        },
        "cases_sha256": digest(cases_path),
    }
    results = tmp_path / "results.json"
    results.write_text(json.dumps(report))
    return results, cases_path


def test_report_uses_recorded_host_and_actual_decisions(tmp_path: Path) -> None:
    results, cases = _report_fixture(tmp_path)
    summarise(results, cases, tmp_path / "report")
    prose = (tmp_path / "report/report.md").read_text()
    assert "review-test-host" in prose
    assert "Interface decisions changed on 1 of 4 inputs" in prose
    assert "Core decisions changed on 1 of 4 inputs" in prose
    assert "known_collision_repairs_verified: failed" in prose


def test_report_rejects_different_case_file(tmp_path: Path) -> None:
    results, cases = _report_fixture(tmp_path)
    cases.write_text(cases.read_text() + "\n")
    with pytest.raises(ValueError, match=r"case.*hash"):
        summarise(results, cases, tmp_path / "report")


def test_report_rejects_a_nonboolean_decision(tmp_path: Path) -> None:
    results, _ = _report_fixture(tmp_path)
    # A string "false" must not become True through bool(value).
    report = object_json(results.read_text())
    worker = object_value(object_value(report["workers"])["current"])
    list_value(object_value(worker["outcomes"])["core_python"])[0] = "false"
    results.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="boolean"):
        read_benchmark_report(results)


def test_report_rejects_missing_timing_samples(tmp_path: Path) -> None:
    results, _ = _report_fixture(tmp_path)
    report = read_benchmark_report(results)
    report["raw_wall_us"]["core_python"][0].pop()
    results.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="sample counts"):
        read_benchmark_report(results)


@pytest.mark.parametrize(
    ("failed_method", "case_index"),
    [
        (None, 0),
        ("old_python", 0),
        ("old_python", 1),
        *[(method, index) for method in ("core_python", "interface_prebuilt") for index in range(5)],
    ],
)
def test_optimised_benchmark_validates_controls(failed_method: str | None, case_index: int) -> None:
    probe = """
import json
import math
import sys
from paper.benchmark_library_versions import Case, checks
names = ('small-turn-collision', 'half-turn-collision', 'small-turn-safe', 'half-turn-exact-control')
# Validation consumes recorded decisions; motion fields are not evaluated here.
cases = [Case(name, 'regression', *([0.0] * 13)) for name in names]
cases.append(Case('saved-unsafe', 'saved', *([0.0] * 13), reference='saved constant-speed unsafe'))
outcomes = {
    'old_python': [True, True, True, True, False],
    'core_python': [False, False, True, True, False],
    'interface_prebuilt': [False, False, True, True, False],
}
method, index = sys.argv[1], int(sys.argv[2])
if method:
    outcomes[method][index] = not outcomes[method][index]
print(json.dumps(checks(cases, outcomes)))
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", probe, failed_method or "", str(case_index)],
        text=True,
        capture_output=True,
        check=False,
    )
    if failed_method is None:
        assert result.returncode == 0, result.stderr
        checks = object_json(result.stdout)
        for key in (
            "known_collision_repairs_verified",
            "known_separated_controls_verified",
            "no_current_acceptance_of_saved_unsafe_cases",
        ):
            assert checks[key] is True
    else:
        assert result.returncode != 0, result.stdout
        if case_index == 4:
            failure = "a saved unsafe case was accepted"
        else:
            negation = "not " if case_index < 2 and failed_method != "old_python" else ""
            failure = f"{negation}outcomes['{failed_method}'][index]"
        assert f"Benchmark validation failed: {failure}" in result.stderr


@pytest.mark.parametrize("kind", ["unavailable", "interval"])
def test_parity_roundtrip_preserves_unavailable_diagnostics(
    tmp_path: Path, parity_report: ParityReport, kind: str
) -> None:
    case = Case(
        "diagnostic",
        "regression",
        2.0 * ra.NUMERICAL_SCALAR_LIMIT,
        0.0,
        0.0,
        90.0,
        300.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        600.0,
        9260.0,
    )
    if kind == "interval":
        case = replace(
            case,
            x=-0.75 * 2.0**60 * 1e-18,
            y=1.0,
            ah=89.0,
            ar=1.0,
            av=1e-18 / ra.KT_TO_MPS,
            horizon=2.0**60,
            threshold=1.0 - 5e-7,
        )
    report = copy.deepcopy(parity_report)
    report["records"] = evaluate_cases([case])
    record = report["records"][0]
    assert not record["core"][0]
    assert math.isnan(record["core"][2])
    assert record["interface"]["base_closest_time_s"] is None
    if kind == "interval":
        assert record["interface"]["diagnostic_kind"] == "interval_bound"
        assert record["interface"]["base_bound_interval_s"] is not None
    else:
        assert math.isnan(record["core"][1])
        assert record["interface"]["base_min_distance_m"] is None
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    restored = read_parity_report(path)
    comparison = compare_reports(report, restored)
    assert comparison["maximum_distance_change_m"] == {"core": 0.0, "interface": 0.0}
    assert comparison["diagnostic_time_changes"] == {"core": 0, "interface": 0}
    changed = copy.deepcopy(restored)
    changed["records"][0]["core"] = (False, 0.0, 0.0)
    with pytest.raises(ValueError, match="distance"):
        compare_reports(restored, changed)
    changed = copy.deepcopy(restored)
    changed["records"][0]["interface"]["diagnostic_kind"] = "straight_minimum"
    with pytest.raises(ValueError, match="diagnostic_kind"):
        compare_reports(restored, changed)
    if kind == "interval":
        changed = copy.deepcopy(restored)
        changed["records"][0]["interface"]["base_bound_interval_s"] = (0.0, 1.0)
        assert compare_reports(restored, changed)["diagnostic_time_changes"]["interface"] == 1
