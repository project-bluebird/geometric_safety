"""Failure controls for reusable benchmark validation and reporting."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paper.benchmark_library_versions import Case, Json, digest, list_value, object_json, object_value
from paper.summarise_performance import summarise
from paper.validate_performance_parity import compare_reports, corpus, evaluate_cases


@pytest.fixture(scope="module")
def parity_report() -> dict[str, Json]:
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
    assert records[0]["interface"].status == "certified_safe"
    assert not records[1]["core"][0]
    assert records[1]["interface"].status == "unresolved"
    return object_json(json.dumps({"records": records}, default=asdict))


def _parity_process(
    tmp_path: Path, report: dict[str, Json], *, optimised: bool, empty: bool
) -> subprocess.CompletedProcess[str]:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report))
    changed = copy.deepcopy(report)
    if empty:
        changed["records"] = []
    else:
        first = object_value(list_value(changed["records"])[0])
        list_value(first["core"])[0] = False
    candidate.write_text(json.dumps(changed))
    probe = """
import sys
from pathlib import Path
from paper.benchmark_library_versions import object_json
from paper.validate_performance_parity import compare_reports
before, after = (object_json(Path(path).read_text()) for path in sys.argv[1:])
compare_reports(before, after)
"""
    command = [sys.executable, *(["-O"] if optimised else []), "-c", probe, str(baseline), str(candidate)]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_optimised_parity_comparison_rejects_a_changed_decision(tmp_path: Path, parity_report: dict[str, Json]) -> None:
    result = _parity_process(tmp_path, parity_report, optimised=True, empty=False)
    assert result.returncode != 0, result.stdout
    assert "decisions differ" in result.stderr


def test_parity_comparison_rejects_empty_evidence(tmp_path: Path, parity_report: dict[str, Json]) -> None:
    result = _parity_process(tmp_path, parity_report, optimised=False, empty=True)
    assert result.returncode != 0, result.stdout
    assert "no records" in result.stderr


def test_parity_comparison_accepts_real_production_records(parity_report: dict[str, Json]) -> None:
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
        Case(f"case-{i}", suite, 0.0, 10000.0, 0.0, 90.0, 300.0, 3.0, 0.0, 90.0, 300.0, 3.0, 0.0, 60.0, 9260.0)
        for i, suite in enumerate(("broad", "targeted", "straight", "circular_clearance"))
    ]
    methods = ["old_python", "old_compiled"] + [
        prefix + name
        for prefix in ("", "before_")
        for name in ("core_python", "core_compiled", "interface_prebuilt", "interface_construct", "interface_core_only")
    ]
    outcomes = {method: [True] * len(cases) for method in methods}
    outcomes["core_python"][0] = False
    outcomes["interface_prebuilt"][0] = False
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [asdict(case) for case in cases]}))
    report = {
        "protocol": {
            "host": "review-test-host",
            "started_at_utc": "2026-10-12T12:00:00+00:00",
            "cpu_model": "test CPU",
            "rounds": 1,
            "calls_per_batch": 1,
            "tail_rounds": 1,
        },
        "raw_wall_us": {method: [[1.0] * len(cases)] for method in methods},
        "workers": {"test": {"outcomes": outcomes}},
        "slow_indices": [0],
        "slow_raw_wall_us": {method: [[1.0]] for method in methods},
        "slow_raw_cpu_us": {method: [[1.0]] for method in methods},
        "slow_summary": {method: {"p95": 1.0, "p99": 1.0, "max": 1.0} for method in methods},
        "checks": {"before_after_decision_parity": [], "known_collision_repairs_verified": False},
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
    assert "retained their core and interface decisions" not in prose
    assert "Core decisions changed on 1 of 4 inputs" in prose
    assert "known_collision_repairs_verified: failed" in prose


def test_report_rejects_different_case_file(tmp_path: Path) -> None:
    results, cases = _report_fixture(tmp_path)
    cases.write_text(cases.read_text() + "\n")
    with pytest.raises(ValueError, match=r"case.*hash"):
        summarise(results, cases, tmp_path / "report")


def test_optimised_benchmark_rejects_failed_controls() -> None:
    probe = """
from paper.benchmark_library_versions import Case, checks
names = ('small-turn-collision', 'half-turn-collision', 'small-turn-safe', 'half-turn-exact-control')
cases = [Case(name, 'regression', *([0.0] * 13)) for name in names]
outcomes = {name: [False] * 4 for name in ('old_python', 'core_python', 'interface_prebuilt')}
checks(cases, outcomes)
"""
    result = subprocess.run([sys.executable, "-O", "-c", probe], text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "Benchmark validation failed" in result.stderr
