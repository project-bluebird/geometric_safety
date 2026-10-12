"""Failure controls for reusable benchmark validation and reporting."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paper.benchmark_library_versions import Case, digest
from paper.summarise_performance import summarise
from paper.validate_performance_parity import corpus


def _parity_process(tmp_path: Path, *, optimised: bool, empty: bool) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "validate_performance_parity.py"
    probe = """
import json, runpy, sys
from pathlib import Path
from unittest.mock import patch
script, work, empty = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == 'True'
sys.path.insert(0, str(script.parent))
namespace = runpy.run_path(str(script), run_name='parity_regression')
cases = work / 'cases.json'
cases.write_text('{"cases": []}')
def fake_worker(command, **kwargs):
    output = Path(command[command.index('--output') + 1])
    interface = dict(status='certified_safe', reason='same', inflation_m=0.0,
                     base_min_distance_m=10.0, base_closest_time_s=0.0)
    row = dict(id='different-decision', core=[output.stem == 'candidate', 10.0, 0.0], interface=interface)
    records = [] if empty else [row]
    output.write_text(json.dumps(dict(records=records, source_sha256={})))
args = ['parity', '--cases', str(cases), '--baseline', str(work / 'before'),
        '--candidate', str(work / 'after'), '--output', str(work / 'output')]
with patch.object(sys, 'argv', args), patch('subprocess.run', fake_worker):
    namespace['main']()
"""
    command = [sys.executable, *(["-O"] if optimised else []), "-c", probe, str(script), str(tmp_path), str(empty)]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_optimised_parity_run_rejects_a_changed_decision(tmp_path: Path) -> None:
    result = _parity_process(tmp_path, optimised=True, empty=False)
    assert result.returncode != 0, result.stdout
    assert "decisions differ" in result.stderr
    assert not (tmp_path / "output/summary.json").exists()


def test_parity_run_rejects_empty_evidence(tmp_path: Path) -> None:
    result = _parity_process(tmp_path, optimised=False, empty=True)
    assert result.returncode != 0, result.stdout
    assert "no records" in result.stderr


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
