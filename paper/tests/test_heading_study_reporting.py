"""Exercise saved-study reporting with real decisions and non-default protocols."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import geometric_safety
from geometric_safety.heading import PROFILE_REFINEMENT_MARGIN_M
from paper.benchmark_contracts import Json, digest, module_hashes, object_json, object_value, parse_list, string
from paper.plot_heading_study import plot
from paper.run_heading_study import evaluate, load_protocol, run
from paper.summarise_heading_study import read_outcomes, summarise

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, Json]:
    protocol = object_json((ROOT / "paper/fixtures/heading_study_aviation_2027.json").read_text())
    pairs = parse_list(protocol["pair_scenes"], object_value)
    crossing, overtaking = pairs[0], pairs[4]
    crossing["id"], overtaking["id"] = "CROSSING", "OVERTAKING"
    protocol["pair_scenes"] = [crossing, overtaking]
    protocol["multi_traffic_scenes"] = [
        {"id": "CROSS", "pairs": ["CROSSING"]},
        {"id": "OVERTAKE", "pairs": ["OVERTAKING"]},
    ]
    protocol["target_headings_deg"] = [0, 30, 90]
    protocol["speed_half_widths_kt"] = [50]
    protocol["query_count_per_model"] = 12
    protocol["complete_menu_count"] = 4
    protocol["pair_checks_per_complete_menu"] = 3
    timing = object_value(protocol["timing"])
    timing["rounds"], timing["individually_timed_repetitions_per_round"] = 3, 2
    reference = object_value(protocol["reference"])
    reference["precision_decimal_digits"], reference["decisive_result_recheck_digits"] = 40, 60
    return protocol


def _save_study(directory: Path, protocol: dict[str, Json]) -> None:
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps(protocol))
    cases, settings = load_protocol(directory / "protocol.json")
    records = [evaluate(case, settings) for case in cases]
    (directory / "results.jsonl").write_text("".join(json.dumps(asdict(row)) + "\n" for row in records))
    (directory / "complete.json").write_text(
        json.dumps({"count": len(records), "results_sha256": digest(directory / "results.jsonl")})
    )
    (directory / "manifest.json").write_text(
        json.dumps({"package_sha256": module_hashes(Path(geometric_safety.__file__).resolve().parent)})
    )


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("heading-report") / "study"
    _save_study(directory, _protocol())
    return directory


def test_summary_identifies_acceptance_without_decisive_reference(tmp_path: Path) -> None:
    protocol = _protocol()
    scene = parse_list(protocol["pair_scenes"], object_value)[0]
    scene.update(
        traffic_position_nm=[0, 5 + 10 / 1852],
        traffic_initial_heading_deg=90,
        traffic_target_heading_deg=90,
        traffic_nominal_speed_kt=250,
    )
    protocol["pair_scenes"] = [scene]
    protocol["multi_traffic_scenes"] = [{"id": "PARALLEL", "pairs": ["CROSSING"]}]
    protocol["target_headings_deg"], protocol["speed_half_widths_kt"], protocol["horizon_s"] = [90], [0], [1]
    protocol["query_count_per_model"] = 1
    # Further subdivision cannot resolve a constant distance inside the exclusion
    # band. Keep this reporting regression from exhausting the full study budget.
    object_value(protocol["reference"])["max_time_cells_per_query"] = 2
    directory = tmp_path / "boundary"
    _save_study(directory, protocol)
    (outcome,) = read_outcomes(directory)
    # The exact straight-motion check accepts 10 m of clearance. The reference's
    # 25 m exclusion band deliberately leaves this genuine boundary case unresolved.
    assert outcome.variable
    assert outcome.variable_reference == "unresolved"
    record = object_json((directory / "results.jsonl").read_text())
    reference = object_value(object_value(record["variable_reference"])["primary"])
    assert float(string(reference["upper_m"])) == pytest.approx(9270.0)
    result = summarise(directory)
    assert result["0"].variable_accepted_reference_unresolved == 1
    saved = object_json((directory / "summary.json").read_text())
    assert saved["accepted_reference_unresolved_cases"] == [outcome.case.identifier]
    assert saved["acceptances_match_reference_separated"] is False
    assert "Accepted queries with unresolved varying-speed references: 1." in (directory / "report.md").read_text()


def test_summary_reports_actual_design_and_agreement(study: Path, tmp_path: Path) -> None:
    directory = tmp_path / "study"
    shutil.copytree(study, directory)
    outcomes = read_outcomes(directory)
    assert any(r.constant_reference == "separated" and r.variable_reference == "conflict" for r in outcomes)
    assert any(r.variable and r.variable_reference == "separated" for r in outcomes)
    summarise(directory)
    report = (directory / "report.md").read_text()
    assert "All 12 predeclared candidate queries" in report
    assert "40 and 60 digits" in report
    assert "912" not in report
    summary = object_json((directory / "summary.json").read_text())
    assert summary["acceptances_match_reference_separated"] is True
    assert summary["accepted_reference_unresolved_cases"] == []
    assert summary["constant_accepted_reference_unresolved_cases"] == []
    assert "Accepted queries with unresolved constant-speed references: 0." in report


def test_summary_identifies_constant_acceptance_without_decisive_reference(tmp_path: Path) -> None:
    protocol = _protocol()
    scene = parse_list(protocol["pair_scenes"], object_value)[0]
    scene["traffic_target_heading_deg"] = 30
    protocol["pair_scenes"] = [scene]
    protocol["multi_traffic_scenes"] = [{"id": "TURNING", "pairs": ["CROSSING"]}]
    protocol["target_headings_deg"], protocol["speed_half_widths_kt"], protocol["horizon_s"] = [110], [10], [180]
    protocol["query_count_per_model"] = 1
    protocol["complete_menu_count"] = 1
    protocol["pair_checks_per_complete_menu"] = 1
    directory = tmp_path / "constant-boundary"
    _save_study(directory, protocol)
    (outcome,) = read_outcomes(directory)
    assert outcome.constant
    assert outcome.constant_reference == "unresolved"
    assert not outcome.variable
    result = summarise(directory)
    assert result["10"].constant_accepted_reference_unresolved == 1
    assert result["10"].variable_accepted_reference_unresolved == 0
    saved = object_json((directory / "summary.json").read_text())
    assert saved["constant_accepted_reference_unresolved_cases"] == [outcome.case.identifier]
    assert saved["accepted_reference_unresolved_cases"] == []
    report = (directory / "report.md").read_text()
    assert "Accepted queries with unresolved constant-speed references: 1." in report
    assert "Accepted queries with unresolved varying-speed references: 0." in report


def test_execution_rejects_margin_different_from_validation_protocol(tmp_path: Path) -> None:
    protocol = _protocol()
    object_value(protocol["reference"])["classification_exclusion_band_m"] = PROFILE_REFINEMENT_MARGIN_M + 1
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match=r"exclusion band.*refinement margin"):
        run(path, tmp_path / "results", workers=1)
    assert not (tmp_path / "results").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="The actual timing tool records Linux CPU affinity and load")
@pytest.mark.parametrize("has_effects", [True, False])
def test_plot_report_uses_actual_scenes_and_measurements(study: Path, tmp_path: Path, has_effects: bool) -> None:
    directory = tmp_path / "study"
    if has_effects:
        shutil.copytree(study, directory)
    else:
        protocol = _protocol()
        protocol["target_headings_deg"] = [0, 90]
        protocol["query_count_per_model"] = 8
        protocol["pair_checks_per_complete_menu"] = 2
        _save_study(directory, protocol)
    outcomes = read_outcomes(directory)
    assert (
        any(row.constant_reference == "separated" and row.variable_reference == "conflict" for row in outcomes)
        == has_effects
    )
    summarise(directory)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    for extra in ([], ["--cold"]):
        subprocess.run(
            [sys.executable, "-m", "paper.measure_heading_study", str(directory), *extra],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    plot(directory)
    report = (directory / "study-results.md").read_text()
    heading_count = 3 if has_effects else 2
    assert f"{heading_count} candidate headings" in report
    assert f"CROSS: CROSSING ({heading_count} pair checks)" in report
    assert f"OVERTAKE: OVERTAKING ({heading_count} pair checks)" in report
    assert "6 individually timed repetitions in 3 rounds" in report
    effect_header = "| Scene | Query | Constant lower bound (NM) |"
    if has_effects:
        assert effect_header in report
        assert "| CROSSING | CROSSING_t180_w50_h30 |" in report
    else:
        assert "No query has constant-speed separation and an attainable varying-speed conflict." in report
        assert effect_header not in report
    assert "C1 crossing geometry" not in report
    assert "57 public-API calls" not in report
    assert "40 and 60 digits" in report
    for suffix in ("png", "pdf", "svg"):
        assert (directory / f"heading-menus.{suffix}").stat().st_size > 1000
