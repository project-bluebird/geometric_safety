"""Exercise the actual frozen inputs and the production/reference adapter."""

from pathlib import Path

import mpmath as mp

from paper.run_heading_study import decimal, evaluate, load_protocol


def protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "heading_study_aviation_2027.json"


def test_frozen_design_has_all_inputs_and_correct_relative_frame() -> None:
    cases, settings = load_protocol(protocol_path())
    assert len(cases) == 912
    assert len({case.identifier for case in cases}) == 912
    assert {case.scene for case in cases} == {"C1", "C2", "H1", "H2", "O1", "O2"}
    for scene in {case.scene for case in cases}:
        selected = [case for case in cases if case.scene == scene]
        assert len(selected) == 152
        assert {case.target_deg for case in selected} == set(range(0, 181, 10))
        assert {case.width_kt for case in selected} == {0, 10, 25, 50}
        assert {case.query.horizon_s for case in selected} == {180, 300}
    crossing = next(case for case in cases if case.identifier == "C1_t180_w50_h90")
    assert crossing.query.relative_position_m == (-18520.0, 18520.0)
    assert crossing.query.a.lower_mps == 200 * (1852 / 3600)
    assert crossing.query.a.upper_mps == 300 * (1852 / 3600)
    assert crossing.query.threshold_m == 9260.0
    assert settings.digits == 80
    assert settings.recheck_digits == 120


def test_known_crossing_runs_real_production_and_both_references() -> None:
    cases, settings = load_protocol(protocol_path())
    case = next(case for case in cases if case.identifier == "C1_t180_w0_h90")
    # Both aircraft reach (10, 0) NM after 144 seconds at 250 kt.
    result = evaluate(case, settings)
    assert not result.nominal_accepted
    assert not result.constant_accepted
    assert result.variable_status == "unresolved"
    assert result.constant_reference.classification == "conflict"
    assert result.variable_reference.classification == "conflict"
    for reference in (result.constant_reference, result.variable_reference):
        assert reference.recheck is not None
        assert reference.recheck.classification == "conflict"
        assert reference.primary.witness is not None
        assert float(reference.primary.upper_m) < 9235
    assert float.fromhex(result.input_hex["relative_east"]) == -18520
    assert float.fromhex(result.input_hex["a_lower"]) == 250 * (1852 / 3600)


def test_saved_high_precision_values_survive_default_context() -> None:
    with mp.workdps(80):
        value = mp.mpf(1) / 7
    encoded = decimal(value)
    with mp.workdps(120):
        assert abs(mp.mpf(encoded) - value) < mp.mpf("1e-100")
