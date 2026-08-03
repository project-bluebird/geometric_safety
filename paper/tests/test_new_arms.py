"""Tests for the generation-2 experiment arms.

Covers the certified uniform-sampling baseline, the just-unsafe suite, and
the combined-class campaign configuration.
"""

import dataclasses
import math

import numpy as np
import pytest

from geometric_safety.util import KT_TO_MPS, NMI_TO_M
from paper.evaluation import (
    BANK_LIMIT_DEG,
    GRAVITY_MPS2,
    EvaluationConfig,
    bank_limited_turn_rate_deg_s,
    generate_just_unsafe_suite,
    generate_mixed_turn_suite,
    run_just_unsafe_suite,
)
from paper.run_robustness import build_combined_configs
from paper.run_sampling_sweep import _evaluate_chunk, summarise


class TestCertifiedUniformSampling:
    def test_rule_is_sound_and_more_conservative_than_plain_sampling(self):
        encounters = generate_mixed_turn_suite(60, 20260731)
        rows = _evaluate_chunk((list(enumerate(encounters)), (3.0, 6.0, 12.0)))

        for row in rows:
            encounter = encounters[row["encounter_idx"]]
            expected_l = (
                (encounter.a_speed_kt + encounter.speed_diff_kt)
                + (encounter.b_speed_kt + encounter.speed_diff_kt)
            ) * KT_TO_MPS
            assert row["global_lipschitz_mps"] == expected_l
            for dt_s in (3.0, 6.0, 12.0):
                certuni = bool(row[f"certuni_is_safe_dt{dt_s:g}"])
                # The certified rule is exactly a margin shift of L*dt/2.
                assert certuni == (row[f"proxy_margin_m_dt{dt_s:g}"] >= 0.5 * expected_l * dt_s)
                # Certifying is strictly harder than plain sampling agreement.
                if certuni:
                    assert bool(row[f"proxy_is_safe_dt{dt_s:g}"])
                # Soundness: a certified-safe verdict never lands on a
                # reference-unsafe encounter outside the ambiguity band.
                if row["eligible"] and not row["reference_is_safe"]:
                    assert not certuni

        summaries = summarise(rows, (3.0, 6.0, 12.0))
        for entry in summaries:
            assert entry["certuni_false_safe_n"] == 0
            assert entry["certuni_false_unsafe_n"] >= entry["proxy_false_unsafe_n"]


class TestJustUnsafeSuite:
    def test_generation_and_scoring(self):
        config = EvaluationConfig.quick_defaults()
        referenced = generate_just_unsafe_suite(10, 20260731, config)
        assert len(referenced) == 10
        for item in referenced:
            assert not item.reference.is_safe
            assert -0.5 * NMI_TO_M <= item.reference.margin_m <= -0.05 * NMI_TO_M

        results = run_just_unsafe_suite(referenced)
        summary = results["summary"]
        assert summary["n"] == 10
        # Theorem-backed: the certified solver may never certify a genuinely
        # unsafe encounter.
        assert summary["solver_false_safe_n"] == 0
        assert summary["margin_max_m"] <= -0.05 * NMI_TO_M
        assert len(results["rows"]) == 10


class TestCombinedConfigs:
    def test_combined_configs_cover_licensed_classes_only(self):
        configs = build_combined_configs()
        assert [config.label for config in configs] == ["combined_modest", "combined_larger"]
        for config in configs:
            assert config.campaign == "combined"
            # Simultaneous draws execute the class being certified.
            assert config.draw_bounds is None
            bounds = config.bounds
            assert bounds.differential_wind_kt > 0.0
            assert bounds.position_error_m > 0.0
            assert bounds.heading_error_deg > 0.0
            assert bounds.turn_delay_max_s > 0.0
            assert bounds.turn_rate_error_max > 0.0
            # The in-turn speed class is excluded: no combined derivation with the
            # execution class exists.
            assert bounds.schedule_uncertainty_kt == 0.0


class TestFlyableTurnRates:
    def test_bank_limited_rate_formula(self):
        speed_mps = 480.0 * KT_TO_MPS
        expected = math.degrees(GRAVITY_MPS2 * math.tan(math.radians(BANK_LIMIT_DEG)) / speed_mps)
        assert bank_limited_turn_rate_deg_s(480.0) == pytest.approx(expected, rel=1e-12)
        # About 1.06 deg/s at the suite's fastest admissible speed: the 3 deg/s
        # standard rate is unflyable at jet cruise under a 25 degree bank.
        assert 1.0 < bank_limited_turn_rate_deg_s(480.0) < 1.1
        assert bank_limited_turn_rate_deg_s(220.0) > 2.0

    def test_generated_suites_are_flyable(self):
        encounters = generate_mixed_turn_suite(500, 20260731)
        turning = 0
        for encounter in encounters:
            for speed_kt, rate in (
                (encounter.a_speed_kt, encounter.a_turn_rate_deg_sec),
                (encounter.b_speed_kt, encounter.b_turn_rate_deg_sec),
            ):
                if rate == 0.0:
                    continue
                turning += 1
                cap = bank_limited_turn_rate_deg_s(speed_kt + encounter.speed_diff_kt)
                assert rate <= cap + 1e-9
                assert 1.0 - 1e-9 <= rate <= 3.5
        assert turning > 100

    def test_threshold_flows_through_suites(self):
        threshold_m = 3.0 * NMI_TO_M
        encounters = generate_mixed_turn_suite(20, 7, threshold_m=threshold_m)
        assert all(encounter.separation_threshold_m == threshold_m for encounter in encounters)

        config = dataclasses.replace(EvaluationConfig.quick_defaults(), threshold_m=threshold_m)
        referenced = generate_just_unsafe_suite(4, 11, config)
        assert all(item.encounter.separation_threshold_m == threshold_m for item in referenced)
        for item in referenced:
            assert -0.5 * NMI_TO_M <= item.reference.margin_m <= -0.05 * NMI_TO_M


class TestEnvelopeBudgetCurves:
    def test_wind_curve_matches_worked_example(self):
        from paper.render_envelope_budget_figure import build_budget_curves

        curves = build_budget_curves(dt_s=100.0)
        times = curves["times_s"]
        idx_600 = int(np.argmin(np.abs(times - 600.0)))
        # The manuscript's worked example: a 5 kt wind bound over 600 s is
        # about 1543 m, 0.83 NMI.
        assert curves["wind 5 kt"][idx_600] == pytest.approx(1543.0 / NMI_TO_M, rel=1e-3)
        # Every curve is nondecreasing and the totals dominate each class.
        for name in ("wind 5 kt", "position 150 m", "stated total", "adverse total"):
            assert float(np.diff(curves[name]).min()) >= -1e-12
        assert np.all(curves["stated total"] >= curves["wind 5 kt"] - 1e-12)
        assert np.all(curves["adverse total"] >= curves["stated total"] - 1e-12)

    def test_statistical_summary_fields(self):
        from paper.run_robustness import wilson_interval

        low, high = wilson_interval(0, 1000)
        assert low == 0.0
        assert 0.0 < high < 0.005
        low, high = wilson_interval(300, 1000)
        assert low < 0.3 < high


class TestRuntimeWithEnvelope:
    def test_mixed_suite_reports_inflated_runtime(self):
        from paper.evaluation import run_mixed_turn_suite

        config = EvaluationConfig.quick_defaults()
        encounters = generate_mixed_turn_suite(40, 20260731)
        results = run_mixed_turn_suite(encounters, config=config)
        summary = results["summary"]
        assert "runtime_with_envelope" in summary
        assert summary["runtime_with_envelope"]["median_us"] > 0.0
        assert summary["runtime_envelope_bounds"]["differential_wind_kt"] == 10.0
        assert summary["in_band"]["n"] + summary["eligible_n"] == summary["n"]
        assert np.isfinite(summary["runtime"]["median_us"])
