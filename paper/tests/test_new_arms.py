"""Tests for the generation-2 experiment arms.

Covers the certified uniform-sampling baseline, the just-unsafe suite, and
the combined-class campaign configuration.
"""

import numpy as np

from geometric_safety.util import KT_TO_MPS, NMI_TO_M
from paper.evaluation import (
    EvaluationConfig,
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

        results = run_just_unsafe_suite(referenced, config=config)
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
            # The schedule class is excluded: no combined derivation with the
            # execution class exists.
            assert bounds.schedule_uncertainty_kt == 0.0


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
