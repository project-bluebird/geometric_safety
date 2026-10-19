"""Smoke tests for the manuscript evaluation helpers."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from geometric_safety import relevant_aircraft as ra
from paper.evaluation import (
    Encounter,
    EvaluationConfig,
    _split_counts,
    build_refinement_times,
    build_sampled_proxy_counterexample_row,
    build_uniform_times,
    count_fixed_time_evals,
    encounter_to_turn_kwargs,
    evaluate_nominal_proxy,
    evaluate_proposed,
    evaluate_straight_oracle,
    generate_mixed_turn_suite,
    generate_near_threshold_suite,
    outcomes_equivalent,
)
from paper.run_evaluation import reconstruct_run_record_from_summary, render_existing_figures, write_csv


def test_turn_local_core_matches_public_api_default_interval() -> None:
    encounter = Encounter(
        rel_east_m=-18000.0,
        rel_north_m=9000.0,
        a_heading0_deg=25.0,
        a_target_heading_deg=55.0,
        a_speed_kt=360.0,
        a_turn_rate_deg_sec=2.5,
        b_heading0_deg=210.0,
        b_target_heading_deg=180.0,
        b_speed_kt=330.0,
        b_turn_rate_deg_sec=1.5,
        speed_diff_kt=20.0,
        projection_time_s=900.0,
    )

    public_result = ra.catch_up_projection_interval_with_turns(**encounter_to_turn_kwargs(encounter, 51.0, -1.0))
    local_result = ra._catch_up_projection_interval_with_turns_local(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        separation_threshold_m=encounter.separation_threshold_m,
        speed_diff_kt=encounter.speed_diff_kt,
        projection_time_s=encounter.projection_time_s,
        min_cert_interval_s=ra.TURN_TIME_CERT_MIN_INTERVAL_S,
    )

    assert public_result[0] == local_result[0]
    assert public_result[1] == pytest.approx(local_result[1], abs=1e-9)
    assert public_result[2] == pytest.approx(local_result[2], abs=1e-9)


def test_count_fixed_time_evals_matches_compiled_outcome() -> None:
    encounter = generate_mixed_turn_suite(1, seed=20260325)[0]
    counted_evals, counted_outcome = count_fixed_time_evals(
        encounter,
        dt_min_s=3.0,
        use_interval_local_lipschitz=True,
    )
    compiled_outcome = evaluate_proposed(
        encounter,
        dt_min_s=3.0,
        use_interval_local_lipschitz=True,
    )

    assert counted_evals > 0
    assert outcomes_equivalent(counted_outcome, compiled_outcome)


def test_outcomes_equivalent_tolerates_roundoff_only() -> None:
    baseline = evaluate_proposed(
        generate_mixed_turn_suite(1, seed=20260325)[0],
        dt_min_s=1.0,
        use_interval_local_lipschitz=True,
    )
    perturbed = type(baseline)(
        is_safe=baseline.is_safe,
        min_distance_m=baseline.min_distance_m + 2e-12,
        closest_time_s=baseline.closest_time_s,
        margin_m=baseline.margin_m + 2e-12,
    )

    assert outcomes_equivalent(baseline, perturbed)


def test_nominal_proxy_matches_exact_nominal_straight_case() -> None:
    encounter = Encounter(
        rel_east_m=-22000.0,
        rel_north_m=-8000.0,
        a_heading0_deg=45.0,
        a_target_heading_deg=45.0,
        a_speed_kt=350.0,
        a_turn_rate_deg_sec=0.0,
        b_heading0_deg=120.0,
        b_target_heading_deg=120.0,
        b_speed_kt=310.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=25.0,
        projection_time_s=600.0,
    )

    nominal = evaluate_nominal_proxy(encounter)
    oracle = evaluate_straight_oracle(
        Encounter(
            rel_east_m=encounter.rel_east_m,
            rel_north_m=encounter.rel_north_m,
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_speed_kt=encounter.a_speed_kt,
            a_turn_rate_deg_sec=0.0,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_speed_kt=encounter.b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            speed_diff_kt=0.0,
            projection_time_s=encounter.projection_time_s,
            separation_threshold_m=encounter.separation_threshold_m,
        )
    )

    assert nominal.is_safe == oracle.is_safe
    assert nominal.min_distance_m == pytest.approx(oracle.min_distance_m, abs=1e-9)
    assert nominal.closest_time_s == pytest.approx(oracle.closest_time_s, abs=1e-9)


def test_uniform_and_refined_times_include_projection_endpoint() -> None:
    coarse_times = build_uniform_times(12.0, 5.0)
    coarse_distances = np.asarray([8.0, 3.0, 8.5, 9.0], dtype=np.float64)
    refined_times = build_refinement_times(
        coarse_times=coarse_times,
        coarse_distances=coarse_distances,
        projection_time_s=12.0,
        threshold_m=5.0,
        coarse_dt_s=5.0,
        refined_dt_s=1.0,
        near_threshold_band_m=3.5,
    )

    assert coarse_times[-1] == pytest.approx(12.0)
    assert np.all(refined_times <= 12.0 + 1e-12)
    assert np.any(np.isclose(refined_times, 6.0))


def test_sampled_proxy_counterexample_shows_sampling_is_heuristic() -> None:
    row = build_sampled_proxy_counterexample_row(EvaluationConfig())

    assert row["coarse_proxy_dt_s"] == pytest.approx(6.0)
    assert row["coarse_proxy_is_safe"] == 1
    assert row["reference_is_safe"] == 0
    assert row["proposed_is_safe"] == 0
    assert row["nominal_proxy_is_safe"] == 1


def test_split_counts_balances_targets() -> None:
    assert _split_counts(10, 3) == [4, 3, 3]
    assert _split_counts(5, 5) == [1, 1, 1, 1, 1]


def test_generate_near_threshold_suite_parallel_smoke() -> None:
    config = EvaluationConfig(n_jobs=2, quick=True)
    suite = generate_near_threshold_suite(4, seed=20260325, config=config)

    assert len(suite) == 4
    for referenced in suite:
        assert referenced.encounter.turn_count() >= 1
        assert referenced.reference.is_safe
        assert config.near_threshold_margin_min_m <= referenced.reference.margin_m <= config.near_threshold_margin_max_m


def test_reconstruct_run_record_from_summary_shape() -> None:
    summary = {
        "config": {
            "straight_n": 100,
            "mixed_turn_n": 95,
            "near_threshold_n": 50,
            "seed": 20260325,
        },
        "straight": {
            "n": 100,
            "oracle_vs_proposed_agreement_rate": 1.0,
            "max_distance_error_m": 0.0,
            "max_time_error_s": 0.0,
            "nominal_proxy": {
                "eligible_n": 100.0,
                "false_safe_rate": 0.2,
                "false_unsafe_rate": 0.0,
                "overall_disagreement_rate": 0.04,
            },
        },
        "mixed_turn": {
            "n": 95,
            "eligible_n": 90,
            "certification_rate_given_reference_safe": 78.0 / 80.0,
            "proposed": {
                "eligible_n": 90.0,
                "false_safe_rate": 0.0,
                "false_unsafe_rate": 2.0 / 80.0,
                "overall_disagreement_rate": 2.0 / 90.0,
            },
            "nominal_proxy": {
                "eligible_n": 90.0,
                "false_safe_rate": 0.3,
                "false_unsafe_rate": 0.0,
                "overall_disagreement_rate": 3.0 / 90.0,
            },
            "coarse_proxy": {
                "eligible_n": 90.0,
                "false_safe_rate": 0.0,
                "false_unsafe_rate": 0.0,
                "overall_disagreement_rate": 0.0,
            },
            "runtime": {
                "mean_us": 8.0,
                "median_us": 6.0,
                "p50_us": 6.0,
                "p95_us": 15.0,
                "p99_us": 25.0,
                "max_us": 30.0,
            },
        },
        "near_threshold": {
            "summary_rows": [
                {
                    "lipschitz_mode": "local",
                    "dt_min_s": 3.0,
                    "n": 50,
                    "certification_rate": 47.0 / 50.0,
                    "median_fixed_time_evals": 9.0,
                    "p95_fixed_time_evals": 20.0,
                    "median_runtime_us": 10.0,
                }
            ],
            "margin_summary": [],
        },
        "kernel_runtime": {
            "mean_us": 2.0,
            "median_us": 2.0,
            "p50_us": 2.0,
            "p95_us": 3.0,
            "p99_us": 4.0,
            "max_us": 5.0,
        },
    }

    record = reconstruct_run_record_from_summary(summary)

    assert record["straight"]["oracle_safe_n"] == 80
    assert record["straight"]["oracle_unsafe_n"] == 20
    assert record["straight"]["nominal_proxy_false_safe_n"] == 4
    assert record["mixed_turn"]["reference_safe_n"] == 80
    assert record["mixed_turn"]["reference_unsafe_n"] == 10
    assert record["mixed_turn"]["proposed_false_unsafe_n"] == 2
    assert record["mixed_turn"]["nominal_proxy_false_safe_n"] == 3
    assert record["near_threshold"]["default_local_dt3_certified_n"] == 47


def test_render_existing_figures_writes_pdf_only(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "rendered"
    source_dir.mkdir()

    sampled_row = build_sampled_proxy_counterexample_row(EvaluationConfig())
    summary = {
        "narrative_scenarios": {
            "crossing": sampled_row,
            "proxy_miss": sampled_row,
            "sampled_proxy_miss": sampled_row,
            "dt_sensitive": sampled_row,
        },
        "near_threshold": {
            "summary_rows": [
                {
                    "lipschitz_mode": "local",
                    "dt_min_s": 1.0,
                    "certification_rate": 0.99,
                    "median_fixed_time_evals": 8.0,
                },
                {
                    "lipschitz_mode": "local",
                    "dt_min_s": 3.0,
                    "certification_rate": 0.95,
                    "median_fixed_time_evals": 9.0,
                },
                {
                    "lipschitz_mode": "global",
                    "dt_min_s": 1.0,
                    "certification_rate": 0.97,
                    "median_fixed_time_evals": 12.0,
                },
                {
                    "lipschitz_mode": "global",
                    "dt_min_s": 3.0,
                    "certification_rate": 0.91,
                    "median_fixed_time_evals": 13.0,
                },
            ]
        },
    }
    (source_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with (source_dir / "mixed_turn_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reference_margin_m",
                "max_turn_angle_deg",
                "reference_is_safe",
                "proposed_is_safe",
                "speed_diff_kt",
                "eligible_for_boolean_tallies",
                "nominal_proxy_is_safe",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "reference_margin_m": 0.4 * 1852.0,
                "max_turn_angle_deg": 25.0,
                "reference_is_safe": 1,
                "proposed_is_safe": 1,
                "speed_diff_kt": 10.0,
                "eligible_for_boolean_tallies": 1,
                "nominal_proxy_is_safe": 1,
            }
        )
        writer.writerow(
            {
                "reference_margin_m": -0.2 * 1852.0,
                "max_turn_angle_deg": 40.0,
                "reference_is_safe": 0,
                "proposed_is_safe": 0,
                "speed_diff_kt": 20.0,
                "eligible_for_boolean_tallies": 1,
                "nominal_proxy_is_safe": 1,
            }
        )

    render_existing_figures(source_dir, output_dir, figure_formats=("pdf",))

    for stem in (
        "fig_crossing",
        "fig_nominal_proxy_miss",
        "fig_stress_maps",
        "fig_ablation",
        "fig_sampled_proxy_miss",
    ):
        assert (output_dir / f"{stem}.pdf").exists()
        assert not (output_dir / f"{stem}.png").exists()


def test_write_csv_creates_valid_empty_gzip(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv.gz"

    write_csv(path, [])

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.read() == ""
