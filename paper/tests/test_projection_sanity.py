"""Tests for the projection sanity-check helpers."""

import csv
import gzip
from pathlib import Path

import numpy as np

from geometric_safety.util import NMI_TO_M
from paper.evaluation import Encounter
from paper.plot_projection_sweep import load_row_files
from paper.projection_plotting import render_projection_diagnostic_plots
from paper.projection_sanity import (
    ProjectionDeterministicSweepConfig,
    ProjectionSanityConfig,
    build_projection_times,
    embed_local_track_on_sphere,
    evaluate_projection_encounter,
    generate_deterministic_mixed_turn_sweep,
    generate_deterministic_straight_sweep,
    generate_projection_stress_suite,
    great_circle_distance_m,
    run_projection_deterministic_sweep,
    run_projection_sanity_check,
    shift_tracks_for_anchor_mode,
)


def test_build_projection_times_includes_endpoint() -> None:
    times_s = build_projection_times(10.5, 1.0)
    assert times_s[0] == 0.0
    assert times_s[-1] == 10.5
    assert np.all(np.diff(times_s) > 0.0)


def test_embedding_preserves_range_from_anchor() -> None:
    track_xy = np.asarray(
        [
            [0.0, 0.0],
            [1_000.0, 0.0],
            [0.0, 2_000.0],
        ],
        dtype=np.float64,
    )
    lat_deg, lon_deg = embed_local_track_on_sphere(track_xy, 0.0, 0.0)
    anchor_lat = np.zeros(lat_deg.shape, dtype=np.float64)
    anchor_lon = np.zeros(lon_deg.shape, dtype=np.float64)
    ranges_m = great_circle_distance_m(anchor_lat, anchor_lon, lat_deg, lon_deg)

    np.testing.assert_allclose(ranges_m[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(ranges_m[1], 1_000.0, atol=1e-6)
    np.testing.assert_allclose(ranges_m[2], 2_000.0, atol=1e-6)


def test_zero_motion_encounter_is_exact_at_equator() -> None:
    encounter = Encounter(
        rel_east_m=10.0 * NMI_TO_M,
        rel_north_m=0.0,
        a_heading0_deg=0.0,
        a_target_heading_deg=0.0,
        a_speed_kt=0.0,
        a_turn_rate_deg_sec=0.0,
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_speed_kt=0.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=0.0,
        projection_time_s=600.0,
    )

    row = evaluate_projection_encounter(
        encounter,
        anchor_lat_deg=0.0,
        anchor_lon_deg=0.0,
        time_step_s=1.0,
        suite_name="representative",
    )

    np.testing.assert_allclose(row["worst_pointwise_error_m"], 0.0, atol=1e-9)
    np.testing.assert_allclose(row["worst_min_separation_error_m"], 0.0, atol=1e-9)
    assert row["threshold_disagreement_any_corner"] == 0


def test_shift_tracks_for_anchor_mode_reanchors_tracks() -> None:
    encounter = Encounter(
        rel_east_m=10.0 * NMI_TO_M,
        rel_north_m=4.0 * NMI_TO_M,
        a_heading0_deg=0.0,
        a_target_heading_deg=0.0,
        a_speed_kt=0.0,
        a_turn_rate_deg_sec=0.0,
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_speed_kt=0.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=0.0,
        projection_time_s=60.0,
    )
    a_track_xy = np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    b_track_xy = np.asarray([[-encounter.rel_east_m, -encounter.rel_north_m], [3.0, 4.0]], dtype=np.float64)

    a_a, b_a = shift_tracks_for_anchor_mode(a_track_xy, b_track_xy, encounter, "a")
    np.testing.assert_allclose(a_a, a_track_xy)
    np.testing.assert_allclose(b_a, b_track_xy)

    a_b, b_b = shift_tracks_for_anchor_mode(a_track_xy, b_track_xy, encounter, "b")
    np.testing.assert_allclose(b_b[0], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(a_b[0], np.array([encounter.rel_east_m, encounter.rel_north_m]), atol=1e-12)

    a_mid, b_mid = shift_tracks_for_anchor_mode(a_track_xy, b_track_xy, encounter, "midpoint")
    midpoint_xy = 0.5 * np.array([encounter.rel_east_m, encounter.rel_north_m], dtype=np.float64)
    np.testing.assert_allclose(a_mid[0], midpoint_xy, atol=1e-12)
    np.testing.assert_allclose(b_mid[0], -midpoint_xy, atol=1e-12)


def test_east_west_projection_error_grows_away_from_equator() -> None:
    track_xy = np.asarray([[10.0 * NMI_TO_M, 0.0]], dtype=np.float64)
    lat_eq, lon_eq = embed_local_track_on_sphere(track_xy, 0.0, 0.0)
    lat_hi, lon_hi = embed_local_track_on_sphere(track_xy, 60.0, 0.0)

    distance_eq = great_circle_distance_m(np.asarray([0.0]), np.asarray([0.0]), lat_eq, lon_eq)[0]
    distance_hi = great_circle_distance_m(np.asarray([60.0]), np.asarray([0.0]), lat_hi, lon_hi)[0]

    assert abs(distance_eq - 10.0 * NMI_TO_M) < abs(distance_hi - 10.0 * NMI_TO_M)


def test_generate_projection_stress_suite_default_size() -> None:
    encounters = generate_projection_stress_suite()
    assert len(encounters) == 8 * 4 * 4 * 5 * 5
    assert encounters[0].projection_time_s == 1_200.0
    assert encounters[0].speed_diff_kt == 30.0


def test_run_projection_sanity_check_includes_straight_mixed_and_turn_count_breakdown() -> None:
    result = run_projection_sanity_check(ProjectionSanityConfig.quick_defaults())

    assert len(result["straight"]) == 2
    assert len(result["mixed_turn"]) == 2
    assert len(result["stress"]) == 2
    assert set(result["mixed_turn_turn_count"]) == {0, 1, 2}
    for turn_count, rows in result["mixed_turn_turn_count"].items():
        assert len(rows) == 2
        assert all(row["subset_name"] == f"turn_count_{turn_count}" for row in rows)


def test_generate_deterministic_sweeps_quick_sizes() -> None:
    config = ProjectionDeterministicSweepConfig.quick_defaults()

    straight = generate_deterministic_straight_sweep(config)
    mixed = generate_deterministic_mixed_turn_sweep(config)

    assert len(straight) == 2 * 2 * 4 * 4 * 4 * 2 * 1
    assert len(mixed) == 2 * 2 * 2 * 2 * 2 * 3 * 3 * 2 * 1


def test_run_projection_deterministic_sweep_includes_excursion_breakdowns() -> None:
    result = run_projection_deterministic_sweep(ProjectionDeterministicSweepConfig.quick_defaults())

    assert len(result["straight_summary"]) == 1
    assert len(result["mixed_turn_summary"]) == 1
    assert result["straight_by_excursion"]
    assert result["mixed_turn_by_excursion"]
    assert result["mixed_turn_by_excursion_and_turn_count"]
    assert any(row["subset_name"] == "turn_count_0" for row in result["mixed_turn_by_excursion_and_turn_count"])


def test_render_projection_diagnostic_plots_smoke(tmp_path: Path) -> None:
    result = run_projection_deterministic_sweep(ProjectionDeterministicSweepConfig.quick_defaults())

    output_paths = render_projection_diagnostic_plots(
        straight_rows=result["straight_rows_by_latitude"][51.0],
        mixed_turn_rows=result["mixed_turn_rows_by_latitude"][51.0],
        output_dir=tmp_path,
    )

    assert len(output_paths) == 4
    assert all(path.exists() for path in output_paths)


def test_projection_row_loader_ignores_partial_csv_suffixes(tmp_path: Path) -> None:
    csv_path = tmp_path / "straight_rows_lat_51.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["suite_name", "value"])
        writer.writeheader()
        writer.writerow({"suite_name": "straight_sweep", "value": "plain"})

    gz_path = tmp_path / "straight_rows_lat_52.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["suite_name", "value"])
        writer.writeheader()
        writer.writerow({"suite_name": "straight_sweep", "value": "gzip"})

    partial_path = tmp_path / "straight_rows_lat_53.csv.tmp"
    partial_path.write_text("suite_name,value\nstraight_sweep,partial\n", encoding="utf-8")

    rows = load_row_files(tmp_path, "straight_rows_lat_*.csv")

    assert [row["value"] for row in rows] == ["plain", "gzip"]
