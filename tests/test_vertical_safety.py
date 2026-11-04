import math

import pytest

from geometric_safety import (
    VERTICAL_OVERLAP_NEVER_RESOLVES_S,
    catch_up_projection_interval_with_turns_and_vertical,
    catch_up_projection_interval_with_vertical,
    level_at_time_fl,
    time_to_vertical_overlap_resolution,
    vertical_band_gap_fl,
    vertical_bands_are_resolved,
)
from geometric_safety.relevant_aircraft import catch_up_projection_interval
from geometric_safety.util import DEG_TO_RAD, EARTH_RADIUS_IN_METERS, NMI_TO_M, RAD_TO_DEG


def _local_xy_to_latlon(east_m: float, north_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    ref_lat_rad = ref_lat * DEG_TO_RAD
    lat = ref_lat + (north_m / EARTH_RADIUS_IN_METERS) * RAD_TO_DEG
    lon = ref_lon + (east_m / (EARTH_RADIUS_IN_METERS * math.cos(ref_lat_rad))) * RAD_TO_DEG
    return lat, lon


def _brute_force_vertical_resolution_s(
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    vertical_rate_fpm: float,
    required_gap_fl: float,
    rounded: int,
    max_time_s: float,
    step_s: float,
) -> float:
    # This is intentionally a simple grid scan for test coverage. It gives an
    # independent upper-bound oracle for the exact rounded boundary calculation.
    t_s = 0.0
    while t_s <= max_time_s:
        if vertical_bands_are_resolved(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=vertical_rate_fpm,
            required_gap_fl=required_gap_fl,
            t_s=t_s,
            rounded=rounded,
        ):
            return t_s
        t_s += step_s
    return VERTICAL_OVERLAP_NEVER_RESOLVES_S


class TestVerticalOverlapResolution:
    def test_level_at_time_accepts_rate_magnitude_from_public_api(self) -> None:
        assert level_at_time_fl(100.0, 120.0, -1000.0, 60.0) == pytest.approx(110.0)

    def test_level_at_time_clamps_after_selected_level_in_descent(self) -> None:
        assert level_at_time_fl(200.0, 180.0, 1000.0, 180.0) == pytest.approx(180.0)

    def test_already_distinct_bands_return_zero(self) -> None:
        assert vertical_band_gap_fl(100.0, 100.0, 120.0, 120.0) == pytest.approx(20.0)
        assert vertical_bands_are_resolved(100.0, 100.0, 120.0, 120.0)
        assert time_to_vertical_overlap_resolution(100.0, 100.0, 120.0, 120.0) == pytest.approx(0.0)

    def test_rounded_band_gap_tracks_remaining_vertical_band_over_time(self) -> None:
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=0.0) == pytest.approx(0.0)
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=30.0) == pytest.approx(0.0)
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=60.0) == pytest.approx(10.0)

        assert not vertical_bands_are_resolved(100.0, 100.0, 100.0, 120.0, t_s=30.0)
        assert vertical_bands_are_resolved(100.0, 100.0, 100.0, 120.0, t_s=60.0)

    def test_exact_band_gap_tracks_remaining_vertical_band_when_rounding_disabled(self) -> None:
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=0.0, rounded=0) == pytest.approx(0.0)
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=30.0, rounded=0) == pytest.approx(5.0)
        assert vertical_band_gap_fl(100.0, 100.0, 100.0, 120.0, t_s=60.0, rounded=0) == pytest.approx(10.0)

        assert not vertical_bands_are_resolved(100.0, 100.0, 100.0, 120.0, t_s=30.0, rounded=0)
        assert vertical_bands_are_resolved(100.0, 100.0, 100.0, 120.0, t_s=60.0, rounded=0)

    def test_default_rounding_closes_non_multiple_level_gap(self) -> None:
        assert vertical_band_gap_fl(127.0, 127.0, 137.0, 137.0) == pytest.approx(0.0)
        assert vertical_band_gap_fl(137.0, 137.0, 127.0, 127.0) == pytest.approx(0.0)
        assert not vertical_bands_are_resolved(127.0, 127.0, 137.0, 137.0)

        assert vertical_band_gap_fl(127.0, 127.0, 137.0, 137.0, rounded=0) == pytest.approx(10.0)
        assert vertical_bands_are_resolved(127.0, 127.0, 137.0, 137.0, rounded=0)

    def test_rounded_band_gap_uses_tolerance_at_grid_boundaries(self) -> None:
        within_tolerance_fl = 5.0e-10
        outside_tolerance_fl = 2.0e-9

        assert vertical_band_gap_fl(130.0, 130.0, 140.0, 140.0) == pytest.approx(10.0)
        assert vertical_band_gap_fl(
            120.0, 120.0, 130.0 - within_tolerance_fl, 130.0 - within_tolerance_fl
        ) == pytest.approx(10.0)
        assert vertical_band_gap_fl(
            130.0 + within_tolerance_fl, 130.0 + within_tolerance_fl, 140.0, 140.0
        ) == pytest.approx(10.0)
        assert vertical_band_gap_fl(
            120.0, 120.0, 130.0 - outside_tolerance_fl, 130.0 - outside_tolerance_fl
        ) == pytest.approx(0.0)
        assert vertical_band_gap_fl(
            130.0 + outside_tolerance_fl, 130.0 + outside_tolerance_fl, 140.0, 140.0
        ) == pytest.approx(0.0)

    def test_rounded_resolution_waits_for_next_rounded_gap(self) -> None:
        raw_resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=127.0,
            a_selected_fl=127.0,
            b_current_fl=127.0,
            b_selected_fl=147.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
            rounded=0,
        )
        rounded_resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=127.0,
            a_selected_fl=127.0,
            b_current_fl=127.0,
            b_selected_fl=147.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )

        assert raw_resolution_s == pytest.approx(60.0)
        assert rounded_resolution_s == pytest.approx(78.0)

    def test_rounded_boundary_event_timing_counts_coincident_crossings(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=135.0,
            a_selected_fl=100.0,
            b_current_fl=125.0,
            b_selected_fl=200.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
            rounded=10,
        )

        assert resolution_s == pytest.approx(90.0)
        assert vertical_band_gap_fl(
            a_current_fl=135.0,
            a_selected_fl=100.0,
            b_current_fl=125.0,
            b_selected_fl=200.0,
            vertical_rate_fpm=1000.0,
            t_s=30.0,
            rounded=10,
        ) == pytest.approx(0.0)
        assert not vertical_bands_are_resolved(
            a_current_fl=135.0,
            a_selected_fl=100.0,
            b_current_fl=125.0,
            b_selected_fl=200.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
            t_s=30.0,
            rounded=10,
        )

    @pytest.mark.parametrize(
        (
            "a_current_fl",
            "a_selected_fl",
            "b_current_fl",
            "b_selected_fl",
            "required_gap_fl",
            "rounded",
            "expected_resolution_s",
        ),
        [
            (127.0, 127.0, 127.0, 147.0, 10.0, 10, 78.0),
            (147.0, 100.0, 137.0, 137.0, 10.0, 10, 162.0),
            (142.0, 100.0, 126.0, 200.0, 10.0, 10, 84.0),
            (135.0, 130.0, 125.0, 150.0, 10.0, 10, 90.0),
            (127.0, 127.0, 127.0, 157.0, 15.0, 10, 138.0),
            (126.0, 126.0, 126.0, 141.0, 10.0, 5, 84.0),
        ],
    )
    def test_rounded_boundary_event_timing(
        self,
        a_current_fl: float,
        a_selected_fl: float,
        b_current_fl: float,
        b_selected_fl: float,
        required_gap_fl: float,
        rounded: int,
        expected_resolution_s: float,
    ) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=1000.0,
            required_gap_fl=required_gap_fl,
            rounded=rounded,
        )

        assert resolution_s == pytest.approx(expected_resolution_s)
        assert not vertical_bands_are_resolved(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=1000.0,
            required_gap_fl=required_gap_fl,
            t_s=expected_resolution_s - 1.0e-6,
            rounded=rounded,
        )
        assert vertical_bands_are_resolved(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=1000.0,
            required_gap_fl=required_gap_fl,
            t_s=expected_resolution_s,
            rounded=rounded,
        )

    @pytest.mark.parametrize(
        (
            "a_current_fl",
            "a_selected_fl",
            "b_current_fl",
            "b_selected_fl",
            "vertical_rate_fpm",
            "required_gap_fl",
            "rounded",
            "max_time_s",
        ),
        [
            (123.0, 123.0, 119.0, 151.0, 750.0, 10.0, 5, 360.0),
            (178.0, 132.0, 166.0, 166.0, 1500.0, 10.0, 10, 360.0),
            (301.0, 290.0, 287.0, 325.0, 800.0, 20.0, 10, 480.0),
            (240.0, 210.0, 235.0, 260.0, 600.0, 15.0, 5, 600.0),
        ],
    )
    def test_rounded_resolution_matches_small_brute_force_sweep(
        self,
        a_current_fl: float,
        a_selected_fl: float,
        b_current_fl: float,
        b_selected_fl: float,
        vertical_rate_fpm: float,
        required_gap_fl: float,
        rounded: int,
        max_time_s: float,
    ) -> None:
        step_s = 0.5
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=vertical_rate_fpm,
            required_gap_fl=required_gap_fl,
            rounded=rounded,
        )
        brute_force_s = _brute_force_vertical_resolution_s(
            a_current_fl=a_current_fl,
            a_selected_fl=a_selected_fl,
            b_current_fl=b_current_fl,
            b_selected_fl=b_selected_fl,
            vertical_rate_fpm=vertical_rate_fpm,
            required_gap_fl=required_gap_fl,
            rounded=rounded,
            max_time_s=max_time_s,
            step_s=step_s,
        )

        assert brute_force_s != VERTICAL_OVERLAP_NEVER_RESOLVES_S
        assert resolution_s <= brute_force_s + 1.0e-9
        assert resolution_s > brute_force_s - step_s - 1.0e-9

    def test_both_climbing_resolves_when_lower_band_edge_clears(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=150.0,
            b_current_fl=140.0,
            b_selected_fl=200.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(120.0)

    def test_both_descending_resolves_when_upper_band_edge_clears(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=200.0,
            a_selected_fl=140.0,
            b_current_fl=150.0,
            b_selected_fl=100.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(120.0)

    def test_opposite_climb_and_descent_resolves_in_closed_form(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=200.0,
            b_current_fl=250.0,
            b_selected_fl=150.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(480.0)

    def test_one_aircraft_levels_before_resolution(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=150.0,
            b_current_fl=200.0,
            b_selected_fl=140.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(360.0)

    def test_zero_required_gap_resolves_at_first_touch(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=150.0,
            b_current_fl=200.0,
            b_selected_fl=140.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=0.0,
        )
        assert resolution_s == pytest.approx(300.0)

    def test_rate_parameter_scales_time(self) -> None:
        slow_resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            vertical_rate_fpm=500.0,
            required_gap_fl=10.0,
        )
        fast_resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            vertical_rate_fpm=2000.0,
            required_gap_fl=10.0,
        )

        assert slow_resolution_s == pytest.approx(120.0)
        assert fast_resolution_s == pytest.approx(30.0)

    def test_returns_never_sentinel_when_selected_bands_never_clear(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=150.0,
            b_current_fl=200.0,
            b_selected_fl=145.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(VERTICAL_OVERLAP_NEVER_RESOLVES_S)

    def test_zero_rate_keeps_unresolved_overlap_forever(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=105.0,
            b_selected_fl=105.0,
            vertical_rate_fpm=0.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(VERTICAL_OVERLAP_NEVER_RESOLVES_S)

    def test_zero_rate_keeps_already_resolved_bands_safe(self) -> None:
        resolution_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=120.0,
            b_selected_fl=120.0,
            vertical_rate_fpm=0.0,
            required_gap_fl=10.0,
        )
        assert resolution_s == pytest.approx(0.0)

    def test_negative_rate_is_treated_as_magnitude(self) -> None:
        positive_rate_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            vertical_rate_fpm=1000.0,
            required_gap_fl=10.0,
        )
        negative_rate_s = time_to_vertical_overlap_resolution(
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            vertical_rate_fpm=-1000.0,
            required_gap_fl=10.0,
        )

        assert negative_rate_s == pytest.approx(positive_rate_s)


class TestCombinedSafety:
    def test_vertical_resolution_before_lateral_loss_is_safe(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        lateral_is_sep, _, _ = catch_up_projection_interval(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
        )
        assert not lateral_is_sep

        combined_is_safe, min_distance_m, closest_time_s, vertical_time_s = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
        )

        assert combined_is_safe
        assert vertical_time_s == pytest.approx(60.0)
        assert min_distance_m > separation_m
        assert closest_time_s == pytest.approx(60.0)

    def test_rounded_vertical_resolution_controls_combined_horizon(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        rounded_combined = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=127.0,
            a_selected_fl=127.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=127.0,
            b_selected_fl=147.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
        )
        exact_combined = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=127.0,
            a_selected_fl=127.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=127.0,
            b_selected_fl=147.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
            vertical_rounding_fl=0,
        )

        assert not rounded_combined[0]
        assert rounded_combined[3] == pytest.approx(78.0)
        assert exact_combined[0]
        assert exact_combined[3] == pytest.approx(60.0)

    def test_lateral_loss_before_vertical_resolution_is_unsafe(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        combined_is_safe, min_distance_m, _, vertical_time_s = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=500.0,
        )

        assert not combined_is_safe
        assert vertical_time_s == pytest.approx(120.0)
        assert min_distance_m < separation_m

    def test_vertical_resolution_after_projection_uses_full_lateral_horizon(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        lateral = catch_up_projection_interval(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=60.0,
        )

        combined = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=60.0,
            vertical_rate_fpm=500.0,
        )

        assert combined[0] == lateral[0]
        assert combined[1] == pytest.approx(lateral[1], abs=1e-6)
        assert combined[2] == pytest.approx(lateral[2], abs=1e-6)
        assert combined[3] == pytest.approx(120.0)

    def test_already_resolved_vertical_bands_shortcut_to_safe(self) -> None:
        is_safe, min_distance_m, closest_time_s, vertical_time_s = catch_up_projection_interval_with_vertical(
            a_lat=51.0,
            a_lon=-1.0,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=51.0,
            b_lon=-1.0,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=120.0,
            b_selected_fl=120.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
        )

        assert is_safe
        assert min_distance_m == pytest.approx(0.0)
        assert closest_time_s == pytest.approx(0.0)
        assert vertical_time_s == pytest.approx(0.0)

    def test_turn_aware_already_resolved_bands_return_t0_diagnostics(self) -> None:
        is_safe, min_distance_m, closest_time_s, vertical_time_s = catch_up_projection_interval_with_turns_and_vertical(
            a_lat=51.0,
            a_lon=-1.0,
            a_heading0_deg=0.0,
            a_target_heading_deg=270.0,
            a_speed_kt=360.0,
            a_turn_rate_deg_sec=1.5,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=51.0,
            b_lon=-1.0,
            b_heading0_deg=180.0,
            b_target_heading_deg=90.0,
            b_speed_kt=360.0,
            b_turn_rate_deg_sec=1.5,
            b_current_fl=120.0,
            b_selected_fl=120.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
        )

        assert is_safe
        assert min_distance_m == pytest.approx(0.0)
        assert closest_time_s == pytest.approx(0.0)
        assert vertical_time_s == pytest.approx(0.0)

    def test_never_resolving_vertical_bands_use_full_lateral_horizon(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        combined_is_safe, min_distance_m, _, vertical_time_s = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=150.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=200.0,
            b_selected_fl=145.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
        )

        assert not combined_is_safe
        assert vertical_time_s == pytest.approx(VERTICAL_OVERLAP_NEVER_RESOLVES_S)
        assert min_distance_m < separation_m

    def test_turn_aware_combined_matches_fixed_heading_without_turns(self) -> None:
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(0.0, 20.0 * NMI_TO_M, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        fixed = catch_up_projection_interval_with_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=0.0,
            a_speed_kt=360.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=180.0,
            b_speed_kt=360.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
        )
        turn_aware = catch_up_projection_interval_with_turns_and_vertical(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=0.0,
            a_speed_kt=360.0,
            a_turn_rate_deg_sec=0.0,
            a_current_fl=100.0,
            a_selected_fl=100.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=360.0,
            b_turn_rate_deg_sec=0.0,
            b_current_fl=100.0,
            b_selected_fl=120.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
            vertical_rate_fpm=1000.0,
        )

        assert turn_aware[0] == fixed[0]
        assert turn_aware[1] == pytest.approx(fixed[1], abs=1e-6)
        assert turn_aware[2] == pytest.approx(fixed[2], abs=1e-6)
        assert turn_aware[3] == pytest.approx(fixed[3], abs=1e-6)
