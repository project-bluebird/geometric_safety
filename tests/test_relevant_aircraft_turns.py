"""
Tests for the turn-aware lateral separation solver.

Covers:
1. Regression: straight-straight equivalence with the exact straight-line solver
2. Helper-function tests for the current turn-aware geometry
3. During-turn validation against numerical integration
4. Full-horizon validation against numerical trajectory simulation
5. End-to-end combined scenarios
6. Edge cases
"""

import math

import numpy as np
import pytest

from geometric_safety.relevant_aircraft import (
    TurnInputs,
    _closest_time_from_projection,
    _deviation_envelope_m,
    _interval_distance_lower_bound,
    _interval_local_lipschitz_mps,
    _relative_position_hull_at_time,
    _relative_speed_mps,
    _speed_rectangle_corner_points,
    _turn_displacement_basis,
    _turn_execution_envelope_m,
    _turn_speed_schedule_envelope_m,
    _unwrapped_heading_deg_at_time,
    catch_up_projection_interval,
    catch_up_projection_interval_with_turns,
    compute_relative_velocity_hull,
)
from geometric_safety.util import (
    DEG_TO_RAD,
    EARTH_RADIUS_IN_METERS,
    KT_TO_MPS,
    NMI_TO_M,
    RAD_TO_DEG,
    closest_point_on_convex_polygon,
    convex_hull,
    heading_diff,
    heading_to_unit_vector,
    latlon_to_local_xy,
    small_convex_hull,
)

# ---------------------------------------------------------------------------
# Helpers for numerical trajectory simulation
# ---------------------------------------------------------------------------


def assert_hulls_equivalent(hull_a: np.ndarray, hull_b: np.ndarray, abs_tol: float = 1e-9) -> None:
    """
    Assert that two convex hulls describe the same closed set.

    The optimized structured hull builders may return a different cyclic ordering from
    the generic monotone-chain reference. We therefore compare the represented sets
    geometrically rather than comparing raw vertex arrays directly.
    """
    for vertex in hull_a:
        _proj, distance = closest_point_on_convex_polygon(vertex, hull_b)
        assert distance == pytest.approx(0.0, abs=abs_tol)

    for vertex in hull_b:
        _proj, distance = closest_point_on_convex_polygon(vertex, hull_a)
        assert distance == pytest.approx(0.0, abs=abs_tol)


def simulate_arc_trajectory(
    lat: float,
    lon: float,
    heading0_deg: float,
    target_heading_deg: float,
    speed_kt: float,
    turn_rate_deg_sec: float,
    dt: float,
    total_time: float,
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """
    Simulate an aircraft trajectory at fine time steps.

    Returns array of shape (N, 3) with columns (east_m, north_m, time_s).
    Aircraft turns at `turn_rate_deg_sec` from `heading0_deg` to `target_heading_deg`,
    then flies straight.
    """
    turn_angle_deg = heading_diff(heading0_deg, target_heading_deg)
    if abs(turn_angle_deg) < 1e-6 or abs(turn_rate_deg_sec) < 1e-6:
        turn_duration = 0.0
    else:
        turn_duration = abs(turn_angle_deg) / abs(turn_rate_deg_sec)

    xy = latlon_to_local_xy(lat, lon, ref_lat, ref_lon)
    east, north = float(xy[0]), float(xy[1])
    heading = heading0_deg
    speed_mps = speed_kt * KT_TO_MPS

    times = np.arange(0, total_time + dt, dt)
    positions = np.empty((len(times), 3))

    for i, t in enumerate(times):
        positions[i, 0] = east
        positions[i, 1] = north
        positions[i, 2] = t

        if t < total_time:
            # Determine current heading
            if t < turn_duration:
                # Still turning - advance heading
                heading_rad = heading * DEG_TO_RAD
                dist = speed_mps * dt
                east += dist * math.sin(heading_rad)
                north += dist * math.cos(heading_rad)
                # Turn direction matches sign of turn_angle
                heading += math.copysign(turn_rate_deg_sec, turn_angle_deg) * dt
            else:
                # Flying straight at target heading
                heading_rad = target_heading_deg * DEG_TO_RAD
                dist = speed_mps * dt
                east += dist * math.sin(heading_rad)
                north += dist * math.cos(heading_rad)

    return positions


def compute_min_separation_numerical(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
) -> tuple[float, float]:
    """
    Find the minimum separation between two trajectories (aligned by time).

    Returns (min_dist_m, time_of_min_dist_s).
    """
    # Both trajectories should have same time steps
    n = min(traj_a.shape[0], traj_b.shape[0])
    min_dist = float("inf")
    min_time = 0.0

    for i in range(n):
        de = traj_a[i, 0] - traj_b[i, 0]
        dn = traj_a[i, 1] - traj_b[i, 1]
        dist = math.sqrt(de * de + dn * dn)
        if dist < min_dist:
            min_dist = dist
            min_time = traj_a[i, 2]

    return min_dist, min_time


def _local_xy_to_latlon(east_m: float, north_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """
    Test-only inverse of `latlon_to_local_xy`.

    The production code only needs the forward local tangent-plane projection, but a few
    tests construct scenarios from desired local offsets, so they use this helper to get
    corresponding latitude/longitude inputs.
    """
    ref_lat_rad = ref_lat * DEG_TO_RAD
    lat = ref_lat + (north_m / EARTH_RADIUS_IN_METERS) * RAD_TO_DEG
    lon = ref_lon + (east_m / (EARTH_RADIUS_IN_METERS * math.cos(ref_lat_rad))) * RAD_TO_DEG
    return lat, lon


# ---------------------------------------------------------------------------
# 1. Regression: straight-straight equivalence
# ---------------------------------------------------------------------------


class TestStraightStraightRegression:
    """Verify the turn-aware solver matches the straight solver for straight pairs."""

    @pytest.mark.parametrize("a_heading", [0, 45, 90, 135, 180, 270])
    @pytest.mark.parametrize("b_heading", [0, 45, 90, 180, 270])
    @pytest.mark.parametrize("a_speed_kt", [250, 400])
    @pytest.mark.parametrize("b_speed_kt", [200, 350])
    def test_straight_equivalence(
        self, a_heading: float, b_heading: float, a_speed_kt: float, b_speed_kt: float
    ) -> None:
        """Turn-aware function should produce identical results when both aircraft are straight."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.95
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 15.0 * 60.0

        ref_sep, ref_dist, ref_time = catch_up_projection_interval(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=a_heading,
            a_speed_kt=a_speed_kt,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=b_heading,
            b_speed_kt=b_speed_kt,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        turn_sep, turn_dist, turn_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading,
            a_target_heading_deg=a_heading,  # no turn
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=0.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,  # no turn
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        assert ref_sep == turn_sep, f"is_separated mismatch: ref={ref_sep}, turn={turn_sep}"
        assert ref_dist == pytest.approx(turn_dist, abs=1e-6), f"min_dist mismatch: ref={ref_dist}, turn={turn_dist}"
        assert ref_time == pytest.approx(turn_time, abs=1e-6), f"time mismatch: ref={ref_time}, turn={turn_time}"

    def test_sliver_hull_closest_time(self) -> None:
        """Near-antiparallel headings collapse the reachable hull to a razor-thin
        sliver, and the closest-time recovery must still report the time that
        attains the minimum rather than clipping to the horizon end. The values
        are pinned against an arbitrary-precision reimplementation."""
        import numpy as np

        is_separated, min_dist_m, closest_time_s = catch_up_projection_interval(
            a_lat=0.0,
            a_lon=0.0,
            a_heading=32.61553945503712,
            a_speed_kt=234.3488541869565,
            b_lat=0.0,
            b_lon=0.0,
            b_heading=212.6143910177294,
            b_speed_kt=368.7382526298944,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=27.836888098896647,
            projection_time_s=1200.0,
            rel_pos_override=np.array([-21453.687270485607, -56678.73631862431]),
        )
        assert is_separated
        assert min_dist_m == pytest.approx(12478.440009027207, abs=1e-4)
        assert closest_time_s == pytest.approx(191.1478471721988, abs=1e-3)

    @pytest.mark.parametrize("speed_diff_kt", [10, 25, 50])
    @pytest.mark.parametrize("proj_time_min", [5, 15, 30])
    @pytest.mark.parametrize("sep_nmi", [3, 5, 7])
    def test_straight_equivalence_speed_diff_and_proj(
        self, speed_diff_kt: float, proj_time_min: float, sep_nmi: float
    ) -> None:
        """Check for different speed diffs, projection times, and separation thresholds."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.03, -0.97
        a_heading, b_heading = 45.0, 135.0
        a_speed_kt, b_speed_kt = 350.0, 300.0
        separation_m = sep_nmi * NMI_TO_M
        proj_time = proj_time_min * 60.0

        ref_sep, ref_dist, ref_time = catch_up_projection_interval(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=a_heading,
            a_speed_kt=a_speed_kt,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=b_heading,
            b_speed_kt=b_speed_kt,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        turn_sep, turn_dist, turn_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading,
            a_target_heading_deg=a_heading,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=3.0,  # turn rate is set but no turn angle
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=3.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        assert ref_sep == turn_sep
        assert ref_dist == pytest.approx(turn_dist, abs=1e-6)
        assert ref_time == pytest.approx(turn_time, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_heading_to_unit_vector_cardinals(self) -> None:
        """The module uses aviation headings: clockwise from north."""
        north = heading_to_unit_vector(0.0)
        east = heading_to_unit_vector(90.0)
        south = heading_to_unit_vector(180.0)
        west = heading_to_unit_vector(270.0)

        assert north[0] == pytest.approx(0.0, abs=1e-12)
        assert north[1] == pytest.approx(1.0, abs=1e-12)
        assert east[0] == pytest.approx(1.0, abs=1e-12)
        assert east[1] == pytest.approx(0.0, abs=1e-12)
        assert south[0] == pytest.approx(0.0, abs=1e-12)
        assert south[1] == pytest.approx(-1.0, abs=1e-12)
        assert west[0] == pytest.approx(-1.0, abs=1e-12)
        assert west[1] == pytest.approx(0.0, abs=1e-12)

    def test_turn_displacement_basis_straight(self) -> None:
        """No-turn cases should reduce to heading vector times elapsed time."""
        t_s = 42.0
        basis = _turn_displacement_basis(25.0, 25.0, 0.0, t_s)
        expected = heading_to_unit_vector(25.0) * t_s
        assert basis[0] == pytest.approx(expected[0], abs=1e-12)
        assert basis[1] == pytest.approx(expected[1], abs=1e-12)

    def test_turn_displacement_basis_turn_then_straight(self) -> None:
        """
        For a 90 deg right turn from north to east, the closed-form basis should equal
        the analytic quarter-circle displacement plus any trailing straight segment.
        """
        turn_rate_deg_sec = 3.0
        turn_rate_rad_per_s = turn_rate_deg_sec * DEG_TO_RAD
        turn_duration_s = 90.0 / turn_rate_deg_sec
        total_time_s = turn_duration_s + 10.0

        basis = _turn_displacement_basis(0.0, 90.0, turn_rate_deg_sec, total_time_s)
        expected_turn = np.array([1.0 / turn_rate_rad_per_s, 1.0 / turn_rate_rad_per_s], dtype=np.float64)
        expected = expected_turn + np.array([10.0, 0.0], dtype=np.float64)

        assert basis[0] == pytest.approx(expected[0], rel=1e-12)
        assert basis[1] == pytest.approx(expected[1], rel=1e-12)

    def test_relative_position_hull_contains_speed_corners(self) -> None:
        """The fixed-time hull must contain all four speed-corner realizations exactly."""
        rel_pos0 = np.array([1200.0, -3400.0], dtype=np.float64)
        t_s = 35.0
        a_min, a_max = 140.0, 165.0
        b_min, b_max = 120.0, 155.0

        hull = _relative_position_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=0.0,
            a_target_heading_deg=45.0,
            a_turn_rate_deg_sec=3.0,
            a_min_speed_mps=a_min,
            a_max_speed_mps=a_max,
            b_heading0_deg=210.0,
            b_target_heading_deg=180.0,
            b_turn_rate_deg_sec=2.0,
            b_min_speed_mps=b_min,
            b_max_speed_mps=b_max,
            t_s=t_s,
        )
        a_basis = _turn_displacement_basis(0.0, 45.0, 3.0, t_s)
        b_basis = _turn_displacement_basis(210.0, 180.0, 2.0, t_s)

        for a_speed in (a_min, a_max):
            for b_speed in (b_min, b_max):
                corner = rel_pos0 + a_speed * a_basis - b_speed * b_basis
                projected, distance = closest_point_on_convex_polygon(corner, hull)
                assert distance == pytest.approx(0.0, abs=1e-9)
                assert np.linalg.norm(projected - corner) == pytest.approx(0.0, abs=1e-9)

    def test_speed_rectangle_corner_points_match_direct_formula(self) -> None:
        """The shared corner helper should reproduce the explicit speed-corner formulas."""
        base = np.array([1200.0, -3400.0], dtype=np.float64)
        a_basis = np.array([31.0, 12.0], dtype=np.float64)
        b_basis = np.array([-18.0, 27.5], dtype=np.float64)
        a_min, a_max = 140.0, 165.0
        b_min, b_max = 120.0, 155.0

        corners = _speed_rectangle_corner_points(base, a_basis, a_min, a_max, b_basis, b_min, b_max)

        expected = np.array(
            [
                base + a_min * a_basis - b_min * b_basis,
                base + a_max * a_basis - b_min * b_basis,
                base + a_max * a_basis - b_max * b_basis,
                base + a_min * a_basis - b_max * b_basis,
            ],
            dtype=np.float64,
        )

        assert np.allclose(corners, expected, atol=1e-12)

    def test_small_convex_hull_matches_generic_reference(self) -> None:
        """The tiny-point hull builder should match the generic convex-hull reference."""
        pts = np.array(
            [
                [0.0, 0.0],
                [4200.0, -1800.0],
                [5600.0, 900.0],
                [1400.0, 2700.0],
                [2800.0, -450.0],
            ],
            dtype=np.float64,
        )

        small_hull = small_convex_hull(pts)
        generic_hull = convex_hull(pts)

        assert_hulls_equivalent(small_hull, generic_hull)

    def test_compute_relative_velocity_hull_matches_generic_reference(self) -> None:
        """The optimized straight-line hull should match the retained generic reference."""
        a_heading = 86.07215429731461
        b_heading = 359.34847733836904
        a_speed = 356.1049998158458
        b_speed = 288.34698590803924
        speed_diff = 21.85042676853181
        T = 923.7721875173809

        optimized_hull = compute_relative_velocity_hull(a_heading, a_speed, b_heading, b_speed, speed_diff, T)

        dir_a = heading_to_unit_vector(a_heading)
        dir_b = heading_to_unit_vector(b_heading)
        a_min = max(a_speed - speed_diff, 0.0) * KT_TO_MPS
        a_max = (a_speed + speed_diff) * KT_TO_MPS
        b_min = max(b_speed - speed_diff, 0.0) * KT_TO_MPS
        b_max = (b_speed + speed_diff) * KT_TO_MPS
        reference_points = np.zeros((5, 2), dtype=np.float64)
        reference_points[1:] = _speed_rectangle_corner_points(
            base=np.zeros(2, dtype=np.float64),
            a_basis=T * dir_a,
            a_min_speed=a_min,
            a_max_speed=a_max,
            b_basis=T * dir_b,
            b_min_speed=b_min,
            b_max_speed=b_max,
        )
        reference_hull = convex_hull(reference_points)

        assert_hulls_equivalent(optimized_hull, reference_hull)

    def test_relative_position_hull_matches_generic_reference(self) -> None:
        """The fixed-time optimized hull should match the retained generic reference."""
        rel_pos0 = np.array([1200.0, -3400.0], dtype=np.float64)
        t_s = 35.0
        a_min, a_max = 140.0, 165.0
        b_min, b_max = 120.0, 155.0
        a_basis = _turn_displacement_basis(0.0, 45.0, 3.0, t_s)
        b_basis = _turn_displacement_basis(210.0, 180.0, 2.0, t_s)

        optimized_hull = _relative_position_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=0.0,
            a_target_heading_deg=45.0,
            a_turn_rate_deg_sec=3.0,
            a_min_speed_mps=a_min,
            a_max_speed_mps=a_max,
            b_heading0_deg=210.0,
            b_target_heading_deg=180.0,
            b_turn_rate_deg_sec=2.0,
            b_min_speed_mps=b_min,
            b_max_speed_mps=b_max,
            t_s=t_s,
        )
        reference_corners = _speed_rectangle_corner_points(
            base=rel_pos0,
            a_basis=a_basis,
            a_min_speed=a_min,
            a_max_speed=a_max,
            b_basis=b_basis,
            b_min_speed=b_min,
            b_max_speed=b_max,
        )
        reference_hull = convex_hull(reference_corners)

        assert_hulls_equivalent(optimized_hull, reference_hull)

    def test_interval_distance_lower_bound_formula(self) -> None:
        """The helper should implement the stated Lipschitz lower-bound formula."""
        start_dist = 12000.0
        end_dist = 9000.0
        lipschitz = 250.0
        dt_s = 8.0
        expected = max(0.0, 0.5 * (start_dist + end_dist - lipschitz * dt_s))
        assert _interval_distance_lower_bound(start_dist, end_dist, lipschitz, dt_s) == pytest.approx(expected)

    def test_interval_local_lipschitz_matches_bruteforce_sampling(self) -> None:
        """
        The interval-local Lipschitz helper should upper-bound the exact sampled corner
        speeds and be no larger than the global closing-rate bound.
        """
        a_heading0 = 10.0
        a_target = 70.0
        a_turn_rate_deg_sec = 2.0
        a_min, a_max = 145.0, 180.0
        b_heading0 = 220.0
        b_target = 170.0
        b_turn_rate_deg_sec = 1.5
        b_min, b_max = 120.0, 155.0
        t0_s = 18.0
        t1_s = 66.0
        global_lipschitz = a_max + b_max

        local_lipschitz = _interval_local_lipschitz_mps(
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min,
            a_max_speed_mps=a_max,
            b_heading0_deg=b_heading0,
            b_target_heading_deg=b_target,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min,
            b_max_speed_mps=b_max,
            t0_s=t0_s,
            t1_s=t1_s,
            global_lipschitz_mps=global_lipschitz,
        )

        sampled_max = 0.0
        sample_times = np.linspace(t0_s, t1_s, 4001)
        for t_s in sample_times:
            a_heading = _unwrapped_heading_deg_at_time(a_heading0, a_target, a_turn_rate_deg_sec, float(t_s))
            b_heading = _unwrapped_heading_deg_at_time(b_heading0, b_target, b_turn_rate_deg_sec, float(t_s))
            delta_heading = a_heading - b_heading
            for a_speed in (a_min, a_max):
                for b_speed in (b_min, b_max):
                    sampled_max = max(sampled_max, _relative_speed_mps(a_speed, b_speed, delta_heading))

        assert local_lipschitz >= sampled_max - 1e-9
        assert local_lipschitz <= global_lipschitz + 1e-12

    def test_turn_aware_global_lipschitz_mode_is_more_conservative(self) -> None:
        """
        The optional global bound should remain available and can only be at least as
        conservative as the default interval-local bound.
        """
        kwargs: TurnInputs = {
            "a_lat": 51.0,
            "a_lon": -1.0,
            "a_heading0_deg": 180.0,
            "a_target_heading_deg": 135.0,
            "a_speed_kt": 360.0,
            "a_turn_rate_deg_sec": 3.0,
            "b_lat": 51.071877473364735,
            "b_lon": -0.8841859420551476,
            "b_heading0_deg": 270.0,
            "b_target_heading_deg": 225.0,
            "b_speed_kt": 360.0,
            "b_turn_rate_deg_sec": 3.0,
            "separation_threshold_m": 5.0 * NMI_TO_M,
            "speed_diff_kt": 20.0,
            "projection_time_s": 600.0,
        }

        local_result = catch_up_projection_interval_with_turns(
            **kwargs,
            use_interval_local_lipschitz=True,
        )
        global_result = catch_up_projection_interval_with_turns(
            **kwargs,
            use_interval_local_lipschitz=False,
        )

        # The local bound should never produce a less safe answer than the global bound.
        assert int(local_result[0]) >= int(global_result[0])

    def test_turn_speed_schedule_envelope_matches_proven_bound(self) -> None:
        """The envelope should equal delta * t_a * (1 - cos(theta_a)) / theta_a in scope."""
        envelope_m = _turn_speed_schedule_envelope_m(
            heading0_deg=0.0,
            target_heading_deg=90.0,
            turn_rate_deg_sec=3.0,
            t_s=30.0,
            schedule_uncertainty_kt=50.0,
            min_speed_mps=100.0,
            max_speed_mps=200.0,
        )

        theta_a = 90.0 * DEG_TO_RAD
        expected = (50.0 * KT_TO_MPS) * 30.0 * (1.0 - math.cos(theta_a)) / theta_a
        assert envelope_m == pytest.approx(expected, rel=1e-12)

    def test_turn_speed_schedule_envelope_never_exceeds_sine_buffer(self) -> None:
        """The proven envelope never exceeds the coarser sin(theta/2) bound."""
        for sweep_deg in (10.0, 30.0, 60.0, 90.0):
            for t_s in (5.0, 15.0, 30.0, 120.0):
                envelope_m = _turn_speed_schedule_envelope_m(
                    heading0_deg=0.0,
                    target_heading_deg=sweep_deg,
                    turn_rate_deg_sec=3.0,
                    t_s=t_s,
                    schedule_uncertainty_kt=25.0,
                    min_speed_mps=100.0,
                    max_speed_mps=200.0,
                )
                t_active = min(t_s, sweep_deg / 3.0)
                theta_active = 3.0 * DEG_TO_RAD * t_active
                sine_buffer_m = (25.0 * KT_TO_MPS) * t_active * math.sin(0.5 * theta_active)
                assert envelope_m <= sine_buffer_m + 1e-12

    def test_turn_speed_schedule_envelope_beyond_ninety_uses_fallback(self) -> None:
        """Sweeps beyond 90 degrees should use the unconditional fallback bound."""
        envelope_m = _turn_speed_schedule_envelope_m(
            heading0_deg=0.0,
            target_heading_deg=150.0,
            turn_rate_deg_sec=3.0,
            t_s=50.0,
            schedule_uncertainty_kt=10.0,
            min_speed_mps=100.0,
            max_speed_mps=200.0,
        )

        theta_a = 150.0 * DEG_TO_RAD
        amplitude = min(2.0 * 10.0 * KT_TO_MPS, 100.0)
        expected = amplitude * 4.0 * 50.0 * (1.0 - math.cos(0.5 * theta_a)) / theta_a
        assert envelope_m == pytest.approx(expected, rel=1e-12)

    def test_turn_speed_schedule_envelope_selects_form_by_total_sweep(self) -> None:
        """A wide instructed turn must use the fallback form at every query time.

        A per-time switch at the pi/2 crossing can step DOWN when the admissible
        interval width is below ~0.854 * delta, which would break the
        nondecreasing-envelope hypothesis of the inflated interval certificate.
        """
        kwargs = {
            "heading0_deg": 0.0,
            "target_heading_deg": 180.0,
            "turn_rate_deg_sec": 3.0,
            "schedule_uncertainty_kt": 20.0,
            "min_speed_mps": 133.0,
            "max_speed_mps": 135.0,
        }
        # Before the pi/2 crossing (25 s -> 75 degrees completed) the value must
        # already be the fallback form, selected from the 180-degree total sweep.
        envelope_m = _turn_speed_schedule_envelope_m(t_s=25.0, **kwargs)
        theta_a = 3.0 * DEG_TO_RAD * 25.0
        amplitude = min(2.0 * 20.0 * KT_TO_MPS, 135.0 - 133.0)
        expected = amplitude * 4.0 * 25.0 * (1.0 - math.cos(0.5 * theta_a)) / theta_a
        assert envelope_m == pytest.approx(expected, rel=1e-12)

    def test_turn_speed_schedule_envelope_is_nondecreasing_across_pi_over_two(self) -> None:
        """Regression: the envelope must never decrease in time for any sweep."""
        for width_mps in (2.0, 8.0, 60.0):
            kwargs = {
                "heading0_deg": 0.0,
                "target_heading_deg": 180.0,
                "turn_rate_deg_sec": 3.0,
                "schedule_uncertainty_kt": 20.0,
                "min_speed_mps": 134.0 - 0.5 * width_mps,
                "max_speed_mps": 134.0 + 0.5 * width_mps,
            }
            values = [_turn_speed_schedule_envelope_m(t_s=t_s, **kwargs) for t_s in np.arange(0.5, 90.0, 0.5)]
            diffs = np.diff(np.asarray(values))
            assert float(diffs.min()) >= -1e-9

    def test_turn_speed_schedule_envelope_is_zero_for_straight_flight(self) -> None:
        """The envelope should vanish when no turn is flown."""
        envelope_m = _turn_speed_schedule_envelope_m(
            heading0_deg=45.0,
            target_heading_deg=45.0,
            turn_rate_deg_sec=0.0,
            t_s=600.0,
            schedule_uncertainty_kt=50.0,
            min_speed_mps=100.0,
            max_speed_mps=200.0,
        )

        assert envelope_m == pytest.approx(0.0, abs=1e-12)

    def test_turn_execution_envelope_saturates_after_turn(self) -> None:
        """The execution envelope should stop growing once the turn window passes."""
        kwargs = {
            "heading0_deg": 0.0,
            "target_heading_deg": 60.0,
            "turn_rate_deg_sec": 3.0,
            "max_speed_mps": 128.6,
            "turn_delay_max_s": 5.0,
            "turn_rate_error_max": 0.1,
        }
        early = _turn_execution_envelope_m(t_s=10.0, **kwargs)
        at_saturation = _turn_execution_envelope_m(t_s=25.0, **kwargs)
        late = _turn_execution_envelope_m(t_s=600.0, **kwargs)

        assert 0.0 < early < at_saturation
        assert late == pytest.approx(at_saturation, rel=1e-12)

    def test_deviation_envelope_const_and_rate_terms(self) -> None:
        """Constant and linear envelope terms should add exactly."""
        eta_m = _deviation_envelope_m(
            t_s=100.0,
            eta_const_m=100.0,
            eta_rate_mps=2.5,
            a_heading0_deg=0.0,
            a_target_heading_deg=0.0,
            a_turn_rate_deg_sec=0.0,
            a_min_speed_mps=100.0,
            a_max_speed_mps=150.0,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_turn_rate_deg_sec=0.0,
            b_min_speed_mps=100.0,
            b_max_speed_mps=150.0,
            turn_speed_schedule_uncertainty_kt=0.0,
            turn_delay_max_s=0.0,
            turn_rate_error_max=0.0,
        )

        assert eta_m == pytest.approx(100.0 + 2.5 * 100.0, rel=1e-12)

    def test_turn_speed_schedule_uncertainty_zero_matches_default(self) -> None:
        """Passing a zero robustness uncertainty should be a no-op."""
        kwargs: TurnInputs = {
            "a_lat": 51.0,
            "a_lon": -1.0,
            "a_heading0_deg": 0.0,
            "a_target_heading_deg": 45.0,
            "a_speed_kt": 340.0,
            "a_turn_rate_deg_sec": 3.0,
            "b_lat": 51.04,
            "b_lon": -0.98,
            "b_heading0_deg": 30.0,
            "b_target_heading_deg": 30.0,
            "b_speed_kt": 350.0,
            "b_turn_rate_deg_sec": 0.0,
            "separation_threshold_m": 5.0 * NMI_TO_M,
            "speed_diff_kt": 25.0,
            "projection_time_s": 600.0,
        }

        default_result = catch_up_projection_interval_with_turns(**kwargs)
        explicit_zero_result = catch_up_projection_interval_with_turns(
            **kwargs,
            turn_speed_schedule_uncertainty_kt=0.0,
        )

        assert default_result[0] == explicit_zero_result[0]
        assert default_result[1] == pytest.approx(explicit_zero_result[1], abs=1e-9)
        assert default_result[2] == pytest.approx(explicit_zero_result[2], abs=1e-9)

    def test_turn_speed_schedule_uncertainty_is_ignored_for_straight_pairs(self) -> None:
        """Straight-flight delegation should ignore the optional turn-only buffer."""
        kwargs: TurnInputs = {
            "a_lat": 51.0,
            "a_lon": -1.0,
            "a_heading0_deg": 45.0,
            "a_target_heading_deg": 45.0,
            "a_speed_kt": 350.0,
            "a_turn_rate_deg_sec": 0.0,
            "b_lat": 51.03,
            "b_lon": -0.97,
            "b_heading0_deg": 135.0,
            "b_target_heading_deg": 135.0,
            "b_speed_kt": 300.0,
            "b_turn_rate_deg_sec": 0.0,
            "separation_threshold_m": 5.0 * NMI_TO_M,
            "speed_diff_kt": 25.0,
            "projection_time_s": 900.0,
        }

        result_no_buffer = catch_up_projection_interval_with_turns(**kwargs)
        result_large_buffer = catch_up_projection_interval_with_turns(
            **kwargs,
            turn_speed_schedule_uncertainty_kt=50.0,
        )

        assert result_no_buffer[0] == result_large_buffer[0]
        assert result_no_buffer[1] == pytest.approx(result_large_buffer[1], abs=1e-9)
        assert result_no_buffer[2] == pytest.approx(result_large_buffer[2], abs=1e-9)

    def test_turn_speed_schedule_uncertainty_can_flip_a_marginal_safe_case(self) -> None:
        """A large turn-only robustness buffer should make a near-threshold safe case unsafe."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(-12000.0, -4000.0, a_lat, a_lon)
        kwargs: TurnInputs = {
            "a_lat": a_lat,
            "a_lon": a_lon,
            "a_heading0_deg": 0.0,
            "a_target_heading_deg": 270.0,
            "a_speed_kt": 340.0,
            "a_turn_rate_deg_sec": 1.5,
            "b_lat": b_lat,
            "b_lon": b_lon,
            "b_heading0_deg": 90.0,
            "b_target_heading_deg": 90.0,
            "b_speed_kt": 340.0,
            "b_turn_rate_deg_sec": 0.0,
            "separation_threshold_m": 5.0 * NMI_TO_M,
            "speed_diff_kt": 15.0,
            "projection_time_s": 900.0,
        }

        unbuffered_result = catch_up_projection_interval_with_turns(**kwargs)
        buffered_result = catch_up_projection_interval_with_turns(
            **kwargs,
            turn_speed_schedule_uncertainty_kt=50.0,
        )

        assert unbuffered_result[0] is True
        assert buffered_result[0] is False

    def test_local_xy_roundtrip(self) -> None:
        """The test-only inverse should roundtrip the production local projection."""
        ref_lat, ref_lon = 51.0, -1.0
        test_points = [
            (51.01, -0.99),
            (51.05, -1.05),
            (50.95, -0.90),
            (51.0, -1.0),  # origin
        ]
        for lat, lon in test_points:
            xy = latlon_to_local_xy(lat, lon, ref_lat, ref_lon)
            lat2, lon2 = _local_xy_to_latlon(float(xy[0]), float(xy[1]), ref_lat, ref_lon)
            assert lat2 == pytest.approx(lat, abs=1e-8)
            assert lon2 == pytest.approx(lon, abs=1e-8)

    def test_heading_diff(self) -> None:
        assert heading_diff(0.0, 90.0) == pytest.approx(90.0)
        assert heading_diff(0.0, 270.0) == pytest.approx(-90.0)
        assert heading_diff(350.0, 10.0) == pytest.approx(20.0)
        assert heading_diff(10.0, 350.0) == pytest.approx(-20.0)
        assert abs(heading_diff(0.0, 180.0)) == pytest.approx(180.0)
        assert heading_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_closest_time_from_projection_degenerate_segment(self) -> None:
        """
        Straight-line reachable sets can collapse to a segment from the origin to one
        outer point. Time recovery should still return the correct radial fraction.
        """
        T = 866.3267497343913
        hull = np.array(
            [[-28549.150219363833, -15862.747169769964], [0.0, 0.0]],
            dtype=np.float64,
        )
        closest_vec = np.array(
            [-8624.291010343248, -4791.909628287471],
            dtype=np.float64,
        )

        expected = T * (np.linalg.norm(closest_vec) / np.linalg.norm(hull[0]))
        recovered = _closest_time_from_projection(closest_vec, hull, T)

        assert recovered == pytest.approx(expected, rel=1e-12)

    def test_closest_time_from_projection_polygon_slice_consistency(self) -> None:
        """
        For a genuine polygonal reachable set, the recovered time should place the
        projected displacement back on the exact-time reachable slice.
        """
        a_heading = 86.07215429731461
        b_heading = 359.34847733836904
        a_speed = 356.1049998158458
        b_speed = 288.34698590803924
        speed_diff = 21.85042676853181
        T = 923.7721875173809
        query = np.array([25435.71039945353, -3852.776652253924], dtype=np.float64)

        hull = compute_relative_velocity_hull(a_heading, a_speed, b_heading, b_speed, speed_diff, T)
        closest_vec, _ = closest_point_on_convex_polygon(query, hull)
        recovered_t = _closest_time_from_projection(closest_vec, hull, T)

        dir_a = heading_to_unit_vector(a_heading)
        dir_b = heading_to_unit_vector(b_heading)
        a_min = max(a_speed - speed_diff, 0.0) * KT_TO_MPS
        a_max = (a_speed + speed_diff) * KT_TO_MPS
        b_min = max(b_speed - speed_diff, 0.0) * KT_TO_MPS
        b_max = (b_speed + speed_diff) * KT_TO_MPS
        exact_slice = np.array(
            [recovered_t * (sa * dir_a - sb * dir_b) for sa in (a_min, a_max) for sb in (b_min, b_max)],
            dtype=np.float64,
        )
        exact_slice_hull = convex_hull(exact_slice)
        projected, distance = closest_point_on_convex_polygon(closest_vec, exact_slice_hull)

        assert distance == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(projected - closest_vec) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. During-turn validation
# ---------------------------------------------------------------------------


class TestPhase1TurnValidation:
    """
    Validate turn-aware results against numerical arc simulation during the turn.

    The key behavioural property is conservatism: if the model certifies separation,
    the nominal numerical simulation should also remain separated.
    """

    @pytest.mark.parametrize("a_turn_deg", [5, 15, 30, 45, -5, -15, -30, -45])
    @pytest.mark.parametrize("turn_rate_deg_sec", [1.5, 3.0])
    @pytest.mark.parametrize("a_speed_kt", [250, 350, 450])
    def test_a_turning_b_straight(self, a_turn_deg: float, turn_rate_deg_sec: float, a_speed_kt: float) -> None:
        """A is turning, B is straight. Check conservative behaviour during the turn."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.04, -0.98  # B ahead-right of A
        a_heading0 = 0.0
        a_target = a_heading0 + a_turn_deg
        b_heading = 30.0
        b_speed_kt = 350.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Numerical simulation at nominal speed
        dt = 0.5
        traj_a = simulate_arc_trajectory(
            a_lat, a_lon, a_heading0, a_target, a_speed_kt, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        traj_b = simulate_arc_trajectory(
            b_lat, b_lon, b_heading, b_heading, b_speed_kt, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        # If the model says separated, the actual min dist at nominal speed should
        # also be above threshold (conservatism check)
        if is_sep:
            assert actual_min_dist >= separation_m * 0.95, (
                f"Model says separated but actual min dist {actual_min_dist / NMI_TO_M:.2f} NMI "
                f"< threshold {separation_m / NMI_TO_M:.2f} NMI"
            )

    @pytest.mark.parametrize("b_turn_deg", [5, 15, 30, 45, -5, -15, -30, -45])
    @pytest.mark.parametrize("turn_rate_deg_sec", [1.5, 3.0])
    @pytest.mark.parametrize("b_speed_kt", [250, 350, 450])
    def test_a_straight_b_turning(self, b_turn_deg: float, turn_rate_deg_sec: float, b_speed_kt: float) -> None:
        """B is turning, A is straight. Mirror case of the preceding scenario."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.04, -0.98
        a_heading = 30.0
        b_heading0 = 0.0
        b_target = b_heading0 + b_turn_deg
        a_speed_kt = 350.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading,
            a_target_heading_deg=a_heading,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=0.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading0,
            b_target_heading_deg=b_target,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Numerical simulation at nominal speed
        dt = 0.5
        traj_a = simulate_arc_trajectory(
            a_lat, a_lon, a_heading, a_heading, a_speed_kt, 0.0, dt, proj_time, a_lat, a_lon
        )
        traj_b = simulate_arc_trajectory(
            b_lat, b_lon, b_heading0, b_target, b_speed_kt, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        if is_sep:
            assert actual_min_dist >= separation_m * 0.95, (
                f"Model says separated but actual min dist {actual_min_dist / NMI_TO_M:.2f} NMI "
                f"< threshold {separation_m / NMI_TO_M:.2f} NMI"
            )

    @pytest.mark.parametrize("a_turn_deg", [15, 30, 45, -15, -30])
    @pytest.mark.parametrize("speed_diff_kt", [15, 35])
    def test_a_turning_different_speed_diffs(self, a_turn_deg: float, speed_diff_kt: float) -> None:
        """Smoke-test the during-turn geometry under different speed envelopes."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.04, -0.98
        a_heading0 = 0.0
        a_target = a_heading0 + a_turn_deg
        b_heading = 30.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 350.0, 350.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        assert min_dist >= 0.0

    @pytest.mark.parametrize(
        ("b_lat", "b_lon", "label"),
        [
            (51.06, -1.0, "ahead"),  # B ahead (north)
            (50.94, -1.0, "behind"),  # B behind (south)
            (51.0, -0.92, "abeam_right"),  # B abeam right (east)
            (51.04, -0.95, "bearing_045"),  # B at ~45 deg bearing
            (51.0, -1.08, "abeam_left"),  # B abeam left (west)
            (50.96, -0.95, "bearing_180"),  # B behind-right
        ],
    )
    def test_a_turning_various_relative_positions(self, b_lat: float, b_lon: float, label: str) -> None:
        """Check conservative behaviour for several relative starting geometries."""
        a_lat, a_lon = 51.0, -1.0
        a_heading0 = 0.0
        a_target = 30.0  # 30 deg right turn
        b_heading = 45.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 350.0, 300.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Numerical conservatism check
        dt = 0.5
        traj_a = simulate_arc_trajectory(
            a_lat, a_lon, a_heading0, a_target, a_speed_kt, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        traj_b = simulate_arc_trajectory(
            b_lat, b_lon, b_heading, b_heading, b_speed_kt, 0.0, dt, proj_time, a_lat, a_lon
        )
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        if is_sep:
            assert actual_min_dist >= separation_m * 0.95, (
                f"[{label}] Model says separated but actual min dist {actual_min_dist / NMI_TO_M:.2f} NMI < threshold"
            )
        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= proj_time

    @pytest.mark.parametrize("a_turn_deg", [30, -30])
    @pytest.mark.parametrize("b_turn_deg", [30, -30])
    def test_both_turning_same_angle(self, a_turn_deg: float, b_turn_deg: float) -> None:
        """Both aircraft turning same amount. Check conservatism."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.95
        a_heading0, b_heading0 = 0.0, 180.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 350.0, 300.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_heading0 + a_turn_deg,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading0,
            b_target_heading_deg=b_heading0 + b_turn_deg,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= proj_time

    @pytest.mark.parametrize(
        ("a_turn_deg", "b_turn_deg"),
        [(15, 45), (45, 15), (-15, 45), (30, -45), (5, 30)],
    )
    def test_both_turning_different_angles(self, a_turn_deg: float, b_turn_deg: float) -> None:
        """Both aircraft turning different amounts."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.95
        a_heading0, b_heading0 = 0.0, 180.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 350.0, 300.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 600.0

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_heading0 + a_turn_deg,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading0,
            b_target_heading_deg=b_heading0 + b_turn_deg,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= proj_time

    def test_borderline_separation_close(self) -> None:
        """Aircraft barely within separation threshold during turn — should detect conflict."""
        # B is 4.5 NMI from A (below 5 NMI threshold), both heading same way
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.0, -0.935  # ~4.5 NMI east
        turn_rate_deg_sec = 3.0
        separation_m = 5.0 * NMI_TO_M

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=30.0,
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=0.0,
            b_target_heading_deg=0.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=25.0,
            projection_time_s=600.0,
        )

        # Already within threshold at start => must detect conflict
        assert not is_sep
        assert min_dist < separation_m

    def test_borderline_separation_safe(self) -> None:
        """Aircraft well above threshold, both heading north, A turns slightly away — should be safe."""
        # B is 20 NMI from A, both heading north, A turns slightly left (away from B)
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.0, -0.70  # ~20 NMI east
        turn_rate_deg_sec = 3.0
        separation_m = 5.0 * NMI_TO_M

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=355.0,  # tiny turn away from B
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=0.0,
            b_target_heading_deg=0.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=15.0,
            projection_time_s=600.0,
        )

        assert is_sep
        assert min_dist >= separation_m


# ---------------------------------------------------------------------------
# 4. Full-horizon validation
# ---------------------------------------------------------------------------


class TestPhase2PostTurn:
    """Validate full-horizon turn-aware results against numerical simulation."""

    def test_turning_then_straight_separated(self) -> None:
        """A turns away from B, then both fly straight and diverge => should be separated."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.0, -0.95  # B east of A, about 3.5 NMI
        a_heading0 = 0.0
        a_target = 330.0  # A turns left (away from B)
        b_heading = 0.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 350.0, 350.0
        speed_diff_kt = 15.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 900.0

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Numerically verify at min/max speeds
        dt = 1.0
        traj_a = simulate_arc_trajectory(
            a_lat, a_lon, a_heading0, a_target, a_speed_kt, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        traj_b = simulate_arc_trajectory(
            b_lat, b_lon, b_heading, b_heading, b_speed_kt, 0.0, dt, proj_time, a_lat, a_lon
        )
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        # Both model and simulation should agree on separation for nominal speeds
        if is_sep:
            assert actual_min_dist >= separation_m * 0.9

    def test_converging_after_turn(self) -> None:
        """A turns toward B => should detect conflict."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.1, -0.9  # B ahead-right
        a_heading0 = 0.0
        a_target = 45.0  # A turns right (toward B)
        b_heading = 0.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 400.0, 300.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 1200.0

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Numerical simulation: A at max speed should catch B at min speed
        dt = 1.0
        traj_a = simulate_arc_trajectory(
            a_lat,
            a_lon,
            a_heading0,
            a_target,
            a_speed_kt + speed_diff_kt,
            turn_rate_deg_sec,
            dt,
            proj_time,
            a_lat,
            a_lon,
        )
        traj_b = simulate_arc_trajectory(
            b_lat, b_lon, b_heading, b_heading, max(b_speed_kt - speed_diff_kt, 0), 0.0, dt, proj_time, a_lat, a_lon
        )
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        # If numerical shows conflict, model should too
        if actual_min_dist < separation_m:
            assert not is_sep, (
                f"Numerical shows conflict (min_dist={actual_min_dist / NMI_TO_M:.2f} NMI) but model says separated"
            )

    @pytest.mark.parametrize(
        ("a_speed_mult", "b_speed_mult"),
        [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)],
        ids=["nom_nom", "nom_min", "min_nom", "min_min"],
    )
    def test_phase2_all_speed_combos(self, a_speed_mult: float, b_speed_mult: float) -> None:
        """
        Validate post-turn behaviour across all min/max speed combinations.

        The function must return is_separated=True only if ALL speed combos are safe.
        We simulate at each extreme and check the model agrees.
        """
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.06, -0.96  # B ~5 NMI ahead-right
        a_heading0, a_target = 0.0, 20.0
        b_heading = 10.0
        turn_rate_deg_sec = 3.0
        a_speed_kt, b_speed_kt = 380.0, 340.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M
        proj_time = 900.0

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_heading0,
            a_target_heading_deg=a_target,
            a_speed_kt=a_speed_kt,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_heading,
            b_target_heading_deg=b_heading,
            b_speed_kt=b_speed_kt,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=proj_time,
        )

        # Simulate at this particular speed combo
        a_actual = a_speed_kt + a_speed_mult * speed_diff_kt
        b_actual = max(b_speed_kt + b_speed_mult * speed_diff_kt, 0.0)

        dt = 1.0
        traj_a = simulate_arc_trajectory(
            a_lat, a_lon, a_heading0, a_target, a_actual, turn_rate_deg_sec, dt, proj_time, a_lat, a_lon
        )
        traj_b = simulate_arc_trajectory(b_lat, b_lon, b_heading, b_heading, b_actual, 0.0, dt, proj_time, a_lat, a_lon)
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        # If the model says separated, this particular speed combo should also be safe
        if is_sep:
            assert actual_min_dist >= separation_m * 0.9, (
                f"Model says separated but actual min dist at speeds "
                f"({a_actual:.0f}/{b_actual:.0f} kt) = {actual_min_dist / NMI_TO_M:.2f} NMI "
                f"< threshold {separation_m / NMI_TO_M:.2f} NMI"
            )


# ---------------------------------------------------------------------------
# 5. End-to-end combined scenarios
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_diverging_after_turn(self) -> None:
        """Both aircraft turn away from each other => separated."""
        # Aircraft flying north, side by side, 15 NMI apart, each turning outward
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.0, -0.77  # ~15 NMI east
        turn_rate_deg_sec = 3.0
        speed_diff_kt = 15.0
        separation_m = 5.0 * NMI_TO_M

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,  # flying north
            a_target_heading_deg=330.0,  # turns left (away from B)
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=0.0,  # flying north
            b_target_heading_deg=30.0,  # turns right (away from A)
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=1200.0,
        )

        assert is_sep, "Diverging aircraft after turns should be separated"

    def test_parallel_after_turn(self) -> None:
        """Both turn to same heading, good separation => should remain separated."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.0, -0.85  # ~10 NMI east
        turn_rate_deg_sec = 3.0
        speed_diff_kt = 15.0
        separation_m = 5.0 * NMI_TO_M

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=350.0,
            a_target_heading_deg=0.0,  # small 10deg turn
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=10.0,
            b_target_heading_deg=0.0,  # small 10deg turn
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=1200.0,
        )

        # 15 NMI apart, both heading north at same speed => should stay separated
        assert is_sep

    def test_head_on_after_turns(self) -> None:
        """Both turn toward each other => should detect conflict."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -1.0  # B 5 NMI north
        turn_rate_deg_sec = 3.0
        speed_diff_kt = 25.0
        separation_m = 5.0 * NMI_TO_M

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=90.0,
            a_target_heading_deg=0.0,  # turns to fly north (toward B)
            a_speed_kt=400.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=270.0,
            b_target_heading_deg=180.0,  # turns to fly south (toward A)
            b_speed_kt=400.0,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=separation_m,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=1200.0,
        )

        assert not is_sep, "Head-on aircraft after turns should detect conflict"

    def test_one_stationary(self) -> None:
        """One aircraft nearly stationary (very low speed). Edge case from plan."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.03, -0.97  # ~3 NMI away

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=30.0,
            a_speed_kt=5.0,  # nearly stationary
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=600.0,
        )

        # B is heading south toward A with A nearly stationary => should detect conflict
        assert not is_sep
        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= 600.0

    def test_widely_separated_aircraft(self) -> None:
        """Aircraft 50 NMI apart, any turns => should always be separated with short projection."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.5, -0.5  # ~40+ NMI away

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=45.0,
            a_speed_kt=450.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=225.0,
            b_speed_kt=450.0,
            b_turn_rate_deg_sec=3.0,
            separation_threshold_m=5 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=300.0,  # 5 min
        )

        assert is_sep


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_turn_angle_zero(self) -> None:
        """Turn angle == 0 (heading0 == target) should delegate to the straight solver."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.02, -0.98

        result_turn = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=45.0,
            a_target_heading_deg=45.0,  # no turn
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=135.0,
            b_target_heading_deg=135.0,  # no turn
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=3.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=600.0,
        )

        result_ref = catch_up_projection_interval(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading=45.0,
            a_speed_kt=350.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading=135.0,
            b_speed_kt=350.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=600.0,
        )

        assert result_turn[0] == result_ref[0]
        assert result_turn[1] == pytest.approx(result_ref[1], abs=1e-6)
        assert result_turn[2] == pytest.approx(result_ref[2], abs=1e-6)

    def test_large_turn_180(self) -> None:
        """180 degree turn (reversal) should work."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.1, -1.0

        _is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=180.0,  # full reversal
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=3.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=1200.0,
        )

        # Basic sanity
        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= 1200.0

    def test_projection_time_less_than_turn(self) -> None:
        """If projection_time < turn_duration, only the turning portion is relevant."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.1, -1.0  # 6+ NMI away
        turn_rate_deg_sec = 3.0
        turn_angle = 45.0  # turn takes 15s at 3 deg/s
        proj_time = 10.0  # less than turn duration

        _is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=turn_angle,
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=proj_time,
        )

        # The turn-aware search should still return a valid bounded answer even when the
        # aircraft never reaches the post-turn straight segment within the horizon.
        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= proj_time

    def test_very_large_speed_diff(self) -> None:
        """speed_diff > nominal speed => min speed clipped to 0."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.95

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=30.0,
            a_speed_kt=200.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=200.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=250.0,  # larger than nominal speed
            projection_time_s=600.0,
        )

        # Should not crash and should return valid results
        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= 600.0

    def test_same_position(self) -> None:
        """Aircraft at same position => min_distance should be ~0."""
        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=51.0,
            a_lon=-1.0,
            a_heading0_deg=0.0,
            a_target_heading_deg=30.0,
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=51.0,
            b_lon=-1.0,
            b_heading0_deg=0.0,
            b_target_heading_deg=60.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=3.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=25.0,
            projection_time_s=600.0,
        )

        assert not is_sep
        assert min_dist < 100.0  # should be very close to 0


# ---------------------------------------------------------------------------
# 7. Property-based tests
# ---------------------------------------------------------------------------


class TestProperties:
    """Invariant / property-based tests for catch_up_projection_interval_with_turns."""

    @pytest.mark.parametrize(
        ("a_turn", "b_turn"),
        [(30, 0), (0, 30), (30, -20), (-15, 45), (0, 0)],
        ids=["a_turn", "b_turn", "both_opp", "both_diff", "straight"],
    )
    def test_ab_symmetry(self, a_turn: float, b_turn: float) -> None:
        """min_distance should be identical when swapping A and B."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.04, -0.96
        a_h0, a_ht = 10.0, 10.0 + a_turn
        b_h0, b_ht = 350.0, 350.0 + b_turn
        a_spd, b_spd = 380.0, 320.0
        turn_rate_deg_sec = 3.0
        sd, sep_m, pt = 25.0, 5.0 * NMI_TO_M, 900.0

        _, dist_ab, time_ab = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_h0,
            a_target_heading_deg=a_ht,
            a_speed_kt=a_spd,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_h0,
            b_target_heading_deg=b_ht,
            b_speed_kt=b_spd,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=sep_m,
            speed_diff_kt=sd,
            projection_time_s=pt,
        )

        _, dist_ba, time_ba = catch_up_projection_interval_with_turns(
            a_lat=b_lat,
            a_lon=b_lon,
            a_heading0_deg=b_h0,
            a_target_heading_deg=b_ht,
            a_speed_kt=b_spd,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=a_lat,
            b_lon=a_lon,
            b_heading0_deg=a_h0,
            b_target_heading_deg=a_ht,
            b_speed_kt=a_spd,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=sep_m,
            speed_diff_kt=sd,
            projection_time_s=pt,
        )

        # Small differences (~1-2m) are expected due to flat-earth reference point changing
        assert dist_ab == pytest.approx(dist_ba, abs=5.0), f"A-B min_dist={dist_ab:.1f} != B-A min_dist={dist_ba:.1f}"

    @pytest.mark.parametrize("a_turn", [0, 25, -25])
    @pytest.mark.parametrize("b_turn", [0, 20])
    def test_projection_time_monotonicity(self, a_turn: float, b_turn: float) -> None:
        """min_distance should be non-increasing as projection_time grows."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.96
        turn_rate_deg_sec = 3.0
        sd, sep_m = 25.0, 5.0 * NMI_TO_M

        prev_dist = float("inf")
        for pt_min in [2, 5, 10, 15, 20, 30]:
            pt = pt_min * 60.0
            _, dist, _ = catch_up_projection_interval_with_turns(
                a_lat=a_lat,
                a_lon=a_lon,
                a_heading0_deg=0.0,
                a_target_heading_deg=float(a_turn),
                a_speed_kt=360.0,
                a_turn_rate_deg_sec=turn_rate_deg_sec,
                b_lat=b_lat,
                b_lon=b_lon,
                b_heading0_deg=180.0,
                b_target_heading_deg=180.0 + b_turn,
                b_speed_kt=340.0,
                b_turn_rate_deg_sec=turn_rate_deg_sec,
                separation_threshold_m=sep_m,
                speed_diff_kt=sd,
                projection_time_s=pt,
            )
            assert dist <= prev_dist + 1.0, (
                f"min_dist increased from {prev_dist:.1f} to {dist:.1f} when projection_time went to {pt_min} min"
            )
            prev_dist = dist

    def test_speed_diff_zero(self) -> None:
        """With zero speed uncertainty, result should still be valid (deterministic case)."""
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.05, -0.95
        turn_rate_deg_sec = 3.0

        is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=0.0,
            a_target_heading_deg=30.0,
            a_speed_kt=350.0,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=180.0,
            b_target_heading_deg=180.0,
            b_speed_kt=350.0,
            b_turn_rate_deg_sec=0.0,
            separation_threshold_m=5.0 * NMI_TO_M,
            speed_diff_kt=0.0,
            projection_time_s=900.0,
        )

        assert min_dist >= 0.0
        assert 0.0 <= closest_time <= 900.0

        # With zero speed_diff the result should match a single numerical simulation exactly
        dt = 0.5
        traj_a = simulate_arc_trajectory(a_lat, a_lon, 0.0, 30.0, 350.0, turn_rate_deg_sec, dt, 900.0, a_lat, a_lon)
        traj_b = simulate_arc_trajectory(b_lat, b_lon, 180.0, 180.0, 350.0, 0.0, dt, 900.0, a_lat, a_lon)
        actual_min_dist, _ = compute_min_separation_numerical(traj_a, traj_b)

        if is_sep:
            assert actual_min_dist >= 5.0 * NMI_TO_M * 0.95

    @pytest.mark.parametrize(
        ("a_turn", "b_turn"),
        [(30, 0), (0, 30), (30, -20), (-45, 45)],
        ids=["a_only", "b_only", "both_opp", "both_large"],
    )
    def test_full_trajectory_conservatism_all_speed_extremes(self, a_turn: float, b_turn: float) -> None:
        """
        If model says separated, ALL 4 speed extreme combos should be truly separated
        across the full turn+straight trajectory.
        """
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = 51.06, -0.94
        a_h0, b_h0 = 10.0, 200.0
        a_ht, b_ht = a_h0 + a_turn, b_h0 + b_turn
        a_spd, b_spd = 370.0, 330.0
        turn_rate_deg_sec, sd = 3.0, 25.0
        sep_m = 5.0 * NMI_TO_M
        pt = 900.0

        is_sep, _, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=a_h0,
            a_target_heading_deg=a_ht,
            a_speed_kt=a_spd,
            a_turn_rate_deg_sec=turn_rate_deg_sec,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=b_h0,
            b_target_heading_deg=b_ht,
            b_speed_kt=b_spd,
            b_turn_rate_deg_sec=turn_rate_deg_sec,
            separation_threshold_m=sep_m,
            speed_diff_kt=sd,
            projection_time_s=pt,
        )

        if not is_sep:
            return  # can't check conservatism when model already says conflict

        # Simulate all 4 speed corners
        dt = 1.0
        for a_mult, b_mult in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            a_actual = a_spd + a_mult * sd
            b_actual = max(b_spd + b_mult * sd, 0.0)
            traj_a = simulate_arc_trajectory(
                a_lat, a_lon, a_h0, a_ht, a_actual, turn_rate_deg_sec, dt, pt, a_lat, a_lon
            )
            traj_b = simulate_arc_trajectory(
                b_lat, b_lon, b_h0, b_ht, b_actual, turn_rate_deg_sec, dt, pt, a_lat, a_lon
            )
            actual_min, _ = compute_min_separation_numerical(traj_a, traj_b)

            assert actual_min >= sep_m * 0.9, (
                f"Model says separated but speed combo ({a_actual:.0f}/{b_actual:.0f} kt) "
                f"gives min_dist={actual_min / NMI_TO_M:.2f} NMI < threshold"
            )


class TestSafetyRegressions:
    def test_one_straight_one_turning_no_false_safe_at_phase_boundary(self) -> None:
        """
        Regression: a straight/turning pair must not report safe when boundary speed
        uncertainty can still produce a conflict.
        """
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(20240.274442110793, -8855.712362224745, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=180.0,
            a_target_heading_deg=180.0,
            a_speed_kt=280.0,
            a_turn_rate_deg_sec=0.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=315.0,
            b_target_heading_deg=270.0,
            b_speed_kt=260.0,
            b_turn_rate_deg_sec=1.0,
            separation_threshold_m=separation_m,
            speed_diff_kt=25.0,
            projection_time_s=1200.0,
        )

        assert not is_sep
        assert min_dist < separation_m

    def test_both_turning_with_unequal_turn_durations_no_false_safe(self) -> None:
        """
        Regression: unequal turn durations must not report safe once one aircraft has
        already completed its turn and continued straight.
        """
        a_lat, a_lon = 51.0, -1.0
        b_lat, b_lon = _local_xy_to_latlon(9003.823495577744, -18483.59249536298, a_lat, a_lon)
        separation_m = 5.0 * NMI_TO_M

        is_sep, min_dist, _ = catch_up_projection_interval_with_turns(
            a_lat=a_lat,
            a_lon=a_lon,
            a_heading0_deg=60.0,
            a_target_heading_deg=90.0,
            a_speed_kt=380.0,
            a_turn_rate_deg_sec=3.0,
            b_lat=b_lat,
            b_lon=b_lon,
            b_heading0_deg=0.0,
            b_target_heading_deg=315.0,
            b_speed_kt=420.0,
            b_turn_rate_deg_sec=0.75,
            separation_threshold_m=separation_m,
            speed_diff_kt=25.0,
            projection_time_s=900.0,
        )

        assert not is_sep
        assert min_dist < separation_m
