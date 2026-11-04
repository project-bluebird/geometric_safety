"""Matplotlib drawing functions and standalone demo entrypoint.

Run with ``uv run --locked python scripts/visualise.py heatmap`` or
``uv run --locked python scripts/visualise.py turns``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from geometric_safety.demo_utils import (
    TURN_DEMO_SCENARIOS,
    corridor_polygon,
    envelope_distance_curve,
    heading_summary,
    local_xy_to_latlon,
    sample_trajectory_xy,
    speed_bounds_mps,
    speed_segment_at_time,
    turn_duration_s,
)
from geometric_safety.relevant_aircraft import (
    _relative_position_hull_at_time,
    catch_up_projection_interval_with_turns,
)
from geometric_safety.util import (
    DEG_TO_RAD,
    EARTH_RADIUS_IN_METERS,
    KT_TO_MPS,
    NMI_TO_M,
    RAD_TO_DEG,
    closest_point_on_convex_polygon,
    heading_to_unit_vector,
    latlon_to_local_xy,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


@numba.jit(nopython=True, fastmath=True, cache=True, inline="always", error_model="numpy")
def move_location(
    lon: float,
    lat: float,
    dist_m: float,
    heading_deg: float,
    earth_radius_in_meters: float = EARTH_RADIUS_IN_METERS,
) -> tuple[float, float]:
    """
    Propagate a latitude/longitude point along a great-circle heading.

    This helper is only used by the standalone plotting demos, so it prioritises
    readability over the main module's strict local tangent-plane approximation.
    """
    if dist_m == 0.0:
        return lon, lat

    lat_rad = lat * DEG_TO_RAD
    lon_rad = lon * DEG_TO_RAD
    heading_rad = heading_deg * DEG_TO_RAD

    cos_lat_rad = math.cos(lat_rad)
    sin_lat_rad = math.sin(lat_rad)
    norm_dist = dist_m / earth_radius_in_meters
    cos_norm_dist = math.cos(norm_dist)
    sin_norm_dist = math.sin(norm_dist)
    cos_heading_rad = math.cos(heading_rad)
    sin_heading_rad = math.sin(heading_rad)

    lat_new = math.asin(sin_lat_rad * cos_norm_dist + cos_lat_rad * sin_norm_dist * cos_heading_rad)
    lon_new = lon_rad + math.atan2(
        sin_heading_rad * sin_norm_dist * cos_lat_rad,
        cos_norm_dist - sin_lat_rad * math.sin(lat_new),
    )

    return lon_new * RAD_TO_DEG, lat_new * RAD_TO_DEG


def draw_demo_heading_arrow(
    ax: Axes,
    xy: np.ndarray,
    heading_deg: float,
    length_m: float,
    color: str,
    style: str = "-",
    label: str | None = None,
) -> None:
    """Draw a heading arrow in the local XY plotting frame."""
    d = heading_to_unit_vector(heading_deg)
    dx, dy = float(d[0]) * length_m, float(d[1]) * length_m
    ax.annotate(
        "",
        xytext=(xy[0], xy[1]),
        xy=(xy[0] + dx, xy[1] + dy),
        arrowprops={"arrowstyle": "->", "color": color, "linestyle": style, "lw": 1.5},
    )
    if label:
        ax.annotate(label, (xy[0] + dx * 0.5, xy[1] + dy * 0.5), fontsize=8, color=color)


def set_demo_square_limits(ax: Axes, points_xy: np.ndarray, pad_m: float) -> float:
    """
    Fit a square plotting window around a point cloud.

    A square view preserves turn geometry visually and avoids long straight tails
    dominating one axis while hiding the interesting turn interaction.
    """
    min_x = float(np.min(points_xy[:, 0]))
    max_x = float(np.max(points_xy[:, 0]))
    min_y = float(np.min(points_xy[:, 1]))
    max_y = float(np.max(points_xy[:, 1]))
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    half_span_m = 0.5 * max(max_x - min_x, max_y - min_y) + pad_m
    ax.set_xlim(centre_x - half_span_m, centre_x + half_span_m)
    ax.set_ylim(centre_y - half_span_m, centre_y + half_span_m)
    return half_span_m


if __name__ == "__main__":
    import argparse

    from matplotlib.patches import Circle, Polygon
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Relevant aircraft projection demos")
    parser.add_argument(
        "demo",
        choices=["heatmap", "turns"],
        help="'heatmap': straight-ahead relevant-region heatmaps; 'turns': concrete turn-aware scenarios",
    )
    args = parser.parse_args()

    # ===================================================================
    # DEMO: heatmap — straight-ahead relevant-region heatmaps
    # ===================================================================
    if args.demo == "heatmap":
        a_lat, a_lon = 0.0, 0.0
        a_speed_kt, b_speed_kt, speed_diff_kt = 400.0, 300.0, 50.0
        separation_threshold_m = 5.0 * NMI_TO_M
        projection_time_s = 30.0 * 60.0

        n_lats, n_lons = 150, 150
        lat_min, lat_max = -0.5, 0.5
        lon_min, lon_max = -0.5, 0.5
        b_lats = np.linspace(lat_min, lat_max, n_lats)
        b_lons = np.linspace(lon_min, lon_max, n_lons)

        for b_heading in [0, 10, 45, 90, 135, 180, 225, 270, 315]:
            safe_matrix = np.zeros((n_lats, n_lons), dtype=bool)

            for i in range(n_lats):
                for j in range(n_lons):
                    is_sep, _, _ = catch_up_projection_interval_with_turns(
                        a_lat=a_lat,
                        a_lon=a_lon,
                        a_heading0_deg=0.0,
                        a_target_heading_deg=0.0,
                        a_speed_kt=a_speed_kt,
                        a_turn_rate_deg_sec=0.0,
                        b_lat=float(b_lats[i]),
                        b_lon=float(b_lons[j]),
                        b_heading0_deg=float(b_heading),
                        b_target_heading_deg=float(b_heading),
                        b_speed_kt=b_speed_kt,
                        b_turn_rate_deg_sec=0.0,
                        separation_threshold_m=separation_threshold_m,
                        speed_diff_kt=speed_diff_kt,
                        projection_time_s=projection_time_s,
                    )
                    safe_matrix[i, j] = is_sep

            b_lat_ex, b_lon_ex = -0.1, 0.1
            b_ext_lon, b_ext_lat = move_location(b_lon_ex, b_lat_ex, 5 * NMI_TO_M, b_heading)
            a_ext_lon, a_ext_lat = move_location(a_lon, a_lat, 5 * NMI_TO_M, 0.0)

            fig, ax = plt.subplots(figsize=(6, 6))
            b_lon_grid, b_lat_grid = np.meshgrid(b_lons, b_lats)
            ax.pcolormesh(b_lon_grid, b_lat_grid, safe_matrix, shading="auto", cmap="RdYlGn", alpha=0.5)
            ax.plot(a_lon, a_lat, "bo", label="Aircraft A")
            ax.annotate(
                "", xytext=(a_lon, a_lat), xy=(a_ext_lon, a_ext_lat), arrowprops={"arrowstyle": "->", "color": "b"}
            )
            ax.plot(b_lon_ex, b_lat_ex, "ko", label="Aircraft B")
            ax.annotate(
                "",
                xytext=(b_lon_ex, b_lat_ex),
                xy=(b_ext_lon, b_ext_lat),
                arrowprops={"arrowstyle": "->", "color": "k"},
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(
                f"relevant region; B flying {b_heading}\u00b0; "
                f"A={a_speed_kt:g}kt, B={b_speed_kt:g}kt, \u0394={speed_diff_kt:g}kt"
            )
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.legend()
            plt.show()

    # ===================================================================
    # DEMO: turns — concrete turn-aware scenario plots
    # ===================================================================
    if args.demo == "turns":
        NMI = NMI_TO_M
        COLOR_A = "tab:blue"
        COLOR_B = "tab:orange"
        COLOR_SAFE = "tab:green"
        COLOR_CONFLICT = "tab:red"

        for scenario in TURN_DEMO_SCENARIOS:
            title = scenario.title
            a_lat_v, a_lon_v = scenario.a_lat, scenario.a_lon
            a_h0, a_ht = scenario.a_heading_deg, scenario.a_target_heading_deg
            a_spd, a_turn_rate_deg_sec = scenario.a_speed_kt, scenario.a_turn_rate_deg_sec
            b_east_m, b_north_m = scenario.b_east_m, scenario.b_north_m
            b_lat_v, b_lon_v = local_xy_to_latlon(b_east_m, b_north_m, a_lat_v, a_lon_v)
            b_h0, b_ht = scenario.b_heading_deg, scenario.b_target_heading_deg
            b_spd, b_turn_rate_deg_sec = scenario.b_speed_kt, scenario.b_turn_rate_deg_sec
            sep_m = scenario.separation_threshold_nm * NMI
            speed_diff = scenario.speed_diff_kt
            proj_s = scenario.projection_time_min * 60.0
            a_nom_speed_mps = a_spd * KT_TO_MPS
            b_nom_speed_mps = b_spd * KT_TO_MPS
            a_min_speed_mps, a_max_speed_mps = speed_bounds_mps(a_spd, speed_diff)
            b_min_speed_mps, b_max_speed_mps = speed_bounds_mps(b_spd, speed_diff)
            a_turn_end_s = turn_duration_s(a_h0, a_ht, a_turn_rate_deg_sec)
            b_turn_end_s = turn_duration_s(b_h0, b_ht, b_turn_rate_deg_sec)

            # Run the turn-aware lateral solver.
            is_sep, min_dist, closest_time = catch_up_projection_interval_with_turns(
                a_lat=a_lat_v,
                a_lon=a_lon_v,
                a_heading0_deg=a_h0,
                a_target_heading_deg=a_ht,
                a_speed_kt=a_spd,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                b_lat=b_lat_v,
                b_lon=b_lon_v,
                b_heading0_deg=b_h0,
                b_target_heading_deg=b_ht,
                b_speed_kt=b_spd,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                separation_threshold_m=sep_m,
                speed_diff_kt=speed_diff,
                projection_time_s=proj_s,
            )

            # Work in the same local XY frame as the lateral-solver calculations.
            a_xy0 = latlon_to_local_xy(a_lat_v, a_lon_v, a_lat_v, a_lon_v)
            b_xy0 = latlon_to_local_xy(b_lat_v, b_lon_v, a_lat_v, a_lon_v)
            rel_pos0 = a_xy0 - b_xy0  # pyright: ignore[reportOperatorIssue]

            # The classifier looks over the full model horizon, but the demo should zoom
            # in on the part of the geometry where the turn actually matters.
            spatial_focus_end_s = min(
                proj_s,
                max(150.0, closest_time + 60.0, a_turn_end_s + 30.0, b_turn_end_s + 30.0),
            )
            time_focus_end_s = min(
                proj_s,
                max(180.0, closest_time + 90.0, a_turn_end_s + 45.0, b_turn_end_s + 45.0),
            )

            traj_sample_count = min(361, max(161, int(spatial_focus_end_s / 1.5) + 1))
            traj_times_s = np.linspace(0.0, spatial_focus_end_s, traj_sample_count)
            if 0.0 < closest_time < spatial_focus_end_s:
                traj_times_s = np.unique(np.concatenate((traj_times_s, np.asarray([closest_time], dtype=np.float64))))

            a_traj_nom = sample_trajectory_xy(a_xy0, a_h0, a_ht, a_nom_speed_mps, a_turn_rate_deg_sec, traj_times_s)
            b_traj_nom = sample_trajectory_xy(b_xy0, b_h0, b_ht, b_nom_speed_mps, b_turn_rate_deg_sec, traj_times_s)
            a_traj_min = sample_trajectory_xy(a_xy0, a_h0, a_ht, a_min_speed_mps, a_turn_rate_deg_sec, traj_times_s)
            a_traj_max = sample_trajectory_xy(a_xy0, a_h0, a_ht, a_max_speed_mps, a_turn_rate_deg_sec, traj_times_s)
            b_traj_min = sample_trajectory_xy(b_xy0, b_h0, b_ht, b_min_speed_mps, b_turn_rate_deg_sec, traj_times_s)
            b_traj_max = sample_trajectory_xy(b_xy0, b_h0, b_ht, b_max_speed_mps, b_turn_rate_deg_sec, traj_times_s)

            curve_sample_count = min(601, max(241, int(proj_s / 2.0) + 1))
            curve_times_s = np.linspace(0.0, proj_s, curve_sample_count)
            if 0.0 < closest_time < proj_s:
                curve_times_s = np.unique(np.concatenate((curve_times_s, np.asarray([closest_time], dtype=np.float64))))

            envelope_distances_m = envelope_distance_curve(
                rel_pos0=rel_pos0,
                a_heading0_deg=a_h0,
                a_target_heading_deg=a_ht,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_h0,
                b_target_heading_deg=b_ht,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                times_s=curve_times_s,
            )
            a_curve_nom = sample_trajectory_xy(a_xy0, a_h0, a_ht, a_nom_speed_mps, a_turn_rate_deg_sec, curve_times_s)
            b_curve_nom = sample_trajectory_xy(b_xy0, b_h0, b_ht, b_nom_speed_mps, b_turn_rate_deg_sec, curve_times_s)
            nominal_distances_m = np.linalg.norm(a_curve_nom - b_curve_nom, axis=1)
            time_focus_mask = curve_times_s <= time_focus_end_s + 1e-9
            curve_times_focus_s = curve_times_s[time_focus_mask]
            envelope_distances_focus_nmi = envelope_distances_m[time_focus_mask] / NMI
            nominal_distances_focus_nmi = nominal_distances_m[time_focus_mask] / NMI
            threshold_nmi = sep_m / NMI

            a_pos_min, a_pos_nom, a_pos_max = speed_segment_at_time(
                a_xy0, a_h0, a_ht, a_turn_rate_deg_sec, a_nom_speed_mps, a_min_speed_mps, a_max_speed_mps, closest_time
            )
            b_pos_min, b_pos_nom, b_pos_max = speed_segment_at_time(
                b_xy0, b_h0, b_ht, b_turn_rate_deg_sec, b_nom_speed_mps, b_min_speed_mps, b_max_speed_mps, closest_time
            )
            rel_hull = _relative_position_hull_at_time(
                rel_pos0=rel_pos0,
                a_heading0_deg=a_h0,
                a_target_heading_deg=a_ht,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_h0,
                b_target_heading_deg=b_ht,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                t_s=closest_time,
            )
            origin = np.zeros(2, dtype=np.float64)
            rel_nom = a_pos_nom - b_pos_nom  # pyright: ignore[reportOperatorIssue]
            rel_closest_point, _ = closest_point_on_convex_polygon(origin, rel_hull)
            a_speed_span_t_m = float(np.linalg.norm(a_pos_max - a_pos_min))
            b_speed_span_t_m = float(np.linalg.norm(b_pos_max - b_pos_min))
            a_speed_span_end_m = float(np.linalg.norm(a_traj_max[-1] - a_traj_min[-1]))
            b_speed_span_end_m = float(np.linalg.norm(b_traj_max[-1] - b_traj_min[-1]))
            a_corridor_xy = corridor_polygon(a_traj_min, a_traj_max)
            b_corridor_xy = corridor_polygon(b_traj_min, b_traj_max)
            inset_half_window_s = min(20.0, max(12.0, 0.18 * spatial_focus_end_s))
            inset_start_s = max(0.0, closest_time - inset_half_window_s)
            inset_end_s = min(spatial_focus_end_s, closest_time + inset_half_window_s)
            inset_mask = (traj_times_s >= inset_start_s - 1e-9) & (traj_times_s <= inset_end_s + 1e-9)
            a_traj_nom_inset = a_traj_nom[inset_mask]
            b_traj_nom_inset = b_traj_nom[inset_mask]
            a_traj_min_inset = a_traj_min[inset_mask]
            a_traj_max_inset = a_traj_max[inset_mask]
            b_traj_min_inset = b_traj_min[inset_mask]
            b_traj_max_inset = b_traj_max[inset_mask]
            a_corridor_inset = corridor_polygon(a_traj_min_inset, a_traj_max_inset)
            b_corridor_inset = corridor_polygon(b_traj_min_inset, b_traj_max_inset)

            a_nom_turn_end_xy = None
            if 0.0 < a_turn_end_s <= spatial_focus_end_s:
                a_nom_turn_end_xy = sample_trajectory_xy(
                    a_xy0,
                    a_h0,
                    a_ht,
                    a_nom_speed_mps,
                    a_turn_rate_deg_sec,
                    np.asarray([a_turn_end_s], dtype=np.float64),
                )[0]
            b_nom_turn_end_xy = None
            if 0.0 < b_turn_end_s <= spatial_focus_end_s:
                b_nom_turn_end_xy = sample_trajectory_xy(
                    b_xy0,
                    b_h0,
                    b_ht,
                    b_nom_speed_mps,
                    b_turn_rate_deg_sec,
                    np.asarray([b_turn_end_s], dtype=np.float64),
                )[0]

            # Plot
            fig = plt.figure(figsize=(18, 8.4))
            grid = fig.add_gridspec(
                2,
                3,
                height_ratios=[1.0, 0.42],
                width_ratios=[1.2, 0.9, 1.15],
                hspace=0.28,
                wspace=0.3,
            )
            ax_xy = fig.add_subplot(grid[0, 0])
            ax_xy_local = fig.add_subplot(grid[1, 0])
            ax_rel = fig.add_subplot(grid[:, 1])
            ax_dist = fig.add_subplot(grid[:, 2])
            result_str = "SEPARATED" if is_sep else "CONFLICT"
            result_color = COLOR_SAFE if is_sep else COLOR_CONFLICT
            nominal_sep_nmi = float(np.linalg.norm(rel_nom)) / NMI
            fig.suptitle(
                f"{title} | {result_str} | "
                f"reported envelope distance {min_dist / NMI:.2f} NMI at t*={closest_time:.0f}s",
                fontsize=14,
                y=0.98,
            )

            # Spatial view: nominal trajectories plus min/max speed envelopes and the
            # speed-reachable segments at the reported critical time `t*`.
            ax_xy.fill(
                a_corridor_xy[:, 0],
                a_corridor_xy[:, 1],
                facecolor=COLOR_A,
                edgecolor="none",
                alpha=0.12,
                zorder=1,
                label="A speed corridor",
            )
            ax_xy.fill(
                b_corridor_xy[:, 0],
                b_corridor_xy[:, 1],
                facecolor=COLOR_B,
                edgecolor="none",
                alpha=0.12,
                zorder=1,
                label="B speed corridor",
            )
            ax_xy.plot(a_traj_nom[:, 0], a_traj_nom[:, 1], color=COLOR_A, lw=2.6, label="A nominal")
            ax_xy.plot(b_traj_nom[:, 0], b_traj_nom[:, 1], color=COLOR_B, lw=2.6, label="B nominal")
            ax_xy.plot(
                a_traj_min[:, 0],
                a_traj_min[:, 1],
                color=COLOR_A,
                lw=1.3,
                ls=":",
                alpha=0.85,
                zorder=2,
                label="_nolegend_",
            )
            ax_xy.plot(
                a_traj_max[:, 0],
                a_traj_max[:, 1],
                color=COLOR_A,
                lw=1.3,
                ls=":",
                alpha=0.85,
                zorder=2,
                label="_nolegend_",
            )
            ax_xy.plot(
                b_traj_min[:, 0],
                b_traj_min[:, 1],
                color=COLOR_B,
                lw=1.3,
                ls=":",
                alpha=0.85,
                zorder=2,
                label="_nolegend_",
            )
            ax_xy.plot(
                b_traj_max[:, 0],
                b_traj_max[:, 1],
                color=COLOR_B,
                lw=1.3,
                ls=":",
                alpha=0.85,
                zorder=2,
                label="_nolegend_",
            )

            # Start positions
            ax_xy.scatter([a_xy0[0]], [a_xy0[1]], c=COLOR_A, s=88, zorder=5, edgecolors="white", linewidths=1.5)
            ax_xy.scatter([b_xy0[0]], [b_xy0[1]], c=COLOR_B, s=88, zorder=5, edgecolors="white", linewidths=1.5)
            ax_xy.annotate("A", (a_xy0[0], a_xy0[1]), xytext=(8, 8), textcoords="offset points", color=COLOR_A)
            ax_xy.annotate("B", (b_xy0[0], b_xy0[1]), xytext=(8, 8), textcoords="offset points", color=COLOR_B)

            # Heading arrows: solid = initial heading, dashed = target heading
            spatial_focus_points = np.vstack(
                (
                    a_traj_nom,
                    b_traj_nom,
                    a_traj_min,
                    a_traj_max,
                    b_traj_min,
                    b_traj_max,
                    a_pos_min[None, :],
                    a_pos_max[None, :],
                    b_pos_min[None, :],
                    b_pos_max[None, :],
                )
            )
            spatial_half_span_m = set_demo_square_limits(
                ax_xy,
                spatial_focus_points,
                pad_m=max(0.35 * sep_m, 1500.0),
            )
            arrow_len = max(1800.0, 0.15 * spatial_half_span_m)
            draw_demo_heading_arrow(ax_xy, a_xy0, a_h0, arrow_len, COLOR_A)
            draw_demo_heading_arrow(ax_xy, b_xy0, b_h0, arrow_len, COLOR_B)
            if a_h0 != a_ht:
                draw_demo_heading_arrow(ax_xy, a_xy0, a_ht, arrow_len * 0.72, COLOR_A, style="--")
            if b_h0 != b_ht:
                draw_demo_heading_arrow(ax_xy, b_xy0, b_ht, arrow_len * 0.72, COLOR_B, style="--")

            if a_nom_turn_end_xy is not None:
                ax_xy.scatter(
                    [a_nom_turn_end_xy[0]],
                    [a_nom_turn_end_xy[1]],
                    marker="^",
                    s=64,
                    color=COLOR_A,
                    zorder=6,
                    label="A turn end",
                )
            if b_nom_turn_end_xy is not None:
                ax_xy.scatter(
                    [b_nom_turn_end_xy[0]],
                    [b_nom_turn_end_xy[1]],
                    marker="^",
                    s=64,
                    color=COLOR_B,
                    zorder=6,
                    label="B turn end",
                )

            # Highlight the reachable speed segment for each aircraft at the reported
            # critical time `t*`. This shows the 1-D position uncertainty induced by
            # speed.
            ax_xy.plot(
                [a_pos_min[0], a_pos_max[0]],
                [a_pos_min[1], a_pos_max[1]],
                color=COLOR_A,
                lw=5,
                alpha=0.24,
                solid_capstyle="round",
                label="A reachable @ t*",
            )
            ax_xy.plot(
                [b_pos_min[0], b_pos_max[0]],
                [b_pos_min[1], b_pos_max[1]],
                color=COLOR_B,
                lw=5,
                alpha=0.24,
                solid_capstyle="round",
                label="B reachable @ t*",
            )
            ax_xy.scatter([a_pos_nom[0]], [a_pos_nom[1]], color=COLOR_A, s=40, zorder=6)
            ax_xy.scatter([b_pos_nom[0]], [b_pos_nom[1]], color=COLOR_B, s=40, zorder=6)
            ax_xy.plot(
                [a_pos_nom[0], b_pos_nom[0]],
                [a_pos_nom[1], b_pos_nom[1]],
                color="0.5",
                lw=1.1,
                ls="--",
                alpha=0.9,
                label="Nominal sep @ t*",
            )
            ax_xy.annotate(
                f"t*={closest_time:.0f}s",
                (0.5 * (a_pos_nom[0] + b_pos_nom[0]), 0.5 * (a_pos_nom[1] + b_pos_nom[1])),
                xytext=(10, -14),
                textcoords="offset points",
                fontsize=8,
                color="0.3",
            )

            ax_xy.text(
                0.02,
                0.98,
                f"A: {heading_summary(a_h0, a_ht, a_turn_rate_deg_sec, a_spd)}\n"
                f"B: {heading_summary(b_h0, b_ht, b_turn_rate_deg_sec, b_spd)}\n"
                f"Speed envelope: +/-{speed_diff:.0f} kt\n"
                f"A span @ t*: {a_speed_span_t_m:.0f} m, shown end: {a_speed_span_end_m:.0f} m\n"
                f"B span @ t*: {b_speed_span_t_m:.0f} m, shown end: {b_speed_span_end_m:.0f} m\n"
                f"Nominal sep @ t*: {nominal_sep_nmi:.2f} NMI\n"
                f"Spatial view: 0-{spatial_focus_end_s:.0f}s\n"
                f"Model horizon: 0-{proj_s:.0f}s",
                transform=ax_xy.transAxes,
                verticalalignment="top",
                fontsize=9.5,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
            )

            # Dedicated local-view panel around the reported critical time. The top
            # spatial plot keeps the large-scale context; this lower panel makes the
            # small speed-induced spread readable without covering the main geometry.
            ax_xy_local.fill(
                a_corridor_inset[:, 0],
                a_corridor_inset[:, 1],
                facecolor=COLOR_A,
                edgecolor="none",
                alpha=0.12,
            )
            ax_xy_local.fill(
                b_corridor_inset[:, 0],
                b_corridor_inset[:, 1],
                facecolor=COLOR_B,
                edgecolor="none",
                alpha=0.12,
            )
            ax_xy_local.plot(a_traj_nom_inset[:, 0], a_traj_nom_inset[:, 1], color=COLOR_A, lw=2.2)
            ax_xy_local.plot(b_traj_nom_inset[:, 0], b_traj_nom_inset[:, 1], color=COLOR_B, lw=2.2)
            ax_xy_local.plot(a_traj_min_inset[:, 0], a_traj_min_inset[:, 1], color=COLOR_A, lw=1.0, ls=":", alpha=0.8)
            ax_xy_local.plot(a_traj_max_inset[:, 0], a_traj_max_inset[:, 1], color=COLOR_A, lw=1.0, ls=":", alpha=0.8)
            ax_xy_local.plot(b_traj_min_inset[:, 0], b_traj_min_inset[:, 1], color=COLOR_B, lw=1.0, ls=":", alpha=0.8)
            ax_xy_local.plot(b_traj_max_inset[:, 0], b_traj_max_inset[:, 1], color=COLOR_B, lw=1.0, ls=":", alpha=0.8)
            ax_xy_local.plot(
                [a_pos_min[0], a_pos_max[0]],
                [a_pos_min[1], a_pos_max[1]],
                color=COLOR_A,
                lw=4.0,
                alpha=0.32,
                solid_capstyle="round",
            )
            ax_xy_local.plot(
                [b_pos_min[0], b_pos_max[0]],
                [b_pos_min[1], b_pos_max[1]],
                color=COLOR_B,
                lw=4.0,
                alpha=0.32,
                solid_capstyle="round",
            )
            ax_xy_local.scatter([a_pos_nom[0]], [a_pos_nom[1]], color=COLOR_A, s=28, zorder=5)
            ax_xy_local.scatter([b_pos_nom[0]], [b_pos_nom[1]], color=COLOR_B, s=28, zorder=5)
            ax_xy_local.plot(
                [a_pos_nom[0], b_pos_nom[0]],
                [a_pos_nom[1], b_pos_nom[1]],
                color="0.45",
                lw=1.0,
                ls="--",
                alpha=0.85,
            )
            inset_points_xy = np.vstack(
                (
                    a_traj_nom_inset,
                    b_traj_nom_inset,
                    a_traj_min_inset,
                    a_traj_max_inset,
                    b_traj_min_inset,
                    b_traj_max_inset,
                    a_pos_min[None, :],
                    a_pos_max[None, :],
                    b_pos_min[None, :],
                    b_pos_max[None, :],
                )
            )
            set_demo_square_limits(
                ax_xy_local,
                inset_points_xy,
                pad_m=max(250.0, 0.3 * max(a_speed_span_end_m, b_speed_span_end_m)),
            )
            ax_xy_local.set_title(f"Local view [{inset_start_s:.0f}, {inset_end_s:.0f}] s", fontsize=10)
            ax_xy_local.set_xlabel("East (m)")
            ax_xy_local.set_ylabel("North (m)")
            ax_xy_local.set_aspect("equal", "box")
            ax_xy_local.grid(True, alpha=0.2)
            ax_xy_local.tick_params(labelsize=8)
            for spine in ax_xy_local.spines.values():
                spine.set_edgecolor("0.5")
                spine.set_linewidth(1.0)

            # Relative-hull view: at the reported critical time `t*`, the reachable
            # relative positions form a small convex set. Conflict means this set
            # touches the threshold disk around the origin.
            ax_rel.add_patch(
                Circle((0.0, 0.0), radius=sep_m, facecolor="none", edgecolor=COLOR_CONFLICT, ls=":", lw=1.4)
            )
            if rel_hull.shape[0] >= 3:
                ax_rel.add_patch(
                    Polygon(rel_hull, closed=True, facecolor=result_color, edgecolor=result_color, alpha=0.18, lw=2.0)
                )
                hull_closed = np.vstack((rel_hull, rel_hull[0]))
                ax_rel.plot(
                    hull_closed[:, 0], hull_closed[:, 1], color=result_color, lw=2.0, label="Reachable hull @ t*"
                )
            elif rel_hull.shape[0] == 2:
                ax_rel.plot(rel_hull[:, 0], rel_hull[:, 1], color=result_color, lw=3.0, label="Reachable hull @ t*")
            else:
                ax_rel.scatter(rel_hull[:, 0], rel_hull[:, 1], color=result_color, s=48, label="Reachable hull @ t*")
            ax_rel.scatter([0.0], [0.0], color="0.2", s=40, zorder=5, label="Protected centre")
            ax_rel.scatter([rel_nom[0]], [rel_nom[1]], color="0.2", s=42, zorder=6, label="Nominal relative @ t*")
            ax_rel.plot(
                [0.0, rel_closest_point[0]],
                [0.0, rel_closest_point[1]],
                color="0.35",
                lw=1.2,
                ls="--",
            )
            ax_rel.scatter([rel_closest_point[0]], [rel_closest_point[1]], color=result_color, s=48, zorder=6)
            rel_extent_m = max(
                1.25 * sep_m,
                float(np.max(np.abs(rel_hull))) if rel_hull.size > 0 else 0.0,
                float(np.max(np.abs(rel_nom))),
            )
            rel_extent_m = rel_extent_m * 1.15 + 500.0
            ax_rel.set_xlim(-rel_extent_m, rel_extent_m)
            ax_rel.set_ylim(-rel_extent_m, rel_extent_m)
            ax_rel.set_aspect("equal", "box")
            ax_rel.set_title("Relative hull at t*")
            ax_rel.set_xlabel("Relative east (m)")
            ax_rel.set_ylabel("Relative north (m)")
            ax_rel.grid(True, alpha=0.25)
            ax_rel.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                fontsize=9,
            )

            # Time view: show the sampled exact envelope distance against the nominal
            # trajectory distance so the certification result is easier to interpret.
            ax_dist.plot(
                curve_times_focus_s,
                envelope_distances_focus_nmi,
                color=result_color,
                lw=2.6,
                label="Envelope d(t)",
            )
            ax_dist.plot(
                curve_times_focus_s,
                nominal_distances_focus_nmi,
                color="0.25",
                lw=1.6,
                ls="--",
                label="Nominal speed",
            )
            ax_dist.axhline(threshold_nmi, color=COLOR_CONFLICT, lw=1.2, ls=":", label="Threshold")
            ax_dist.axvline(closest_time, color="0.4", lw=1.0, ls="--", alpha=0.8)
            ax_dist.scatter([closest_time], [min_dist / NMI], color=result_color, zorder=5, s=42)
            conflict_mask = [bool(v) for v in envelope_distances_focus_nmi < threshold_nmi]
            ax_dist.fill_between(
                curve_times_focus_s,
                envelope_distances_focus_nmi,
                threshold_nmi,
                where=conflict_mask,
                color=COLOR_CONFLICT,
                alpha=0.14,
                interpolate=True,
            )
            if 0.0 < a_turn_end_s <= time_focus_end_s:
                ax_dist.axvline(a_turn_end_s, color=COLOR_A, lw=1.0, ls=":", alpha=0.45)
            if 0.0 < b_turn_end_s <= time_focus_end_s:
                ax_dist.axvline(b_turn_end_s, color=COLOR_B, lw=1.0, ls=":", alpha=0.45)

            ax_xy.set_xlabel("East (m)")
            ax_xy.set_ylabel("North (m)")
            ax_xy.set_title("Trajectories (zoomed)")
            ax_xy.set_aspect("equal", "box")
            ax_xy.grid(True, alpha=0.25)

            ax_dist.set_title("Distance vs time")
            ax_dist.set_xlabel("Time (s)")
            ax_dist.set_ylabel("Distance (NMI)")
            ax_dist.set_xlim(0.0, time_focus_end_s)
            y_max_nmi = max(
                float(np.max(envelope_distances_focus_nmi)),
                float(np.max(nominal_distances_focus_nmi)),
                threshold_nmi,
            )
            y_min_nmi = min(
                float(np.min(envelope_distances_focus_nmi)),
                float(np.min(nominal_distances_focus_nmi)),
                threshold_nmi,
            )
            y_pad_nmi = max(0.25, 0.06 * max(y_max_nmi - y_min_nmi, 1.0))
            ax_dist.set_ylim(max(0.0, y_min_nmi - y_pad_nmi), y_max_nmi + y_pad_nmi)
            ax_dist.grid(True, alpha=0.25)
            ax_dist.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=3,
                fontsize=9,
            )
            ax_xy_local.legend(
                *ax_xy.get_legend_handles_labels(),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=2,
                fontsize=9,
            )
            fig.subplots_adjust(top=0.9, bottom=0.18, left=0.055, right=0.985)
            plt.show()
