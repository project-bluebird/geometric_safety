"""Render the fail-safe metric concept figure for the NeurIPS competition paper."""

from __future__ import annotations

import argparse
from pathlib import Path as FilePath

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.axes import Axes
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from geometric_safety.demo_utils import sample_trajectory_xy, speed_segment_at_time
from geometric_safety.util import KT_TO_MPS, M_TO_NMI, NMI_TO_M
from geometric_safety.vertical import level_at_time_fl, time_to_vertical_overlap_resolution

COLOR_A = "#2563eb"
COLOR_B = "#111827"
COLOR_ENVELOPE = "#f59e0b"
COLOR_CONFLICT = "#dc2626"
COLOR_SAFE = "#047857"
COLOR_MUTED = "#6b7280"

A_CURRENT_FL = 300.0
A_SELECTED_FL = 300.0
B_CURRENT_FL = 250.0
B_SELECTED_FL = 370.0
CONSERVATIVE_VERTICAL_RATE_FPM = 1000.0
REALISED_VERTICAL_RATE_FPM = 2000.0
TEXT_ZORDER = 30
TEXT_EFFECTS = [pe.withStroke(linewidth=3.0, foreground="white")]
LATERAL_VERTICAL_RANGE_MULTIPLIER = 1.30


def to_nm(xy_m: np.ndarray) -> np.ndarray:
    """Convert metre coordinates to nautical miles."""
    return xy_m * M_TO_NMI


def capsule_patch(start_xy: np.ndarray, end_xy: np.ndarray, radius: float, **kwargs: object) -> Circle | PathPatch:
    """Return a circular or capsule-shaped patch in data coordinates."""
    delta = end_xy - start_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return Circle(tuple(start_xy), radius=radius, **kwargs)

    theta = float(np.arctan2(delta[1], delta[0]))
    end_angles = np.linspace(theta + np.pi / 2.0, theta - np.pi / 2.0, 40)
    start_angles = np.linspace(theta - np.pi / 2.0, theta - 3.0 * np.pi / 2.0, 40)
    end_arc = end_xy + radius * np.column_stack((np.cos(end_angles), np.sin(end_angles)))
    start_arc = start_xy + radius * np.column_stack((np.cos(start_angles), np.sin(start_angles)))
    vertices = np.vstack((end_arc, start_arc, end_arc[0]))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (vertices.shape[0] - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), **kwargs)


def draw_heading_arrow(
    ax: Axes,
    xy: np.ndarray,
    heading_deg: float,
    length_nm: float,
    color: str,
    *,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    """Draw a compact heading arrow in aviation heading convention."""
    theta = np.deg2rad(90.0 - heading_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])
    tip = xy + length_nm * direction
    ax.annotate(
        "",
        xy=(float(tip[0]), float(tip[1])),
        xytext=(float(xy[0]), float(xy[1])),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.2,
            "linestyle": linestyle,
            "mutation_scale": 8,
            "alpha": alpha,
        },
        zorder=7,
    )


def draw_lateral_panel(ax: Axes, vertical_resolution_s: float) -> None:
    """Draw the lateral projection concept using the same kinematic helpers as the app."""
    horizon_s = vertical_resolution_s
    times_s = np.linspace(0.0, horizon_s, 220)
    sample_times_s = np.linspace(0.0, horizon_s, 4)
    sample_mark_indices = [round(time_s / horizon_s * (times_s.shape[0] - 1)) for time_s in sample_times_s]

    a_xy0_m = np.array([0.0, 0.0])
    b_xy0_m = np.array([-8.0 * NMI_TO_M, 1.2 * NMI_TO_M])
    heading0_deg = 0.0
    target_heading_deg = 70.0
    turn_rate_deg_sec = 1.0
    speed_kt = 300.0
    speed_diff_kt = 25.0
    nominal_speed_mps = speed_kt * KT_TO_MPS
    min_speed_mps = (speed_kt - speed_diff_kt) * KT_TO_MPS
    max_speed_mps = (speed_kt + speed_diff_kt) * KT_TO_MPS

    a_nom_m = sample_trajectory_xy(
        a_xy0_m, heading0_deg, target_heading_deg, nominal_speed_mps, turn_rate_deg_sec, times_s
    )
    a_min_m = sample_trajectory_xy(a_xy0_m, heading0_deg, target_heading_deg, min_speed_mps, turn_rate_deg_sec, times_s)
    a_max_m = sample_trajectory_xy(a_xy0_m, heading0_deg, target_heading_deg, max_speed_mps, turn_rate_deg_sec, times_s)
    b_nom_m = sample_trajectory_xy(
        b_xy0_m, heading0_deg, target_heading_deg, nominal_speed_mps, turn_rate_deg_sec, times_s
    )
    b_min_m = sample_trajectory_xy(b_xy0_m, heading0_deg, target_heading_deg, min_speed_mps, turn_rate_deg_sec, times_s)
    b_max_m = sample_trajectory_xy(b_xy0_m, heading0_deg, target_heading_deg, max_speed_mps, turn_rate_deg_sec, times_s)

    a_nom = to_nm(a_nom_m)
    a_min = to_nm(a_min_m)
    a_max = to_nm(a_max_m)
    b_nom = to_nm(b_nom_m)
    b_min = to_nm(b_min_m)
    b_max = to_nm(b_max_m)
    for sample_idx, time_s in enumerate(sample_times_s):
        a_min_pos_m, _a_nom_pos_m, a_max_pos_m = speed_segment_at_time(
            a_xy0_m,
            heading0_deg,
            target_heading_deg,
            turn_rate_deg_sec,
            nominal_speed_mps,
            min_speed_mps,
            max_speed_mps,
            float(time_s),
        )
        style = ":" if time_s == 0.0 else "-"
        ax.add_patch(
            capsule_patch(
                to_nm(a_min_pos_m),
                to_nm(a_max_pos_m),
                5.0,
                facecolor=COLOR_ENVELOPE,
                edgecolor=COLOR_ENVELOPE,
                lw=1.2,
                linestyle=style,
                alpha=0.12 + 0.08 * sample_idx / max(sample_times_s.shape[0] - 1, 1),
                zorder=3,
            )
        )
        ax.add_patch(
            capsule_patch(
                to_nm(a_min_pos_m),
                to_nm(a_max_pos_m),
                5.0,
                facecolor="none",
                edgecolor=COLOR_ENVELOPE,
                lw=1.3,
                linestyle=style,
                alpha=0.78,
                zorder=4,
            )
        )

    ax.plot(
        a_nom[:, 0],
        a_nom[:, 1],
        color=COLOR_A,
        lw=1.9,
        marker="o",
        markevery=sample_mark_indices,
        ms=3.3,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Nominal path",
        zorder=7,
    )
    ax.plot(
        b_nom[:, 0],
        b_nom[:, 1],
        color=COLOR_B,
        lw=1.7,
        marker="o",
        markevery=sample_mark_indices,
        ms=3.3,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Other aircraft",
        zorder=7,
    )

    a_sample_m = sample_trajectory_xy(
        a_xy0_m, heading0_deg, target_heading_deg, nominal_speed_mps, turn_rate_deg_sec, sample_times_s
    )
    b_sample_m = sample_trajectory_xy(
        b_xy0_m, heading0_deg, target_heading_deg, nominal_speed_mps, turn_rate_deg_sec, sample_times_s
    )
    a_sample = to_nm(a_sample_m)
    b_sample = to_nm(b_sample_m)

    for idx in range(sample_times_s.shape[0]):
        ax.plot(
            [a_sample[idx, 0], b_sample[idx, 0]],
            [a_sample[idx, 1], b_sample[idx, 1]],
            color="#9ca3af",
            lw=0.9,
            ls=":",
            alpha=0.9,
            zorder=4,
        )

    ax.scatter([a_nom[0, 0]], [a_nom[0, 1]], s=34, color=COLOR_A, edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([b_nom[0, 0]], [b_nom[0, 1]], s=34, color=COLOR_B, edgecolor="white", linewidth=0.8, zorder=8)
    ax.text(
        a_nom[0, 0],
        a_nom[0, 1] - 1.45,
        "A",
        color=COLOR_A,
        weight="bold",
        fontsize=8,
        ha="center",
        va="top",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.text(
        b_nom[0, 0],
        b_nom[0, 1] - 1.45,
        "B",
        color=COLOR_B,
        weight="bold",
        fontsize=8,
        ha="center",
        va="top",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )

    _final_min_m, final_nom_m, _final_max_m = speed_segment_at_time(
        a_xy0_m,
        heading0_deg,
        target_heading_deg,
        turn_rate_deg_sec,
        nominal_speed_mps,
        min_speed_mps,
        max_speed_mps,
        float(horizon_s),
    )
    final_nom = to_nm(final_nom_m)
    ax.text(
        final_nom[0] + 1.1,
        final_nom[1] - 0.6,
        r"$t_v$",
        color=COLOR_SAFE,
        fontsize=8,
        weight="bold",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )

    protected_label_xy = (21.0, 2.2)
    ax.text(
        *protected_label_xy,
        "5 NMI protected\nenvelope",
        fontsize=7.2,
        color="#374151",
        ha="left",
        va="center",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.annotate(
        "",
        xy=(15.5, 6.7),
        xytext=(20.7, 3.5),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": COLOR_MUTED,
            "shrinkA": 0,
            "shrinkB": 3,
            "zorder": TEXT_ZORDER - 1,
        },
        zorder=TEXT_ZORDER - 1,
    )
    b_label_xy = (b_sample[-2, 0] - 14.0, b_sample[-2, 1] + 1.6)
    ax.text(
        *b_label_xy,
        "B stays outside at\nmatched times",
        fontsize=7.2,
        color="#374151",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.annotate(
        "",
        xy=(b_sample[-2, 0], b_sample[-2, 1]),
        xytext=(0.6, 11.5),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": COLOR_MUTED,
            "shrinkA": 2,
            "shrinkB": 3,
            "zorder": TEXT_ZORDER - 1,
        },
        zorder=TEXT_ZORDER - 1,
    )
    speed_label_xy = (15.2, 21.0)
    ax.text(
        *speed_label_xy,
        "speed envelope\nlengthens",
        fontsize=7.2,
        color="#374151",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.annotate(
        "",
        xy=(27.9, 15.5),
        xytext=(22.8, 19.4),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": COLOR_MUTED,
            "shrinkA": 0,
            "shrinkB": 3,
            "zorder": TEXT_ZORDER - 1,
        },
        zorder=TEXT_ZORDER - 1,
    )

    points = np.vstack((a_min, a_max, b_min, b_max, a_nom, b_nom))
    x_min, y_min = np.min(points, axis=0) - np.array([6.5, 5.8])
    x_max, y_max = np.max(points, axis=0) + np.array([6.2, 6.0])
    y_mid = (y_min + y_max) / 2.0
    y_range = (y_max - y_min) * LATERAL_VERTICAL_RANGE_MULTIPLIER
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_mid - y_range / 2.0), float(y_mid + y_range / 2.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Lateral projection", fontsize=10, weight="bold")
    ax.set_xlabel("East (NMI)")
    ax.set_ylabel("North (NMI)")
    ax.grid(True, color="#e5e7eb", lw=0.6)
    ax.tick_params(labelsize=7)
    ax.text(
        0.98,
        0.055,
        r"lateral check: $0 \leq t \leq t_v$",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#374151",
        ha="right",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )


def draw_vertical_panel(ax: Axes, vertical_resolution_s: float) -> None:
    """Draw the conservative vertical-band concept."""
    horizon_min = 8.0
    times_s = np.linspace(0.0, horizon_min * 60.0, 241)
    times_min = times_s / 60.0

    a_level = np.array(
        [level_at_time_fl(A_CURRENT_FL, A_SELECTED_FL, CONSERVATIVE_VERTICAL_RATE_FPM, float(t)) for t in times_s]
    )
    b_conservative = np.array(
        [level_at_time_fl(B_CURRENT_FL, B_SELECTED_FL, CONSERVATIVE_VERTICAL_RATE_FPM, float(t)) for t in times_s]
    )
    b_real = np.array(
        [level_at_time_fl(B_CURRENT_FL, B_SELECTED_FL, REALISED_VERTICAL_RATE_FPM, float(t)) for t in times_s]
    )
    b_band_low = np.minimum(b_conservative, B_SELECTED_FL)
    b_band_high = np.maximum(b_conservative, B_SELECTED_FL)

    ax.axhspan(A_CURRENT_FL - 10.0, A_CURRENT_FL + 10.0, color=COLOR_CONFLICT, alpha=0.09, zorder=0)
    ax.fill_between(
        times_min,
        b_band_low,
        b_band_high,
        color=COLOR_ENVELOPE,
        alpha=0.22,
        label="Conservative vertical band",
        zorder=1,
    )
    ax.plot(times_min, a_level, color=COLOR_A, lw=1.9, label="Aircraft A level", zorder=4)
    ax.plot(times_min, b_conservative, color=COLOR_ENVELOPE, lw=2.0, label="B at 1000 ft/min", zorder=5)
    ax.plot(times_min, b_real, color=COLOR_B, lw=1.6, ls="--", label="Possible realised climb", zorder=5)
    ax.axhline(B_SELECTED_FL, color=COLOR_ENVELOPE, lw=0.9, ls=":", alpha=0.9)
    ax.axhline(A_CURRENT_FL + 10.0, color=COLOR_CONFLICT, lw=0.9, ls=":", alpha=0.8)
    ax.axhline(A_CURRENT_FL - 10.0, color=COLOR_CONFLICT, lw=0.9, ls=":", alpha=0.8)

    vertical_resolution_min = vertical_resolution_s / 60.0
    ax.axvline(vertical_resolution_min, color=COLOR_SAFE, lw=1.3, ls="-.", zorder=3)
    ax.text(
        vertical_resolution_min + 0.08,
        B_SELECTED_FL - 6.5,
        rf"$t_v={vertical_resolution_min:.0f}$ min",
        color=COLOR_SAFE,
        fontsize=8.0,
        va="center",
        weight="bold",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.annotate(
        "minimum-rate band\nshrinks slowly",
        xy=(5.0, B_SELECTED_FL),
        xytext=(0.7, 357.0),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLOR_MUTED, "zorder": TEXT_ZORDER - 1},
        fontsize=7.2,
        color="#374151",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.annotate(
        "real climb may\nclear earlier",
        xy=(3.0, 310.0),
        xytext=(4.6, 328.0),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLOR_MUTED, "zorder": TEXT_ZORDER - 1},
        fontsize=7.2,
        color="#374151",
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )
    ax.text(
        0.08,
        A_CURRENT_FL + 11.8,
        "10 FL required gap",
        color=COLOR_CONFLICT,
        fontsize=7.5,
        zorder=TEXT_ZORDER,
        path_effects=TEXT_EFFECTS,
    )

    ax.set_xlim(0.0, horizon_min)
    ax.set_ylim(244.0, 378.5)
    ax.set_title("Vertical projection", fontsize=10, weight="bold")
    ax.set_xlabel("Time after clearance (min)")
    ax.set_ylabel("Flight level")
    ax.grid(True, color="#e5e7eb", lw=0.6)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.94, edgecolor="#d1d5db")


def render(output_pdf: FilePath, output_png: FilePath | None) -> None:
    """Render the figure to disk."""
    vertical_resolution_s = time_to_vertical_overlap_resolution(
        A_CURRENT_FL,
        A_SELECTED_FL,
        B_CURRENT_FL,
        B_SELECTED_FL,
        vertical_rate_fpm=CONSERVATIVE_VERTICAL_RATE_FPM,
        required_gap_fl=10.0,
        rounded=10,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_lateral, ax_vertical) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    draw_lateral_panel(ax_lateral, vertical_resolution_s)
    draw_vertical_panel(ax_vertical, vertical_resolution_s)
    shared_box_aspect = ax_lateral.get_data_ratio()
    ax_lateral.set_box_aspect(shared_box_aspect)
    ax_vertical.set_box_aspect(shared_box_aspect)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", type=FilePath, required=True)
    parser.add_argument("--output-png", type=FilePath)
    args = parser.parse_args()
    render(args.output_pdf, args.output_png)


if __name__ == "__main__":
    main()
