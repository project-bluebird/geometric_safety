"""Render the introductory paper schematic with two protected envelopes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

NMI_TO_M = 1852.0
M_TO_NMI = 1.0 / NMI_TO_M
KT_TO_MPS = 0.5144444444444445

A_COLOUR = "#0072B2"
B_COLOUR = "#333333"
A_FILL = "#56B4E9"
B_FILL = "#8a8f98"
CONFLICT_FILL = "#C73E1D"
MUTED_COLOUR = "#6b7280"
TEXT_EFFECTS = [path_effects.withStroke(linewidth=3.0, foreground="white")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("paper_eval_outputs/application_figures/fig_lateral_overlap_schematic.pdf"),
        help="PDF output path for the paper figure.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional PNG preview path.",
    )
    return parser.parse_args()


def wrap_delta_deg(target_deg: float, initial_deg: float) -> float:
    """Return the signed heading increment on [-180, 180)."""
    return (target_deg - initial_deg + 180.0) % 360.0 - 180.0


def unit_heading_rad(heading_rad: float) -> np.ndarray:
    """Unit vector for headings measured clockwise from north."""
    return np.array([np.sin(heading_rad), np.cos(heading_rad)], dtype=float)


def displacement_basis(
    times_s: np.ndarray,
    initial_heading_deg: float,
    target_heading_deg: float,
    turn_rate_deg_s: float,
) -> np.ndarray:
    """Return the one-turn displacement basis in metres per metre/second."""
    h0 = np.deg2rad(initial_heading_deg)
    htar = np.deg2rad(target_heading_deg)
    delta_deg = wrap_delta_deg(target_heading_deg, initial_heading_deg)
    if abs(delta_deg) < 1e-12:
        return times_s[:, None] * unit_heading_rad(h0)[None, :]

    omega = np.sign(delta_deg) * np.deg2rad(abs(turn_rate_deg_s))
    tau = abs(np.deg2rad(delta_deg)) / abs(omega)
    basis_tau = (1.0 / omega) * np.array([np.cos(h0) - np.cos(h0 + omega * tau), np.sin(h0 + omega * tau) - np.sin(h0)])
    target_unit = unit_heading_rad(htar)

    basis = np.empty((times_s.size, 2), dtype=float)
    active = times_s <= tau
    t_active = times_s[active]
    basis[active, 0] = (np.cos(h0) - np.cos(h0 + omega * t_active)) / omega
    basis[active, 1] = (np.sin(h0 + omega * t_active) - np.sin(h0)) / omega
    basis[~active, :] = basis_tau[None, :] + (times_s[~active, None] - tau) * target_unit[None, :]
    return basis


def sample_trajectory_nm(
    xy0_nm: tuple[float, float],
    initial_heading_deg: float,
    target_heading_deg: float,
    speed_kt: float,
    turn_rate_deg_s: float,
    times_s: np.ndarray,
) -> np.ndarray:
    basis_m = displacement_basis(times_s, initial_heading_deg, target_heading_deg, turn_rate_deg_s)
    xy0_m = np.asarray(xy0_nm, dtype=float) * NMI_TO_M
    return (xy0_m[None, :] + speed_kt * KT_TO_MPS * basis_m) * M_TO_NMI


def heading_from_delta_nm(delta_xy: np.ndarray) -> float:
    """Return aviation heading in degrees for an east--north displacement."""
    return float(np.rad2deg(np.arctan2(delta_xy[0], delta_xy[1])) % 360.0)


def capsule_vertices(start_xy: np.ndarray, end_xy: np.ndarray, radius: float, samples: int = 160) -> np.ndarray:
    """Return polygon vertices for a capsule in data coordinates."""
    delta = end_xy - start_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        angles = np.linspace(0.0, 2.0 * np.pi, 2 * samples, endpoint=False)
        return start_xy + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    theta = float(np.arctan2(delta[1], delta[0]))
    end_angles = np.linspace(theta + np.pi / 2.0, theta - np.pi / 2.0, samples)
    start_angles = np.linspace(theta - np.pi / 2.0, theta - 3.0 * np.pi / 2.0, samples)
    end_arc = end_xy + radius * np.column_stack((np.cos(end_angles), np.sin(end_angles)))
    start_arc = start_xy + radius * np.column_stack((np.cos(start_angles), np.sin(start_angles)))
    return np.vstack((end_arc, start_arc))


def polygon_area(vertices: np.ndarray) -> float:
    """Return the signed area of a closed polygon."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    return vertices if polygon_area(vertices) >= 0.0 else vertices[::-1]


def segment_intersection(
    start: np.ndarray,
    end: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    """Return the intersection of two infinite lines defined by segment endpoints."""
    direction = end - start
    clip_direction = clip_end - clip_start
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(float(denominator)) < 1e-12:
        return end
    offset = clip_start - start
    t = (offset[0] * clip_direction[1] - offset[1] * clip_direction[0]) / denominator
    return start + t * direction


def convex_polygon_intersection(subject_vertices: np.ndarray, clip_vertices: np.ndarray) -> np.ndarray:
    """Clip one convex polygon by another using Sutherland--Hodgman clipping."""
    output = ensure_ccw(subject_vertices)
    clip = ensure_ccw(clip_vertices)

    for clip_start, clip_end in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        if output.size == 0:
            return output

        clip_edge = clip_end - clip_start

        def inside(
            point: np.ndarray,
            *,
            clip_start: np.ndarray = clip_start,
            clip_edge: np.ndarray = clip_edge,
        ) -> bool:
            rel = point - clip_start
            return float(clip_edge[0] * rel[1] - clip_edge[1] * rel[0]) >= -1e-10

        input_vertices = output
        output_points: list[np.ndarray] = []
        start = input_vertices[-1]
        start_inside = inside(start)
        for end in input_vertices:
            end_inside = inside(end)
            if end_inside:
                if not start_inside:
                    output_points.append(segment_intersection(start, end, clip_start, clip_end))
                output_points.append(end)
            elif start_inside:
                output_points.append(segment_intersection(start, end, clip_start, clip_end))
            start = end
            start_inside = end_inside
        output = np.asarray(output_points, dtype=float)

    return output


def polygon_patch(vertices: np.ndarray, **kwargs: object) -> PathPatch:
    """Return a closed patch from polygon vertices."""
    closed = np.vstack((vertices, vertices[0]))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (closed.shape[0] - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(closed, codes), **kwargs)


def capsule_patch(start_xy: np.ndarray, end_xy: np.ndarray, radius: float, **kwargs: object) -> Circle | PathPatch:
    """Return a circular or capsule-shaped patch in data coordinates."""
    vertices = capsule_vertices(start_xy, end_xy, radius)
    if vertices.shape[0] > 72 and float(np.linalg.norm(end_xy - start_xy)) < 1e-9:
        return Circle(tuple(start_xy), radius=radius, **kwargs)
    return polygon_patch(vertices, **kwargs)


def annotate(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    colour: str = "#374151",
    ha: str = "left",
) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha=ha,
        va="center",
        fontsize=8.3,
        color=colour,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 3},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )


def render_figure() -> Figure:
    horizon_s = 6.0 * 60.0
    times_s = np.linspace(0.0, horizon_s, 260)
    sample_times_s = np.array([0.0, 3.0 * 60.0, horizon_s])
    threshold_nm = 5.0
    half_threshold_nm = threshold_nm / 2.0

    a_nom = sample_trajectory_nm((0.0, 0.0), 0.0, 70.0, 300.0, 1.0, times_s)
    a_samples = sample_trajectory_nm((0.0, 0.0), 0.0, 70.0, 300.0, 1.0, sample_times_s)
    a_slow_samples = sample_trajectory_nm((0.0, 0.0), 0.0, 70.0, 275.0, 1.0, sample_times_s)
    a_fast_samples = sample_trajectory_nm((0.0, 0.0), 0.0, 70.0, 325.0, 1.0, sample_times_s)

    final_a = a_samples[-1]
    final_b_nominal = final_a + np.array([-6.0, 2.5])
    b_start_xy = final_b_nominal + np.array([-13.5, 11.5])
    b_delta_xy = final_b_nominal - b_start_xy
    b_heading = heading_from_delta_nm(b_delta_xy)
    b_speed = float(np.linalg.norm(b_delta_xy) / (horizon_s / 3600.0))
    b_start = (float(b_start_xy[0]), float(b_start_xy[1]))
    b_nom = sample_trajectory_nm(b_start, b_heading, b_heading, b_speed, 1.0, times_s)
    b_samples = sample_trajectory_nm(b_start, b_heading, b_heading, b_speed, 1.0, sample_times_s)
    b_slow_samples = sample_trajectory_nm(b_start, b_heading, b_heading, b_speed - 22.0, 1.0, sample_times_s)
    b_fast_samples = sample_trajectory_nm(b_start, b_heading, b_heading, b_speed + 22.0, 1.0, sample_times_s)

    fig, ax = plt.subplots(figsize=(7.4, 4.35))

    for idx in range(sample_times_s.size):
        alpha = 0.18 if idx == 1 else 0.24
        for centre, colour in ((a_samples[idx], A_COLOUR), (b_samples[idx], B_COLOUR)):
            ax.add_patch(
                Circle(
                    tuple(centre),
                    radius=half_threshold_nm,
                    facecolor="none",
                    edgecolor=colour,
                    lw=0.9,
                    linestyle=(0, (2.0, 2.0)),
                    alpha=alpha,
                    zorder=1,
                )
            )

    for idx in range(sample_times_s.size):
        linestyle = ":" if idx == 0 else "-"
        alpha = 0.09 + 0.045 * idx
        ax.add_patch(
            capsule_patch(
                a_slow_samples[idx],
                a_fast_samples[idx],
                half_threshold_nm,
                facecolor=A_FILL,
                edgecolor=A_COLOUR,
                lw=1.1,
                linestyle=linestyle,
                alpha=alpha,
                zorder=2,
            )
        )
        ax.add_patch(
            capsule_patch(
                b_slow_samples[idx],
                b_fast_samples[idx],
                half_threshold_nm,
                facecolor=B_FILL,
                edgecolor=B_COLOUR,
                lw=1.1,
                linestyle=linestyle,
                alpha=alpha,
                zorder=2,
            )
        )

    points = np.vstack((a_nom, b_nom, a_slow_samples, a_fast_samples, b_slow_samples, b_fast_samples))
    x_min, y_min = np.min(points, axis=0) - np.array([4.0, half_threshold_nm + 1.0])
    x_max, y_max = np.max(points, axis=0) + np.array([5.8, half_threshold_nm + 3.2])

    a_final_vertices = capsule_vertices(a_slow_samples[-1], a_fast_samples[-1], half_threshold_nm, samples=220)
    b_final_vertices = capsule_vertices(b_slow_samples[-1], b_fast_samples[-1], half_threshold_nm, samples=220)
    overlap_vertices = convex_polygon_intersection(a_final_vertices, b_final_vertices)
    if overlap_vertices.size:
        ax.add_patch(
            polygon_patch(
                overlap_vertices,
                facecolor=CONFLICT_FILL,
                edgecolor="#8C2D04",
                lw=1.0,
                alpha=0.72,
                joinstyle="round",
                zorder=5,
            )
        )

    ax.plot(a_nom[:, 0], a_nom[:, 1], color=A_COLOUR, lw=2.1, zorder=8)
    ax.plot(b_nom[:, 0], b_nom[:, 1], color=B_COLOUR, lw=1.9, zorder=8)

    for a_xy, b_xy in zip(a_samples, b_samples, strict=True):
        ax.plot([a_xy[0], b_xy[0]], [a_xy[1], b_xy[1]], color="#9ca3af", lw=0.85, ls=":", alpha=0.72, zorder=7)

    ax.scatter(a_samples[:, 0], a_samples[:, 1], s=38, color=A_COLOUR, edgecolor="white", linewidth=0.8, zorder=9)
    ax.scatter(b_samples[:, 0], b_samples[:, 1], s=38, color=B_COLOUR, edgecolor="white", linewidth=0.8, zorder=9)
    ax.scatter([a_samples[0, 0]], [a_samples[0, 1]], s=58, color=A_COLOUR, edgecolor="white", linewidth=0.9, zorder=10)
    ax.scatter([b_samples[0, 0]], [b_samples[0, 1]], s=58, color=B_COLOUR, edgecolor="white", linewidth=0.9, zorder=10)

    ax.text(
        a_samples[0, 0],
        a_samples[0, 1] - 1.05,
        "A",
        color=A_COLOUR,
        weight="bold",
        fontsize=11.0,
        ha="center",
        va="top",
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    ax.text(
        b_samples[0, 0],
        b_samples[0, 1] + 0.55,
        "B",
        color=B_COLOUR,
        weight="bold",
        fontsize=11.0,
        ha="center",
        va="bottom",
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )

    annotate(
        ax,
        "bounded protected\nenvelopes",
        xy=(a_samples[1, 0] - 1.15, a_samples[1, 1] - 2.05),
        xytext=(3.2, 1.8),
    )
    annotate(
        ax,
        "matched-time\nseparations",
        xy=tuple((a_samples[1] + b_samples[1]) / 2.0),
        xytext=(2.8, 17.4),
    )
    annotate(
        ax,
        "nominally safe;\nunsafe under\nspeed bounds",
        xy=(final_a[0] - 2.3, final_a[1] + 1.6),
        xytext=(final_a[0] - 3.6, final_a[1] + 9.4),
        colour="#8C2D04",
        ha="left",
    )

    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (NMI)", fontsize=10.5)
    ax.set_ylabel("North (NMI)", fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, color="#d7dbe0", lw=0.6, alpha=0.7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#aeb7c2")
        ax.spines[spine].set_linewidth(0.8)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    fig = render_figure()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    if args.output_png:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
