"""Render the three-panel method-geometry figure for the paper.

Panel (a) shows the straight-heading relative-displacement hull and the
closest-approach projection. Panel (b) shows fixed-time reachable hulls along
a one-turn encounter in the relative frame. Panel (c) shows the worst-case
distance over time together with the interval-certification lower bounds
produced by the adaptive search.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

NMI_TO_M = 1852.0
M_TO_NMI = 1.0 / NMI_TO_M
KT_TO_MPS = 0.5144444444444445

A_COLOUR = "#0072B2"
A_FILL = "#56B4E9"
B_COLOUR = "#333333"
HULL_FILL = "#56B4E9"
HULL_EDGE = "#0072B2"
CERT_COLOUR = "#009E73"
CONFLICT_COLOUR = "#C73E1D"
MUTED_COLOUR = "#6b7280"
TEXT_EFFECTS = [path_effects.withStroke(linewidth=3.0, foreground="white")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("paper_eval_outputs/application_figures/fig_method_geometry.pdf"),
        help="PDF output path for the paper figure.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional PNG preview path.",
    )
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help="Print scenario diagnostics (minimum distance, certification intervals).",
    )
    return parser.parse_args()


def wrap_delta_deg(target_deg: float, initial_deg: float) -> float:
    """Return the signed heading increment on [-180, 180)."""
    return (target_deg - initial_deg + 180.0) % 360.0 - 180.0


def unit_heading_rad(heading_rad: float) -> np.ndarray:
    """Unit vector for headings measured clockwise from north."""
    return np.array([np.sin(heading_rad), np.cos(heading_rad)], dtype=float)


def heading_profile_rad(
    times_s: np.ndarray,
    initial_heading_deg: float,
    target_heading_deg: float,
    turn_rate_deg_s: float,
) -> np.ndarray:
    """Return the one-turn heading profile in radians at each sample time."""
    h0 = np.deg2rad(initial_heading_deg)
    delta_deg = wrap_delta_deg(target_heading_deg, initial_heading_deg)
    if abs(delta_deg) < 1e-12:
        return np.full(times_s.shape, h0)
    omega = np.sign(delta_deg) * np.deg2rad(abs(turn_rate_deg_s))
    tau = abs(np.deg2rad(delta_deg)) / abs(omega)
    return h0 + omega * np.minimum(times_s, tau)


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
    basis_tau = (1.0 / omega) * np.array(
        [np.cos(h0) - np.cos(h0 + omega * tau), np.sin(h0 + omega * tau) - np.sin(h0)]
    )
    target_unit = unit_heading_rad(htar)

    basis = np.empty((times_s.size, 2), dtype=float)
    active = times_s <= tau
    t_active = times_s[active]
    basis[active, 0] = (np.cos(h0) - np.cos(h0 + omega * t_active)) / omega
    basis[active, 1] = (np.sin(h0 + omega * t_active) - np.sin(h0)) / omega
    basis[~active, :] = basis_tau[None, :] + (times_s[~active, None] - tau) * target_unit[None, :]
    return basis


def convex_hull_ccw(points: np.ndarray) -> np.ndarray:
    """Return the convex hull of a small point set in counter-clockwise order."""
    unique = np.unique(points, axis=0)
    if unique.shape[0] < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def half_hull(candidates: np.ndarray) -> list[np.ndarray]:
        hull: list[np.ndarray] = []
        for point in candidates:
            while len(hull) >= 2:
                cross = (hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (
                    point[0] - hull[-2][0]
                )
                if cross <= 1e-12:
                    hull.pop()
                else:
                    break
            hull.append(point)
        return hull

    lower = half_hull(ordered)
    upper = half_hull(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def project_point_to_segment(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Return the closest point to ``point`` on the segment from ``seg_a`` to ``seg_b``."""
    edge = seg_b - seg_a
    denom = float(edge @ edge)
    if denom < 1e-15:
        return seg_a
    fraction = float(np.clip((point - seg_a) @ edge / denom, 0.0, 1.0))
    return seg_a + fraction * edge


def point_in_convex_polygon(point: np.ndarray, vertices_ccw: np.ndarray) -> bool:
    """Return True if ``point`` lies inside the counter-clockwise polygon."""
    for start, end in zip(vertices_ccw, np.roll(vertices_ccw, -1, axis=0), strict=True):
        edge = end - start
        rel = point - start
        if float(edge[0] * rel[1] - edge[1] * rel[0]) < -1e-10:
            return False
    return True


def project_point_to_convex_polygon(point: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Euclidean projection of ``point`` onto the polygon and its distance."""
    hull = convex_hull_ccw(vertices)
    if hull.shape[0] >= 3 and point_in_convex_polygon(point, hull):
        return point.copy(), 0.0
    best_point = hull[0]
    best_distance = float(np.linalg.norm(point - hull[0]))
    boundary = hull if hull.shape[0] >= 2 else np.vstack((hull, hull))
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0), strict=True):
        candidate = project_point_to_segment(point, start, end)
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_point = candidate
            best_distance = distance
    return best_point, best_distance


@dataclass(frozen=True)
class TurnScenario:
    """One-turn encounter expressed in the relative frame."""

    rel_start_nm: tuple[float, float]
    a_heading0_deg: float
    a_target_heading_deg: float
    a_turn_rate_deg_s: float
    a_speed_bounds_kt: tuple[float, float]
    b_heading0_deg: float
    b_target_heading_deg: float
    b_turn_rate_deg_s: float
    b_speed_bounds_kt: tuple[float, float]
    horizon_s: float
    threshold_nm: float

    def speed_corners_mps(self) -> list[tuple[float, float]]:
        a_lo, a_hi = (value * KT_TO_MPS for value in self.a_speed_bounds_kt)
        b_lo, b_hi = (value * KT_TO_MPS for value in self.b_speed_bounds_kt)
        return [(a_lo, b_lo), (a_hi, b_lo), (a_hi, b_hi), (a_lo, b_hi)]

    def corner_positions_nm(self, times_s: np.ndarray) -> np.ndarray:
        """Return relative positions (n_corners, n_times, 2) in nautical miles."""
        basis_a = displacement_basis(times_s, self.a_heading0_deg, self.a_target_heading_deg, self.a_turn_rate_deg_s)
        basis_b = displacement_basis(times_s, self.b_heading0_deg, self.b_target_heading_deg, self.b_turn_rate_deg_s)
        rel0_m = np.asarray(self.rel_start_nm, dtype=float) * NMI_TO_M
        positions = [
            (rel0_m[None, :] + speed_a * basis_a - speed_b * basis_b) * M_TO_NMI
            for speed_a, speed_b in self.speed_corners_mps()
        ]
        return np.asarray(positions)

    def distance_nm(self, time_s: float) -> float:
        corners = self.corner_positions_nm(np.array([time_s]))[:, 0, :]
        _, distance = project_point_to_convex_polygon(np.zeros(2), corners)
        return distance

    def interval_lipschitz_nm_s(self, t0_s: float, t1_s: float, samples: int = 96) -> float:
        """Interval-local bound on |d'| from the corner trajectories, in NMI per second."""
        times = np.linspace(t0_s, t1_s, samples)
        heading_a = heading_profile_rad(times, self.a_heading0_deg, self.a_target_heading_deg, self.a_turn_rate_deg_s)
        heading_b = heading_profile_rad(times, self.b_heading0_deg, self.b_target_heading_deg, self.b_turn_rate_deg_s)
        best = 0.0
        for speed_a, speed_b in self.speed_corners_mps():
            velocity_east = speed_a * np.sin(heading_a) - speed_b * np.sin(heading_b)
            velocity_north = speed_a * np.cos(heading_a) - speed_b * np.cos(heading_b)
            best = max(best, float(np.max(np.hypot(velocity_east, velocity_north))))
        return best * M_TO_NMI


@dataclass(frozen=True)
class CertifiedInterval:
    """One leaf interval of the adaptive certification search."""

    t0_s: float
    t1_s: float
    d0_nm: float
    d1_nm: float
    lipschitz_nm_s: float
    certified: bool


def adaptive_certification(scenario: TurnScenario, dt_min_s: float = 3.0) -> list[CertifiedInterval]:
    """Emulate the adaptive certified search and return its leaf intervals in time order."""
    leaves: list[CertifiedInterval] = []

    def visit(t0_s: float, t1_s: float, d0_nm: float, d1_nm: float) -> None:
        lipschitz = scenario.interval_lipschitz_nm_s(t0_s, t1_s)
        lower_bound = 0.5 * (d0_nm + d1_nm - lipschitz * (t1_s - t0_s))
        if lower_bound >= scenario.threshold_nm:
            leaves.append(CertifiedInterval(t0_s, t1_s, d0_nm, d1_nm, lipschitz, certified=True))
            return
        if (t1_s - t0_s) <= dt_min_s:
            leaves.append(CertifiedInterval(t0_s, t1_s, d0_nm, d1_nm, lipschitz, certified=False))
            return
        t_mid = 0.5 * (t0_s + t1_s)
        d_mid = scenario.distance_nm(t_mid)
        visit(t0_s, t_mid, d0_nm, d_mid)
        visit(t_mid, t1_s, d_mid, d1_nm)

    visit(0.0, scenario.horizon_s, scenario.distance_nm(0.0), scenario.distance_nm(scenario.horizon_s))
    return leaves


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.975,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        weight="bold",
        va="top",
        ha="left",
        path_effects=TEXT_EFFECTS,
        zorder=40,
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d7dbe0", lw=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8.6)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#aeb7c2")
        ax.spines[spine].set_linewidth(0.8)


def polygon_patch(vertices: np.ndarray, **kwargs: object) -> PathPatch:
    closed = np.vstack((vertices, vertices[0]))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (closed.shape[0] - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(closed, codes), **kwargs)


def draw_straight_panel(ax: plt.Axes) -> None:
    """Panel (a): straight-heading displacement hull and closest-approach projection."""
    heading_a_deg = 35.0
    heading_b_deg = 305.0
    speeds_a_kt = (235.0, 275.0)
    speeds_b_kt = (185.0, 225.0)
    horizon_s = 300.0

    unit_a = unit_heading_rad(np.deg2rad(heading_a_deg))
    unit_b = unit_heading_rad(np.deg2rad(heading_b_deg))
    corner_displacements_nm = np.asarray(
        [
            horizon_s * (speed_a * KT_TO_MPS * unit_a - speed_b * KT_TO_MPS * unit_b) * M_TO_NMI
            for speed_a in speeds_a_kt
            for speed_b in speeds_b_kt
        ]
    )
    hull_points = np.vstack((np.zeros((1, 2)), corner_displacements_nm))
    hull = convex_hull_ccw(hull_points)

    centroid = hull.mean(axis=0)
    query_point = centroid + np.array([1.0, 9.5])
    projection, _ = project_point_to_convex_polygon(query_point, hull_points)

    ax.add_patch(polygon_patch(hull, facecolor=HULL_FILL, edgecolor=HULL_EDGE, lw=1.2, alpha=0.30, zorder=2))
    ax.scatter(
        corner_displacements_nm[:, 0],
        corner_displacements_nm[:, 1],
        s=22,
        color=HULL_EDGE,
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
    )
    ax.scatter([0.0], [0.0], s=34, color=B_COLOUR, edgecolor="white", linewidth=0.7, zorder=6)
    ax.plot(
        [query_point[0], projection[0]],
        [query_point[1], projection[1]],
        color=CONFLICT_COLOUR,
        lw=1.4,
        ls="--",
        zorder=5,
    )
    ax.scatter([query_point[0]], [query_point[1]], s=34, color=CONFLICT_COLOUR, edgecolor="white", zorder=6)
    ax.scatter([projection[0]], [projection[1]], s=26, color=CONFLICT_COLOUR, edgecolor="white", zorder=6)

    ax.text(
        -0.9,
        -1.7,
        "$0$",
        fontsize=9.5,
        ha="center",
        va="top",
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    ax.text(
        float(centroid[0] - 1.5),
        float(centroid[1] + 0.4),
        "$S(T)$",
        fontsize=10.0,
        ha="center",
        color=HULL_EDGE,
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    ax.text(
        float(query_point[0] + 0.9),
        float(query_point[1] + 0.3),
        "$-\\boldsymbol{r}_0$",
        fontsize=9.5,
        ha="left",
        color=CONFLICT_COLOUR,
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    midpoint = 0.5 * (query_point + projection)
    ax.annotate(
        "$d_*$",
        xy=tuple(midpoint),
        xytext=(midpoint[0] - 7.0, midpoint[1] + 1.8),
        fontsize=9.5,
        color=CONFLICT_COLOUR,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 2},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    corner_label = corner_displacements_nm[np.argmax(corner_displacements_nm[:, 0])]
    ax.annotate(
        "$T \\boldsymbol{v}_i$",
        xy=tuple(corner_label),
        xytext=(corner_label[0] + 1.2, corner_label[1] - 4.4),
        fontsize=9.5,
        color=HULL_EDGE,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 3},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )

    all_points = np.vstack((hull_points, query_point[None, :]))
    ax.set_xlim(float(all_points[:, 0].min()) - 3.0, float(all_points[:, 0].max()) + 5.5)
    ax.set_ylim(float(all_points[:, 1].min()) - 3.2, float(all_points[:, 1].max()) + 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Relative east displacement (NMI)", fontsize=9.2)
    ax.set_ylabel("Relative north displacement (NMI)", fontsize=9.2)
    style_axes(ax)
    panel_label(ax, "(a)")


def draw_turn_panel(ax: plt.Axes, scenario: TurnScenario, hull_times_s: np.ndarray) -> None:
    """Panel (b): fixed-time reachable hulls along a one-turn encounter."""
    dense_times = np.linspace(0.0, scenario.horizon_s, 480)
    corner_paths = scenario.corner_positions_nm(dense_times)

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius=scenario.threshold_nm,
            facecolor="none",
            edgecolor=CONFLICT_COLOUR,
            lw=1.0,
            linestyle=(0, (2.0, 2.0)),
            alpha=0.75,
            zorder=3,
        )
    )
    ax.scatter([0.0], [0.0], s=30, color=CONFLICT_COLOUR, edgecolor="white", linewidth=0.7, zorder=6)

    for corner_path in corner_paths:
        ax.plot(corner_path[:, 0], corner_path[:, 1], color=A_FILL, lw=0.8, alpha=0.85, zorder=4)

    for time_s in hull_times_s:
        corners = scenario.corner_positions_nm(np.array([time_s]))[:, 0, :]
        hull = convex_hull_ccw(corners)
        ax.add_patch(polygon_patch(hull, facecolor=HULL_FILL, edgecolor=HULL_EDGE, lw=1.1, alpha=0.32, zorder=5))

    start = np.asarray(scenario.rel_start_nm)
    ax.scatter([start[0]], [start[1]], s=40, color=A_COLOUR, edgecolor="white", linewidth=0.8, zorder=7)
    ax.text(
        float(start[0] + 1.3),
        float(start[1] + 0.2),
        "$\\boldsymbol{r}_0$",
        fontsize=9.5,
        ha="left",
        color=A_COLOUR,
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )

    closest_times = np.linspace(0.0, scenario.horizon_s, 1200)
    closest_distances = np.asarray([scenario.distance_nm(t) for t in closest_times])
    t_closest = float(closest_times[int(np.argmin(closest_distances))])
    closest_corners = scenario.corner_positions_nm(np.array([t_closest]))[:, 0, :]
    closest_point, _ = project_point_to_convex_polygon(np.zeros(2), closest_corners)
    ax.plot(
        [0.0, closest_point[0]],
        [0.0, closest_point[1]],
        color=CONFLICT_COLOUR,
        lw=1.2,
        ls="--",
        zorder=6,
    )
    ax.annotate(
        "$d(t)$",
        xy=tuple(0.55 * closest_point),
        xytext=(closest_point[0] + 3.6, 2.6),
        fontsize=9.5,
        color=CONFLICT_COLOUR,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 2},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )

    ax.annotate(
        "$R(t)$",
        xy=tuple(scenario.corner_positions_nm(np.array([hull_times_s[-3]]))[:, 0, :].mean(axis=0)),
        xytext=(6.6, -14.6),
        fontsize=10.0,
        color=HULL_EDGE,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 3},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )
    ax.annotate(
        "$\\|\\boldsymbol{r}\\| = D$",
        xy=(scenario.threshold_nm * np.sin(np.deg2rad(285.0)), scenario.threshold_nm * np.cos(np.deg2rad(285.0))),
        xytext=(-13.6, 3.4),
        fontsize=9.5,
        color=CONFLICT_COLOUR,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED_COLOUR, "shrinkA": 2, "shrinkB": 3},
        path_effects=TEXT_EFFECTS,
        zorder=30,
    )

    threshold = scenario.threshold_nm
    all_points = np.vstack(
        (
            corner_paths.reshape(-1, 2),
            np.array([[-threshold, -threshold], [threshold, threshold]]),
        )
    )
    ax.set_xlim(float(all_points[:, 0].min()) - 7.6, float(all_points[:, 0].max()) + 2.6)
    ax.set_ylim(float(all_points[:, 1].min()) - 2.2, float(all_points[:, 1].max()) + 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Relative east position (NMI)", fontsize=9.2)
    ax.set_ylabel("Relative north position (NMI)", fontsize=9.2)
    style_axes(ax)
    panel_label(ax, "(b)")


def draw_certification_panel(ax: plt.Axes, scenario: TurnScenario, leaves: list[CertifiedInterval]) -> None:
    """Panel (c): worst-case distance over time with interval-certification bounds."""
    dense_times = np.linspace(0.0, scenario.horizon_s, 600)
    distances = np.asarray([scenario.distance_nm(t) for t in dense_times])

    ax.plot(dense_times, distances, color=A_COLOUR, lw=1.7, zorder=6, label="$d(t)$")
    ax.axhline(scenario.threshold_nm, color=CONFLICT_COLOUR, lw=1.1, ls="--", zorder=4, label="$D$")

    evaluated_times = [leaves[0].t0_s] + [leaf.t1_s for leaf in leaves]
    evaluated_distances = [leaves[0].d0_nm] + [leaf.d1_nm for leaf in leaves]

    wedge_label_added = False
    for leaf in leaves:
        width = leaf.t1_s - leaf.t0_s
        apex_time = 0.5 * (leaf.t0_s + leaf.t1_s) + 0.5 * (leaf.d1_nm - leaf.d0_nm) / leaf.lipschitz_nm_s
        apex_value = 0.5 * (leaf.d0_nm + leaf.d1_nm - leaf.lipschitz_nm_s * width)
        label = None
        if not wedge_label_added:
            label = "certified lower bound"
            wedge_label_added = True
        ax.plot(
            [leaf.t0_s, apex_time, leaf.t1_s],
            [leaf.d0_nm, apex_value, leaf.d1_nm],
            color=CERT_COLOUR,
            lw=1.1,
            alpha=0.9,
            zorder=5,
            label=label,
        )

    ax.scatter(
        evaluated_times,
        evaluated_distances,
        s=20,
        color=A_COLOUR,
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
        label="evaluated $d(t_k)$",
    )

    ax.set_xlabel("Time (s)", fontsize=9.2)
    ax.set_ylabel("Distance (NMI)", fontsize=9.2)
    ax.set_xlim(0.0, scenario.horizon_s)
    ax.set_xticks(np.arange(0.0, scenario.horizon_s + 1.0, 60.0))
    y_max = float(np.max(distances)) + 1.4
    ax.set_ylim(scenario.threshold_nm - 1.2, y_max)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
    style_axes(ax)
    panel_label(ax, "(c)")


def build_scenario() -> TurnScenario:
    return TurnScenario(
        rel_start_nm=(-7.5, -17.8),
        a_heading0_deg=0.0,
        a_target_heading_deg=90.0,
        a_turn_rate_deg_s=3.0,
        a_speed_bounds_kt=(240.0, 280.0),
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_turn_rate_deg_s=0.0,
        b_speed_bounds_kt=(200.0, 240.0),
        horizon_s=240.0,
        threshold_nm=5.0,
    )


def render_figure(print_diagnostics: bool = False) -> plt.Figure:
    scenario = build_scenario()
    leaves = adaptive_certification(scenario)

    if print_diagnostics:
        dense_times = np.linspace(0.0, scenario.horizon_s, 2000)
        distances = np.asarray([scenario.distance_nm(t) for t in dense_times])
        arg_min = int(np.argmin(distances))
        print(f"min d(t) = {distances[arg_min]:.3f} NMI at t = {dense_times[arg_min]:.1f} s")
        print(f"leaf intervals: {len(leaves)}; all certified: {all(leaf.certified for leaf in leaves)}")
        for leaf in leaves:
            print(
                f"  [{leaf.t0_s:7.2f}, {leaf.t1_s:7.2f}] d0={leaf.d0_nm:6.3f} d1={leaf.d1_nm:6.3f} "
                f"L={leaf.lipschitz_nm_s:6.4f} certified={leaf.certified}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55))
    draw_straight_panel(axes[0])
    draw_turn_panel(axes[1], scenario, hull_times_s=np.array([45.0, 90.0, 135.0, 180.0, 225.0]))
    draw_certification_panel(axes[2], scenario, leaves)

    turn_x0, turn_x1 = axes[1].get_xlim()
    turn_y0, turn_y1 = axes[1].get_ylim()
    axes[2].set_box_aspect((turn_y1 - turn_y0) / (turn_x1 - turn_x0))

    fig.tight_layout(w_pad=2.2)
    return fig


def main() -> None:
    args = parse_args()
    fig = render_figure(print_diagnostics=args.print_diagnostics)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    if args.output_png:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
