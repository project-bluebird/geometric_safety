"""Render the usage guide's figures from its runnable examples and shared geometry."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples import heading_filter, uncertainty_bounds, vertical_progress
from geometric_safety import CombinedHeadingAssuranceResult, HeadingMotion
from geometric_safety.demo_utils import sample_trajectory_xy
from geometric_safety.util import KT_TO_MPS, NMI_TO_M, heading_to_unit_vector, latlon_to_local_xy
from geometric_safety.vertical import level_at_time_fl

BLUE = "#2563b4"
AMBER = "#ad6b00"
INK = "#243247"
MUTED = "#66758a"
GREEN = "#176b4a"
RED = "#b52b39"
STATUS_COLORS = {"certified_safe": GREEN, "unresolved": RED, "unsupported": MUTED}
RADAR_BACKGROUND = "#101e2d"
RADAR_TEXT = "#e5edf5"
RADAR_MUTED = "#a5b6c9"
RADAR_A = "#80bdff"
RADAR_B = "#f0bc69"


def style_axes(ax: Axes, title: str) -> None:
    """Use a consistent, unobtrusive frame for quantitative panels."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5eaf0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)


def radar_canvas(
    title: str,
    limits_nm: tuple[float, float, float, float],
    origin: str,
    notes: tuple[tuple[str, str], ...],
) -> tuple[Figure, Axes]:
    """Lay out a north-up starting snapshot beside its query declarations."""
    fig = plt.figure(figsize=(12, 6.8), facecolor=RADAR_BACKGROUND)
    radar = fig.add_axes((0.08, 0.18, 0.54, 0.65), facecolor=RADAR_BACKGROUND)
    brief = fig.add_axes((0.68, 0.19, 0.30, 0.64))
    brief.axis("off")
    fig.suptitle(title, x=0.08, y=0.97, ha="left", fontsize=19, fontweight="bold", color=RADAR_TEXT)
    fig.text(0.08, 0.87, "STARTING SNAPSHOT  /  NORTH UP", color=RADAR_MUTED, fontsize=10)
    radar.set(
        xlim=limits_nm[:2], ylim=limits_nm[2:], xlabel=f"East from {origin} (NM)", ylabel=f"North from {origin} (NM)"
    )
    radar.set_aspect("equal")
    radar.grid(color="#2c4055", linewidth=0.7)
    radar.set_axisbelow(True)
    radar.tick_params(colors=RADAR_MUTED, length=0, labelsize=10, pad=7)
    radar.xaxis.label.set_color(RADAR_MUTED)
    radar.yaxis.label.set_color(RADAR_MUTED)
    for spine in radar.spines.values():
        spine.set_color("#40546b")
    radar.annotate(
        "",
        (0.06, 0.94),
        (0.06, 0.83),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": RADAR_MUTED, "lw": 1.5},
    )
    radar.text(0.06, 0.96, "N", transform=radar.transAxes, ha="center", color=RADAR_MUTED, fontsize=10)
    for index, (heading, body) in enumerate(notes):
        y = 0.98 - index / len(notes)
        brief.text(0, y, heading, va="top", fontsize=10, fontweight="bold", color=RADAR_MUTED)
        brief.text(0, y - 0.055, body, va="top", fontsize=11.5, linespacing=1.5, color=RADAR_TEXT)
    fig.text(
        0.08,
        0.035,
        "Blips show starting positions. Arrows show current velocity direction, not travel distance.",
        fontsize=10,
        color=RADAR_MUTED,
    )
    return fig, radar


def radar_track(
    ax: Axes,
    position_nm: tuple[float, float],
    heading_deg: float,
    data_block: str,
    label_nm: tuple[float, float],
    color: str,
) -> None:
    """Attach a readable data block and heading vector to one radar blip."""
    direction = heading_to_unit_vector(heading_deg)
    vector_length_nm = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.055
    tip = (position_nm[0] + direction[0] * vector_length_nm, position_nm[1] + direction[1] * vector_length_nm)
    ax.annotate("", tip, position_nm, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6}, zorder=4)
    ax.scatter(*position_nm, marker="s", s=40, facecolor=RADAR_BACKGROUND, edgecolor=color, linewidth=1.6, zorder=5)
    ax.annotate(
        data_block,
        position_nm,
        xytext=label_nm,
        va="top",
        color=color,
        fontsize=10.5,
        fontfamily="DejaVu Sans Mono",
        linespacing=1.5,
        bbox={"boxstyle": "square,pad=0.45", "facecolor": RADAR_BACKGROUND, "edgecolor": color, "linewidth": 0.6},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "shrinkA": 0, "shrinkB": 5},
        zorder=6,
    )


def heading_setup_figure() -> Figure:
    ownship, traffic = heading_filter.example_snapshot()
    targets = heading_filter.CANDIDATE_HEADINGS_DEG
    fig, radar = radar_canvas(
        "Example 1  /  Choose a heading for A",
        (-5, 31, -6, 30),
        "A",
        (
            ("QUERY", f"5 NM separation over 180 s\nSnapshot time: {ownship.timestamp_s:,.0f} s"),
            (
                "A'S TARGET HEADINGS",
                f"{targets[0]:03d}° to {targets[-1]:03d}° in {targets[1] - targets[0]}° steps\n"
                "000° = north; 090° = east\nTurns start now at 3°/s",
            ),
            ("TRAFFIC", "B: opposing, continues 270°\nC: parallel, continues 090°"),
            ("FRAME", "A: 51.00°N, 1.40°W\nShared local east/north plane\nAltitude is not used here."),
        ),
    )
    positions: dict[str, tuple[float, float]] = {}
    aircraft = {"A": ownship, "B": traffic["opposing"], "C": traffic["parallel"]}
    labels = {"A": (2.0, 6.0), "B": (18.0, 6.0), "C": (2.0, 27.0)}
    roles = {"A": "OWNSHIP", "B": "OPPOSING", "C": "PARALLEL"}
    for identifier, snapshot in aircraft.items():
        xy = latlon_to_local_xy(
            snapshot.latitude_deg, snapshot.longitude_deg, ownship.latitude_deg, ownship.longitude_deg
        )
        position = (float(xy[0] / NMI_TO_M), float(xy[1] / NMI_TO_M))
        positions[identifier] = position
        motion = snapshot.motion
        block = (
            f"{identifier}  {roles[identifier]}\nHDG {motion.heading0_deg:03.0f}°\n"
            f"SPD {motion.speed_min_mps / KT_TO_MPS:.0f}-{motion.speed_max_mps / KT_TO_MPS:.0f} kt"
        )
        color = RADAR_A if identifier == "A" else RADAR_B if identifier == "B" else RADAR_TEXT
        radar_track(radar, position, motion.heading0_deg, block, labels[identifier], color)
    opposing_east_nm = positions["B"][0]
    radar.annotate("", (0, -3), (opposing_east_nm, -3), arrowprops={"arrowstyle": "<->", "color": RADAR_MUTED})
    radar.text(opposing_east_nm / 2, -4.5, f"{opposing_east_nm:.1f} NM", ha="center", color=RADAR_MUTED, fontsize=10)
    return fig


def uncertainty_setup_figure() -> Figure:
    fig, radar = radar_canvas(
        "Example 2  /  Declare the position and velocity errors",
        (-6, 12, -4, 12),
        "B",
        (
            ("QUERY", "5 NM separation over 120 s\nA - B = (0, 10,000) m"),
            ("FIRST ERROR BUDGET", "Position: 80 + 120 = 200 m\nResidual velocity: 1 + 1 = 2 kt"),
            ("SECOND ERROR BUDGET", "Position: 400 + 400 = 800 m\nSame velocity errors and motions"),
            (
                "DECLARATIONS",
                "Error entries are norm bounds.\nSpeeds are absolute intervals.\nAltitude is not used here.",
            ),
        ),
    )
    north_nm = 10000.0 / NMI_TO_M
    radar_track(
        radar, (0, north_nm), 90.0, "A\nHDG 090°  SPD 190-210 kt\nPOS ERR ≤80 m\nVEL ERR ≤1 kt", (2, 9.5), RADAR_A
    )
    radar_track(radar, (0, 0), 90.0, "B\nHDG 090°  SPD 195-215 kt\nPOS ERR ≤120 m\nVEL ERR ≤1 kt", (2, 3.5), RADAR_B)
    radar.annotate("", (-2, 0), (-2, north_nm), arrowprops={"arrowstyle": "<->", "color": RADAR_MUTED})
    radar.text(
        -2.6, north_nm / 2, "10 km / 5.40 NM", rotation=90, ha="center", va="center", color=RADAR_MUTED, fontsize=10
    )
    return fig


def vertical_setup_figure() -> Figure:
    speed_label = f"{vertical_progress.SPEED_MIN_KT:g}-{vertical_progress.SPEED_MAX_KT:g} kt"
    fig, radar = radar_canvas(
        "Example 3  /  A holds level while B climbs",
        (-8, 48, -15, 21),
        "B",
        (
            ("QUERY", "5 NM lateral / 10 FL vertical\nFull decision horizon: 600 s"),
            (
                "B'S CLIMB GUARANTEE",
                "Start now; ≥1,000 ft/min\nNo overshoot of FL120\nRemain at FL120 once reached",
            ),
            ("SPEED BOUNDS", f"{speed_label} for each aircraft"),
            ("WITHOUT THAT GUARANTEE", "Use the lateral-only query\nfor the full 600 seconds."),
        ),
    )
    east_nm = 60000.0 / NMI_TO_M
    radar_track(radar, (0, 0), 90.0, f"B\nHDG 090°\nFL100 → FL120\nSPD {speed_label}", (0, 14), RADAR_B)
    radar_track(radar, (east_nm, 0), 270.0, f"A\nHDG 270°\nFL100 → FL100\nSPD {speed_label}", (29, 14), RADAR_A)
    radar.annotate("", (0, -5), (east_nm, -5), arrowprops={"arrowstyle": "<->", "color": RADAR_MUTED})
    radar.text(east_nm / 2, -8, f"60 km / {east_nm:.2f} NM", ha="center", color=RADAR_MUTED, fontsize=11)
    return fig


def nominal_path(
    motion: HeadingMotion, times_s: NDArray[np.float64], start_m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample one midpoint-speed path for illustration, not the reachable set."""
    return (
        sample_trajectory_xy(
            start_m,
            motion.heading0_deg,
            motion.target_heading_deg,
            (motion.speed_min_mps + motion.speed_max_mps) / 2.0,
            motion.turn_rate_deg_s,
            times_s,
        )
        / NMI_TO_M
    )


def heading_figure() -> Figure:
    """Show the discrete candidate paths and each pair's actual decision."""
    ownship, traffic = heading_filter.example_snapshot()
    assessments = heading_filter.run_example()
    passing = heading_filter.passing_candidates(assessments)
    fig = plt.figure(figsize=(12, 7.5), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.1, 1))
    plan = fig.add_subplot(grid[0, 0])
    decisions = fig.add_subplot(grid[0, 1])
    fig.suptitle("Filter a range of heading choices", x=0.07, ha="left", fontweight="bold", fontsize=19)
    times = np.linspace(0.0, 180.0, 181)
    label_offsets = {0.0: (-16, 12), 30.0: (4, 10), 60.0: (10, 5), 90.0: (5, -15)}
    for heading in assessments:
        motion = replace(ownship.motion, target_heading_deg=heading)
        path = nominal_path(motion, times, np.zeros(2))
        accepted = heading in passing
        color = BLUE if accepted else RED
        plan.plot(path[:, 0], path[:, 1], color=color, linestyle="-" if accepted else "--", linewidth=1.6, zorder=3)
        plan.scatter(*path[-1], marker="o" if accepted else "x", color=color, s=22, zorder=4)
        if heading in label_offsets:
            plan.annotate(
                f"{heading:03.0f}°",
                path[-1],
                xytext=label_offsets[heading],
                textcoords="offset points",
                color=color,
                fontsize=10,
            )
    plan.scatter(0.0, 0.0, color=BLUE, zorder=5)
    plan.annotate("A starts on 090°", (0, 0), xytext=(0, -2), color=BLUE, fontsize=11)
    for name, color in (("opposing", AMBER), ("parallel", MUTED)):
        aircraft = traffic[name]
        start = latlon_to_local_xy(
            aircraft.latitude_deg, aircraft.longitude_deg, ownship.latitude_deg, ownship.longitude_deg
        )
        path = nominal_path(aircraft.motion, times, start)
        plan.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5)
        plan.scatter(*path[0], color=color, zorder=4)
        plan.scatter(*path[-1], marker="s", facecolor="white", edgecolor=color, zorder=4)
        plan.annotate("", xy=path[120], xytext=path[85], arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
    style_axes(plan, "Ten candidate paths over 180 seconds")
    plan.set(xlim=(-2, 18), ylim=(-3, 24), xlabel="East from A at start (NM)", ylabel="North from A at start (NM)")
    plan.set_aspect("equal")
    plan.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", label="A: passes both pairs"),
            Line2D([], [], color=RED, linestyle="--", marker="x", label="A: unresolved"),
            Line2D([], [], color=AMBER, label="B: opposing"),
            Line2D([], [], color=MUTED, label="C: parallel"),
        ],
        loc="center right",
        frameon=False,
        fontsize=10.5,
    )
    plan.text(0.02, 0.98, "Endpoints show 180 seconds", transform=plan.transAxes, va="top", color=MUTED, fontsize=10)

    decisions.axis("off")
    decisions.set_title("Each heading must pass both pairs", loc="left", fontsize=13, fontweight="bold", pad=14)
    for x, label in ((0.02, "Target"), (0.48, "B: opposing"), (0.84, "C: parallel")):
        decisions.text(x, 0.92, label, transform=decisions.transAxes, ha="left" if x == 0.02 else "center", color=MUTED)
    for row, (heading, pairs) in enumerate(assessments.items()):
        y = 0.82 - row * 0.075
        decisions.text(0.02, y, f"{heading:03.0f}°", transform=decisions.transAxes, va="center", fontsize=12)
        for x, name in ((0.48, "opposing"), (0.84, "parallel")):
            status = pairs[name].status
            decisions.text(
                x,
                y,
                status.replace("certified_safe", "certified"),
                transform=decisions.transAxes,
                ha="center",
                va="center",
                fontsize=10.5,
                color=STATUS_COLORS[status],
                fontweight="bold",
            )
    decisions.text(
        0.02,
        0.02,
        "Passing headings\n" + ", ".join(f"{heading:03.0f}°" for heading in passing),
        transform=decisions.transAxes,
        color=GREEN,
        fontsize=11,
        va="bottom",
        linespacing=1.5,
    )
    fig.text(
        0.07,
        0.025,
        "Paths show 200 kt; decisions cover 190 to 210 kt. Only the sampled target headings are assessed.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.87, wspace=0.35)
    return fig


def uncertainty_figure() -> Figure:
    """Zoom into the separation boundary for a common-speed allowed trajectory."""
    results = uncertainty_bounds.run_example()
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.3), facecolor="white")
    fig.suptitle("The same gap can pass or remain unresolved", x=0.08, ha="left", fontweight="bold", fontsize=19)
    east_km = np.linspace(-1.2, 1.2, 401)
    boundary_km = np.sqrt((5 * NMI_TO_M / 1000) ** 2 - east_km**2)
    for ax, result, position_m in zip(axes, results.values(), (200, 800), strict=True):
        radius_km = result.inflation_m / 1000
        ax.fill_between(east_km, 8.75, boundary_km, color=RED, alpha=0.10)
        ax.plot(east_km, boundary_km, color=RED, linewidth=1.5)
        ax.add_patch(Circle((0, 10), radius_km, facecolor=BLUE, edgecolor=BLUE, alpha=0.18, linewidth=2))
        ax.add_patch(Circle((0, 10), radius_km, fill=False, edgecolor=BLUE, linewidth=1.8))
        ax.scatter(0, 10, color=INK, s=22, zorder=4)
        ax.annotate("Nominal A - B", (0, 10), xytext=(10, 8), textcoords="offset points", fontsize=10.5)
        ax.text(0, 8.88, "Inside 5 NM of B", ha="center", color=RED, fontsize=11)
        ax.text(0, 11.08, f"{result.inflation_m:.2f} m total allowance", ha="center", color=BLUE, fontsize=11)
        style_axes(ax, f"{position_m} m initial relative-position error")
        ax.set(
            xlim=(-1.2, 1.2),
            ylim=(8.75, 11.25),
            xlabel="Relative east position (km)",
            ylabel="Relative north position (km)",
        )
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([9, 10, 11])
        ax.set_aspect("equal")
        ax.text(
            0.5,
            -0.25,
            result.status,
            ha="center",
            transform=ax.transAxes,
            color=STATUS_COLORS[result.status],
            fontsize=13,
            fontweight="bold",
        )
    fig.text(
        0.08,
        0.065,
        "At 120 s: position error + residual velocity error over 120 s = position error + 123.47 m.",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.08,
        0.025,
        "Both aircraft fly at 200 kt here. Circles show additive error bounds, not the full reachable sets.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.25, top=0.83, wspace=0.35)
    return fig


def vertical_figure() -> Figure:
    """Show the remaining altitude band and the corresponding lateral horizon."""
    result = vertical_progress.assess_encounter(1000.0)
    if not isinstance(result, CombinedHeadingAssuranceResult):
        raise TypeError("The guaranteed-progress example must return a combined result")
    fallback = vertical_progress.assess_encounter(None)
    resolution_s = result.lateral_horizon_s
    times = np.linspace(0.0, 600.0, 601)
    lower_fl = np.array([level_at_time_fl(100.0, 120.0, 1000.0, float(t)) for t in times])
    fig, (altitude, lateral) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, facecolor="white")
    fig.suptitle(
        "Guaranteed vertical progress can shorten the lateral check", x=0.08, ha="left", fontweight="bold", fontsize=18
    )
    altitude.fill_between(times, lower_fl, 120.0, color=AMBER, alpha=0.23)
    altitude.plot(times, lower_fl, color=AMBER, linewidth=2.5)
    altitude.axhline(120, color=AMBER, linewidth=1.5, linestyle="--")
    altitude.axhline(100, color=BLUE, linewidth=2.5)
    altitude.text(210, 118, "B: FL120 selected; no overshoot", color=AMBER, fontsize=11)
    altitude.text(210, 101.5, "A: level at FL100", color=BLUE, fontsize=11)
    altitude.annotate(
        "B's remaining band",
        (35, 116),
        xytext=(140, 112),
        arrowprops={"arrowstyle": "->", "color": AMBER},
        color=AMBER,
        fontsize=11,
    )
    altitude.annotate(
        "", (resolution_s, 110), (resolution_s, 100), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.5}
    )
    altitude.text(77, 104, "10 FL gap\nat 60 s", fontsize=11)
    style_axes(altitude, "B progresses at ≥1,000 ft/min until reaching its selected level")
    altitude.set(ylim=(97, 124), ylabel="Flight level")
    altitude.set_yticks([100, 110, 120])

    # Head-on closing speeds add; after crossing, distance is the absolute relative position.
    initial_distance_m = 60000.0
    speed_min_mps = vertical_progress.SPEED_MIN_KT * KT_TO_MPS
    speed_max_mps = vertical_progress.SPEED_MAX_KT * KT_TO_MPS
    nominal_speed_kt = (vertical_progress.SPEED_MIN_KT + vertical_progress.SPEED_MAX_KT) / 2
    nominal_closing_speed_mps = 2 * nominal_speed_kt * KT_TO_MPS
    low_signed_km = (initial_distance_m - 2 * speed_max_mps * times) / 1000
    high_signed_km = (initial_distance_m - 2 * speed_min_mps * times) / 1000
    minimum_km = np.maximum(np.maximum(low_signed_km, -high_signed_km), 0)
    maximum_km = np.maximum(np.abs(low_signed_km), np.abs(high_signed_km))
    lateral.fill_between(
        times,
        minimum_km,
        maximum_km,
        color=MUTED,
        alpha=0.17,
        label=(
            "Distances allowed by "
            f"{vertical_progress.SPEED_MIN_KT:g} to {vertical_progress.SPEED_MAX_KT:g} kt per aircraft"
        ),
    )
    lateral.plot(
        times,
        np.abs(initial_distance_m - nominal_closing_speed_mps * times) / 1000,
        color=MUTED,
        linewidth=1.6,
        linestyle="--",
        label=f"One allowed trajectory: {nominal_speed_kt:g} kt each",
    )
    checked = times <= resolution_s
    lateral.plot(times[checked], minimum_km[checked], color=BLUE, linewidth=3)
    lateral.axhline(5 * NMI_TO_M / 1000, color=RED, linewidth=1.5)
    lateral.text(465, 11.5, "5 NM required", color=RED, fontsize=10.5)
    meeting_time_s = initial_distance_m / nominal_closing_speed_mps
    lateral.scatter(meeting_time_s, 0, color=RED, zorder=4)
    lateral.annotate(
        f"Paths can meet in plan view\nat about {meeting_time_s:.0f} s",
        (meeting_time_s, 0),
        xytext=(335, 24),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10.5,
    )
    minimum_at_resolution_km = (initial_distance_m - 2 * speed_max_mps * resolution_s) / 1000
    lateral.annotate(
        f"{minimum_at_resolution_km:.1f} km minimum\nat {resolution_s:g} s",
        (resolution_s, minimum_at_resolution_km),
        xytext=(110, 52),
        arrowprops={"arrowstyle": "->", "color": BLUE},
        color=BLUE,
        fontsize=11,
    )
    style_axes(lateral, f"Combined check: {result.status}   ·   Full-horizon lateral fallback: {fallback.status}")
    lateral.set(
        xlim=(0, 600),
        ylim=(-3, float(maximum_km.max()) + 3),
        ylabel="Lateral distance (km)",
        xlabel="Seconds from the common starting time",
    )
    lateral.set_xticks([0, 60, 120, 240, 360, 480, 600])
    lateral.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in (altitude, lateral):
        ax.axvspan(0, resolution_s, color=BLUE, alpha=0.07)
        ax.axvline(resolution_s, color=BLUE, linewidth=1.2, linestyle=":")
    fig.text(
        0.08,
        0.025,
        "Blue shading: check lateral separation through 60 s. The declared climb then provides the vertical gap.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.86, hspace=0.42)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "figures")
    parser.add_argument("--preview-dir", type=Path, help="Also write PNGs here for visual inspection")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#b5c0cd",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "path",
            "svg.hashsalt": "geometric-safety-usage-guide",
        }
    )
    for name, render in (
        ("heading-setup", heading_setup_figure),
        ("uncertainty-setup", uncertainty_setup_figure),
        ("vertical-setup", vertical_setup_figure),
        ("heading-filter", heading_figure),
        ("uncertainty-bounds", uncertainty_figure),
        ("vertical-progress", vertical_figure),
    ):
        fig = render()
        svg_path = args.output_dir / f"{name}.svg"
        fig.savefig(svg_path, metadata={"Date": None}, facecolor=fig.get_facecolor())
        svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
        svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
        if args.preview_dir is not None:
            fig.savefig(args.preview_dir / f"{name}.png", dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Rendered {name}")


if __name__ == "__main__":
    main()
