"""Render the lateral clearance-evaluation grid used in the paper.

The figure uses the constant-speed turn model and its saved source grid,
independently of the interactive heading-certificate visualiser.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager, suppress
import gzip
import json
import os
from pathlib import Path
import sys
from typing import TypedDict

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SAFE_COLOUR = "#0072B2"
UNSAFE_COLOUR = "#D55E00"
A_COLOUR = "#1f4e79"
B_COLOUR = "#4b4b4b"
LABEL_COLOUR = "white"
TEXT_OUTLINE = [path_effects.withStroke(linewidth=1.8, foreground="#303030")]
NMI_TO_M = 1852.0
X_RANGE_NM = (-17.5, 10.0)
Y_RANGE_NM = (-10.0, 17.5)
REPO_ROOT = Path(__file__).resolve().parents[1]


class AircraftAnnotation(TypedDict):
    lat: float
    lon: float
    heading: float
    target_heading_deg: float
    turn_rate_deg_sec: float
    speed_kt: float


class OffsetAircraftAnnotation(AircraftAnnotation):
    east_offset_nm: float
    north_offset_nm: float


class GridStatistics(TypedDict):
    safe_percentage: float
    mean_closest_time_min: float


class GridPayload(TypedDict):
    model_mode: str
    turn_preset_id: str
    grid_density: int
    x_range_nmi: list[float]
    y_range_nmi: list[float]
    lats: list[float]
    lons: list[float]
    safe: list[list[bool]]
    stats: GridStatistics
    lat_bounds: list[float]
    lon_bounds: list[float]
    view_half_extent_nm: float
    a: AircraftAnnotation
    b: OffsetAircraftAnnotation
    speed_diff_kt: float
    projection_time_min: float
    separation_threshold_nm: float


@contextmanager
def prepended_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prefer imports from a local repository checkout."""
    path_string = str(path)
    sys.path.insert(0, path_string)
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(path_string)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path(os.environ.get("GEOMETRIC_SAFETY_ROOT", str(REPO_ROOT))),
        help="Path to the geometric_safety code repository.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("paper_eval_outputs/application_figures/fig_clearance_grid.pdf"),
        help="PDF output path for the paper figure.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional PNG preview path.",
    )
    parser.add_argument(
        "--grid-density",
        type=int,
        default=600,
        help="Number of grid samples in each lateral direction; 600 evaluates 360,000 grid cells.",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=None,
        help="Optional JSON or JSON.GZ source grid path.",
    )
    parser.add_argument(
        "--read-source-json",
        action="store_true",
        help="Render from --source-json instead of recomputing the grid.",
    )
    parser.add_argument(
        "--write-source-json",
        action="store_true",
        help="Write --source-json after computing the grid.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only write source data; do not render the PDF/PNG.",
    )
    return parser.parse_args()


def centres_to_edges(values: np.ndarray) -> np.ndarray:
    diffs = np.diff(values)
    if diffs.size == 0:
        raise ValueError("At least two grid coordinates are required.")
    mids = values[:-1] + 0.5 * diffs
    return np.concatenate(([values[0] - 0.5 * diffs[0]], mids, [values[-1] + 0.5 * diffs[-1]]))


def heading_vector(heading_deg: float, length: float) -> tuple[float, float]:
    heading_rad = np.deg2rad(heading_deg)
    return float(length * np.sin(heading_rad)), float(length * np.cos(heading_rad))


def draw_heading_arrow(
    ax: Axes,
    origin: tuple[float, float],
    heading_deg: float,
    *,
    length: float,
    colour: str,
    label: str,
    linestyle: str = "-",
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    dx, dy = heading_vector(heading_deg, length)
    ax.annotate(
        "",
        xy=(origin[0] + dx, origin[1] + dy),
        xytext=origin,
        arrowprops={
            "arrowstyle": "-|>",
            "color": colour,
            "lw": 1.8,
            "linestyle": linestyle,
            "mutation_scale": 13.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=5,
    )
    ax.text(
        origin[0] + dx + text_offset[0],
        origin[1] + dy + text_offset[1],
        label,
        color=LABEL_COLOUR,
        fontsize=10.2,
        ha="center",
        va="center",
        path_effects=TEXT_OUTLINE,
        zorder=6,
    )


def load_grid_payload(library_root: Path, grid_density: int) -> GridPayload:
    if grid_density < 2:
        raise ValueError("At least two grid coordinates are required.")
    with prepended_sys_path(library_root):
        from geometric_safety.demo_utils import TURN_DEMO_SCENARIOS
        from geometric_safety.relevant_aircraft import catch_up_projection_interval_with_turns

    scenario = next(item for item in TURN_DEMO_SCENARIOS if item.preset_id == "single_turn_dodge")
    b_lat, b_lon = scenario.b_latlon()

    x_offsets_nm = np.linspace(X_RANGE_NM[0], X_RANGE_NM[1], grid_density, dtype=np.float64)
    y_offsets_nm = np.linspace(Y_RANGE_NM[0], Y_RANGE_NM[1], grid_density, dtype=np.float64)
    ref_lat_rad = np.deg2rad(scenario.a_lat)
    cos_lat = max(float(np.cos(ref_lat_rad)), 1e-6)
    b_lats = scenario.a_lat + y_offsets_nm / 60.0
    b_lons = scenario.a_lon + x_offsets_nm / (60.0 * cos_lat)

    separation_threshold_m = scenario.separation_threshold_nm * NMI_TO_M
    projection_time_s = scenario.projection_time_min * 60.0
    safe_matrix = np.zeros((b_lats.size, b_lons.size), dtype=bool)
    time_matrix_mins = np.zeros_like(safe_matrix, dtype=np.float64)

    for i, cell_lat in enumerate(b_lats):
        for j, cell_lon in enumerate(b_lons):
            is_sep, _, closest_time_s = catch_up_projection_interval_with_turns(
                a_lat=scenario.a_lat,
                a_lon=scenario.a_lon,
                a_heading0_deg=scenario.a_heading_deg,
                a_target_heading_deg=scenario.a_target_heading_deg,
                a_speed_kt=scenario.a_speed_kt,
                a_turn_rate_deg_sec=scenario.a_turn_rate_deg_sec,
                b_lat=float(cell_lat),
                b_lon=float(cell_lon),
                b_heading0_deg=scenario.b_heading_deg,
                b_target_heading_deg=scenario.b_target_heading_deg,
                b_speed_kt=scenario.b_speed_kt,
                b_turn_rate_deg_sec=scenario.b_turn_rate_deg_sec,
                separation_threshold_m=separation_threshold_m,
                speed_diff_kt=scenario.speed_diff_kt,
                projection_time_s=projection_time_s,
                turn_speed_schedule_uncertainty_kt=0.0,
            )
            safe_matrix[i, j] = is_sep
            time_matrix_mins[i, j] = closest_time_s / 60.0

    payload: GridPayload = {
        "model_mode": "turn_aware",
        "turn_preset_id": scenario.preset_id,
        "grid_density": int(grid_density),
        "x_range_nmi": list(X_RANGE_NM),
        "y_range_nmi": list(Y_RANGE_NM),
        "lats": b_lats.tolist(),
        "lons": b_lons.tolist(),
        "safe": safe_matrix.tolist(),
        "stats": {
            "safe_percentage": float(100.0 * safe_matrix.mean()),
            "mean_closest_time_min": float(np.mean(time_matrix_mins)),
        },
        "lat_bounds": [float(b_lats[0]), float(b_lats[-1])],
        "lon_bounds": [float(b_lons[0]), float(b_lons[-1])],
        "view_half_extent_nm": scenario.view_half_extent_nm,
        "a": {
            "lat": scenario.a_lat,
            "lon": scenario.a_lon,
            "heading": scenario.a_heading_deg,
            "target_heading_deg": scenario.a_target_heading_deg,
            "turn_rate_deg_sec": scenario.a_turn_rate_deg_sec,
            "speed_kt": scenario.a_speed_kt,
        },
        "b": {
            "lat": b_lat,
            "lon": b_lon,
            "east_offset_nm": float(scenario.b_east_m) / NMI_TO_M,
            "north_offset_nm": float(scenario.b_north_m) / NMI_TO_M,
            "heading": scenario.b_heading_deg,
            "target_heading_deg": scenario.b_target_heading_deg,
            "turn_rate_deg_sec": scenario.b_turn_rate_deg_sec,
            "speed_kt": scenario.b_speed_kt,
        },
        "speed_diff_kt": scenario.speed_diff_kt,
        "projection_time_min": scenario.projection_time_min,
        "separation_threshold_nm": scenario.separation_threshold_nm,
    }
    return payload


def write_source_payload(path: Path, payload: GridPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
    else:
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def read_source_payload(path: Path) -> GridPayload:
    # This boundary reads the source schema written by write_source_payload.
    payload: GridPayload
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    return payload


def render_figure(payload: GridPayload) -> Figure:
    safe = np.asarray(payload["safe"], dtype=bool)
    lats = np.asarray(payload["lats"], dtype=np.float64)
    lons = np.asarray(payload["lons"], dtype=np.float64)

    ref_lat_rad = np.deg2rad(payload["a"]["lat"])
    x_centres_nm = (lons - payload["a"]["lon"]) * 60.0 * np.cos(ref_lat_rad)
    y_centres_nm = (lats - payload["a"]["lat"]) * 60.0
    x_edges_nm = centres_to_edges(x_centres_nm)
    y_edges_nm = centres_to_edges(y_centres_nm)

    fig, ax = plt.subplots(figsize=(6.2, 5.25))
    ax.pcolormesh(
        x_edges_nm,
        y_edges_nm,
        safe.astype(int),
        cmap=ListedColormap([UNSAFE_COLOUR, SAFE_COLOUR]),
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    ax.contour(
        x_centres_nm,
        y_centres_nm,
        safe.astype(float),
        levels=[0.5],
        colors="#1f1f1f",
        linewidths=1.0,
        zorder=3,
    )

    ax.scatter([0.0], [0.0], s=70, marker="o", facecolor="white", edgecolor=A_COLOUR, linewidth=1.5, zorder=7)
    ax.text(
        0.65,
        0.0,
        "A fixed",
        color=LABEL_COLOUR,
        fontsize=10.6,
        ha="left",
        va="center",
        path_effects=TEXT_OUTLINE,
        zorder=7,
    )
    draw_heading_arrow(
        ax,
        (0.0, 0.0),
        payload["a"]["heading"],
        length=3.8,
        colour=A_COLOUR,
        label=r"$h_A^0$",
        text_offset=(0.0, 0.9),
    )
    draw_heading_arrow(
        ax,
        (0.0, 0.0),
        payload["a"]["target_heading_deg"],
        length=3.8,
        colour=A_COLOUR,
        label=r"$h_A^{\mathrm{tar}}$",
        linestyle="--",
        text_offset=(-1.0, 0.0),
    )

    b_x_nm = payload["b"]["east_offset_nm"]
    b_y_nm = payload["b"]["north_offset_nm"]
    ax.scatter(
        [b_x_nm],
        [b_y_nm],
        s=58,
        marker="s",
        facecolor="white",
        edgecolor=B_COLOUR,
        linewidth=1.3,
        zorder=7,
    )
    draw_heading_arrow(
        ax,
        (b_x_nm, b_y_nm),
        payload["b"]["heading"],
        length=3.2,
        colour=B_COLOUR,
        label=r"$h_B$",
        text_offset=(0.9, 0.0),
    )
    label_y_nm = b_y_nm - 1.2
    ax.text(
        b_x_nm - 0.88,
        label_y_nm,
        "example",
        color=LABEL_COLOUR,
        fontsize=10.2,
        ha="right",
        va="center",
        path_effects=TEXT_OUTLINE,
        zorder=7,
    )
    ax.text(
        b_x_nm,
        label_y_nm,
        "B",
        color=LABEL_COLOUR,
        fontsize=10.2,
        ha="center",
        va="center",
        path_effects=TEXT_OUTLINE,
        zorder=7,
    )

    legend_handles = [
        Patch(facecolor=SAFE_COLOUR, edgecolor="none", label="Certified safe"),
        Patch(facecolor=UNSAFE_COLOUR, edgecolor="none", label="Not certified safe"),
        Line2D([0], [0], color="#1f1f1f", lw=1.0, label="Decision boundary"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=9.5,
        borderpad=0.55,
    )

    annotation_lines = [
        r"A clearance: $0^\circ \rightarrow 270^\circ$",
        r"B plan at each cell: $90^\circ$ straight",
        rf"Speed envelope: +/- {payload['speed_diff_kt']:.0f} kt",
        (
            rf"Horizon {payload['projection_time_min']:.0f} min; "
            rf"threshold {payload['separation_threshold_nm']:.0f} NMI"
        ),
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"facecolor": "white", "edgecolor": "#c8c8c8", "boxstyle": "round,pad=0.3", "alpha": 0.94},
        zorder=8,
    )

    ax.set_xlabel("Initial east offset of aircraft B (NMI)", fontsize=11.5)
    ax.set_ylabel("Initial north offset of aircraft B (NMI)", fontsize=11.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_xlim(X_RANGE_NM)
    ax.set_ylim(Y_RANGE_NM)
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.22, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b7b7b7")
    ax.spines["bottom"].set_color("#b7b7b7")

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    if args.read_source_json:
        if args.source_json is None:
            raise ValueError("--read-source-json requires --source-json.")
        payload = read_source_payload(args.source_json)
    else:
        payload = load_grid_payload(args.library_root, args.grid_density)
        if args.write_source_json:
            if args.source_json is None:
                raise ValueError("--write-source-json requires --source-json.")
            write_source_payload(args.source_json, payload)

    if args.skip_render:
        return

    fig = render_figure(payload)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    if args.output_png:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
