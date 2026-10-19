"""Render the robustness-study figures from campaign summaries.

Reads the per-seed ``robustness_summary.json`` files of a campaign output tree
(full or compact) and renders the exceedance degradation curves and the
misspecification comparison. Aggregation across seeds is encounter-weighted.

    uv run python -m paper.render_robustness_figures \
        --results-dir paper_eval_outputs/robustness_full_20260720 \
        --output-dir paper_eval_outputs/robustness_figures
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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

import matplotlib.pyplot as plt

EXCEEDANCE_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EXCEEDANCE_CLASSES = (
    ("wind_5kt", "Differential wind 5 kt", "#0072B2"),
    ("heading_1deg", "Heading estimate 1$^\\circ$", "#D55E00"),
    ("position_150m", "Relative position 150 m", "#009E73"),
    ("execution_5s_0.1", "Turn execution 5 s, 10%", "#CC79A7"),
    ("schedule_10kt", "In-turn speed 10 kt", "#56B4E9"),
)
MISSPEC_GROUPS = (
    ("roll", "Roll-in transient", ("3", "6", "10"), "s"),
    ("jitter", "Heading jitter", ("0.5", "1", "2"), "$^\\circ$"),
    ("ou", "OU speed drift", ("5", "10", "20"), " kt"),
)
FLOOR_PER_1K = 8e-3


def load_aggregates(results_dir: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Aggregate the per-seed summaries, weighted by encounter and sim counts."""
    aggregate: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"viols": 0.0, "sims": 0.0, "venc": 0.0, "n": 0.0}
    )
    summary_paths = sorted(results_dir.glob("run_*_seed_*/robustness_summary.json"))
    if not summary_paths:
        raise SystemExit(f"no run summaries under {results_dir}")
    for path in summary_paths:
        summary = json.loads(path.read_text())
        for entry in summary["summaries"]:
            bucket = aggregate[(entry["campaign"], entry["label"], entry["suite"])]
            bucket["viols"] += entry["total_violations"]
            bucket["sims"] += entry["total_sims"]
            bucket["venc"] += entry["n_encounters_with_violation"]
            bucket["n"] += entry["n_encounters"]
    return aggregate


def rate_per_1k(bucket: dict[str, float]) -> float:
    return 1000.0 * bucket["viols"] / bucket["sims"] if bucket["sims"] else 0.0


def render_exceedance(aggregate: dict, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for key, label, colour in EXCEEDANCE_CLASSES:
        rates = [
            rate_per_1k(aggregate[("exceedance", f"{key}_x{factor:g}", "boundary")]) for factor in EXCEEDANCE_FACTORS
        ]
        plotted = [max(rate, FLOOR_PER_1K) for rate in rates]
        ax.plot(EXCEEDANCE_FACTORS, plotted, marker="o", ms=4.5, lw=1.6, color=colour, label=label)

    ax.axvline(1.0, color="#6b7280", lw=1.0, ls=":")
    ax.text(
        0.985,
        0.03,
        "stated bound",
        fontsize=8.2,
        color="#6b7280",
        ha="right",
        va="bottom",
        rotation=90,
    )
    ax.set_yscale("log")
    ax.set_ylim(FLOOR_PER_1K * 0.8, 90.0)
    ax.set_yticks([0.01, 0.1, 1.0, 10.0])
    ax.set_yticklabels(["0", "0.1", "1", "10"])
    ax.set_xlabel("Applied perturbation relative to stated bound", fontsize=9.5)
    ax.set_ylabel("Violations per 1000 simulations", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.6)
    ax.grid(True, color="#d7dbe0", lw=0.6, alpha=0.7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#aeb7c2")
        ax.spines[spine].set_linewidth(0.8)
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def render_misspec(aggregate: dict, output_pdf: Path) -> None:
    labels = []
    plain_rates = []
    proxy_rates = []
    for prefix, title, magnitudes, unit in MISSPEC_GROUPS:
        for magnitude in magnitudes:
            proxy_suffix = {"roll": "proxy_delay", "jitter": "proxy_heading", "ou": "proxy_schedule"}[prefix]
            unit_key = {"roll": "s", "jitter": "deg", "ou": "kt"}[prefix]
            base = f"{prefix}_{magnitude}{unit_key}"
            labels.append(f"{title} {magnitude}{unit}")
            plain_rates.append(rate_per_1k(aggregate[("misspec", f"{base}_plain", "boundary")]))
            proxy_rates.append(rate_per_1k(aggregate[("misspec", f"{base}_{proxy_suffix}", "boundary")]))

    positions = list(range(len(labels)))[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for pos, plain, proxy in zip(positions, plain_rates, proxy_rates, strict=True):
        plain_x = max(plain, FLOOR_PER_1K)
        proxy_x = max(proxy, FLOOR_PER_1K)
        ax.plot([proxy_x, plain_x], [pos, pos], color="#9ca3af", lw=1.2, zorder=3)
        ax.plot(plain_x, pos, "o", ms=6.0, color="#C73E1D", zorder=5)
        ax.plot(proxy_x, pos, "o", ms=6.0, mfc="white", mec="#0072B2", mew=1.6, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(FLOOR_PER_1K * 0.8, 120.0)
    ax.set_xticks([0.01, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Violations per 1000 simulations", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.grid(True, axis="x", color="#d7dbe0", lw=0.6, alpha=0.7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#aeb7c2")
        ax.spines[spine].set_linewidth(0.8)
    ax.plot([], [], "o", ms=6.0, color="#C73E1D", label="plain certificate")
    ax.plot([], [], "o", ms=6.0, mfc="white", mec="#0072B2", mew=1.6, label="proxy-inflated certificate")
    ax.legend(fontsize=7.8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate = load_aggregates(args.results_dir)
    render_exceedance(aggregate, args.output_dir / "fig_robustness_exceedance.pdf")
    render_misspec(aggregate, args.output_dir / "fig_robustness_misspec.pdf")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
