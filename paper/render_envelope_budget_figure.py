"""Render the envelope budget against the separation threshold over horizon.

Every value here is a direct evaluation of the manuscript's closed-form
envelopes; no simulation is involved. The figure answers the operational
question the paper otherwise leaves implicit: how much of the protected
threshold the total allowance consumes as look-ahead grows, and where the
inflated certificate stops being able to certify anything.

The representative encounter follows the manuscript's worked example with a
flyable turn: one aircraft makes a 60 degree turn at the bank-limited rate
for its speed, the other flies straight, both at 300 kt. The stated bound
set is the manuscript's mid configuration (5 kt wind, 150 m position,
1 degree heading, 5 s / 10 percent execution); the adverse set (15 kt,
300 m, 3 degrees, 10 s / 20 percent) is the parameter regime a skeptical
operational reading would insist on. Both totals use only the classes whose
composition the manuscript's lemma licenses.

    uv run python -m paper.render_envelope_budget_figure \
        --output paper_eval_outputs/envelope_budget/fig_envelope_budget.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometric_safety.util import NMI_TO_M
from paper.evaluation import Encounter, bank_limited_turn_rate_deg_s
from paper.robustness import PerturbationBounds, deviation_envelope_m

REPRESENTATIVE_SPEED_KT = 300.0
REPRESENTATIVE_TURN_DEG = 60.0
HORIZON_MAX_S = 1200.0

STATED_CLASSES: tuple[tuple[str, PerturbationBounds], ...] = (
    ("wind 5 kt", PerturbationBounds(differential_wind_kt=5.0)),
    ("position 150 m", PerturbationBounds(position_error_m=150.0)),
    ("heading 1$^\\circ$", PerturbationBounds(heading_error_deg=1.0)),
    ("execution 5 s / 10%", PerturbationBounds(turn_delay_max_s=5.0, turn_rate_error_max=0.10)),
)
STATED_TOTAL = PerturbationBounds(
    differential_wind_kt=5.0,
    position_error_m=150.0,
    heading_error_deg=1.0,
    turn_delay_max_s=5.0,
    turn_rate_error_max=0.10,
)
ADVERSE_TOTAL = PerturbationBounds(
    differential_wind_kt=15.0,
    position_error_m=300.0,
    heading_error_deg=3.0,
    turn_delay_max_s=10.0,
    turn_rate_error_max=0.20,
)


def representative_encounter() -> Encounter:
    """One turning and one straight aircraft at a flyable rate, 300 kt each."""
    turn_rate = min(3.0, bank_limited_turn_rate_deg_s(REPRESENTATIVE_SPEED_KT))
    return Encounter(
        rel_east_m=15.0 * NMI_TO_M,
        rel_north_m=0.0,
        a_heading0_deg=0.0,
        a_target_heading_deg=REPRESENTATIVE_TURN_DEG,
        a_speed_kt=REPRESENTATIVE_SPEED_KT,
        a_turn_rate_deg_sec=turn_rate,
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_speed_kt=REPRESENTATIVE_SPEED_KT,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=0.0,
        projection_time_s=HORIZON_MAX_S,
    )


def build_budget_curves(dt_s: float = 5.0) -> dict[str, np.ndarray]:
    """Evaluate every envelope curve in NMI over the horizon grid."""
    encounter = representative_encounter()
    times_s = np.arange(0.0, HORIZON_MAX_S + dt_s, dt_s)
    curves: dict[str, np.ndarray] = {"times_s": times_s}
    for name, bounds in STATED_CLASSES:
        curves[name] = np.asarray(
            [deviation_envelope_m(encounter, bounds, float(t)) / NMI_TO_M for t in times_s]
        )
    curves["stated total"] = np.asarray(
        [deviation_envelope_m(encounter, STATED_TOTAL, float(t)) / NMI_TO_M for t in times_s]
    )
    curves["adverse total"] = np.asarray(
        [deviation_envelope_m(encounter, ADVERSE_TOTAL, float(t)) / NMI_TO_M for t in times_s]
    )
    return curves


PLOT_HORIZON_S = 600.0


def render(curves: dict[str, np.ndarray], output_pdf: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    times_s = curves["times_s"]
    mask = times_s <= PLOT_HORIZON_S + 1e-9
    t = times_s[mask]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    class_colours = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    series = [
        (name, curves[name][mask], colour, 1.2)
        for (name, _), colour in zip(STATED_CLASSES, class_colours, strict=True)
    ]
    series.append(("total, all four", curves["stated total"][mask], "#1f2a44", 2.2))
    for _name, values, colour, lw in series:
        ax.plot(t, values, color=colour, lw=lw)

    # Direct labels at the right edge instead of a legend box, nudged apart
    # where curve endpoints nearly coincide.
    ends = sorted((float(values[-1]), name, colour) for name, values, colour, _ in series)
    placed_y = -10.0
    for y, name, colour in ends:
        label_y = max(y, placed_y + 0.32)
        placed_y = label_y
        ax.annotate(
            name,
            xy=(float(t[-1]), y),
            xytext=(float(t[-1]) + 12.0, label_y),
            fontsize=8.0,
            color=colour,
            va="center",
            annotation_clip=False,
        )

    ax.axhline(5.0, color="#6b7280", lw=1.0, ls=":")
    ax.axhline(3.0, color="#9ca3af", lw=1.0, ls=":")
    ax.text(15.0, 5.08, "D = 5 NMI", fontsize=8.0, color="#6b7280")
    ax.text(15.0, 3.08, "D = 3 NMI", fontsize=8.0, color="#9ca3af")
    ax.set_xlim(0.0, PLOT_HORIZON_S)
    ax.set_ylim(0.0, 5.6)
    ax.set_xlabel("Look-ahead time (s)", fontsize=9.5)
    ax.set_ylabel("Envelope allowance (NMI)", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.6)
    ax.grid(True, color="#d7dbe0", lw=0.6, alpha=0.7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_color("#aeb7c2")
        ax.spines[spine].set_linewidth(0.8)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_eval_outputs/envelope_budget/fig_envelope_budget.pdf"),
    )
    args = parser.parse_args(argv)
    curves = build_budget_curves()
    render(curves, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
