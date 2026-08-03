"""Exact worst-case in-turn speed deviation via support-function duality.

The manuscript's in-turn speed-variation proposition states a deviation envelope
together with a conservatism remark. This module computes the exact worst case
over the full measurable speed-profile class, not a discretisation of it, and
writes the comparison against the proven envelope and the superseded buffer as
a pipeline artefact.

The computation: for one aircraft with unit turn duration and unit excursion
bound, the deviation of a speed profile ``f`` with ``f(xi)`` in ``[-beta, alpha]``
from the constant-speed position is ``v = integral f(xi) u(phi(xi)) dxi``. The
distance from the perturbed position to the admissible constant-speed segment is
a maximum over unit directions ``g`` of ``g . v`` minus the segment's support,
and the supremum over schedules of ``g . v`` is attained by the schedule that
holds ``alpha`` where ``g . u`` is positive and ``-beta`` where it is negative.
Swapping the two maximisations gives the exact worst case as a maximum over a
direction grid of one-dimensional integrals, with no schedule discretisation.

    uv run python -m paper.run_schedule_enumeration \
        --output paper_eval_outputs/schedule_enumeration/schedule_enumeration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

THETA_GRID_DEG = tuple(range(10, 181, 10))
T_OVER_TAU_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 20.0)
SLACK_GRID = ((8.0, 8.0), (8.0, 1.0), (1.0, 1.0), (8.0, 0.0), (0.5, 0.5))
N_DIRECTIONS = 720
N_XI = 2000


def unit_speed_basis(theta_rad: float, t_over_tau: float) -> np.ndarray:
    """Unit-speed displacement at normalised time t for a unit-duration turn."""
    omega = theta_rad
    t = t_over_tau
    if omega < 1e-12:
        return np.array([0.0, t])
    if t <= 1.0:
        return np.array([(1.0 - math.cos(omega * t)) / omega, math.sin(omega * t) / omega])
    turn_end = np.array([(1.0 - math.cos(omega)) / omega, math.sin(omega) / omega])
    return turn_end + (t - 1.0) * np.array([math.sin(omega), math.cos(omega)])


def exact_worst_case(
    theta_rad: float,
    t_over_tau: float,
    slack_lo: float,
    slack_hi: float,
) -> float:
    """Exact supremum of the deviation distance, in units of the excursion bound.

    ``slack_lo`` and ``slack_hi`` are the distances from the realised speed to the
    interval edges in units of the excursion bound. The in-turn schedule may move
    by at most one unit but never beyond the interval, so its pointwise range is
    ``[-min(1, slack_lo), min(1, slack_hi)]``, and the comparison constant speed
    may shift by any amount inside ``[-slack_lo, slack_hi]``.
    """
    t_active = min(t_over_tau, 1.0)
    if t_active <= 0.0:
        return 0.0
    alpha = min(1.0, slack_hi)
    beta = min(1.0, slack_lo)

    xi = np.linspace(0.0, t_active, N_XI + 1)
    phi = theta_rad * xi
    u = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    b_t = unit_speed_basis(theta_rad, t_over_tau)

    angles = np.linspace(0.0, 2.0 * math.pi, N_DIRECTIONS, endpoint=False)
    directions = np.stack([np.sin(angles), np.cos(angles)], axis=1)

    k = directions @ u.T
    positive = np.clip(k, 0.0, None)
    negative = np.clip(-k, 0.0, None)
    gains = np.trapezoid(alpha * positive + beta * negative, xi, axis=1)

    projection = directions @ b_t
    segment_support = np.maximum(slack_hi * projection, -slack_lo * projection)
    return float(np.max(gains - segment_support).clip(min=0.0))


def proven_envelope(theta_rad: float, t_over_tau: float, slack_lo: float, slack_hi: float) -> float:
    """The manuscript's envelope in the same normalised units."""
    t_active = min(t_over_tau, 1.0)
    sweep = theta_rad * t_active
    if t_active <= 0.0 or sweep < 1e-12:
        return 0.0
    if sweep <= 0.5 * math.pi + 1e-9:
        return t_active * (1.0 - math.cos(sweep)) / sweep
    width = slack_lo + slack_hi
    amplitude = min(2.0, width)
    return amplitude * 4.0 * t_active * (1.0 - math.cos(0.5 * sweep)) / sweep


def sharp_envelope(theta_rad: float, t_over_tau: float) -> float:
    """The sharp (1 - cos)/theta formula regardless of proof scope."""
    t_active = min(t_over_tau, 1.0)
    sweep = theta_rad * t_active
    if t_active <= 0.0 or sweep < 1e-12:
        return 0.0
    return t_active * (1.0 - math.cos(sweep)) / sweep


def sine_buffer(theta_rad: float, t_over_tau: float) -> float:
    """The coarser sin(theta/2) buffer in the same units."""
    t_active = min(t_over_tau, 1.0)
    sweep = theta_rad * t_active
    return t_active * math.sin(0.5 * sweep)


def run_grid() -> list[dict[str, Any]]:
    rows = []
    for theta_deg in THETA_GRID_DEG:
        theta_rad = math.radians(theta_deg)
        for t_over_tau in T_OVER_TAU_GRID:
            for slack_lo, slack_hi in SLACK_GRID:
                exact = exact_worst_case(theta_rad, t_over_tau, slack_lo, slack_hi)
                envelope = proven_envelope(theta_rad, t_over_tau, slack_lo, slack_hi)
                sharp = sharp_envelope(theta_rad, t_over_tau)
                buffer_value = sine_buffer(theta_rad, t_over_tau)
                sweep_rad = theta_rad * min(t_over_tau, 1.0)
                rows.append(
                    {
                        "theta_deg": theta_deg,
                        "t_over_tau": t_over_tau,
                        "slack_lo": slack_lo,
                        "slack_hi": slack_hi,
                        "sweep_deg": math.degrees(sweep_rad),
                        "in_proof_scope": int(sweep_rad <= 0.5 * math.pi + 1e-9),
                        "exact_worst": exact,
                        "proven_envelope": envelope,
                        "sharp_envelope": sharp,
                        "sine_buffer": buffer_value,
                        "exact_over_envelope": exact / envelope if envelope > 0.0 else 0.0,
                        "exact_over_sharp": exact / sharp if sharp > 0.0 else 0.0,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_eval_outputs/schedule_enumeration/schedule_enumeration.json"),
    )
    args = parser.parse_args(argv)

    rows = run_grid()
    worst_ratio = max(row["exact_over_envelope"] for row in rows)
    worst_in_scope = max(row["exact_over_envelope"] for row in rows if row["in_proof_scope"])
    worst_sharp = max(row["exact_over_sharp"] for row in rows)
    long_horizon = [row for row in rows if row["t_over_tau"] == max(T_OVER_TAU_GRID) and row["slack_lo"] >= 1.0]
    tightest = max(row["exact_over_envelope"] for row in long_horizon)

    payload = {
        "n_directions": N_DIRECTIONS,
        "n_xi": N_XI,
        "worst_exact_over_envelope": worst_ratio,
        "worst_exact_over_envelope_in_scope": worst_in_scope,
        "worst_exact_over_sharp_all_sweeps": worst_sharp,
        "long_horizon_max_ratio": tightest,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"grid points: {len(rows)}")
    print(f"max exact/envelope overall: {worst_ratio:.6f} (validity requires <= 1)")
    print(f"max exact/envelope in proof scope: {worst_in_scope:.6f}")
    print(f"max exact/sharp over all sweeps: {worst_sharp:.6f} (remark claims <= 1)")
    print(f"long-horizon max ratio (tightness): {tightest:.6f}")


if __name__ == "__main__":
    main()
