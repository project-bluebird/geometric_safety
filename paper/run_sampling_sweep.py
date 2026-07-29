"""Sampling-interval sweep for the uniform-sampling baseline.

The manuscript's TR-C reviews asked how the uniform-sampling baseline behaves
across sampling intervals, not only at the radar-scan-aligned 6 s step, and how
the published counterexample encounter was found. This runner answers both. For
each sampling interval it scores the baseline against the dense reference on the
broad mixed-turn suite and reports miss counts alongside the work of the
certified solver measured in fixed-time hull evaluations. Encounters where the
baseline is false-safe are recorded verbatim, which is exactly the search that
produced the fixed counterexample in ``paper.evaluation``.

    uv run python -m paper.run_sampling_sweep --quick \
        --output-dir paper_eval_outputs/sampling_sweep
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from paper.evaluation import (
    AMBIGUITY_BAND_M,
    Encounter,
    EvaluationConfig,
    count_fixed_time_evals,
    evaluate_coarse_turn_proxy,
    evaluate_dense_reference,
    generate_mixed_turn_suite,
)

DEFAULT_SEED = 20260325
DEFAULT_SEED_STEP = 10_000
DEFAULT_N_ENCOUNTERS = 20_000
QUICK_N_ENCOUNTERS = 200
SAMPLE_DTS_S = (0.5, 1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 12.0)
CERT_DT_MIN_S = 3.0


def _evaluate_chunk(args: tuple[list[tuple[int, Encounter]], tuple[float, ...]]) -> list[dict[str, Any]]:
    """Worker: dense reference, per-interval baseline outcomes, and solver work."""
    indexed_encounters, sample_dts_s = args
    rows = []
    for encounter_idx, encounter in indexed_encounters:
        reference = evaluate_dense_reference(encounter)
        evals, certified = count_fixed_time_evals(
            encounter, dt_min_s=CERT_DT_MIN_S, use_interval_local_lipschitz=True
        )
        row: dict[str, Any] = {
            "encounter_idx": encounter_idx,
            "projection_time_s": encounter.projection_time_s,
            "reference_is_safe": int(reference.is_safe),
            "reference_margin_m": reference.margin_m,
            "eligible": int(abs(reference.margin_m) >= AMBIGUITY_BAND_M),
            "certified_is_safe": int(certified.is_safe),
            "certified_fixed_time_evals": evals,
        }
        for dt_s in sample_dts_s:
            outcome = evaluate_coarse_turn_proxy(encounter, sample_dt_s=dt_s)
            row[f"proxy_is_safe_dt{dt_s:g}"] = int(outcome.is_safe)
            row[f"proxy_margin_m_dt{dt_s:g}"] = outcome.margin_m
        rows.append(row)
    return rows


def summarise(rows: list[dict[str, Any]], sample_dts_s: tuple[float, ...]) -> list[dict[str, Any]]:
    """Aggregate miss counts and work per sampling interval."""
    eligible = [row for row in rows if row["eligible"]]
    unsafe = [row for row in eligible if not row["reference_is_safe"]]
    certified_evals = [row["certified_fixed_time_evals"] for row in eligible]
    summaries = []
    for dt_s in sample_dts_s:
        false_safe = sum(
            1 for row in unsafe if row[f"proxy_is_safe_dt{dt_s:g}"]
        )
        false_unsafe = sum(
            1
            for row in eligible
            if row["reference_is_safe"] and not row[f"proxy_is_safe_dt{dt_s:g}"]
        )
        evals_per_encounter = statistics.mean(
            row["projection_time_s"] / dt_s + 1.0 for row in eligible
        )
        summaries.append(
            {
                "sample_dt_s": dt_s,
                "n_eligible": len(eligible),
                "n_reference_unsafe": len(unsafe),
                "proxy_false_safe_n": false_safe,
                "proxy_false_unsafe_n": false_unsafe,
                "proxy_mean_evals": evals_per_encounter,
                "certified_false_safe_n": sum(
                    1 for row in unsafe if row["certified_is_safe"]
                ),
                "certified_median_evals": float(statistics.median(certified_evals)),
                "certified_p95_evals": float(np.percentile(np.asarray(certified_evals, dtype=np.float64), 95.0)),
            }
        )
    return summaries


def collect_misses(
    rows: list[dict[str, Any]],
    encounters: list[Encounter],
    sample_dts_s: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Record every false-safe baseline encounter verbatim.

    This search over the mixed-turn generator is how the manuscript's fixed
    sampling-miss counterexample was obtained, so its output documents the
    construction and the observed frequency in one artifact.
    """
    misses = []
    for row in rows:
        if not row["eligible"] or row["reference_is_safe"]:
            continue
        missed_dts = [dt_s for dt_s in sample_dts_s if row[f"proxy_is_safe_dt{dt_s:g}"]]
        if not missed_dts:
            continue
        record: dict[str, Any] = {
            "missed_sample_dts_s": missed_dts,
            "reference_margin_m": row["reference_margin_m"],
        }
        record.update(asdict(encounters[row["encounter_idx"]]))
        misses.append(record)
    return misses


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_eval_outputs/sampling_sweep"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument("--n-encounters", type=int, default=DEFAULT_N_ENCOUNTERS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    n_encounters = QUICK_N_ENCOUNTERS if args.quick else args.n_encounters
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Warm the compiled kernels once so worker forks inherit them where possible.
    from paper.evaluation import warm_up_numba

    warm_up_numba(EvaluationConfig())

    pool = ProcessPoolExecutor(max_workers=args.n_jobs) if args.n_jobs > 1 else None
    for repeat in range(args.repeat_seeds):
        seed = args.seed + repeat * args.seed_step
        run_dir = args.output_dir / f"run_{repeat:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        encounters = generate_mixed_turn_suite(n_encounters, seed)
        indexed = list(enumerate(encounters))
        if pool is None:
            rows = _evaluate_chunk((indexed, SAMPLE_DTS_S))
        else:
            chunk_size = max(len(indexed) // (args.n_jobs * 2), 1)
            tasks = [
                (indexed[start : start + chunk_size], SAMPLE_DTS_S)
                for start in range(0, len(indexed), chunk_size)
            ]
            rows = []
            for chunk_rows in pool.map(_evaluate_chunk, tasks):
                rows.extend(chunk_rows)

        summaries = summarise(rows, SAMPLE_DTS_S)
        misses = collect_misses(rows, encounters, SAMPLE_DTS_S)
        elapsed_s = time.perf_counter() - started

        write_rows_csv(run_dir / "sampling_sweep_rows.csv", rows)
        write_rows_csv(run_dir / "sampling_sweep_summary.csv", summaries)
        payload = {
            "seed": seed,
            "n_encounters": n_encounters,
            "cert_dt_min_s": CERT_DT_MIN_S,
            "elapsed_s": elapsed_s,
            "summaries": summaries,
            "false_safe_encounters": misses,
        }
        (run_dir / "sampling_sweep_summary.json").write_text(json.dumps(payload, indent=2))
        total_misses = sum(entry["proxy_false_safe_n"] for entry in summaries)
        print(
            f"run {repeat}: seed={seed} encounters={n_encounters} elapsed={elapsed_s:.1f}s "
            f"proxy_false_safe_total={total_misses} recorded_miss_encounters={len(misses)}"
        )
    if pool is not None:
        pool.shutdown()


if __name__ == "__main__":
    main()
