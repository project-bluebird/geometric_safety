"""Robustness campaign runner for the manuscript's out-of-model study.

Three campaign layers, run over the broad mixed-turn suite:

1. ``inclass``: executed motion drawn from the stated perturbation classes with adversarial
   draws. The propositions require zero violations of the inflated certificate, so
   this layer verifies the envelope implementation and measures the certification
   cost of inflation.
2. ``misspec``: executed motion drawn from outside every stated class (OU speed drift over the
   whole horizon, turn-rate roll-in transients, track-keeping heading jitter). Each
   configuration is scored against the plain certificate and against a certificate
   inflated with a misspecified proxy envelope. Violations are possible and
   informative.
3. ``exceedance``: applied perturbations scaled past the stated bounds while the
   certificate inflates for the stated bounds, producing graceful-degradation
   curves of violation rate against exceedance ratio.

Usage:

    uv run python -m paper.run_robustness --quick --output-dir paper_eval_outputs/robustness
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zlib

import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety.util import NMI_TO_M
from paper.evaluation import (
    DEFAULT_REF_LAT,
    DEFAULT_REF_LON,
    Encounter,
    encounter_to_row,
    encounter_to_turn_kwargs,
    evaluate_dense_reference,
    generate_mixed_turn_suite,
)
from paper.robustness import (
    ExecutionModelConfig,
    PerturbationBounds,
    deviation_envelope_m,
    evaluate_robustness_case,
    solver_envelope_kwargs,
)

DEFAULT_SEED = 20260325
DEFAULT_SEED_STEP = 10_000
DEFAULT_N_ENCOUNTERS = 2_000
DEFAULT_N_BOUNDARY = 200
DEFAULT_N_SIMS = 100
QUICK_N_ENCOUNTERS = 60
QUICK_N_BOUNDARY = 8
QUICK_N_SIMS = 10

EXCEEDANCE_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BOUNDARY_BAND_M = 0.3 * NMI_TO_M
BOUNDARY_CANDIDATE_BATCH = 512
BOUNDARY_MAX_BATCHES = 400


def _encounter_from_json_dict(payload: dict[str, Any]) -> Encounter:
    return Encounter(**{field.name: float(payload[field.name]) for field in fields(Encounter)})


def _boundary_batch_worker(args: tuple[int, PerturbationBounds, float]) -> list[Encounter]:
    """Scan one candidate batch and return the encounters that pass the boundary filter."""
    batch_seed, bounds, band_m = args
    accepted = []
    for encounter in generate_mixed_turn_suite(BOUNDARY_CANDIDATE_BATCH, batch_seed):
        kwargs = encounter_to_turn_kwargs(encounter, DEFAULT_REF_LAT, DEFAULT_REF_LON)
        inflated = ra.catch_up_projection_interval_with_turns(**kwargs, **solver_envelope_kwargs(encounter, bounds))
        if not bool(inflated[0]):
            continue
        reference = evaluate_dense_reference(encounter)
        eta_at_critical_m = deviation_envelope_m(encounter, bounds, reference.closest_time_s)
        if reference.margin_m <= eta_at_critical_m + band_m:
            accepted.append(encounter)
    return accepted


def _boundary_cache_path(cache_dir: Path, bounds: PerturbationBounds, seed: int, band_m: float) -> Path:
    key = json.dumps({"bounds": asdict(bounds), "seed": seed, "band_m": band_m}, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"boundary_{digest}.json"


def generate_boundary_suite(
    n: int,
    seed: int,
    bounds: PerturbationBounds,
    *,
    band_m: float = BOUNDARY_BAND_M,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> list[Encounter]:
    """Rejection-sample encounters certified safe with margin near the envelope.

    Violations of a certificate can only occur on encounters that are certified
    while the true worst case sits close to the threshold, so broad random suites
    barely exercise the phenomenon. This sampler keeps encounters whose inflated
    certificate returns safe and whose dense-reference margin exceeds the envelope
    at the dense-reference critical time by at most ``band_m``.

    Candidate batches are scanned in deterministic seed order, so the accepted
    suite does not depend on ``n_jobs``. The optional cache stores accepted
    encounters and scan progress per (bounds, seed, band) so repeated runs and
    larger ``n`` requests resume instead of rescanning.
    """
    accepted: list[Encounter] = []
    batches_done = 0
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _boundary_cache_path(cache_dir, bounds, seed, band_m)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            accepted = [_encounter_from_json_dict(entry) for entry in payload["encounters"]]
            batches_done = int(payload["batches_done"])

    while len(accepted) < n and batches_done < BOUNDARY_MAX_BATCHES:
        wave = min(max(n_jobs, 1), BOUNDARY_MAX_BATCHES - batches_done)
        tasks = [(seed + batches_done + offset, bounds, band_m) for offset in range(wave)]
        if pool is None:
            results = [_boundary_batch_worker(task) for task in tasks]
        else:
            results = list(pool.map(_boundary_batch_worker, tasks))
        for batch_accepted in results:
            accepted.extend(batch_accepted)
        batches_done += wave
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "batches_done": batches_done,
                        "encounters": [asdict(encounter) for encounter in accepted],
                    }
                )
            )
    return accepted[:n]


@dataclass(frozen=True)
class CampaignConfig:
    """One named configuration inside a campaign layer."""

    campaign: str
    label: str
    bounds: PerturbationBounds
    execution: ExecutionModelConfig

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"campaign": self.campaign, "label": self.label}
        row.update({f"bound_{key}": value for key, value in asdict(self.bounds).items()})
        row.update({f"execution_{key}": value for key, value in asdict(self.execution).items()})
        return row


def build_inclass_configs() -> list[CampaignConfig]:
    def config(label: str, bounds: PerturbationBounds) -> CampaignConfig:
        return CampaignConfig(campaign="inclass", label=label, bounds=bounds, execution=ExecutionModelConfig())

    configs = [
        config(f"wind_{wind_kt:g}kt", PerturbationBounds(differential_wind_kt=wind_kt))
        for wind_kt in (1.0, 2.0, 5.0, 10.0)
    ]
    configs.extend(
        config(f"position_{position_m:g}m", PerturbationBounds(position_error_m=position_m))
        for position_m in (50.0, 150.0, 300.0)
    )
    configs.extend(
        config(f"heading_{heading_deg:g}deg", PerturbationBounds(heading_error_deg=heading_deg))
        for heading_deg in (0.5, 1.0, 2.0)
    )
    configs.extend(
        config(
            f"execution_{delay_s:g}s_{rate_error:g}",
            PerturbationBounds(turn_delay_max_s=delay_s, turn_rate_error_max=rate_error),
        )
        for delay_s, rate_error in ((3.0, 0.05), (5.0, 0.10), (10.0, 0.20))
    )
    configs.extend(
        config(f"schedule_{schedule_kt:g}kt", PerturbationBounds(schedule_uncertainty_kt=schedule_kt))
        for schedule_kt in (5.0, 10.0, 20.0)
    )
    return configs


def build_misspec_configs() -> list[CampaignConfig]:
    """Out-of-class executed motion against plain and proxy-inflated certificates.

    Each execution model appears twice: once with null bounds, so the inflated
    certificate coincides with the plain one, and once with a misspecified proxy
    envelope (schedule uncertainty for OU drift, initiation delay for roll
    transients, heading error for jitter).
    """
    configs = []
    for sigma_kt in (5.0, 10.0, 20.0):
        execution = ExecutionModelConfig(ou_speed_sigma_kt=sigma_kt, ou_speed_timescale_s=60.0)
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"ou_{sigma_kt:g}kt_plain",
                bounds=PerturbationBounds(),
                execution=execution,
            )
        )
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"ou_{sigma_kt:g}kt_proxy_schedule",
                bounds=PerturbationBounds(schedule_uncertainty_kt=sigma_kt),
                execution=execution,
            )
        )
    for roll_s in (3.0, 6.0, 10.0):
        execution = ExecutionModelConfig(roll_time_s=roll_s)
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"roll_{roll_s:g}s_plain",
                bounds=PerturbationBounds(),
                execution=execution,
            )
        )
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"roll_{roll_s:g}s_proxy_delay",
                bounds=PerturbationBounds(turn_delay_max_s=roll_s),
                execution=execution,
            )
        )
    for jitter_deg in (0.5, 1.0, 2.0):
        execution = ExecutionModelConfig(heading_jitter_deg=jitter_deg)
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"jitter_{jitter_deg:g}deg_plain",
                bounds=PerturbationBounds(),
                execution=execution,
            )
        )
        configs.append(
            CampaignConfig(
                campaign="misspec",
                label=f"jitter_{jitter_deg:g}deg_proxy_heading",
                bounds=PerturbationBounds(heading_error_deg=jitter_deg),
                execution=execution,
            )
        )
    return configs


def build_exceedance_configs() -> list[CampaignConfig]:
    base_bounds = (
        ("wind_5kt", PerturbationBounds(differential_wind_kt=5.0)),
        ("position_150m", PerturbationBounds(position_error_m=150.0)),
        ("heading_1deg", PerturbationBounds(heading_error_deg=1.0)),
        ("execution_5s_0.1", PerturbationBounds(turn_delay_max_s=5.0, turn_rate_error_max=0.10)),
        ("schedule_10kt", PerturbationBounds(schedule_uncertainty_kt=10.0)),
    )
    return [
        CampaignConfig(
            campaign="exceedance",
            label=f"{label}_x{factor:g}",
            bounds=bounds,
            execution=ExecutionModelConfig(exceedance_factor=factor),
        )
        for label, bounds in base_bounds
        for factor in EXCEEDANCE_FACTORS
    ]


def _evaluate_chunk(
    args: tuple[CampaignConfig, str, list[tuple[int, Encounter]], int, int],
) -> list[dict[str, Any]]:
    """Worker: evaluate one config over a chunk of indexed encounters."""
    config, suite_name, indexed_encounters, seed, n_sims = args
    label_key = zlib.crc32(config.label.encode("utf-8"))
    suite_key = zlib.crc32(suite_name.encode("utf-8"))
    rows = []
    for encounter_idx, encounter in indexed_encounters:
        rng = np.random.default_rng([seed, label_key, suite_key, encounter_idx])
        result = evaluate_robustness_case(
            encounter,
            config.bounds,
            n_sims=n_sims,
            rng=rng,
            ref_lat=DEFAULT_REF_LAT,
            ref_lon=DEFAULT_REF_LON,
            adversarial=True,
            execution=config.execution,
        )
        row = config.to_row()
        row["suite"] = suite_name
        row["encounter_idx"] = encounter_idx
        row.update(encounter_to_row(encounter))
        row.update(result.to_row())
        row["eta_horizon_m"] = deviation_envelope_m(encounter, config.bounds, encounter.projection_time_s)
        rows.append(row)
    return rows


def run_campaign(
    config_suites: list[tuple[CampaignConfig, str, list[Encounter]]],
    *,
    seed: int,
    n_sims: int,
    n_jobs: int,
    pool: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    """Run every (config, suite) pairing, parallelised over encounter chunks."""
    rows: list[dict[str, Any]] = []
    if pool is None:
        for config, suite_name, encounters in config_suites:
            rows.extend(_evaluate_chunk((config, suite_name, list(enumerate(encounters)), seed, n_sims)))
        return rows

    tasks = []
    for config, suite_name, encounters in config_suites:
        indexed = list(enumerate(encounters))
        chunk_size = max(len(indexed) // (n_jobs * 2), 1)
        tasks.extend(
            (config, suite_name, indexed[start : start + chunk_size], seed, n_sims)
            for start in range(0, len(indexed), chunk_size)
        )
    for chunk_rows in pool.map(_evaluate_chunk, tasks):
        rows.extend(chunk_rows)
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-config violation and certification statistics."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["campaign"], row["label"], row["suite"]), []).append(row)

    summaries = []
    for (campaign, label, suite), group in sorted(grouped.items()):
        n = len(group)
        n_certified_plain = sum(1 for row in group if row["certified_safe"])
        n_certified_inflated = sum(1 for row in group if row["inflated_certified_safe"])
        n_encounters_violated = sum(1 for row in group if row["n_violations"] > 0)
        total_sims = sum(row["n_sims"] for row in group)
        total_violations = sum(row["n_violations"] for row in group)
        inflated_sims = sum(row["n_sims"] for row in group if row["inflated_certified_safe"])
        summaries.append(
            {
                "campaign": campaign,
                "label": label,
                "suite": suite,
                "n_encounters": n,
                "certified_rate_plain": n_certified_plain / n if n else 0.0,
                "certified_rate_inflated": n_certified_inflated / n if n else 0.0,
                "n_encounters_with_violation": n_encounters_violated,
                "violation_rate_over_certified_sims": (total_violations / inflated_sims) if inflated_sims else 0.0,
                "total_violations": total_violations,
                "total_sims": total_sims,
            }
        )
    return summaries


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
    parser.add_argument("--output-dir", type=Path, default=Path("paper_eval_outputs/robustness"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument("--n-encounters", type=int, default=DEFAULT_N_ENCOUNTERS)
    parser.add_argument("--n-boundary", type=int, default=DEFAULT_N_BOUNDARY)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--campaign", choices=("inclass", "misspec", "exceedance", "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="Small suite for smoke runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    n_encounters = QUICK_N_ENCOUNTERS if args.quick else args.n_encounters
    n_boundary = QUICK_N_BOUNDARY if args.quick else args.n_boundary
    n_sims = QUICK_N_SIMS if args.quick else args.n_sims

    configs: list[CampaignConfig] = []
    if args.campaign in ("inclass", "all"):
        configs.extend(build_inclass_configs())
    if args.campaign in ("misspec", "all"):
        configs.extend(build_misspec_configs())
    if args.campaign in ("exceedance", "all"):
        configs.extend(build_exceedance_configs())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=args.n_jobs) if args.n_jobs > 1 else None
    for repeat in range(args.repeat_seeds):
        seed = args.seed + repeat * args.seed_step
        run_dir = args.output_dir / f"run_{repeat:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        broad_suite = generate_mixed_turn_suite(n_encounters, seed)
        boundary_cache: dict[PerturbationBounds, list[Encounter]] = {}
        config_suites: list[tuple[CampaignConfig, str, list[Encounter]]] = []
        for config in configs:
            config_suites.append((config, "broad", broad_suite))
            if config.bounds not in boundary_cache:
                boundary_cache[config.bounds] = generate_boundary_suite(
                    n_boundary,
                    seed + 777,
                    config.bounds,
                    n_jobs=args.n_jobs,
                    cache_dir=args.output_dir / "boundary_cache",
                    pool=pool,
                )
            config_suites.append((config, "boundary", boundary_cache[config.bounds]))
        rows = run_campaign(config_suites, seed=seed, n_sims=n_sims, n_jobs=args.n_jobs, pool=pool)
        summaries = summarise(rows)
        elapsed_s = time.perf_counter() - started

        write_rows_csv(run_dir / "robustness_rows.csv", rows)
        write_rows_csv(run_dir / "robustness_summary.csv", summaries)
        inclass_violations = sum(entry["total_violations"] for entry in summaries if entry["campaign"] == "inclass")
        summary_json = {
            "seed": seed,
            "n_encounters": n_encounters,
            "n_sims": n_sims,
            "n_configs": len(configs),
            "elapsed_s": elapsed_s,
            "inclass_total_violations": inclass_violations,
            "summaries": summaries,
        }
        (run_dir / "robustness_summary.json").write_text(json.dumps(summary_json, indent=2))
        print(
            f"run {repeat}: seed={seed} configs={len(configs)} encounters={n_encounters} "
            f"sims={n_sims} elapsed={elapsed_s:.1f}s inclass_violations={inclass_violations}"
        )
        if inclass_violations:
            print("WARNING: in-class violations found; the envelope implementation is broken.")
    if pool is not None:
        pool.shutdown()


if __name__ == "__main__":
    main()
