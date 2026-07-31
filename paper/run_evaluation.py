#!/usr/bin/env python3
"""Run the manuscript evaluation package and write paper-ready artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

from matplotlib import ticker
import matplotlib.pyplot as plt
import numba
import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety.util import KT_TO_MPS, NMI_TO_M
from paper.evaluation import (
    COARSE_PROXY_DT_S,
    DEFAULT_DT_MIN_VALUES_S,
    MARGIN_BINS_NMI,
    SPEED_UNCERTAINTY_BINS_KT,
    TURN_ANGLE_BINS_DEG,
    Encounter,
    EvaluationConfig,
    _bounded_distances_at_times,
    benchmark_fixed_time_kernel,
    build_sampled_proxy_counterexample_row,
    build_uniform_times,
    generate_just_unsafe_suite,
    generate_mixed_turn_suite,
    generate_near_threshold_suite,
    generate_straight_suite,
    run_just_unsafe_suite,
    run_mixed_turn_suite,
    run_near_threshold_suite,
    run_straight_suite,
    select_narrative_scenarios,
    warm_up_numba,
)

SUITE_PROFILES = {
    "paper": {"straight_n": 50_000, "mixed_turn_n": 20_000, "near_threshold_n": 5_000, "just_unsafe_n": 5_000},
    "large100k": {
        "straight_n": 100_000,
        "mixed_turn_n": 100_000,
        "near_threshold_n": 100_000,
        "just_unsafe_n": 100_000,
    },
}
MANUSCRIPT_FONT_FAMILY = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
ROW_OUTPUT_MODES = ("all", "none", "figure", "first-figure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the geometric-safety paper evaluation package.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_eval_outputs/full"))
    parser.add_argument(
        "--aggregate-run-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Aggregate exact campaign outputs from existing evaluation run directories.",
    )
    parser.add_argument(
        "--render-figures-from",
        type=Path,
        default=None,
        help="Regenerate figures from an existing evaluation output directory without rerunning the suites.",
    )
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=None,
        help="Target directory for render-only figure output. Defaults to --output-dir when omitted.",
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["pdf"],
        help="Figure formats for render-only output.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a smoke-sized evaluation package.")
    parser.add_argument(
        "--profile",
        choices=sorted(SUITE_PROFILES),
        default="paper",
        help="Named suite-size profile for non-quick runs.",
    )
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--repeat-seeds", type=int, default=1, help="Number of independent seed runs to execute.")
    parser.add_argument("--seed-step", type=int, default=10_000, help="Seed increment between repeated runs.")
    parser.add_argument("--straight-n", type=int, default=None)
    parser.add_argument("--mixed-turn-n", type=int, default=None)
    parser.add_argument("--near-threshold-n", type=int, default=None)
    parser.add_argument(
        "--dt-min-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_DT_MIN_VALUES_S),
        help="Certification widths for the ablation sweep.",
    )
    parser.add_argument(
        "--coarse-proxy-dt",
        type=float,
        default=COARSE_PROXY_DT_S,
        help="Sampling step in seconds for the sampled turn-aware heuristic baseline.",
    )
    parser.add_argument(
        "--threshold-nmi",
        type=float,
        default=5.0,
        help="Protected lateral separation threshold in NMI; 5 is the "
        "manuscript setting and 3 feeds the threshold sensitivity arm.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for evaluation and generation.",
    )
    parser.add_argument("--skip-rows", action="store_true", help="Skip writing row-level CSV outputs.")
    parser.add_argument(
        "--row-output-mode",
        choices=ROW_OUTPUT_MODES,
        default="all",
        help=(
            "Row-level CSV policy: all writes every suite row; none writes only summaries; "
            "figure writes only rows needed to re-render figures; first-figure does that only for the first seed."
        ),
    )
    parser.add_argument("--compress-rows", action="store_true", help="Write row-level CSV outputs as gzip files.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip writing figure outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_run_dirs:
        aggregate_existing_runs(
            args.aggregate_run_dirs,
            args.output_dir,
            figure_output_dir=args.figure_output_dir,
            figure_formats=tuple(args.figure_formats),
            render_figures_from=args.render_figures_from,
        )
        return
    if args.render_figures_from is not None:
        render_existing_figures(
            args.render_figures_from,
            args.figure_output_dir or args.output_dir,
            figure_formats=tuple(args.figure_formats),
        )
        return
    if args.repeat_seeds < 1:
        raise ValueError("--repeat-seeds must be at least 1.")
    if args.seed_step < 1:
        raise ValueError("--seed-step must be at least 1.")

    seeds = campaign_seeds(args.seed, args.repeat_seeds, args.seed_step)
    output_dir = args.output_dir
    row_output_mode = effective_row_output_mode(args)

    if len(seeds) == 1:
        config = build_config(args, seed=seeds[0])
        run_single_evaluation(
            config,
            output_dir,
            row_output_mode="figure" if row_output_mode == "first-figure" else row_output_mode,
            compress_rows=args.compress_rows,
            write_plots=not args.skip_plots,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for index, seed in enumerate(seeds):
        config = build_config(args, seed=seed)
        run_output_dir = output_dir / f"run_{index:02d}_seed_{seed}"
        result = run_single_evaluation(
            config,
            run_output_dir,
            row_output_mode=row_output_mode_for_run(row_output_mode, index),
            compress_rows=args.compress_rows,
            write_plots=not args.skip_plots,
        )
        run_records.append(result["run_record"])

    campaign_summary = build_campaign_summary(
        run_records,
        profile=args.profile,
        repeat_seeds=args.repeat_seeds,
        seed_step=args.seed_step,
        row_output_mode=row_output_mode,
        write_plots=not args.skip_plots,
    )
    write_json(output_dir / "campaign_summary.json", campaign_summary)
    write_csv(output_dir / "campaign_runs.csv", build_campaign_rows(run_records))


def effective_row_output_mode(args: argparse.Namespace) -> str:
    if args.skip_rows:
        return "none"
    return str(args.row_output_mode)


def row_output_mode_for_run(row_output_mode: str, run_index: int) -> str:
    if row_output_mode == "first-figure":
        return "figure" if run_index == 0 else "none"
    return row_output_mode


def build_config(args: argparse.Namespace, *, seed: int) -> EvaluationConfig:
    threshold_m = args.threshold_nmi * NMI_TO_M
    if args.quick:
        return EvaluationConfig(
            straight_n=200,
            mixed_turn_n=120,
            near_threshold_n=24,
            seed=seed,
            threshold_m=threshold_m,
            coarse_proxy_dt_s=args.coarse_proxy_dt,
            dt_min_values_s=tuple(args.dt_min_values),
            runtimes_rounds=2,
            n_jobs=args.n_jobs,
            quick=True,
        )
    profile_sizes = SUITE_PROFILES[args.profile]
    return EvaluationConfig(
        straight_n=profile_sizes["straight_n"] if args.straight_n is None else args.straight_n,
        mixed_turn_n=profile_sizes["mixed_turn_n"] if args.mixed_turn_n is None else args.mixed_turn_n,
        near_threshold_n=profile_sizes["near_threshold_n"] if args.near_threshold_n is None else args.near_threshold_n,
        just_unsafe_n=profile_sizes["just_unsafe_n"],
        seed=seed,
        threshold_m=threshold_m,
        coarse_proxy_dt_s=args.coarse_proxy_dt,
        dt_min_values_s=tuple(args.dt_min_values),
        n_jobs=args.n_jobs,
    )


def campaign_seeds(seed: int, repeat_seeds: int, seed_step: int) -> list[int]:
    return [seed + seed_step * index for index in range(repeat_seeds)]


def run_single_evaluation(
    config: EvaluationConfig,
    output_dir: Path,
    *,
    row_output_mode: str,
    compress_rows: bool,
    write_plots: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    warm_up_numba(config)

    straight_encounters = generate_straight_suite(config.straight_n, config.seed, threshold_m=config.threshold_m)
    mixed_encounters = generate_mixed_turn_suite(
        config.mixed_turn_n,
        config.seed + 1,
        threshold_m=config.threshold_m,
        turn_zero_probability=config.turn_zero_probability,
    )
    near_threshold_encounters = generate_near_threshold_suite(config.near_threshold_n, config.seed + 2, config)
    just_unsafe_encounters = generate_just_unsafe_suite(config.just_unsafe_n, config.seed + 3, config)

    straight_results = run_straight_suite(straight_encounters, n_jobs=config.n_jobs)
    mixed_results = run_mixed_turn_suite(mixed_encounters, config=config)
    near_threshold_results = run_near_threshold_suite(near_threshold_encounters, config=config)
    just_unsafe_results = run_just_unsafe_suite(just_unsafe_encounters)
    narrative = select_narrative_scenarios(
        straight_results["rows"],
        mixed_results["rows"],
        near_threshold_results["rows"],
    )
    narrative["sampled_proxy_miss"] = build_sampled_proxy_counterexample_row(config)

    kernel_runtime = benchmark_fixed_time_kernel(rounds=config.runtimes_rounds)
    environment = build_environment_summary(config)
    summary = {
        "config": asdict(config),
        "environment": environment,
        "straight": straight_results["summary"],
        "mixed_turn": mixed_results["summary"],
        "near_threshold": {
            "summary_rows": near_threshold_results["summary_rows"],
            "margin_summary": near_threshold_results["margin_summary"],
            "default_margin_summary": near_threshold_results["default_margin_summary"],
        },
        "just_unsafe": just_unsafe_results["summary"],
        "narrative_scenarios": narrative,
        "kernel_runtime": kernel_runtime,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "stress_map_source.csv", build_stress_map_source_rows(mixed_results["rows"]))
    write_csv(
        output_dir / "near_threshold_default_margin_summary.csv",
        near_threshold_results["default_margin_summary"],
    )
    write_csv(output_dir / "just_unsafe_rows.csv", just_unsafe_results["rows"])

    write_row_outputs(
        output_dir,
        row_output_mode=row_output_mode,
        compress_rows=compress_rows,
        straight_rows=straight_results["rows"],
        mixed_rows=mixed_results["rows"],
        near_threshold_rows=near_threshold_results["rows"],
        sampled_proxy_miss=narrative["sampled_proxy_miss"],
    )

    write_csv(output_dir / "table_1_experiment_design.csv", build_table_1(config))
    write_csv(output_dir / "table_2_straight_exactness.csv", build_table_2(straight_results["summary"]))
    write_csv(output_dir / "table_3_mixed_turn_results.csv", build_table_3(mixed_results["summary"]))
    write_csv(
        output_dir / "table_4_runtime_summary.csv",
        build_table_4(
            kernel_runtime=kernel_runtime,
            mixed_runtime=mixed_results["summary"]["runtime"],
            ablation_rows=near_threshold_results["summary_rows"],
            environment=environment,
        ),
    )

    if write_plots:
        render_figure_set(
            output_dir,
            narrative=narrative,
            mixed_rows=mixed_results["rows"],
            ablation_rows=near_threshold_results["summary_rows"],
            figure_formats=("pdf",),
            manuscript_names=False,
        )

    run_record = build_run_record(
        config,
        straight_results=straight_results,
        mixed_results=mixed_results,
        near_threshold_results=near_threshold_results,
        just_unsafe_results=just_unsafe_results,
        kernel_runtime=kernel_runtime,
    )
    write_json(output_dir / "run_record.json", run_record)
    return {"summary": summary, "run_record": run_record}


def build_run_record(
    config: EvaluationConfig,
    *,
    straight_results: dict[str, Any],
    mixed_results: dict[str, Any],
    near_threshold_results: dict[str, Any],
    just_unsafe_results: dict[str, Any],
    kernel_runtime: dict[str, float],
) -> dict[str, Any]:
    straight_rows = straight_results["rows"]
    mixed_rows = mixed_results["rows"]
    near_threshold_rows = near_threshold_results["rows"]

    straight_oracle_safe_n = sum(int(row["oracle_is_safe"]) for row in straight_rows)
    straight_oracle_unsafe_n = len(straight_rows) - straight_oracle_safe_n
    straight_nominal = confusion_counts(straight_rows, "oracle_is_safe", "nominal_proxy_is_safe")
    straight_proposed = confusion_counts(straight_rows, "oracle_is_safe", "proposed_is_safe")

    eligible_mixed_rows = [row for row in mixed_rows if int(row["eligible_for_boolean_tallies"]) == 1]
    mixed_reference_safe_n = sum(int(row["reference_is_safe"]) for row in eligible_mixed_rows)
    mixed_reference_unsafe_n = len(eligible_mixed_rows) - mixed_reference_safe_n
    mixed_proposed = confusion_counts(eligible_mixed_rows, "reference_is_safe", "proposed_is_safe")
    mixed_nominal = confusion_counts(eligible_mixed_rows, "reference_is_safe", "nominal_proxy_is_safe")
    mixed_coarse = confusion_counts(eligible_mixed_rows, "reference_is_safe", "coarse_proxy_is_safe")

    near_default_rows = [
        row for row in near_threshold_rows if row["lipschitz_mode"] == "local" and float(row["dt_min_s"]) == 3.0
    ]
    near_default_certified_n = sum(int(row["proposed_is_safe"]) for row in near_default_rows)
    near_default_total_n = len(near_default_rows)

    return {
        "seed": int(config.seed),
        "config": asdict(config),
        "straight": {
            "n": len(straight_rows),
            "oracle_safe_n": straight_oracle_safe_n,
            "oracle_unsafe_n": straight_oracle_unsafe_n,
            "proposed_disagreement_n": straight_proposed["disagreement_n"],
            "nominal_proxy_false_safe_n": straight_nominal["false_safe_n"],
            "nominal_proxy_false_unsafe_n": straight_nominal["false_unsafe_n"],
            "nominal_proxy_disagreement_n": straight_nominal["disagreement_n"],
            "max_distance_error_m": float(straight_results["summary"]["max_distance_error_m"]),
            "max_time_error_s": float(straight_results["summary"]["max_time_error_s"]),
            "independent_agreement_rate": float(
                straight_results["summary"]["independent_vs_proposed_agreement_rate"]
            ),
            "max_distance_error_vs_independent_m": float(
                straight_results["summary"]["max_distance_error_vs_independent_m"]
            ),
            "max_time_error_vs_independent_s": float(
                straight_results["summary"]["max_time_error_vs_independent_s"]
            ),
        },
        "just_unsafe": dict(just_unsafe_results["summary"]),
        "mixed_turn": {
            "n": len(mixed_rows),
            "eligible_n": len(eligible_mixed_rows),
            "reference_safe_n": mixed_reference_safe_n,
            "reference_unsafe_n": mixed_reference_unsafe_n,
            "proposed_false_safe_n": mixed_proposed["false_safe_n"],
            "proposed_false_unsafe_n": mixed_proposed["false_unsafe_n"],
            "proposed_disagreement_n": mixed_proposed["disagreement_n"],
            "nominal_proxy_false_safe_n": mixed_nominal["false_safe_n"],
            "nominal_proxy_false_unsafe_n": mixed_nominal["false_unsafe_n"],
            "nominal_proxy_disagreement_n": mixed_nominal["disagreement_n"],
            "coarse_proxy_false_safe_n": mixed_coarse["false_safe_n"],
            "coarse_proxy_false_unsafe_n": mixed_coarse["false_unsafe_n"],
            "coarse_proxy_disagreement_n": mixed_coarse["disagreement_n"],
            "runtime": mixed_results["summary"]["runtime"],
            "runtime_with_envelope": mixed_results["summary"]["runtime_with_envelope"],
            "in_band": mixed_results["summary"]["in_band"],
        },
        "near_threshold": {
            "accepted_n": config.near_threshold_n,
            "default_local_dt3_certified_n": near_default_certified_n,
            "default_local_dt3_total_n": near_default_total_n,
        },
        "kernel_runtime": kernel_runtime,
    }


def write_row_outputs(
    output_dir: Path,
    *,
    row_output_mode: str,
    compress_rows: bool,
    straight_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    near_threshold_rows: list[dict[str, Any]],
    sampled_proxy_miss: dict[str, Any],
) -> None:
    suffix = ".csv.gz" if compress_rows else ".csv"
    if row_output_mode == "none":
        return
    if row_output_mode == "figure":
        write_csv(output_dir / f"mixed_turn_rows{suffix}", mixed_rows)
        write_csv(output_dir / "sampled_proxy_miss_counterexample.csv", [sampled_proxy_miss])
        return
    if row_output_mode == "all":
        write_csv(output_dir / f"straight_suite_rows{suffix}", straight_rows)
        write_csv(output_dir / f"mixed_turn_rows{suffix}", mixed_rows)
        write_csv(output_dir / f"near_threshold_rows{suffix}", near_threshold_rows)
        write_csv(output_dir / "sampled_proxy_miss_counterexample.csv", [sampled_proxy_miss])
        return
    raise ValueError(f"Unsupported row_output_mode={row_output_mode!r}")


def aggregate_existing_runs(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    figure_output_dir: Path | None,
    figure_formats: tuple[str, ...],
    render_figures_from: Path | None,
) -> dict[str, Any]:
    loaded = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected summary.json in {run_dir}")
        loaded.append({"run_dir": run_dir, "summary": read_json(summary_path)})

    validate_existing_run_summaries(loaded)
    run_records = sorted(
        (reconstruct_run_record_from_summary(item["summary"]) for item in loaded),
        key=lambda item: int(item["seed"]),
    )
    inferred_profile = infer_profile_name(run_records[0]["config"])
    seed_step = infer_seed_step([int(record["seed"]) for record in run_records])

    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_summary = build_campaign_summary(
        run_records,
        profile=inferred_profile,
        repeat_seeds=len(run_records),
        seed_step=seed_step,
        row_output_mode="none",
        write_plots=False,
    )
    write_json(output_dir / "campaign_summary.json", campaign_summary)
    write_csv(output_dir / "campaign_runs.csv", build_campaign_rows(run_records))
    write_json(output_dir / "campaign_run_records.json", {"runs": run_records})

    if render_figures_from is not None:
        render_existing_figures(
            render_figures_from,
            figure_output_dir or output_dir,
            figure_formats=figure_formats,
        )
    return campaign_summary


def render_existing_figures(
    run_dir: Path,
    output_dir: Path,
    *,
    figure_formats: tuple[str, ...],
) -> None:
    summary = read_json(run_dir / "summary.json")
    straight_rows_path = find_csv_or_gzip(run_dir / "straight_suite_rows.csv")
    mixed_rows_path = find_csv_or_gzip(run_dir / "mixed_turn_rows.csv")
    near_threshold_rows_path = find_csv_or_gzip(run_dir / "near_threshold_rows.csv")
    stress_map_source_path = find_csv_or_gzip(run_dir / "stress_map_source.csv")
    mixed_rows = read_csv_rows(mixed_rows_path) if mixed_rows_path is not None else []
    stress_map_rows = read_csv_rows(stress_map_source_path) if stress_map_source_path is not None else []
    if not mixed_rows and not stress_map_rows:
        raise FileNotFoundError(
            f"Expected mixed_turn_rows.csv(.gz) or stress_map_source.csv(.gz) in {run_dir}"
        )
    if straight_rows_path is not None and near_threshold_rows_path is not None:
        narrative = select_narrative_scenarios(
            read_csv_rows(straight_rows_path),
            mixed_rows,
            read_csv_rows(near_threshold_rows_path),
        )
        narrative["sampled_proxy_miss"] = summary["narrative_scenarios"]["sampled_proxy_miss"]
    else:
        narrative = summary["narrative_scenarios"]

    output_dir.mkdir(parents=True, exist_ok=True)
    render_figure_set(
        output_dir,
        narrative=narrative,
        mixed_rows=mixed_rows,
        stress_map_rows=stress_map_rows,
        ablation_rows=summary["near_threshold"]["summary_rows"],
        figure_formats=figure_formats,
        manuscript_names=True,
    )


def render_figure_set(
    output_dir: Path,
    *,
    narrative: dict[str, dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    stress_map_rows: list[dict[str, Any]] | None = None,
    ablation_rows: list[dict[str, Any]],
    figure_formats: tuple[str, ...],
    manuscript_names: bool,
) -> None:
    configure_manuscript_plot_style()
    specs = [
        (
            "fig_crossing" if manuscript_names else "figure_1_crossing",
            lambda path: plot_crossing_scenario(path, narrative["crossing"]),
        ),
        (
            "fig_nominal_proxy_miss" if manuscript_names else "figure_2_proxy_miss",
            lambda path: plot_proxy_miss(path, narrative["proxy_miss"]),
        ),
        (
            "fig_stress_maps" if manuscript_names else "figure_3_stress_maps",
            lambda path: plot_stress_maps_from_source(path, stress_map_rows)
            if stress_map_rows
            else plot_stress_maps(path, mixed_rows),
        ),
        (
            "fig_ablation" if manuscript_names else "figure_4_ablation",
            lambda path: plot_ablation(path, ablation_rows),
        ),
        (
            "fig_sampled_proxy_miss" if manuscript_names else "figure_sampled_proxy_miss",
            lambda path: plot_sampled_proxy_miss(
                path,
                narrative["sampled_proxy_miss"],
                sample_dt_s=float(narrative["sampled_proxy_miss"]["coarse_proxy_dt_s"]),
            ),
        ),
    ]

    for base_name, plotter in specs:
        for figure_format in figure_formats:
            plotter(output_dir / f"{base_name}.{figure_format.lower()}")


def configure_manuscript_plot_style() -> None:
    """Use manuscript-style fonts and embed TrueType text in generated PDFs."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": MANUSCRIPT_FONT_FAMILY,
            "mathtext.fontset": "stix",
            "font.size": 12.5,
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def validate_existing_run_summaries(loaded: list[dict[str, Any]]) -> None:
    if not loaded:
        raise ValueError("At least one existing run directory is required.")

    reference = normalized_config(loaded[0]["summary"]["config"])
    for item in loaded[1:]:
        candidate = normalized_config(item["summary"]["config"])
        if candidate != reference:
            raise ValueError(f"Config mismatch between existing runs: {loaded[0]['run_dir']} and {item['run_dir']}")


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "seed"}


def infer_profile_name(config: dict[str, Any]) -> str:
    for name, sizes in SUITE_PROFILES.items():
        if (
            int(config["straight_n"]) == sizes["straight_n"]
            and int(config["mixed_turn_n"]) == sizes["mixed_turn_n"]
            and int(config["near_threshold_n"]) == sizes["near_threshold_n"]
            and not bool(config["quick"])
        ):
            return name
    return "custom"


def infer_seed_step(seeds: list[int]) -> int:
    if len(seeds) < 2:
        return 0
    ordered = sorted(seeds)
    deltas = {ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)}
    return deltas.pop() if len(deltas) == 1 else 0


def reconstruct_run_record_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary["config"]
    straight = reconstruct_straight_record(summary["straight"])
    mixed_turn = reconstruct_mixed_turn_record(summary["mixed_turn"])
    near_threshold = reconstruct_near_threshold_record(summary["near_threshold"], config)
    return {
        "seed": int(config["seed"]),
        "config": config,
        "straight": straight,
        "mixed_turn": mixed_turn,
        "near_threshold": near_threshold,
        "kernel_runtime": summary["kernel_runtime"],
    }


def reconstruct_straight_record(straight_summary: dict[str, Any]) -> dict[str, Any]:
    total_n = int(straight_summary["n"])
    nominal = straight_summary["nominal_proxy"]
    confusion = reconstruct_binary_confusion(
        total_n=total_n,
        false_safe_rate=float(nominal["false_safe_rate"]),
        false_unsafe_rate=float(nominal["false_unsafe_rate"]),
        overall_disagreement_rate=float(nominal["overall_disagreement_rate"]),
    )
    proposed_disagreement_n = recover_count_from_rate(
        1.0 - float(straight_summary["oracle_vs_proposed_agreement_rate"]),
        total_n,
    )
    return {
        "n": total_n,
        "oracle_safe_n": confusion["reference_safe_n"],
        "oracle_unsafe_n": confusion["reference_unsafe_n"],
        "proposed_disagreement_n": proposed_disagreement_n,
        "nominal_proxy_false_safe_n": confusion["false_safe_n"],
        "nominal_proxy_false_unsafe_n": confusion["false_unsafe_n"],
        "nominal_proxy_disagreement_n": confusion["disagreement_n"],
        "max_distance_error_m": float(straight_summary["max_distance_error_m"]),
        "max_time_error_s": float(straight_summary["max_time_error_s"]),
    }


def reconstruct_mixed_turn_record(mixed_summary: dict[str, Any]) -> dict[str, Any]:
    total_n = int(mixed_summary["n"])
    eligible_n = int(mixed_summary["eligible_n"])
    proposed = reconstruct_binary_confusion(
        total_n=eligible_n,
        false_safe_rate=float(mixed_summary["proposed"]["false_safe_rate"]),
        false_unsafe_rate=float(mixed_summary["proposed"]["false_unsafe_rate"]),
        overall_disagreement_rate=float(mixed_summary["proposed"]["overall_disagreement_rate"]),
    )
    certification_rate = float(mixed_summary["certification_rate_given_reference_safe"])
    expected_cert_rate = 1.0 - safe_divide(proposed["false_unsafe_n"], proposed["reference_safe_n"])
    if not math.isclose(expected_cert_rate, certification_rate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Proposed mixed-turn certification rate does not match reconstructed counts.")

    nominal = reconstruct_binary_confusion(
        total_n=eligible_n,
        false_safe_rate=float(mixed_summary["nominal_proxy"]["false_safe_rate"]),
        false_unsafe_rate=float(mixed_summary["nominal_proxy"]["false_unsafe_rate"]),
        overall_disagreement_rate=float(mixed_summary["nominal_proxy"]["overall_disagreement_rate"]),
        reference_safe_n=proposed["reference_safe_n"],
        reference_unsafe_n=proposed["reference_unsafe_n"],
    )
    coarse = reconstruct_binary_confusion(
        total_n=eligible_n,
        false_safe_rate=float(mixed_summary["coarse_proxy"]["false_safe_rate"]),
        false_unsafe_rate=float(mixed_summary["coarse_proxy"]["false_unsafe_rate"]),
        overall_disagreement_rate=float(mixed_summary["coarse_proxy"]["overall_disagreement_rate"]),
        reference_safe_n=proposed["reference_safe_n"],
        reference_unsafe_n=proposed["reference_unsafe_n"],
    )
    return {
        "n": total_n,
        "eligible_n": eligible_n,
        "reference_safe_n": proposed["reference_safe_n"],
        "reference_unsafe_n": proposed["reference_unsafe_n"],
        "proposed_false_safe_n": proposed["false_safe_n"],
        "proposed_false_unsafe_n": proposed["false_unsafe_n"],
        "proposed_disagreement_n": proposed["disagreement_n"],
        "nominal_proxy_false_safe_n": nominal["false_safe_n"],
        "nominal_proxy_false_unsafe_n": nominal["false_unsafe_n"],
        "nominal_proxy_disagreement_n": nominal["disagreement_n"],
        "coarse_proxy_false_safe_n": coarse["false_safe_n"],
        "coarse_proxy_false_unsafe_n": coarse["false_unsafe_n"],
        "coarse_proxy_disagreement_n": coarse["disagreement_n"],
        "runtime": mixed_summary["runtime"],
    }


def reconstruct_near_threshold_record(
    near_threshold_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    default_row = next(
        row
        for row in near_threshold_summary["summary_rows"]
        if row["lipschitz_mode"] == "local" and float(row["dt_min_s"]) == 3.0
    )
    total_n = int(default_row["n"])
    certified_n = recover_count_from_rate(float(default_row["certification_rate"]), total_n)
    expected_total_n = int(config["near_threshold_n"])
    if total_n != expected_total_n:
        raise ValueError("Near-threshold summary row does not match config near_threshold_n.")
    return {
        "accepted_n": expected_total_n,
        "default_local_dt3_certified_n": certified_n,
        "default_local_dt3_total_n": total_n,
    }


def reconstruct_binary_confusion(
    *,
    total_n: int,
    false_safe_rate: float,
    false_unsafe_rate: float,
    overall_disagreement_rate: float,
    reference_safe_n: int | None = None,
    reference_unsafe_n: int | None = None,
) -> dict[str, int]:
    disagreement_n = recover_count_from_rate(overall_disagreement_rate, total_n)

    if reference_safe_n is not None and reference_unsafe_n is not None:
        false_safe_n = recover_count_from_rate(false_safe_rate, reference_unsafe_n)
        false_unsafe_n = recover_count_from_rate(false_unsafe_rate, reference_safe_n)
        if false_safe_n + false_unsafe_n != disagreement_n:
            raise ValueError("Stored disagreement rate does not match reconstructed confusion counts.")
        return {
            "reference_safe_n": int(reference_safe_n),
            "reference_unsafe_n": int(reference_unsafe_n),
            "false_safe_n": false_safe_n,
            "false_unsafe_n": false_unsafe_n,
            "disagreement_n": disagreement_n,
        }

    false_safe_candidates = range(disagreement_n + 1)
    if is_zero_rate(false_safe_rate):
        false_safe_candidates = [0]
    elif is_zero_rate(false_unsafe_rate):
        false_safe_candidates = [disagreement_n]

    for false_safe_n in false_safe_candidates:
        false_unsafe_n = disagreement_n - false_safe_n
        unsafe_candidate = recover_denominator_from_rate(false_safe_n, false_safe_rate)
        safe_candidate = recover_denominator_from_rate(false_unsafe_n, false_unsafe_rate)

        if unsafe_candidate is None and safe_candidate is None:
            continue
        if unsafe_candidate is None:
            assert safe_candidate is not None
            unsafe_candidate = total_n - safe_candidate
        if safe_candidate is None:
            assert unsafe_candidate is not None
            safe_candidate = total_n - unsafe_candidate
        if safe_candidate < 0 or unsafe_candidate < 0 or safe_candidate + unsafe_candidate != total_n:
            continue

        if not rates_match(false_safe_n, unsafe_candidate, false_safe_rate):
            continue
        if not rates_match(false_unsafe_n, safe_candidate, false_unsafe_rate):
            continue
        return {
            "reference_safe_n": int(safe_candidate),
            "reference_unsafe_n": int(unsafe_candidate),
            "false_safe_n": false_safe_n,
            "false_unsafe_n": false_unsafe_n,
            "disagreement_n": disagreement_n,
        }

    raise ValueError("Could not reconstruct exact confusion counts from stored summary rates.")


def recover_count_from_rate(rate: float, denominator: int) -> int:
    if denominator <= 0 or is_zero_rate(rate):
        return 0
    estimated = rate * float(denominator)
    candidate = round(estimated)
    if not math.isclose(float(candidate) / float(denominator), rate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Rate {rate} does not invert cleanly for denominator {denominator}.")
    return candidate


def recover_denominator_from_rate(count: int, rate: float) -> int | None:
    if is_zero_rate(rate):
        return None if count == 0 else -1
    estimated = float(count) / rate
    candidate = round(estimated)
    if candidate <= 0:
        return -1
    if not math.isclose(float(count) / float(candidate), rate, rel_tol=1e-12, abs_tol=1e-12):
        return -1
    return candidate


def is_zero_rate(rate: float) -> bool:
    return math.isclose(rate, 0.0, abs_tol=1e-15)


def rates_match(count: int, denominator: int, rate: float) -> bool:
    if denominator < 0:
        return False
    if denominator == 0:
        return count == 0 and is_zero_rate(rate)
    return math.isclose(float(count) / float(denominator), rate, rel_tol=1e-12, abs_tol=1e-12)


def confusion_counts(rows: list[dict[str, Any]], reference_key: str, method_key: str) -> dict[str, int]:
    false_safe_n = 0
    false_unsafe_n = 0
    disagreement_n = 0
    for row in rows:
        reference_safe = bool(int(row[reference_key]))
        method_safe = bool(int(row[method_key]))
        if method_safe != reference_safe:
            disagreement_n += 1
            if method_safe:
                false_safe_n += 1
            else:
                false_unsafe_n += 1
    return {
        "false_safe_n": false_safe_n,
        "false_unsafe_n": false_unsafe_n,
        "disagreement_n": disagreement_n,
    }


def build_campaign_summary(
    run_records: list[dict[str, Any]],
    *,
    profile: str,
    repeat_seeds: int,
    seed_step: int,
    row_output_mode: str,
    write_plots: bool,
) -> dict[str, Any]:
    straight_total_n = sum(int(record["straight"]["n"]) for record in run_records)
    straight_nominal_false_safe_n = sum(int(record["straight"]["nominal_proxy_false_safe_n"]) for record in run_records)
    straight_nominal_false_unsafe_n = sum(
        int(record["straight"]["nominal_proxy_false_unsafe_n"]) for record in run_records
    )
    straight_nominal_disagreement_n = sum(
        int(record["straight"]["nominal_proxy_disagreement_n"]) for record in run_records
    )
    straight_oracle_unsafe_n = sum(int(record["straight"]["oracle_unsafe_n"]) for record in run_records)

    mixed_total_n = sum(int(record["mixed_turn"]["n"]) for record in run_records)
    mixed_total_eligible_n = sum(int(record["mixed_turn"]["eligible_n"]) for record in run_records)
    mixed_reference_safe_n = sum(int(record["mixed_turn"]["reference_safe_n"]) for record in run_records)
    mixed_reference_unsafe_n = sum(int(record["mixed_turn"]["reference_unsafe_n"]) for record in run_records)

    proposed_false_safe_n = sum(int(record["mixed_turn"]["proposed_false_safe_n"]) for record in run_records)
    proposed_false_unsafe_n = sum(int(record["mixed_turn"]["proposed_false_unsafe_n"]) for record in run_records)
    proposed_disagreement_n = sum(int(record["mixed_turn"]["proposed_disagreement_n"]) for record in run_records)

    nominal_false_safe_n = sum(int(record["mixed_turn"]["nominal_proxy_false_safe_n"]) for record in run_records)
    nominal_false_unsafe_n = sum(int(record["mixed_turn"]["nominal_proxy_false_unsafe_n"]) for record in run_records)
    nominal_disagreement_n = sum(int(record["mixed_turn"]["nominal_proxy_disagreement_n"]) for record in run_records)

    coarse_false_safe_n = sum(int(record["mixed_turn"]["coarse_proxy_false_safe_n"]) for record in run_records)
    coarse_false_unsafe_n = sum(int(record["mixed_turn"]["coarse_proxy_false_unsafe_n"]) for record in run_records)
    coarse_disagreement_n = sum(int(record["mixed_turn"]["coarse_proxy_disagreement_n"]) for record in run_records)

    near_default_total_n = sum(int(record["near_threshold"]["default_local_dt3_total_n"]) for record in run_records)
    near_default_certified_n = sum(
        int(record["near_threshold"]["default_local_dt3_certified_n"]) for record in run_records
    )

    just_unsafe_total_n = sum(int(record["just_unsafe"]["n"]) for record in run_records)
    just_unsafe_solver_false_safe_n = sum(
        int(record["just_unsafe"]["solver_false_safe_n"]) for record in run_records
    )
    just_unsafe_nominal_false_safe_n = sum(
        int(record["just_unsafe"]["nominal_proxy_false_safe_n"]) for record in run_records
    )

    return {
        "profile": profile,
        "repeat_seeds": repeat_seeds,
        "seed_step": seed_step,
        "row_output_mode": row_output_mode,
        "write_rows": row_output_mode != "none",
        "write_plots": write_plots,
        "seeds": [int(record["seed"]) for record in run_records],
        "runs": run_records,
        "aggregate": {
            "straight": {
                "total_n": straight_total_n,
                "oracle_unsafe_n": straight_oracle_unsafe_n,
                "nominal_proxy_false_safe_n": straight_nominal_false_safe_n,
                "nominal_proxy_false_unsafe_n": straight_nominal_false_unsafe_n,
                "nominal_proxy_disagreement_n": straight_nominal_disagreement_n,
                "nominal_proxy_false_safe_rate": safe_divide(straight_nominal_false_safe_n, straight_oracle_unsafe_n),
                "nominal_proxy_disagreement_rate": safe_divide(straight_nominal_disagreement_n, straight_total_n),
                "proposed_exactness_upper_95_rate_rule_of_three": rule_of_three_upper_bound(0, straight_total_n),
            },
            "mixed_turn": {
                "total_n": mixed_total_n,
                "total_eligible_n": mixed_total_eligible_n,
                "reference_safe_n": mixed_reference_safe_n,
                "reference_unsafe_n": mixed_reference_unsafe_n,
                "proposed_false_safe_n": proposed_false_safe_n,
                "proposed_false_unsafe_n": proposed_false_unsafe_n,
                "proposed_disagreement_n": proposed_disagreement_n,
                "nominal_proxy_false_safe_n": nominal_false_safe_n,
                "nominal_proxy_false_unsafe_n": nominal_false_unsafe_n,
                "nominal_proxy_disagreement_n": nominal_disagreement_n,
                "coarse_proxy_false_safe_n": coarse_false_safe_n,
                "coarse_proxy_false_unsafe_n": coarse_false_unsafe_n,
                "coarse_proxy_disagreement_n": coarse_disagreement_n,
                "proposed_false_safe_rate": safe_divide(proposed_false_safe_n, mixed_reference_unsafe_n),
                "proposed_false_unsafe_rate": safe_divide(proposed_false_unsafe_n, mixed_reference_safe_n),
                "proposed_disagreement_rate": safe_divide(proposed_disagreement_n, mixed_total_eligible_n),
                "proposed_certification_rate_given_reference_safe": (
                    1.0 - safe_divide(proposed_false_unsafe_n, mixed_reference_safe_n)
                ),
                "proposed_certification_rate_ci95": wilson_interval(
                    mixed_reference_safe_n - proposed_false_unsafe_n, mixed_reference_safe_n
                ),
                "nominal_proxy_false_safe_rate_ci95": wilson_interval(
                    nominal_false_safe_n, mixed_reference_unsafe_n
                ),
                "per_seed_certification_rate_given_reference_safe": [
                    1.0
                    - safe_divide(
                        int(record["mixed_turn"]["proposed_false_unsafe_n"]),
                        int(record["mixed_turn"]["reference_safe_n"]),
                    )
                    for record in run_records
                ],
                "just_unsafe": {
                    "total_n": just_unsafe_total_n,
                    "solver_false_safe_n": just_unsafe_solver_false_safe_n,
                    "nominal_proxy_false_safe_n": just_unsafe_nominal_false_safe_n,
                    "solver_false_safe_upper_95_rate_rule_of_three": rule_of_three_upper_bound(
                        just_unsafe_solver_false_safe_n, just_unsafe_total_n
                    ),
                },
                "proposed_false_safe_upper_95_rate_rule_of_three": rule_of_three_upper_bound(
                    proposed_false_safe_n,
                    mixed_reference_unsafe_n,
                ),
                "nominal_proxy_false_safe_rate": safe_divide(nominal_false_safe_n, mixed_reference_unsafe_n),
                "nominal_proxy_disagreement_rate": safe_divide(nominal_disagreement_n, mixed_total_eligible_n),
                "coarse_proxy_false_safe_rate": safe_divide(coarse_false_safe_n, mixed_reference_unsafe_n),
                "coarse_proxy_disagreement_rate": safe_divide(coarse_disagreement_n, mixed_total_eligible_n),
            },
            "near_threshold": {
                "default_local_dt3_total_n": near_default_total_n,
                "default_local_dt3_certified_n": near_default_certified_n,
                "default_local_dt3_certification_rate": safe_divide(near_default_certified_n, near_default_total_n),
                "default_local_dt3_certification_rate_ci95": wilson_interval(
                    near_default_certified_n, near_default_total_n
                ),
            },
        },
    }


def build_campaign_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in run_records:
        mixed = record["mixed_turn"]
        straight = record["straight"]
        near_threshold = record["near_threshold"]
        rows.append(
            {
                "seed": record["seed"],
                "straight_n": straight["n"],
                "straight_nominal_proxy_disagreement_n": straight["nominal_proxy_disagreement_n"],
                "mixed_turn_n": mixed["n"],
                "mixed_turn_eligible_n": mixed["eligible_n"],
                "mixed_turn_reference_unsafe_n": mixed["reference_unsafe_n"],
                "mixed_turn_proposed_false_safe_n": mixed["proposed_false_safe_n"],
                "mixed_turn_proposed_false_unsafe_n": mixed["proposed_false_unsafe_n"],
                "mixed_turn_nominal_proxy_false_safe_n": mixed["nominal_proxy_false_safe_n"],
                "mixed_turn_coarse_proxy_false_safe_n": mixed["coarse_proxy_false_safe_n"],
                "near_threshold_default_local_dt3_total_n": near_threshold["default_local_dt3_total_n"],
                "near_threshold_default_local_dt3_certified_n": near_threshold["default_local_dt3_certified_n"],
            }
        )
    return rows


def safe_divide(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 percent score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2.0 * trials)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three_upper_bound(failures: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    if failures == 0:
        return 3.0 / float(trials)
    return None


def build_environment_summary(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os_cpu_count(),
        "warm_path_policy": "First-call JIT excluded from main runtime tables.",
        "reference_lat_lon": [config.ref_lat, config.ref_lon],
    }


def build_table_1(config: EvaluationConfig) -> list[dict[str, Any]]:
    return [
        {
            "section": "Encounter generation",
            "item": "Straight suite",
            "definition": (
                f"N={config.straight_n}; range 6-40 NMI; headings 0-360 deg; speeds 220-450 kt; "
                "speed uncertainty 5-30 kt; horizons 300/600/900/1200 s."
            ),
        },
        {
            "section": "Encounter generation",
            "item": "Mixed-turn suite",
            "definition": (
                f"N={config.mixed_turn_n}; same ranges as straight suite; zero-turn probability "
                f"{config.turn_zero_probability:.2f}; nonzero turns 10-60 deg at 1.0-3.5 deg/s."
            ),
        },
        {
            "section": "Encounter generation",
            "item": "Near-threshold suite",
            "definition": (
                f"N={config.near_threshold_n}; accepted by dense reference; at least one turning aircraft; "
                "safe margin 0.05-1.50 NMI."
            ),
        },
        {
            "section": "Comparators",
            "item": "Dense bounded-speed reference",
            "definition": "0.25 s coarse grid, 0.05 s local refinement around minima and near-threshold neighborhoods.",
        },
        {
            "section": "Comparators",
            "item": "Nominal-motion proxy",
            "definition": "Same commanded turn law, one nominal speed per aircraft, no speed envelope.",
        },
        {
            "section": "Comparators",
            "item": "Sampled turn-aware heuristic",
            "definition": (
                f"Exact fixed-time bounded hull sampled every {config.coarse_proxy_dt_s:.1f} s "
                "with no interval certification."
            ),
        },
        {
            "section": "Ablation",
            "item": "Turn-aware certifier",
            "definition": f"Lipschitz modes local/global; dt_min sweep {list(config.dt_min_values_s)} s.",
        },
    ]


def build_table_2(straight_summary: dict[str, Any]) -> list[dict[str, Any]]:
    nominal = straight_summary["nominal_proxy"]
    return [
        {"metric": "Oracle vs proposed agreement", "value": straight_summary["oracle_vs_proposed_agreement_rate"]},
        {"metric": "Max distance error (m)", "value": straight_summary["max_distance_error_m"]},
        {"metric": "Max closest-time error (s)", "value": straight_summary["max_time_error_s"]},
        {
            "metric": "Independent oracle vs proposed agreement",
            "value": straight_summary["independent_vs_proposed_agreement_rate"],
        },
        {
            "metric": "Max distance error vs independent (m)",
            "value": straight_summary["max_distance_error_vs_independent_m"],
        },
        {
            "metric": "Max closest-time error vs independent (s)",
            "value": straight_summary["max_time_error_vs_independent_s"],
        },
        {"metric": "Nominal proxy false-safe rate", "value": nominal["false_safe_rate"]},
        {"metric": "Nominal proxy false-unsafe rate", "value": nominal["false_unsafe_rate"]},
        {"metric": "Nominal proxy disagreement rate", "value": nominal["overall_disagreement_rate"]},
    ]


def build_table_3(mixed_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in ("proposed", "nominal_proxy", "coarse_proxy"):
        entry = mixed_summary[name]
        rows.append(
            {
                "method": name,
                "false_safe_rate": entry["false_safe_rate"],
                "false_unsafe_rate": entry["false_unsafe_rate"],
                "overall_disagreement_rate": entry["overall_disagreement_rate"],
                "eligible_n": entry["eligible_n"],
            }
        )
    rows.append(
        {
            "method": "proposed_certification_given_reference_safe",
            "false_safe_rate": "",
            "false_unsafe_rate": "",
            "overall_disagreement_rate": mixed_summary["certification_rate_given_reference_safe"],
            "eligible_n": mixed_summary["eligible_n"],
        }
    )
    return rows


def build_table_4(
    *,
    kernel_runtime: dict[str, float],
    mixed_runtime: dict[str, float],
    ablation_rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {"component": "Fixed-time kernel", **kernel_runtime},
        {"component": "Full solver broad sweep", **mixed_runtime},
    ]
    rows.extend(
        {
            "component": "Near-threshold ablation (local, dt_min=3)",
            "mean_us": "",
            "median_us": row["median_runtime_us"],
            "p50_us": "",
            "p95_us": row["p95_fixed_time_evals"],
            "p99_us": "",
            "max_us": row["median_fixed_time_evals"],
        }
        for row in ablation_rows
        if row["dt_min_s"] == 3.0 and row["lipschitz_mode"] == "local"
    )
    rows.append(
        {
            "component": "Environment",
            "mean_us": environment["platform"],
            "median_us": environment["processor"],
            "p50_us": environment["cpu_count"],
            "p95_us": environment["python"],
            "p99_us": environment["numpy"],
            "max_us": environment["numba"],
        }
    )
    return rows


def plot_crossing_scenario(path: Path, row: dict[str, Any]) -> None:
    encounter = encounter_from_row(row)
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_nominal_trajectories(ax, encounter)
    min_distance_key = next(
        key for key in ("oracle_min_distance_m", "reference_min_distance_m", "proposed_min_distance_m") if key in row
    )
    closest_time_key = next(
        key for key in ("oracle_closest_time_s", "reference_closest_time_s", "proposed_closest_time_s") if key in row
    )
    add_annotation_box(
        ax,
        [
            f"$d_{{\\min}} = {float(row[min_distance_key]) / NMI_TO_M:.2f}$ NMI",
            f"$t_{{\\min}} = {float(row[closest_time_key]):.1f}$ s",
        ],
        anchor="lower_left",
    )
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_proxy_miss(path: Path, row: dict[str, Any]) -> None:
    encounter = encounter_from_row(row)
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_nominal_trajectories(ax, encounter)
    add_annotation_box(
        ax,
        [
            f"Nominal: {'safe' if int(row['nominal_proxy_is_safe']) else 'unsafe'} "
            f"({float(row['nominal_proxy_margin_m']) / NMI_TO_M:+.2f} NMI)",
            f"Bounded model: {'safe' if int(row['reference_is_safe']) else 'unsafe'} "
            f"({float(row['reference_margin_m']) / NMI_TO_M:+.2f} NMI)",
        ],
        anchor="lower_left",
    )
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_sampled_proxy_miss(path: Path, row: dict[str, Any], *, sample_dt_s: float) -> None:
    encounter = encounter_from_row(row)
    center_s = float(row["reference_closest_time_s"])
    left_s = max(0.0, center_s - 18.0)
    right_s = min(encounter.projection_time_s, center_s + 18.0)
    dense_times = np.arange(left_s, right_s + 0.025, 0.05)
    dense_distances = _bounded_distances_at_times(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        speed_diff_kt=encounter.speed_diff_kt,
        times_s=dense_times,
    )
    sample_times = build_uniform_times(encounter.projection_time_s, sample_dt_s)
    sample_mask = (sample_times >= left_s - 1e-12) & (sample_times <= right_s + 1e-12)
    sample_window_times = sample_times[sample_mask]
    sample_window_distances = _bounded_distances_at_times(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        speed_diff_kt=encounter.speed_diff_kt,
        times_s=sample_window_times,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    style_axes(ax)
    margin_nmi = (dense_distances - encounter.separation_threshold_m) / NMI_TO_M
    ax.axhline(0.0, color="#444444", linestyle=":", linewidth=1.2, label="Threshold")
    ax.fill_between(dense_times, margin_nmi, 0.0, where=margin_nmi < 0.0, color="#d95f5f", alpha=0.5)
    ax.plot(
        dense_times,
        margin_nmi,
        color="#1f4e79",
        linewidth=2.2,
        label="Dense bounded-speed reference",
    )
    ax.vlines(sample_window_times, ymin=min(np.min(margin_nmi), -0.15), ymax=0.0, color="#c7c7c7", linewidth=0.8)
    ax.scatter(
        sample_window_times,
        (sample_window_distances - encounter.separation_threshold_m) / NMI_TO_M,
        s=46,
        marker="s",
        color="#b55d1d",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=f"{sample_dt_s:g} s sampled heuristic",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Clearance margin (NMI)")
    ax.set_title("Sampled heuristic near a short unsafe interval")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_stress_maps(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.55))
    plot_heatmap(
        ax=axes[0],
        rows=rows,
        x_edges=MARGIN_BINS_NMI,
        y_edges=TURN_ANGLE_BINS_DEG[1:],
        x_getter=lambda row: max(row["reference_margin_m"] / NMI_TO_M, 0.0),
        y_getter=lambda row: row["max_turn_angle_deg"],
        mask=lambda row: row["reference_is_safe"] == 1,
        value=lambda row: row["proposed_is_safe"],
        title="Certified solver: certification on safe cases",
        xlabel="Dense-reference safe margin (NMI)",
        ylabel="Maximum turn angle (deg)",
        cmap="Blues",
        colorbar_label="Rate",
        x_decimals=2,
        y_decimals=0,
        x_lower_start=0.0,
        y_lower_start=0.0,
    )
    plot_heatmap(
        ax=axes[1],
        rows=rows,
        x_edges=SPEED_UNCERTAINTY_BINS_KT[1:],
        y_edges=TURN_ANGLE_BINS_DEG[1:],
        x_getter=lambda row: row["speed_diff_kt"],
        y_getter=lambda row: row["max_turn_angle_deg"],
        mask=lambda row: row["eligible_for_boolean_tallies"] == 1 and row["reference_is_safe"] == 0,
        value=lambda row: row["nominal_proxy_is_safe"],
        title="Nominal-motion baseline: false-safe rate on unsafe cases",
        xlabel="Speed uncertainty (kt)",
        ylabel="Maximum turn angle (deg)",
        cmap="OrRd",
        colorbar_label="Rate",
        x_decimals=0,
        y_decimals=0,
        x_lower_start=5.0,
        y_lower_start=0.0,
    )
    fig.tight_layout(w_pad=2.6)
    save_figure(fig, path)
    plt.close(fig)


def build_stress_map_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    output_rows.extend(
        build_heatmap_source_rows(
            rows=rows,
            panel="proposed_certification",
            title="Certified solver: certification on safe cases",
            xlabel="Dense-reference safe margin (NMI)",
            ylabel="Maximum turn angle (deg)",
            x_edges=MARGIN_BINS_NMI,
            y_edges=TURN_ANGLE_BINS_DEG[1:],
            x_getter=lambda row: max(row["reference_margin_m"] / NMI_TO_M, 0.0),
            y_getter=lambda row: row["max_turn_angle_deg"],
            mask=lambda row: row["reference_is_safe"] == 1,
            value=lambda row: row["proposed_is_safe"],
            x_decimals=2,
            y_decimals=0,
            x_lower_start=0.0,
            y_lower_start=0.0,
        )
    )
    output_rows.extend(
        build_heatmap_source_rows(
            rows=rows,
            panel="nominal_false_safe",
            title="Nominal-motion baseline: false-safe rate on unsafe cases",
            xlabel="Speed uncertainty (kt)",
            ylabel="Maximum turn angle (deg)",
            x_edges=SPEED_UNCERTAINTY_BINS_KT[1:],
            y_edges=TURN_ANGLE_BINS_DEG[1:],
            x_getter=lambda row: row["speed_diff_kt"],
            y_getter=lambda row: row["max_turn_angle_deg"],
            mask=lambda row: row["eligible_for_boolean_tallies"] == 1 and row["reference_is_safe"] == 0,
            value=lambda row: row["nominal_proxy_is_safe"],
            x_decimals=0,
            y_decimals=0,
            x_lower_start=5.0,
            y_lower_start=0.0,
        )
    )
    return output_rows


def build_heatmap_source_rows(
    *,
    rows: list[dict[str, Any]],
    panel: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
    x_getter: Callable[[dict[str, Any]], float],
    y_getter: Callable[[dict[str, Any]], float],
    mask: Callable[[dict[str, Any]], bool],
    value: Callable[[dict[str, Any]], float | int],
    x_decimals: int,
    y_decimals: int,
    x_lower_start: float,
    y_lower_start: float,
) -> list[dict[str, Any]]:
    x_bins = np.asarray(x_edges, dtype=np.float64)
    y_bins = np.asarray(y_edges, dtype=np.float64)
    x_labels = format_interval_labels(x_bins, decimals=x_decimals, lower_start=x_lower_start)
    y_labels = format_interval_labels(y_bins, decimals=y_decimals, lower_start=y_lower_start)
    output_rows = []
    for yi, y_upper in enumerate(y_bins):
        y_lower = y_lower_start if yi == 0 else y_bins[yi - 1]
        for xi, x_upper in enumerate(x_bins):
            x_lower = x_lower_start if xi == 0 else x_bins[xi - 1]
            bucket = [
                row
                for row in rows
                if mask(row) and x_lower <= float(x_getter(row)) < x_upper and y_lower <= float(y_getter(row)) < y_upper
            ]
            numerator = float(np.sum([float(value(row)) for row in bucket])) if bucket else 0.0
            numerator_n = round(numerator)
            denominator = len(bucket)
            output_rows.append(
                {
                    "panel": panel,
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "x_index": xi,
                    "y_index": yi,
                    "x_label": x_labels[xi],
                    "y_label": y_labels[yi],
                    "x_lower": float(x_lower),
                    "x_upper": float(x_upper),
                    "y_lower": float(y_lower),
                    "y_upper": float(y_upper),
                    "numerator_n": numerator_n,
                    "denominator_n": denominator,
                    "rate": safe_divide(numerator_n, denominator) if denominator else float("nan"),
                }
            )
    return output_rows


def plot_stress_maps_from_source(path: Path, source_rows: list[dict[str, Any]] | None) -> None:
    if not source_rows:
        raise ValueError("Stress-map source rows are required.")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.55))
    for ax, panel, cmap in (
        (axes[0], "proposed_certification", "Blues"),
        (axes[1], "nominal_false_safe", "OrRd"),
    ):
        panel_rows = [row for row in source_rows if row["panel"] == panel]
        if not panel_rows:
            raise ValueError(f"Missing stress-map source rows for panel={panel!r}.")
        plot_heatmap_source_panel(ax=ax, rows=panel_rows, cmap=cmap)
    fig.tight_layout(w_pad=2.6)
    save_figure(fig, path)
    plt.close(fig)


def plot_ablation(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.45), sharex=True)
    palette = {"local": "#1f4e79", "global": "#b55d1d"}
    labels = {"local": "Interval-local", "global": "Global"}
    for lipschitz_mode in ("local", "global"):
        subset = [row for row in summary_rows if row["lipschitz_mode"] == lipschitz_mode]
        subset.sort(key=lambda row: row["dt_min_s"])
        dt_values = [row["dt_min_s"] for row in subset]
        certification = [row["certification_rate"] for row in subset]
        eval_counts = [row["median_fixed_time_evals"] for row in subset]
        color = palette[lipschitz_mode]
        ax1.plot(
            dt_values,
            certification,
            marker="o",
            linewidth=2.2,
            markersize=6.2,
            color=color,
            label=labels[lipschitz_mode],
        )
        ax2.plot(
            dt_values,
            eval_counts,
            marker="s",
            linewidth=2.2,
            markersize=6.0,
            color=color,
            label=labels[lipschitz_mode],
        )
    style_axes(ax1)
    style_axes(ax2)
    ax1.set_xlabel(r"Minimum interval width, $\Delta t_{\min}$ (s)", fontsize=15.5)
    ax1.set_ylabel(r"Certification rate ($\rightarrow$)", fontsize=15.5)
    ax1.set_title("Certification rate", fontsize=18.0, pad=8)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_ylim(0.5, 1.01)
    ax2.set_xlabel(r"Minimum interval width, $\Delta t_{\min}$ (s)", fontsize=15.5)
    ax2.set_ylabel(r"Median fixed-time hull evaluations ($\leftarrow$)", fontsize=15.5)
    ax2.set_title("Subdivision work", fontsize=18.0, pad=8)
    ax2.set_ylim(0.0, 13.25)
    ax1.set_xticks([1.0, 3.0, 6.0, 12.0])
    ax2.set_xticks([1.0, 3.0, 6.0, 12.0])
    ax1.tick_params(axis="both", labelsize=14.0)
    ax2.tick_params(axis="both", labelsize=14.0)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=14.0)
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, path)
    plt.close(fig)


def draw_nominal_trajectories(ax: plt.Axes, encounter: Encounter) -> None:
    times_s = np.linspace(0.0, encounter.projection_time_s, 200)
    a_path = nominal_path(encounter, aircraft="a", times_s=times_s)
    b_path = nominal_path(encounter, aircraft="b", times_s=times_s)
    color_a = "#1f4e79"
    color_b = "#b55d1d"
    ax.plot(a_path[:, 0] / NMI_TO_M, a_path[:, 1] / NMI_TO_M, color=color_a, linewidth=2.4, label="Aircraft A")
    ax.plot(b_path[:, 0] / NMI_TO_M, b_path[:, 1] / NMI_TO_M, color=color_b, linewidth=2.4, label="Aircraft B")
    ax.scatter(
        [a_path[0, 0] / NMI_TO_M],
        [a_path[0, 1] / NMI_TO_M],
        s=52,
        marker="o",
        facecolor="white",
        edgecolor=color_a,
        linewidth=1.4,
        zorder=3,
    )
    ax.scatter(
        [b_path[0, 0] / NMI_TO_M],
        [b_path[0, 1] / NMI_TO_M],
        s=52,
        marker="o",
        facecolor="white",
        edgecolor=color_b,
        linewidth=1.4,
        zorder=3,
    )
    add_path_arrow(ax, a_path[:, 0] / NMI_TO_M, a_path[:, 1] / NMI_TO_M, color=color_a)
    add_path_arrow(ax, b_path[:, 0] / NMI_TO_M, b_path[:, 1] / NMI_TO_M, color=color_b)
    ax.set_xlabel("East (NMI)", fontsize=14)
    ax.set_ylabel("North (NMI)", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.tick_params(axis="both", labelsize=12)
    style_axes(ax)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, fontsize=12)


def nominal_path(encounter: Encounter, *, aircraft: str, times_s: np.ndarray) -> np.ndarray:
    if aircraft == "a":
        heading0_deg = encounter.a_heading0_deg
        target_heading_deg = encounter.a_target_heading_deg
        turn_rate_deg_sec = encounter.a_turn_rate_deg_sec
        speed_mps = encounter.a_speed_kt * KT_TO_MPS
        origin = np.zeros(2, dtype=np.float64)
    else:
        heading0_deg = encounter.b_heading0_deg
        target_heading_deg = encounter.b_target_heading_deg
        turn_rate_deg_sec = encounter.b_turn_rate_deg_sec
        speed_mps = encounter.b_speed_kt * KT_TO_MPS
        origin = np.array([-encounter.rel_east_m, -encounter.rel_north_m], dtype=np.float64)
    path = np.empty((times_s.shape[0], 2), dtype=np.float64)
    for i, t_s in enumerate(times_s):
        disp = speed_mps * ra._turn_displacement_basis(heading0_deg, target_heading_deg, turn_rate_deg_sec, float(t_s))
        path[i] = origin + disp
    return path


def plot_heatmap_source_panel(
    *,
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    cmap: str,
) -> None:
    x_count = max(int(row["x_index"]) for row in rows) + 1
    y_count = max(int(row["y_index"]) for row in rows) + 1
    grid = np.full((y_count, x_count), np.nan, dtype=np.float64)
    x_labels = [""] * x_count
    y_labels = [""] * y_count
    for row in rows:
        xi = int(row["x_index"])
        yi = int(row["y_index"])
        grid[yi, xi] = float(row["rate"])
        x_labels[xi] = str(row["x_label"])
        y_labels[yi] = str(row["y_label"])

    first_row = rows[0]
    draw_heatmap_grid(
        ax=ax,
        grid=grid,
        x_labels=x_labels,
        y_labels=y_labels,
        title=str(first_row["title"]),
        xlabel=str(first_row["xlabel"]),
        ylabel=str(first_row["ylabel"]),
        cmap=cmap,
        colorbar_label="Rate",
    )


def draw_heatmap_grid(
    *,
    ax: plt.Axes,
    grid: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    masked = np.ma.masked_invalid(grid)
    mesh = ax.pcolormesh(
        np.arange(grid.shape[1] + 1),
        np.arange(grid.shape[0] + 1),
        masked,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.8,
        shading="flat",
    )
    style_axes(ax, grid=False)
    ax.set_xticks(np.arange(grid.shape[1]) + 0.5)
    ax.set_xticklabels(x_labels, rotation=32, ha="right")
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5)
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=18.0, pad=8)
    ax.set_xlabel(xlabel, fontsize=16.0)
    ax.set_ylabel(ylabel, fontsize=16.0)
    ax.tick_params(axis="both", labelsize=13.5)
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            value_at_cell = grid[yi, xi]
            if np.isnan(value_at_cell):
                continue
            text_color = "white" if value_at_cell >= 0.55 else "#222222"
            ax.text(
                xi + 0.5,
                yi + 0.5,
                f"{100.0 * value_at_cell:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13.0,
            )
    colorbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    colorbar.set_label(colorbar_label, fontsize=15.0)
    colorbar.ax.tick_params(labelsize=13.5)


def plot_heatmap(
    *,
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
    x_getter: Callable[[dict[str, Any]], float],
    y_getter: Callable[[dict[str, Any]], float],
    mask: Callable[[dict[str, Any]], bool],
    value: Callable[[dict[str, Any]], float | int],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    colorbar_label: str,
    x_decimals: int,
    y_decimals: int,
    x_lower_start: float,
    y_lower_start: float,
) -> None:
    x_bins = np.asarray(x_edges, dtype=np.float64)
    y_bins = np.asarray(y_edges, dtype=np.float64)
    grid = np.full((len(y_bins), len(x_bins)), np.nan, dtype=np.float64)
    for yi, y_upper in enumerate(y_bins):
        y_lower = y_lower_start if yi == 0 else y_bins[yi - 1]
        for xi, x_upper in enumerate(x_bins):
            x_lower = x_lower_start if xi == 0 else x_bins[xi - 1]
            bucket = [
                row
                for row in rows
                if mask(row) and x_lower <= float(x_getter(row)) < x_upper and y_lower <= float(y_getter(row)) < y_upper
            ]
            if bucket:
                grid[yi, xi] = float(np.mean([float(value(row)) for row in bucket]))
    draw_heatmap_grid(
        ax=ax,
        grid=grid,
        x_labels=format_interval_labels(x_bins, decimals=x_decimals, lower_start=x_lower_start),
        y_labels=format_interval_labels(y_bins, decimals=y_decimals, lower_start=y_lower_start),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        colorbar_label=colorbar_label,
    )


def encounter_from_row(row: dict[str, Any]) -> Encounter:
    return Encounter(
        rel_east_m=float(row["rel_east_m"]),
        rel_north_m=float(row["rel_north_m"]),
        a_heading0_deg=float(row["a_heading0_deg"]),
        a_target_heading_deg=float(row["a_target_heading_deg"]),
        a_speed_kt=float(row["a_speed_kt"]),
        a_turn_rate_deg_sec=float(row["a_turn_rate_deg_sec"]),
        b_heading0_deg=float(row["b_heading0_deg"]),
        b_target_heading_deg=float(row["b_target_heading_deg"]),
        b_speed_kt=float(row["b_speed_kt"]),
        b_turn_rate_deg_sec=float(row["b_turn_rate_deg_sec"]),
        speed_diff_kt=float(row["speed_diff_kt"]),
        projection_time_s=float(row["projection_time_s"]),
        separation_threshold_m=float(row["separation_threshold_m"]),
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path)
        return
    fig.savefig(path, dpi=200)


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor("#fbfbfb")
    if grid:
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b7b7b7")
    ax.spines["bottom"].set_color("#b7b7b7")


def add_annotation_box(ax: plt.Axes, lines: list[str], *, anchor: str = "upper_left") -> None:
    if anchor == "lower_left":
        x_pos, y_pos = 0.02, 0.02
        vertical_alignment = "bottom"
    else:
        x_pos, y_pos = 0.02, 0.98
        vertical_alignment = "top"
    ax.text(
        x_pos,
        y_pos,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va=vertical_alignment,
        fontsize=11.5,
        bbox={"facecolor": "white", "edgecolor": "#c8c8c8", "boxstyle": "round,pad=0.28", "alpha": 0.95},
    )


def add_path_arrow(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, *, color: str) -> None:
    if x_values.size < 3:
        return
    end_index = x_values.size - 1
    start_index = max(0, end_index - 14)
    dx = x_values[end_index] - x_values[end_index - 1]
    dy = y_values[end_index] - y_values[end_index - 1]
    segment_norm = math.hypot(dx, dy)
    if segment_norm > 0.0:
        extend = 0.45
        x_tip = x_values[end_index] + extend * dx / segment_norm
        y_tip = y_values[end_index] + extend * dy / segment_norm
    else:
        x_tip = x_values[end_index]
        y_tip = y_values[end_index]
    ax.annotate(
        "",
        xy=(x_tip, y_tip),
        xytext=(x_values[start_index], y_values[start_index]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 2.0,
            "mutation_scale": 18.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )


def format_interval_labels(edges: np.ndarray, *, decimals: int, lower_start: float = 0.0) -> list[str]:
    labels = []
    lower = lower_start
    for upper in edges:
        labels.append(f"[{lower:.{decimals}f}, {upper:.{decimals}f})")
        lower = upper
    return labels


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_csv_or_gzip(path: Path) -> Path | None:
    if path.exists():
        return path
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        return gzip_path
    return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: coerce_csv_value(value) for key, value in row.items()} for row in reader]


def coerce_csv_value(value: str | None) -> object:
    if value is None or value == "":
        return ""
    lowered = value.lower()
    if lowered in {"nan", "inf", "-inf"}:
        return float(value)
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer() and all(char not in value.lower() for char in (".", "e")):
        return int(numeric)
    return numeric


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if not rows:
        with opener(path, "wt", newline="", encoding="utf-8"):
            pass
        return
    fieldnames = sorted({key for row in rows for key in row})
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def os_cpu_count() -> int:
    count = os.cpu_count()
    return 0 if count is None else count


if __name__ == "__main__":
    main()
