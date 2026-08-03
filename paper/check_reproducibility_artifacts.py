#!/usr/bin/env python3
"""Check that a paper reproducibility run contains the expected artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

RUN_ARTIFACTS = (
    "summary.json",
    "run_record.json",
    "stress_map_source.csv",
    "near_threshold_default_margin_summary.csv",
    "table_1_experiment_design.csv",
    "table_2_straight_exactness.csv",
    "table_3_mixed_turn_results.csv",
    "table_4_runtime_summary.csv",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTION_ARCHIVE = (
    REPO_ROOT / "paper" / "results" / "projection_deterministic_51_cap_100nmi.tar.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Paper reproducibility run root under paper_eval_outputs.")
    parser.add_argument(
        "--projection-archive",
        type=Path,
        default=DEFAULT_PROJECTION_ARCHIVE,
        help="Rendered capped projection archive.",
    )
    parser.add_argument("--require-rendered", action="store_true", help="Require rendered figure PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = check_run(args.run_root, args.projection_archive, require_rendered=args.require_rendered)
    if missing:
        for path in missing:
            print(f"MISSING {path}")
        raise SystemExit(1)
    print(f"OK {args.run_root}")


def check_run(run_root: Path, projection_archive: Path, *, require_rendered: bool) -> list[str]:
    main_eval_dir = run_root / "main_evaluation"
    first_run_dir = first_evaluation_run_dir(main_eval_dir)
    expected = [
        run_root / "logs" / "main_evaluation.log",
        run_root / "logs" / "projection_sweep.log",
        run_root / "projection_full" / "summary.json",
        run_root / "projection_full" / "projection_rows.csv.gz",
    ]
    if is_campaign_run(main_eval_dir):
        expected.extend([main_eval_dir / "campaign_summary.json", main_eval_dir / "campaign_runs.csv"])
    if first_run_dir is None:
        expected.append(main_eval_dir / "summary.json")
    else:
        expected.extend(first_run_dir / name for name in RUN_ARTIFACTS)
    if require_rendered:
        expected.extend(
            [
                run_root / "rendered_figures" / "main" / "fig_stress_maps.pdf",
                run_root / "rendered_figures" / "main" / "fig_ablation.pdf",
                run_root / "rendered_figures" / "main" / "fig_sampled_proxy_miss.pdf",
                run_root / "rendered_figures" / "application" / "fig_lateral_overlap_schematic.pdf",
                run_root / "rendered_figures" / "application" / "fig_clearance_grid.pdf",
                run_root / "rendered_figures" / "application" / "fig_clearance_grid_source.json.gz",
                run_root
                / "rendered_figures"
                / "projection_deterministic_51_cap_100nmi"
                / "projection_min_error_vs_excursion.pdf",
            ]
        )
    missing = [str(path) for path in expected if not path.exists()]
    missing.extend(check_campaign_runs(run_root))
    if require_rendered:
        missing.extend(check_projection_archive(projection_archive))
    return missing


def is_campaign_run(main_eval_dir: Path) -> bool:
    return (main_eval_dir / "campaign_summary.json").exists()


def first_evaluation_run_dir(main_eval_dir: Path) -> Path | None:
    """Return the first per-seed run directory, or the single-run directory."""
    if not is_campaign_run(main_eval_dir):
        return main_eval_dir if (main_eval_dir / "summary.json").exists() else None
    summary = json.loads((main_eval_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    seeds = summary.get("seeds", [])
    if seeds:
        return main_eval_dir / f"run_00_seed_{seeds[0]}"
    candidates = sorted(path for path in main_eval_dir.glob("run_00_seed_*") if path.is_dir())
    return candidates[0] if candidates else None


def check_campaign_runs(run_root: Path) -> list[str]:
    summary_path = run_root / "main_evaluation" / "campaign_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = []
    for index, seed in enumerate(summary.get("seeds", [])):
        run_dir = run_root / "main_evaluation" / f"run_{index:02d}_seed_{seed}"
        for name in RUN_ARTIFACTS:
            path = run_dir / name
            if not path.exists():
                missing.append(str(path))
    return missing


def check_projection_archive(projection_archive: Path) -> list[str]:
    if not projection_archive.exists():
        return [str(projection_archive)]
    expected_members = {
        "summary.json",
        "projection_rows.csv",
        "projection_min_error_vs_excursion.pdf",
        "projection_error_by_range_and_horizon.pdf",
        "projection_orientation_disagreement.pdf",
        "projection_disagreement_by_excursion.pdf",
    }
    with tarfile.open(projection_archive, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    return [f"{projection_archive}:{member}" for member in sorted(expected_members - members)]


if __name__ == "__main__":
    main()
