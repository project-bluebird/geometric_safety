"""Build a compact, repo-committable archive from a full robustness campaign run.

The full campaign output carries wide per-row CSVs whose config columns repeat on
every row. This tool keeps the per-run summaries and boundary caches verbatim and
slims the rows to the columns the figures and the manuscript fact checker consume,
compressed per run. Usage:

    uv run python -m paper.archive_robustness \
        --results-dir paper_eval_outputs/robustness_full_20260720 \
        --output paper/results/robustness_compact_20260720.tar.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import tarfile

SLIM_COLUMNS = (
    "campaign",
    "label",
    "suite",
    "encounter_idx",
    "projection_time_s",
    "max_turn_angle_deg",
    "turn_count",
    "speed_diff_kt",
    "separation_threshold_m",
    "certified_safe",
    "inflated_certified_safe",
    "n_sims",
    "n_below_threshold",
    "n_violations",
    "truth_min_separation_m",
    "eta_horizon_m",
)


def slim_rows_bytes(rows_path: Path) -> bytes:
    """Return the gzip-compressed slimmed rows CSV for one run."""
    buffer = io.StringIO()
    with rows_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(buffer, fieldnames=SLIM_COLUMNS)
        writer.writeheader()
        for row in reader:
            writer.writerow({column: row[column] for column in SLIM_COLUMNS})
    return gzip.compress(buffer.getvalue().encode("utf-8"))


def build_archive(results_dir: Path, output_path: Path) -> None:
    run_dirs = sorted(results_dir.glob("run_*_seed_*"))
    if not run_dirs:
        raise SystemExit(f"no run directories under {results_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(results_dir.name), "runs": [run_dir.name for run_dir in run_dirs]}
    with tarfile.open(output_path, "w:gz") as archive:

        def add_bytes(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        add_bytes("compact_manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        for run_dir in run_dirs:
            for name in ("robustness_summary.json", "robustness_summary.csv"):
                archive.add(run_dir / name, arcname=f"{run_dir.name}/{name}")
            add_bytes(
                f"{run_dir.name}/robustness_rows_slim.csv.gz",
                slim_rows_bytes(run_dir / "robustness_rows.csv"),
            )
        cache_dir = results_dir / "boundary_cache"
        if cache_dir.is_dir():
            for cache_file in sorted(cache_dir.glob("*.json")):
                archive.add(cache_file, arcname=f"boundary_cache/{cache_file.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_archive(args.results_dir, args.output)
    size_mib = args.output.stat().st_size / (1024 * 1024)
    print(f"wrote {args.output} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
