"""Export paper figure files from the public reproduction pipeline.

The expensive empirical source data are produced by this repository's
``paper`` pipeline. This wrapper regenerates the explanatory application
figures from the public code and copies the rendered empirical PDFs from a
saved paper run into a local ``manuscript/figures/...`` output tree.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_ROOT = Path(os.environ.get("GEOMETRIC_SAFETY_ROOT", str(REPO_ROOT)))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PAPER_FIGURE_OUTPUT_ROOT", str(Path.cwd())))

APPLICATION_COMMANDS = [
    (
        "lateral-overlap-schematic",
        "paper.render_lateral_overlap_figure",
        "manuscript/figures/application/fig_lateral_overlap_schematic.pdf",
        None,
    ),
    (
        "method-geometry",
        "paper.render_method_geometry_figure",
        "manuscript/figures/application/fig_method_geometry.pdf",
        None,
    ),
    (
        "clearance-grid",
        "paper.render_clearance_grid_figure",
        "manuscript/figures/application/fig_clearance_grid.pdf",
        "manuscript/figures/application/fig_clearance_grid_source.json.gz",
    ),
]

EMPIRICAL_FIGURES = [
    (
        "mixed-turn stress maps",
        "rendered_figures/main/fig_stress_maps.pdf",
        "manuscript/figures/evaluation/fig_stress_maps.pdf",
    ),
    (
        "near-threshold ablation",
        "rendered_figures/main/fig_ablation.pdf",
        "manuscript/figures/evaluation/fig_ablation.pdf",
    ),
    (
        "sampled-heuristic supplement figure",
        "rendered_figures/main/fig_sampled_proxy_miss.pdf",
        "manuscript/figures/evaluation/fig_sampled_proxy_miss.pdf",
    ),
    (
        "projection-sensitivity supplement figure",
        "rendered_figures/projection_deterministic_51_cap_100nmi/projection_min_error_vs_excursion.pdf",
        "manuscript/figures/evaluation/fig_projection_excursion.pdf",
    ),
    (
        "robustness exceedance degradation",
        "rendered_figures/robustness/fig_robustness_exceedance.pdf",
        "manuscript/figures/evaluation/fig_robustness_exceedance.pdf",
    ),
    (
        "robustness misspecification comparison",
        "rendered_figures/robustness/fig_robustness_misspec.pdf",
        "manuscript/figures/evaluation/fig_robustness_misspec.pdf",
    ),
    (
        "envelope budget against look-ahead time",
        "rendered_figures/budget/fig_envelope_budget.pdf",
        "manuscript/figures/evaluation/fig_envelope_budget.pdf",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=DEFAULT_LIBRARY_ROOT,
        help="Path to the geometric_safety repository.",
    )
    parser.add_argument(
        "--output-root",
        dest="manuscript_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="OUTPUT_ROOT",
        help="Destination root for the generated manuscript/figures layout.",
    )
    parser.add_argument("--manuscript-root", dest="manuscript_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Extracted paper run root with rendered_figures/ present.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Compact paper-run .tar.gz archive to import empirical figures from.",
    )
    parser.add_argument(
        "--grid-density",
        type=int,
        default=600,
        help="Grid density for the clearance-grid application figure; 600 evaluates 360,000 grid cells.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON manifest to write after successful reproduction.",
    )
    parser.add_argument("--skip-application", action="store_true", help="Do not regenerate application figures.")
    parser.add_argument("--skip-empirical", action="store_true", help="Do not import empirical figures.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_paper_run_archive(library_root: Path) -> Path | None:
    """Return the newest compact paper-run archive, if one exists."""
    result_dir = library_root / "paper" / "results"
    archives = sorted(
        (
            path
            for path in result_dir.glob("*.tar.gz")
            if not path.name.startswith("projection_")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return archives[0] if archives else None


def locate_run_root(path: Path) -> Path:
    """Locate the run root within a directory or extracted archive tree."""
    if (path / "main_evaluation" / "campaign_summary.json").exists():
        return path

    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "main_evaluation" / "campaign_summary.json").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Could not find a paper run root under {path}")
    raise RuntimeError(f"Multiple paper run roots found under {path}: {candidates}")


@contextmanager
def resolved_run_root(
    *,
    run_root: Path | None,
    archive: Path | None,
    library_root: Path,
) -> Iterator[Path]:
    """Yield a paper run root from either a directory or an archive."""
    if run_root is not None:
        yield locate_run_root(run_root.resolve())
        return

    archive_path = archive or find_latest_paper_run_archive(library_root)
    if archive_path is None:
        raise FileNotFoundError(
            "No paper-run archive found. Pass --run-root or --archive, "
            "or generate one with ./paper/run_full_results.sh and ./paper/render_paper_figures.sh."
        )

    with tempfile.TemporaryDirectory(prefix="paper_repro_") as temp_dir:
        temp_path = Path(temp_dir)
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(temp_path, filter="data")
        yield locate_run_root(temp_path)


def default_manifest_path(manuscript_root: Path) -> Path:
    return manuscript_root / "manuscript" / "figures" / "figure_manifest.json"


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def run_application_figures(args: argparse.Namespace) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    manuscript_root = args.manuscript_root.resolve()
    library_root = args.library_root.resolve()
    for label, module_name, output_rel, source_rel in APPLICATION_COMMANDS:
        output_path = manuscript_root / output_rel
        command = [
            "uv",
            "run",
            "--project",
            str(library_root),
            "python",
            "-m",
            module_name,
            "--output-pdf",
            str(output_path),
        ]
        source_path = None
        if label == "clearance-grid":
            source_path = manuscript_root / str(source_rel)
            command.extend(
                [
                    "--library-root",
                    str(library_root),
                    "--grid-density",
                    str(args.grid_density),
                    "--source-json",
                    str(source_path),
                    "--write-source-json",
                ]
            )
        print(f"RENDER {label}: {relative_to_root(output_path, manuscript_root)}")
        if not args.dry_run:
            subprocess.run(command, cwd=library_root, check=True)
            records.append(
                {
                    "label": label,
                    "output": relative_to_root(output_path, manuscript_root),
                    "source": module_name if source_path is None else relative_to_root(source_path, manuscript_root),
                    "sha256": sha256_file(output_path),
                }
            )
    return records


def copy_empirical_figures(args: argparse.Namespace) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    manuscript_root = args.manuscript_root.resolve()
    with resolved_run_root(
        run_root=args.run_root,
        archive=args.archive,
        library_root=args.library_root.resolve(),
    ) as run_root:
        for label, source_rel, target_rel in EMPIRICAL_FIGURES:
            source_path = run_root / source_rel
            target_path = manuscript_root / target_rel
            if not source_path.exists():
                raise FileNotFoundError(f"Missing rendered source for {label}: {source_path}")
            print(f"IMPORT {label}: {relative_to_root(target_path, manuscript_root)}")
            if not args.dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                records.append(
                    {
                        "label": label,
                        "output": relative_to_root(target_path, manuscript_root),
                        "source": f"saved-paper-run/{source_rel}",
                        "sha256": sha256_file(target_path),
                    }
                )
    return records


def write_manifest(args: argparse.Namespace, records: list[dict[str, str]]) -> None:
    if args.dry_run:
        return
    manuscript_root = args.manuscript_root.resolve()
    manifest_path = args.manifest or default_manifest_path(manuscript_root)
    payload = {
        "description": "Generated by geometric_safety paper/reproduce_paper_outputs.py.",
        "library_root": "provided at runtime via --library-root or GEOMETRIC_SAFETY_ROOT",
        "empirical_source": "saved paper run supplied via --archive or --run-root",
        "figures": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {relative_to_root(manifest_path, manuscript_root)}")


def main() -> None:
    args = parse_args()
    args.library_root = args.library_root.resolve()
    args.manuscript_root = args.manuscript_root.resolve()

    records: list[dict[str, str]] = []
    if not args.skip_application:
        records.extend(run_application_figures(args))
    if not args.skip_empirical:
        records.extend(copy_empirical_figures(args))
    write_manifest(args, records)


if __name__ == "__main__":
    main()
