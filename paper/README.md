# Paper evaluation

Reproduction pipeline for:

> **Certified Pairwise Lateral Safety for Bounded-Speed Aircraft
> Encounters with Single-Turn Manoeuvres**
> George De Ath and Ben Carvell

The pipeline regenerates synthetic benchmark summaries,
projection-sensitivity data, and figure assets. Core solvers live in
`geometric_safety`; evaluation scripts live in `paper/`.

## Reproducing the empirical paper results

### Prerequisites

```bash
git clone https://github.com/project-bluebird/geometric_safety.git
cd geometric_safety
uv sync
```

### Reproducing from the included archive

Compact submitted-paper archive:

```text
paper/results/paper_full_20260502_0738.tar.gz
```

This archive contains the five-seed main evaluation summaries, the
projection-sensitivity sweep, terminal logs, compact figure-source data,
and rendered figure assets. It supports figure and numerical verification
without rerunning the full evaluation.

To reproduce the figure files into a local output tree:

```bash
mkdir -p paper_figure_repro
uv run python -m paper.reproduce_paper_outputs \
    --output-root paper_figure_repro \
    --archive paper/results/paper_full_20260502_0738.tar.gz
```

Output: `paper_figure_repro/manuscript/figures/`.

### Full pipeline

The paper pipeline is split into a data stage and a rendering stage.
The data stage saves compact summaries, the row-level data needed for
figure regeneration, and terminal logs.
The rendering stage reads those saved outputs to generate figures and
the versioned capped projection archive. It also regenerates the two
explanatory application figures.

The full data stage is CPU-intensive and is only needed to regenerate
the saved archive from scratch. As a reference point, a 128-worker Linux
server completes the run in roughly 30--60 minutes. On smaller machines,
reduce `N_JOBS` or run the smoke test first with `QUICK=1`.

#### Full data stage

Run the expensive data/log generation stage with:

```bash
N_JOBS=128 ./paper/run_full_results.sh \
    paper_eval_outputs/paper_full_$(date +%Y%m%d_%H%M%S)
```

The saved data and log paths are listed in [Output structure](#output-structure).

With the script defaults, the first run directory is
`run_00_seed_20260325`. If `REPEAT_SEEDS=1`, `paper.run_evaluation`
writes the same per-run files directly under `main_evaluation/` rather
than inside `run_00_seed_<seed>/`.

#### Rendering stage

After the data stage completes, render figures from the saved outputs:

```bash
./paper/render_paper_figures.sh paper_eval_outputs/<run>
```

Figures go under `rendered_figures/`. The script also writes the capped projection
archive and a compact archive of the complete run under `paper/results/`; see
[Output structure](#output-structure) and the
[artifact-to-claim mapping](reproducibility_manifest.md).

To check that a completed run contains the expected source artifacts:

```bash
uv run python -m paper.check_reproducibility_artifacts paper_eval_outputs/<run>
uv run python -m paper.check_reproducibility_artifacts \
    paper_eval_outputs/<run> --require-rendered
```

### Running individual stages

#### Step 1: Main evaluation (5 seeds, 100k encounters per suite)

This produces all numerical results without rendering figures. It writes
compact figure-source tables, including the stress-map cell counts and
rates and the default near-threshold margin-bin summary, but avoids the
much larger straight, mixed-turn, and near-threshold row dumps.

```bash
uv run python -m paper.run_evaluation \
    --profile large100k \
    --repeat-seeds 5 \
    --n-jobs 128 \
    --output-dir paper_eval_outputs/paper_full/main_evaluation \
    --row-output-mode none \
    --skip-plots
```

#### Step 2: Projection sensitivity sweep

This produces the saved data for the projection-sensitivity figure.

```bash
uv run python -m paper.run_projection_sweep \
    --n-jobs 128 \
    --output-dir paper_eval_outputs/paper_full/projection_full \
    --combined-rows \
    --compress-rows
```

#### Render figures from saved data

This reads the saved outputs and renders figures without rerunning the
expensive evaluation. The path below is the first seed run produced by
the default full pipeline; use the matching `run_00_seed_<seed>` path if
the seed is changed. For a single-seed run, use
`paper_eval_outputs/paper_full/main_evaluation` directly.

```bash
uv run python -m paper.run_evaluation \
    --render-figures-from paper_eval_outputs/paper_full/main_evaluation/run_00_seed_20260325 \
    --figure-output-dir paper_eval_outputs/paper_full/rendered_figures/main \
    --output-dir paper_eval_outputs/paper_full/render_work/main \
    --figure-formats pdf
```

The projection figure is rendered separately with the 100 NMI excursion
cap:

```bash
uv run python -m paper.plot_projection_sweep \
    --input-dir paper_eval_outputs/paper_full/projection_full \
    --output-dir paper_eval_outputs/paper_full/rendered_figures/projection_deterministic_51_cap_100nmi \
    --max-excursion-nmi 100 \
    --write-rows \
    --combined-rows \
    --archive-output paper/results/projection_deterministic_51_cap_100nmi.tar.gz
```

The application figures can also be rendered directly:

```bash
uv run python -m paper.render_lateral_overlap_figure \
    --output-pdf paper_eval_outputs/paper_full/rendered_figures/application/fig_lateral_overlap_schematic.pdf

uv run python -m paper.render_clearance_grid_figure \
    --output-pdf paper_eval_outputs/paper_full/rendered_figures/application/fig_clearance_grid.pdf \
    --source-json paper_eval_outputs/paper_full/rendered_figures/application/fig_clearance_grid_source.json.gz \
    --write-source-json
```

To inspect or extract the archived result:

```bash
tar -tzf paper/results/projection_deterministic_51_cap_100nmi.tar.gz
mkdir -p paper_eval_outputs/projection_deterministic_51_cap_100nmi_extracted
tar -xzf paper/results/projection_deterministic_51_cap_100nmi.tar.gz \
    -C paper_eval_outputs/projection_deterministic_51_cap_100nmi_extracted
```

### Quick smoke test

To verify the pipeline works without running the full evaluation:

```bash
QUICK=1 N_JOBS=4 ./paper/run_full_results.sh paper_results/quick
PROJECTION_ARCHIVE=paper_results/quick_projection.tar.gz \
FULL_ARCHIVE=paper_results/quick.tar.gz \
    ./paper/render_paper_figures.sh paper_results/quick
uv run python -m paper.check_reproducibility_artifacts \
    paper_results/quick --require-rendered \
    --projection-archive paper_results/quick_projection.tar.gz
```

## Output structure

The default five-seed pipeline writes:

```text
paper_eval_outputs/<run>/
  logs/
    main_evaluation.log
    projection_sweep.log
  main_evaluation/
    campaign_summary.json
    campaign_runs.csv
    run_00_seed_<seed>/
      summary.json
      run_record.json
      stress_map_source.csv
      near_threshold_default_margin_summary.csv
      table_1_experiment_design.csv
      table_2_straight_exactness.csv
      table_3_mixed_turn_results.csv
      table_4_runtime_summary.csv
    run_01_seed_<seed+step>/
      ...
  projection_full/
    summary.json
    projection_rows.csv.gz
    *_summary_by_latitude.csv
    *_by_excursion.csv
    mixed_turn_by_excursion_and_turn_count.csv
  rendered_figures/
    application/
      fig_lateral_overlap_schematic.pdf
      fig_clearance_grid.pdf
      fig_clearance_grid_source.json.gz
    main/
      fig_crossing.pdf
      fig_nominal_proxy_miss.pdf
      fig_stress_maps.pdf
      fig_ablation.pdf
      fig_sampled_proxy_miss.pdf
    projection_deterministic_51_cap_100nmi/
      projection_min_error_vs_excursion.pdf
      projection_error_by_range_and_horizon.pdf
      projection_orientation_disagreement.pdf
      projection_disagreement_by_excursion.pdf
paper/results/
  <run>.tar.gz
  projection_deterministic_51_cap_100nmi.tar.gz
```

The projection archive contains the capped `summary.json`, `projection_rows.csv`,
straight and mixed-turn summary tables, and the four projection PDFs shown above.
The [reproducibility manifest](reproducibility_manifest.md) maps the outputs to
manuscript figures and claims.

### Figure sources

The schematic application figures are deterministic. The clearance-grid figure
is rendered from the archive's saved source grid, and the empirical PDFs are
copied from that archive. Use `--recompute-clearance-grid` to generate a new grid
with the current solver; `--grid-density` applies only in that mode.

The near-threshold Lipschitz/stopping-width ablation explicitly disables the
directional certificate in both its timed solver and evaluation counter. Other
evaluations retain the production default. Near-threshold headline counts and
the default margin-bin summary use a separate production pass with directional
mode 2. Its settings and counts appear in `default_summary`; the ablation rows
remain scalar. Full row output includes `near_threshold_production_rows.csv`
(or `.csv.gz`) for this pass. Archived evaluations retain the settings used to generate them,
and aggregation rejects a mixture of scalar and directional default results.

To export a different archive, use the
[archive reproduction command](#reproducing-from-the-included-archive) with that
archive's path as `--archive`.

## Key paper claims and where to verify them

| Claim | Where to check |
| ----- | -------------- |
| 100% straight exactness | `run_*/summary.json` > `straight` |
| 0 false-safe (mixed-turn) | `run_*/summary.json` > `mixed_turn` |
| 99.83% certification | `run_*/summary.json` > `mixed_turn` |
| Rule-of-three 5.6e-5 | `campaign_summary.json` > `aggregate` |
| 95.36% near-threshold | `run_*/summary.json` > `near_threshold` |
| Near-threshold margin-bin rates | `run_*/near_threshold_default_margin_summary.csv` |
| 1,047 / 535 flips | `projection_full/summary.json` inside `paper/results/paper_full_20260502_0738.tar.gz` |
| 94 / 52 above-band projection flips | `projection_full/projection_rows.csv.gz` inside `paper/results/paper_full_20260502_0738.tar.gz` |
| 2.02 us fixed-time median runtime | `run_*/summary.json` > `kernel_runtime` |
| 5.16 / 13.41 / 23.37 us full-solver runtimes | `run_*/summary.json` > `mixed_turn.runtime` |

## Additional diagnostic scripts

Optional diagnostics:

| Command | Purpose |
| ------- | ------- |
| `uv run python -m paper.run_projection_sanity` | Multi-latitude validation |
| `uv run python -m paper.run_projection_anchors` | Anchor-mode comparison |

## Robustness campaigns

The out-of-model robustness study runs three campaign layers over the broad
mixed-turn suite plus per-configuration boundary suites (encounters certified
safe with margin near the deviation envelope, where violations can actually
occur): `inclass` verification of the envelope implementation, `misspec`
validation against out-of-class execution models, and `exceedance` degradation
curves past the stated bounds.

Quick smoke run:

```bash
uv run python -m paper.run_robustness --quick --n-jobs 8 \
    --output-dir paper_eval_outputs/robustness_smoke
```

Full five-seed campaign (about two hours on a 96-worker server):

```bash
uv run python -m paper.run_robustness \
    --n-encounters 2000 --n-boundary 200 --n-sims 100 \
    --repeat-seeds 5 --n-jobs 96 \
    --output-dir paper_eval_outputs/robustness_full
```

Each per-seed run directory contains `robustness_rows.csv` (one row per
config, suite, and encounter), `robustness_summary.csv`, and
`robustness_summary.json`. Boundary suites are cached under
`<output-dir>/boundary_cache/` and are reused across configs and repeat
runs. In-class total violations must be zero; the runner prints a warning
otherwise.

To archive a campaign compactly for the repository and render its figures:

```bash
uv run python -m paper.archive_robustness \
    --results-dir paper_eval_outputs/robustness_full \
    --output paper/results/robustness_compact_<date>.tar.gz

uv run python -m paper.render_robustness_figures \
    --results-dir paper_eval_outputs/robustness_full \
    --output-dir paper_eval_outputs/robustness_figures
```
