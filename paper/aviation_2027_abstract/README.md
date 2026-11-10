# Heading study reported in the AVIATION 2027 abstract

This folder holds the inputs, saved decisions, scripts and timings for the ten-minute heading study reported in the AIAA AVIATION 2027 extended abstract by De Ath and Carvell. It is a different study from the 912-query protocol in `paper/fixtures/heading_study_aviation_2027.json`, which uses other speeds, another turn rate, shorter horizons and other encounters. The counts quoted in the abstract are the ones below.

The files in `data/` were produced on 9 November 2026 with `geometric_safety` revision `f472fd58d8483fda57251f971c02e0fea5c5f486`. The scripts are the ones used for that run, changed only in their imports and file paths so that they run from this repository, and with the checks in `summarise_results` placed in a function that the test can call. `paper/tests/test_aviation_2027_abstract.py` repeats the 873 library calls and fails if the current package returns a different decision or a different count.

## Inputs

Aircraft A flies at a nominal 450 kt and aircraft B at 360 kt. A is instructed to turn at 1 degree per second and then holds the target heading, and B holds heading 045. The separation distance is 5 NM and the horizon is 600 s. `scale` gives the speed bounds: 0 is the nominal speeds with no uncertainty, 1 is the base bounds of 10 kt either side for A and 30 kt either side for B, and 2 doubles both.

There are 23 distinct relative geometries: nine oblique crossings, nine perpendicular crossings and five overtaking gaps (6, 8, 10, 12 and 14 NM). Translating both aircraft along the overtaking track does not create a different geometry, which is why the 27 starting positions in `data/protocol.json` give 23 geometries. In each row, `da_nm` and `db_nm` are the distances of A and B short of a common point on their initial tracks, and `query` holds the values passed to `certify_heading_separation` in SI units.

The initial range comprises nine heading changes from -20 to +20 degrees at three uncertainty levels, which gives 621 queries. Where no instruction is retained, the range extends in 5-degree increments to 90 degrees in both directions, which adds 56 queries with base bounds and 196 with doubled bounds.

## Results

Across all 873 queries, the filter retains the 396 instructions that the reference classifies as separated and withholds the 476 for which the reference finds a permitted loss of separation. One nominal query is unresolved in both calculations and does not change the smallest retained turn.

With base bounds, four smallest turns are unchanged from the nominal result, 18 increase by 5 degrees and one increases by 10 degrees. With doubled bounds, one is unchanged, six increase by 5 degrees, 12 by 10 degrees and three by 15 degrees. For the remaining case, an overtaking gap of 6 NM, no assessed turn through 90 degrees is retained, and the reference finds a permitted loss for each.

For all 621 initial queries, the four endpoint combinations of constant speeds, the calculation over all constant speeds and the varying-speed reference produce the same classifications: 184 separated, 436 with a permitted loss and one unresolved. This comparison does not cover the wider turns. It does not make the four endpoint combinations sufficient in general, because the analysis first checks a constructed straight-flight example in which every endpoint combination is separated but an interior pair of speeds collides.

The constant-speed and doubled-bound wider-turn calculations agree at 80 and 120 decimal digits. They use `paper.heading_study_reference` and its 25 m exclusion band. Reference-separated cases imply that all four constant-speed endpoint combinations are also separated, and other cases are checked explicitly until a loss is found or all endpoint pairs have been evaluated. No decision is inferred from time samples alone. The reference shares support-function mathematics with the filter and uses a separate implementation, so it is not an independent proof of the floating-point library.

The varying-speed classifications of the 621 initial queries and of the 56 base-bound wider turns were saved from the earlier runs that fixed the inputs. The scripts here read those classifications and do not recompute them.

## Timings

`data/paper-timings.json` contains 100 measurements for each of the 69 nine-heading sets. Aircraft declarations are prepared before timing, and each measurement includes nine public-API calls and collection of their decisions. Every set is evaluated before measurement to exclude compilation, and all measured decisions are checked against the saved results.

On an Intel Xeon Gold 6430, the per-set median elapsed times range from 0.371861 to 1.3555585 ms, with a median across sets of 1.072194 ms. The process is pinned to one logical CPU. The host is shared and its load is recorded in the output, so these are workload measurements, not latency guarantees. CPU times, all samples and package hashes are also saved. The kernel release has been removed from the `platform` string of the saved file.

## Files

| File | Content |
| ---- | ------- |
| `data/protocol.json` | Encounter families, heading changes, speed bounds and reference settings of the initial run |
| `data/library-results.jsonl` | The 1,458 initial rows (27 starting positions, nine heading changes, three uncertainty levels, horizons of 300 s and 600 s) with the saved reference classification and library decision; the study uses the 600 s rows |
| `data/wider-results.jsonl` | The 56 base-bound wider turns with their reference bounds and library decisions |
| `data/model-comparison.jsonl` | The 621 distinct initial queries with the constant-speed and endpoint classifications |
| `data/doubled-wider.jsonl` | The 196 doubled-bound wider turns with their reference bounds and library decisions |
| `data/comparison-manifest.json` | SHA-256 of the sources evaluated on 9 November, under the paths of that run |
| `data/results-summary.json` | The counts quoted above; `script_sha256` is the hash of the summary script as run on 9 November |
| `data/paper-timings.json` | The timing samples and their summary |

Figure code and figure data are not part of this folder.

## Reproduction

From the repository root, using Python 3.13 and uv:

```bash
uv sync --locked
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --locked python -m paper.aviation_2027_abstract.compare_models
uv run --locked python -m paper.aviation_2027_abstract.summarise_results
uv run --locked python -m paper.aviation_2027_abstract.measure_timings
```

The commands rewrite their outputs in `data/`. `compare_models` uses four worker processes and writes rows as they finish, so the row order of `model-comparison.jsonl` and `doubled-wider.jsonl` differs between runs. It also replaces `comparison-manifest.json` with the hashes of the sources in the working tree. `measure_timings` requires Linux.
