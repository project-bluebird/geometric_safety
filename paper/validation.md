# Numerical validation and timing

These tools check the lateral geometry and compare complete package versions.
Run them from the repository root. The tests and numerical audit need the
development dependencies installed by `uv sync --locked`. Each pytest run uses
a fresh Numba cache, because persistent compiled callers can retain an older
version of an imported helper.

The numerical audits also execute in isolated processes with fresh Python bytecode and compiler
caches, including calls to `numerical_checks`. Workers check package and harness
hashes before importing the library and after completing the calculation.
Directional benchmarks also verify their input file across this interval.

## Independent numerical reference

```bash
uv run --locked pytest tests/ paper/tests/
uv run --locked python -m paper.validate_performance_numerics \
  --output numerical_reference.json
```

The audit compares the directional interval bounds with an
independent 80-digit calculation over 512 seeded encounters, including
threshold probes. Selected comparisons are repeated at 120 digits. It uses
the actual binary64 candidate directions, then independently integrates the
motion and evaluates the bounds. The reference treats each supplied binary64
input as exact, converts knots using 1852/3600, and uses an arbitrary-precision
value of pi.

The report records source hashes, comparison counts, accepted certificates,
optimistic error and precision rechecks. Missing instrumentation, empty
expected coverage, disagreement with the production solver and negative
reference residuals on accepted certificates raise exceptions, including
under `python -O`. These finite checks do not prove a floating-point error
bound for every supported input.

`paper.run_turn_numerics` separately audits the scalar certificate with the
directional calculation disabled. It has independent calibration and
validation seeds. Its threshold probes include each candidate guard, and its
report counts completed production-parity checks. Neither audit changes the
production numerical guard.

To run that separate scalar calibration and held-out validation:

```bash
uv run --locked python -m paper.run_turn_numerics \
  --output-dir /path/to/new/scalar-audit
```

## Fixed timing inputs

`fixtures/heading_encounters.json` contains 421 fixed local-plane encounters:
229 broad inputs and 192 inputs selected near the separation boundary. It
retains generation seeds and constant-speed reference information. These
reference labels do not classify arbitrary time-varying speed profiles.
The benchmark adds 128 straight encounters, 168 analytic circular encounters
and six regression controls, giving 723 inputs.

The benchmark uses isolated persistent workers with the same interpreter and
numerical libraries. Compilation, input preparation and interprocess
communication are outside the timed regions. It reports the core called
from Python and from a compiled wrapper, plus the heading interface with and
without constructing its motion objects. A separate measurement calls the
core with the interface's enclosing intervals and inflated threshold.

## Comparing the scalar library and the current package

The Linux benchmark uses CPU affinity and records both wall and thread CPU
time. Set `--cpu` to an available logical CPU. Affinity does not reserve it.
The scalar reference is fixed at commit
`9b7869d363286e69aa45471c11d13b83861320e9`, for which the regression controls
are defined.

Prepare a fresh output directory, the reference package and an installed wheel:

```bash
run_root=$(mktemp -d)
git archive --format=tar \
  9b7869d363286e69aa45471c11d13b83861320e9 geometric_safety \
  > "$run_root/scalar.tar"
mkdir "$run_root/scalar"
tar -xf "$run_root/scalar.tar" -C "$run_root/scalar"
uv build --wheel --out-dir "$run_root/wheels"
uv venv --python 3.13 "$run_root/env"
uv pip install --python "$run_root/env/bin/python" "$run_root"/wheels/*.whl
"$run_root/env/bin/python" -I - "$run_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from geometric_safety import relevant_aircraft

root = Path(sys.argv[1])
wheel, = (root / 'wheels').glob('*.whl')
package = Path(relevant_aircraft.__file__).parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    'wheel_sha256': sha(wheel),
    'provenance': {
        'module_sha256': {path.name: sha(path) for path in package.glob('*.py')}
    },
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$run_root/env/bin/python" -m paper.benchmark_library_versions \
  --inputs paper/fixtures/heading_encounters.json \
  --old-source "$run_root/scalar" --old-archive "$run_root/scalar.tar" \
  --manifest "$run_root/manifest.json" --cpu 0 --output "$run_root/timing"
```

The output includes `cases.json`, package hashes, checked outcomes and raw
measurements in `results.json`. The default protocol uses seven rounds of
16 calls per input. It then measures the union of each method's ten slowest
inputs in 31 separate calls. Record the environment and host load when
interpreting these measurements. They describe synthetic inputs, not a
runtime guarantee or operational traffic.

The core assumes constant speeds through turns. The heading interface covers
arbitrary bounded speed profiles and can therefore reject more inputs. Compare
zero-width speed intervals or the effective-core measurement when separating
interface overhead from the cost of its broader motion class.

## Comparing directional modes

`paper.validate_directional_production` compares the current scalar, endpoint
direction and endpoint-plus-hull direction modes. Its optional `--baseline`
argument takes a directory containing a complete `geometric_safety/` package,
not an individual solver file:

```bash
uv run --locked python -m paper.validate_directional_production \
  --inputs paper/fixtures/heading_encounters.json \
  --baseline /path/to/baseline \
  --output /path/to/directional.json
```

The candidate modes run in a separate process with an empty Numba cache. The
baseline uses the same isolated Linux worker as the package benchmark. Both
record hashes for the complete selected package. The report's `baseline`
object also records runtime versions and checked decisions. Timings exclude
compilation, input preparation and interprocess communication.

The packed and baseline calls use the base constant-speed contract. Nondefault
deviation envelopes, stopping widths or Lipschitz settings are rejected rather
than omitted from those calls.

This comparison requires the baseline's decisions to match the current scalar
mode. A difference stops the comparison. Use `paper.benchmark_library_versions`
when comparing versions with known decision changes. Omitting `--baseline`
compares only the three current modes and records `baseline: null`.

## Comparing two versions of the heading interface

An additional baseline package can be supplied to the benchmark using
`--before-source` and `--before-manifest`, with the same manifest format.
Set `--expected-recovered` to the known count of circular controls accepted by
that baseline's core but rejected by its interface; use zero for a baseline
without that discrepancy. Add `--require-before-decision-parity` when the
change is intended to preserve decisions. The matched report requires this
additional baseline:

```bash
uv run --locked python -m paper.summarise_performance \
  --results /path/to/timing/results.json --cases /path/to/timing/cases.json \
  --output /path/to/report
uv run --locked python -m paper.validate_performance_parity \
  --cases /path/to/timing/cases.json \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --output /path/to/parity
```

Both packages in the parity comparison must expose the heading API. This
check adds 20,000 seeded encounters and 33 numerical-guard boundary cases.
It checks decisions, diagnostic kinds and distance diagnostics, and reports
changes in time or bound-interval diagnostics separately. Matching unavailable
diagnostics are preserved; a change in distance availability fails comparison.
A changed decision, missing records or mismatched
case identifiers prevents a successful summary.

Each source path must contain a `geometric_safety/` package. Package-only
exports and full archived checkouts are supported. Workers import the selected
package before loading the current validation tools, and verify its location
and source hashes. A missing submodule is an error; installed editable packages
cannot fill it in. Workers use the current `paper/` tools.

For a focused regression comparison, `--saved-only` evaluates only the supplied
cases. Its reports record that the seeded and boundary cases were omitted.
Omit this option for the standard comparison described above.
