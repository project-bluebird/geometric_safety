# Numerical validation and timing

These tools check the lateral geometry and compare complete package versions.
Run them from the repository root. The tests and numerical audit need the
development dependencies installed by `uv sync --locked`.

## Independent numerical reference

```bash
uv run --locked pytest tests/ paper/tests/
uv run --locked python -m paper.validate_performance_numerics \
  --output numerical_reference.json
```

The audit compares the scalar and directional interval bounds with an
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
validation seeds. Neither audit changes the production numerical guard.

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
It checks decisions and distance diagnostics, and reports changes in time
diagnostics separately. A changed decision, missing records or mismatched
case identifiers prevents a successful summary.
