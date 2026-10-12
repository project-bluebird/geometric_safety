# geometric_safety

Geometric reachability-based safety filtering for air traffic control.

## Overview

This package implements geometric safety filters for lateral aircraft
separation. Given two aircraft with known positions, headings, speed
bounds, and optional turn profiles, the lateral solvers determine
whether the pair can possibly come within a separation threshold over a
projection horizon.

Core lateral solvers:

- **Fixed heading** (`catch_up_projection_interval`) — exact
  geometric check for straight-line flight. Reduces the problem
  to one convex projection.
- **Turn-aware** (`catch_up_projection_interval_with_turns`) —
  handles one constant-rate turn per aircraft, then straight.
  Uses exact fixed-time hull evaluations with interval
  certification in time. A scalar Lipschitz bound handles easy intervals;
  projected-motion bounds along separating directions can certify additional
  intervals before subdivision. Unresolved intervals are rejected conservatively.

The solver uses only numpy and numba (no ATC domain dependencies).

### Bounded speed profiles through a heading turn

`certify_heading_separation` accepts a prescribed heading turn followed by
straight flight. Each aircraft has its own horizontal speed interval, and its
speed may vary anywhere in that interval throughout the horizon. Positions and
velocities use one local east/north plane, in metres and metres per second.
Headings are degrees clockwise from north.

```python
from geometric_safety import HeadingMotion, certify_heading_separation

a = HeadingMotion(
    heading0_deg=0.0, target_heading_deg=45.0, turn_rate_deg_s=1.5,
    speed_min_mps=140.0, speed_max_mps=160.0,
)
b = HeadingMotion(
    heading0_deg=90.0, target_heading_deg=90.0, turn_rate_deg_s=0.0,
    speed_min_mps=130.0, speed_max_mps=150.0,
)
result = certify_heading_separation(
    relative_position_m=(-20000.0, 20000.0),  # position A minus position B
    a=a, b=b, separation_threshold_m=9260.0, horizon_s=120.0,
)
if result.status == "certified_safe":
    pass  # The declared motion class passes the separation check.
```

The result distinguishes `certified_safe`, `unresolved` and `unsupported`.
Only `certified_safe` permits acceptance. `unresolved` is not a proved collision.
`unsupported` identifies a motion assumption outside this interface's
contract. Malformed numeric inputs raise `ValueError`. The `diagnostic_kind`
field distinguishes a `straight_minimum`, `sampled_minimum` and `interval_bound`
in the underlying constant-speed calculation. For an interval bound,
`base_bound_interval_s` gives its endpoints and `base_closest_time_s` is `None`.
The bound is not an observed distance. These diagnostics do not give an
executed-motion minimum or a collision time.

The supported heading law is an immediate constant-rate shortest turn, then
maintained heading. The interface checks the full lateral horizon and does
not shorten it using vertical motion. A caller must establish that its motion
satisfies this contract. Route following, delayed instructions, explicitly directed
turns and speed-dependent heading laws need separate coverage.

The speed-profile calculation encloses each execution by an admissible
mean-speed position plus a deviation bound. That bound stops growing once the
turn ends, even if speed continues changing. The calculation uses its
maximum over the requested horizon as constant threshold inflation. Turning
queries may also widen the smaller of the two speed intervals to reuse the
constant-speed kernel. Both choices can reject otherwise safe encounters.

Optional position, relative velocity and heading-deviation bounds must hold
throughout the stated horizon. They are supplied by the caller, not inferred
from a simulator configuration. A wind bound covers an additive velocity
perturbation; it does not establish that wind leaves the prescribed heading
law unchanged. The geometric solver uses floating-point
arithmetic and empirical numerical guard. This interface does not establish
validated interval arithmetic for production decisions.

The numerical domain limits the initial relative distance plus maximum total
travel, and the inflated separation threshold, to about 8.8 million metres.
The shared limit is the `1e-6 m` margin divided by 512 times binary64
machine epsilon. It keeps floating-point spacing small relative to that margin;
the factor is an engineering resolution policy, not a bound on accumulated
rounding error. Inputs outside this domain return `unresolved` from the heading
interface. Legacy lateral solvers return `False` as their decision flag, with
unavailable distance and time diagnostics. Separate finite-range checks prevent
overflow in the intermediate speed and time calculations.

### Compatibility

The legacy lateral and combined lateral/vertical entry points retain their
signatures and return tuple shapes. Their decisions can change because the
turn-aware solver now retains every nonzero turn and uses tighter interval
bounds. A changed target heading with zero turn rate raises `ValueError` in
the legacy turn solver and returns `unsupported` from the heading interface.
An exact 180-degree heading difference selects a left turn; values on either
side retain their direction, including near a heading wrap.

The module constants `TURN_HEADING_EPS_DEG` and `TURN_RATE_EPS_DEG_PER_S` are
removed. Callers that applied these cut-offs must retain nonzero instructed turns.

Legacy turn-aware distance and time values are solver diagnostics, not a
guaranteed continuous-time minimum. Use the heading interface when the caller
needs to distinguish a sampled distance from an interval lower bound. Consumers
must check the decision or status before using diagnostics, including the
unavailable values returned for inputs outside the numerical domain.

The turn-aware core assumes each aircraft's speed is constant across the
horizon unless one of its specific deviation envelopes is supplied. Use
`certify_heading_separation` for speed that may vary throughout a turn and its
straight continuation. Its broader motion class can require more separation.

### Additional vertical filtering

The package also includes a vertical cleared-band timing check, which
can be combined with the lateral solver. This vertical filtering is an
engineering extension and is not used in the lateral-only paper
experiments.

Vertical filtering (`time_to_vertical_overlap_resolution`) assumes
each aircraft climbs or descends from current flight level to selected
flight level at the same fixed parameterized rate magnitude. Negative
rates are treated as magnitudes; climb or descent direction comes from
the current and selected levels. It returns the first time at which the
two remaining cleared-to-selected level bands are separated by the
required vertical gap. The defaults are 1,000 ft/min and 10 flight
levels. Before that gap is measured, compared band edges are rounded
toward each other to multiples of 10 flight levels by default; pass
`rounded=0` to the standalone check, or `vertical_rounding_fl=0` to the
combined checks, to use the exact band edges. When using the combined
check as a safety filter, pass a conservative low vertical rate for the
operation being modelled.

The vertical timing model does not include vertical-rate uncertainty:
the supplied rate must be a lower bound on climb/descent
rate throughout the interval, including its beginning. An eventual rate does
not cover an initial response delay or ramp-up. Use a full lateral-horizon
check unless the vertical progress assumption is established.
The combined lateral/vertical check also keeps the supplied lateral
kinematics fixed while the aircraft climb or descend; it does not model
changes in lateral speed caused by vertical profile, altitude, or
speed schedule.

## Installation

Add to an existing uv project:

```bash
uv add geometric-safety \
    --git https://github.com/project-bluebird/geometric_safety.git
```

This adds two entries to your `pyproject.toml`:

```toml
# in [project] dependencies
"geometric_safety>=0.1.0"

# in [tool.uv.sources]
geometric_safety = {
    git = "https://github.com/project-bluebird/geometric_safety"
}
```

To pin a specific commit or tag, pass `--rev`:

```bash
uv add geometric-safety \
    --git https://github.com/project-bluebird/geometric_safety.git \
    --rev <commit-or-tag>
```

For local development of this repo itself:

```bash
uv sync
```

## Usage

Turn-aware lateral check:

```python
from geometric_safety import catch_up_projection_interval_with_turns

is_separated, min_distance_m, closest_time_s = (
    catch_up_projection_interval_with_turns(
        a_lat=51.0, a_lon=-1.0,
        a_heading0_deg=0.0, a_target_heading_deg=270.0,
        a_speed_kt=340.0, a_turn_rate_deg_sec=1.5,
        b_lat=51.03, b_lon=-0.97,
        b_heading0_deg=90.0, b_target_heading_deg=90.0,
        b_speed_kt=340.0, b_turn_rate_deg_sec=0.0,
        separation_threshold_m=9260.0,  # 5 NMI
        speed_diff_kt=15.0,
        projection_time_s=900.0,
    )
)
```

Standalone vertical timing:

```python
from geometric_safety import (
    VERTICAL_OVERLAP_NEVER_RESOLVES_S,
    time_to_vertical_overlap_resolution,
)

vertical_resolution_time_s = time_to_vertical_overlap_resolution(
    a_current_fl=100.0,
    a_selected_fl=100.0,
    b_current_fl=100.0,
    b_selected_fl=120.0,
    vertical_rate_fpm=1000.0,
    required_gap_fl=10.0,
    rounded=10,
)

if vertical_resolution_time_s == VERTICAL_OVERLAP_NEVER_RESOLVES_S:
    ...
```

Combined lateral and vertical check:

```python
from geometric_safety import catch_up_projection_interval_with_turns_and_vertical

is_safe, min_distance_m, closest_time_s, vertical_resolution_time_s = (
    catch_up_projection_interval_with_turns_and_vertical(
        a_lat=51.0, a_lon=-1.0,
        a_heading0_deg=0.0, a_target_heading_deg=270.0,
        a_speed_kt=340.0, a_turn_rate_deg_sec=1.5,
        a_current_fl=100.0, a_selected_fl=100.0,
        b_lat=51.03, b_lon=-0.97,
        b_heading0_deg=90.0, b_target_heading_deg=90.0,
        b_speed_kt=340.0, b_turn_rate_deg_sec=0.0,
        b_current_fl=100.0, b_selected_fl=120.0,
        separation_threshold_m=9260.0,  # 5 NMI
        speed_diff_kt=15.0,
        projection_time_s=900.0,
        vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=10,
    )
)
```

Use `catch_up_projection_interval_with_vertical` for the same combined
logic with the fixed-heading lateral solver.

For combined checks, lateral separation is only evaluated until the
vertical overlap resolves. If the vertical bands are already distinct
by the required gap, the result is safe immediately and the returned
lateral distance/time describe the current geometry at `t=0`. If they
never resolve, reported as `VERTICAL_OVERLAP_NEVER_RESOLVES_S`
(`-1.0`), the lateral solver checks the full projection horizon.

## Tests

```bash
uv run pytest
uv run pytest tests/
uv run pytest paper/tests/
uv run pytest tests/test_relevant_aircraft_turns.py
uv run pytest tests/test_vertical_safety.py
uv run pytest tests/test_relevant_aircraft_app.py
```

Run the independent numerical audit as a module from the repository root:

```bash
uv run --locked python -m paper.validate_performance_numerics --output numerical_reference.json
```

The audit rejects missing instrumentation and empty expected coverage. Benchmark
and parity validation uses explicit exceptions, so Python optimisation does not
disable its checks. Reports derive host details, decision changes and recorded
check outcomes from their input data. See [numerical validation and timing](paper/validation.md)
for the reference assumptions, fixed inputs and package comparison commands.

## Relevant-aircraft visualiser

An interactive FastAPI interface for inspecting the lateral safety geometry.
Supports both fixed-heading and turn-aware modes.

- **Fixed heading mode** — sweeps aircraft B's initial position
  across a configurable grid, showing the straight-line reachable
  hulls over time.
- **Turn aware mode** — uses configured initial headings, target
  headings, turn rates, and an optional in-turn speed buffer.
  The detail view shows spatial trajectories, the relative hull at
  the selected time, and the distance-vs-time curve.

Turn-aware mode includes canned presets:

- Single-turn dodge
- Single-turn cut-in
- Mutual opening turns
- Mutual closing turns

```bash
uv sync --group apps
uv run uvicorn app:app --reload
```

Then browse to <http://127.0.0.1:8000/>.

## Benchmarks

Isolated lateral-solver timing benchmarks:

```bash
uv run python scripts/benchmark_relevant_aircraft.py
uv run python scripts/benchmark_relevant_aircraft.py \
    --random-cases 5000
```

## Paper Results

Reproduce the paper figures and numerical checks with the scripts in
[`paper/`](paper/README.md).

## Documentation

- `relative_aircraft_projection_walkthrough.ipynb` — interactive
  walkthrough notebook
