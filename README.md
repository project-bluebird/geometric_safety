# geometric_safety

Geometric reachability-based safety filtering for air traffic control.

## Overview

This package implements geometric safety filters for lateral aircraft
separation. Given two aircraft with known positions, headings, speed
bounds, and optional turn profiles, the lateral solvers determine
whether the pair can possibly come within a separation threshold over a
projection horizon.

Two lateral solver variants:

- **Fixed heading** (`catch_up_projection_interval`) — exact
  geometric check for straight-line flight. Reduces the problem
  to one convex projection.
- **Turn-aware** (`catch_up_projection_interval_with_turns`) —
  handles one constant-rate turn per aircraft, then straight.
  Uses exact fixed-time hull evaluations with interval
  certification in time.

The solver uses only numpy and numba (no ATC domain dependencies).

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
the supplied rate should be interpreted as a lower-bound climb/descent
rate that all aircraft being filtered are expected to meet or exceed.
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
