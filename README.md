# geometric_safety

Check whether two aircraft remain laterally separated throughout a chosen
horizon, given their positions, heading instructions and speed bounds. A
successful check allows a caller to exclude that pair from further lateral
conflict checking for the stated motion and horizon. Otherwise, the pair
remains for further assessment.

For heading instructions, use `HeadingMotion` and
`certify_heading_separation_from_latlon`. Each aircraft either maintains its
heading or makes one immediate, constant-rate shortest turn and then maintains
the target heading. Its speed may vary anywhere within its declared interval
throughout the horizon, including during climb or descent. The caller must
establish that these motion assumptions and bounds hold.

The existing lateral and combined lateral/vertical functions remain available.
See [migrating existing calls](#migrating-existing-calls) for their relationship
to the heading interface.

## Installation

This revision supports Python 3.13. The project's supported Python range in
`pyproject.toml` must also be compatible with that restriction:

```toml
[project]
requires-python = ">=3.13,<3.14"
```

In an existing uv project, add a checkout containing the interface documented
here. For example, if this repository is in the adjacent directory:

```bash
uv add ../geometric_safety
```

The geometry code has no dependency on an aircraft simulator or controller.
For development in a checkout of this repository, use `uv sync --locked`.

## Check a heading instruction

Aircraft A starts about 20 km west and 20 km north of B. It turns from north to
north-east at 1.5 degrees per second, completing the turn after 30 seconds.
B maintains an easterly heading. We require 5 nautical miles of separation
for the next 120 seconds. The speed bounds below apply during the turn and
the subsequent straight flight.

```python
from geometric_safety import HeadingMotion, certify_heading_separation_from_latlon

a = HeadingMotion(
    heading0_deg=0.0,
    target_heading_deg=45.0,
    turn_rate_deg_s=1.5,
    speed_min_mps=140.0,
    speed_max_mps=160.0,
)
b = HeadingMotion(
    heading0_deg=90.0,
    target_heading_deg=90.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=130.0,
    speed_max_mps=150.0,
)

result = certify_heading_separation_from_latlon(
    a_lat=51.18,
    a_lon=-1.29,
    b_lat=51.0,
    b_lon=-1.0,
    a=a,
    b=b,
    separation_threshold_m=5.0 * 1852.0,
    horizon_s=120.0,
)

if result.status == "certified_safe":
    print("Pair passes the lateral separation check.")
else:
    print(f"Keep the pair for further checking: {result.status}")
```

Output:

```text
Pair passes the lateral separation check.
```

Both positions must refer to the same starting time. Latitude and longitude
are in degrees, with east-positive longitude. The library converts them to
one local east/north frame centred on A and handles the relative-position
sign. Headings are degrees clockwise from north, turn rate is a nonnegative
magnitude in degrees per second, and speed is in metres per second. The check
covers the whole interval from zero to `horizon_s`, including the turn itself.

This conversion uses a local planar approximation. It handles the date line,
but does not calculate geodesic separation or bound projection error. Both
heading descriptions are treated as directions in the common plane. See
[coordinate frames](docs/motion-models.md#coordinate-frames) for the model and
input limits.

This example assumes no additional position or velocity error. Optional
[deviation bounds](docs/motion-models.md#deviation-bounds) cover explicitly
bounded departures from the prescribed motion. Route following, response
delays, explicitly directed turns and speed-dependent heading laws are
outside this interface's supported motion class. Numerical acceptance uses
floating-point arithmetic with an empirical guard; its
[numerical limits](docs/motion-models.md#numerical-limits) apply to these checks.

### Positions already in a local frame

Use `certify_heading_separation` when your application already supplies
positions and headings in one Cartesian frame. Pass **A minus B**, with east
first and north second, in metres. For example, two aircraft maintaining the
same heading and speed keep their initial separation:

```python
from geometric_safety import HeadingMotion, certify_heading_separation

motion = HeadingMotion(
    heading0_deg=0.0,
    target_heading_deg=0.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=150.0,
    speed_max_mps=150.0,
)
result = certify_heading_separation(
    relative_position_m=(-20_000.0, 20_000.0),
    a=motion,
    b=motion,
    separation_threshold_m=5.0 * 1852.0,
    horizon_s=120.0,
)
print(result.status)
```

```text
certified_safe
```

## Use the result

| `result.status` | Meaning | Caller action |
| --- | --- | --- |
| `certified_safe` | Separation passes for the declared motion class over the whole horizon. | The pair may be excluded from further lateral checking for that query. |
| `unresolved` | The calculation did not establish separation. This includes possible conflicts, conservative rejection and inputs outside the numerical domain. | Retain the pair for further checking. |
| `unsupported` | The declared motion is outside the heading interface's contract. | Use a method that covers that motion. |

Check `result.status` explicitly. `if result:` only tests the result object
and does not test separation. Malformed numeric inputs, such as a negative
horizon or reversed speed bounds, raise `ValueError`.

`result.reason` explains the outcome, and `result.inflation_m` reports the
extra separation used to cover the declared speed variation and deviation
bounds. The optional distance and time fields describe the underlying
constant-speed calculation. They do not give an executed aircraft's minimum
separation or the time at which separation is first lost. See
[diagnostics](docs/motion-models.md#diagnostics) before using those fields.

## Include vertical separation

Use `certify_heading_separation_with_vertical_from_latlon` to include current
and selected flight levels and a guaranteed minimum climb/descent rate. It
determines when vertical separation suffices and checks lateral separation
only until that time. If vertical separation cannot be established, it checks
the full requested horizon.

Here the aircraft approach head-on from about 60 km apart. A maintains FL100 while
B climbs from FL100 to FL120 at no less than 1,000 ft/min. The requested
horizon is ten minutes, but the altitude bands separate after one minute.

```python
from geometric_safety import HeadingMotion, certify_heading_separation_with_vertical_from_latlon

a = HeadingMotion(
    heading0_deg=270.0,
    target_heading_deg=270.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=95.0,
    speed_max_mps=105.0,
)
b = HeadingMotion(
    heading0_deg=90.0,
    target_heading_deg=90.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=95.0,
    speed_max_mps=105.0,
)
result = certify_heading_separation_with_vertical_from_latlon(
    a_lat=51.0,
    a_lon=-0.15,
    b_lat=51.0,
    b_lon=-1.0,
    a=a,
    b=b,
    a_current_fl=100.0,
    a_selected_fl=100.0,
    b_current_fl=100.0,
    b_selected_fl=120.0,
    minimum_vertical_rate_fpm=1000.0,
    separation_threshold_m=5.0 * 1852.0,
    required_vertical_gap_fl=10.0,
    horizon_s=600.0,
)
print(result.status, result.vertical_resolution_time_s, result.lateral_horizon_s)
```

Output:

```text
certified_safe 60.0 60.0
```

Check `result.status` as above. `result.lateral_result` contains the lateral
result and its diagnostics over `result.lateral_horizon_s`. It is `None` when
the altitude bands are already separated and no lateral check is needed.
`result.vertical_resolution_time_s` is `None` when no usable vertical
resolution time was established.

The supplied rate is a shared nonnegative minimum. Rates may differ and vary
above it. Each aircraft must progress towards its selected level immediately,
without overshoot, and remain there on arrival. A zero minimum guarantees no
progress, although already separated altitude bands still suffice. The
horizontal speed bounds must cover any climb/descent speed changes during
the lateral check. See [vertical filtering](docs/motion-models.md#vertical-filtering)
for the altitude-band argument and rounding options.

Before shortening the lateral horizon, the heading interface independently
verifies that the unrounded altitude bands meet the requested vertical gap.
The returned time may be moved later to satisfy that check.

For positions already in a common Cartesian frame, use
`certify_heading_separation_with_vertical` with `relative_position_m` in place
of the four latitude/longitude arguments. All other inputs and result fields
are the same.

## Migrating existing calls

The existing functions retain their call signatures and tuple return shapes:

- `catch_up_projection_interval` for fixed headings;
- `catch_up_projection_interval_with_turns` for one turn and straight continuation;
- `catch_up_projection_interval_with_vertical` for fixed headings with vertical filtering;
- `catch_up_projection_interval_with_turns_and_vertical` for turns with vertical filtering.

Updating the library gives existing turn-aware calls the repaired turn
geometry and tighter interval checks. Adopting the heading interface also
changes the motion contract: it covers speed that can vary throughout the
turn and continuation. A wider constant-speed interval alone does not give
that coverage during a turn. The broader contract can require more separation
and can therefore return a different decision for the same speed endpoints.

### From a turn-aware lateral call

This existing call allows each aircraft an unknown constant speed within
20 knots of its nominal speed:

```python
from geometric_safety import catch_up_projection_interval_with_turns

is_separated, _, _ = catch_up_projection_interval_with_turns(
    a_lat=51.0,
    a_lon=-1.0,
    a_heading0_deg=0.0,
    a_target_heading_deg=45.0,
    a_speed_kt=300.0,
    a_turn_rate_deg_sec=1.5,
    b_lat=51.15,
    b_lon=-0.7,
    b_heading0_deg=90.0,
    b_target_heading_deg=90.0,
    b_speed_kt=280.0,
    b_turn_rate_deg_sec=0.0,
    separation_threshold_m=5.0 * 1852.0,
    speed_diff_kt=20.0,
    projection_time_s=120.0,
)
print(bool(is_separated))
```

Output:

```text
True
```

The following call uses the same positions and speed endpoints, allowing
the speed to vary within those bounds. Each example runs independently.
The new entry point handles the local projection and relative-position sign.

```python
from geometric_safety import HeadingMotion, certify_heading_separation_from_latlon
from geometric_safety.util import KT_TO_MPS

a = HeadingMotion(
    heading0_deg=0.0,
    target_heading_deg=45.0,
    turn_rate_deg_s=1.5,
    speed_min_mps=(300.0 - 20.0) * KT_TO_MPS,
    speed_max_mps=(300.0 + 20.0) * KT_TO_MPS,
)
b = HeadingMotion(
    heading0_deg=90.0,
    target_heading_deg=90.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=(280.0 - 20.0) * KT_TO_MPS,
    speed_max_mps=(280.0 + 20.0) * KT_TO_MPS,
)

result = certify_heading_separation_from_latlon(
    a_lat=51.0,
    a_lon=-1.0,
    b_lat=51.15,
    b_lon=-0.7,
    a=a,
    b=b,
    separation_threshold_m=5.0 * 1852.0,
    horizon_s=120.0,
)
print(result.status)
```

Output:

```text
certified_safe
```

The matching decisions in this example do not imply equivalence of the two
motion classes. Both use the same local planar approximation for these
positions, and both wrap longitude differences across the date line.

| Existing input or return value | Heading interface |
| --- | --- |
| Latitude and longitude | The same `a_lat`, `a_lon`, `b_lat`, `b_lon` inputs on the `_from_latlon` functions |
| Nominal speed and uncertainty in knots | Each aircraft's absolute `speed_min_mps` and `speed_max_mps` |
| `a_turn_rate_deg_sec`, `b_turn_rate_deg_sec` | Each aircraft's `turn_rate_deg_s` |
| `projection_time_s` | `horizon_s` |
| `turn_speed_schedule_uncertainty_kt` | Speed-variation enclosure calculated from the declared speed intervals |
| `is_separated` | `result.status == "certified_safe"` |
| Distance and time tuple entries | Named diagnostics with `diagnostic_kind` identifying their meaning |

### Fixed headings and asymmetric speed bounds

For straight flight, set the target equal to the initial heading and the
turn rate to zero. The library handles the reachable geometry internally;
there is no need to construct or pass a velocity hull.

If an existing caller uses separate lower and upper speed allowances, convert
them to absolute bounds. For example, 300 knots with allowances of 25 knots
below and 50 knots above becomes:

```python
from geometric_safety import HeadingMotion
from geometric_safety.util import KT_TO_MPS

nominal_kt = 300.0
lower_allowance_kt = 25.0
upper_allowance_kt = 50.0

a = HeadingMotion(
    heading0_deg=90.0,
    target_heading_deg=90.0,
    turn_rate_deg_s=0.0,
    speed_min_mps=max(0.0, nominal_kt - lower_allowance_kt) * KT_TO_MPS,
    speed_max_mps=(nominal_kt + upper_allowance_kt) * KT_TO_MPS,
)
```

Construct B in the same way and call `certify_heading_separation_from_latlon`.
Use the requested lookahead in seconds as `horizon_s`. The speed bounds must hold
throughout that period, including any level change. The library does not
infer those bounds from a current speed or an aircraft's intent.

A caller using a private copy named `geometric_safety.py` must also change
that import arrangement when adopting the package. A local module with that
name can shadow the installed package.

### Combined lateral and vertical calls

The existing combined functions accept `vertical_rate_fpm` as a shared
minimum climb/descent rate. Each aircraft may move faster, with independently
varying rates, provided it progresses towards its selected level at no less
than that rate and remains there on arrival. The check covers the altitude
bands between the minimum-rate progress and the selected levels.

Use [`certify_heading_separation_with_vertical_from_latlon`](#include-vertical-separation)
to retain this check with the heading interface. Pass the same current and
selected levels, the requested horizon as `horizon_s`, and the guaranteed
rate as `minimum_vertical_rate_fpm`. This minimum is required explicitly.
Negative rates raise `ValueError`; climb or descent follows from the levels.
`vertical_rounding_fl` must be a nonnegative integer, with zero disabling
rounding. The existing combined functions retain their argument conventions.

The combined result reports the resolution time, the lateral horizon used
and the nested lateral result. A missing vertical resolution is represented
by `None`, rather than the older helper's `-1.0` sentinel. Both functions
without `_with_vertical` perform a lateral-only check. An eventual climb/descent
rate does not cover an initial response delay or ramp below the declared
minimum.

### Behaviour changes for existing callers

Every nonzero instructed turn is retained. The constants
`TURN_HEADING_EPS_DEG` and `TURN_RATE_EPS_DEG_PER_S` have been removed, and
callers should remove code that uses those cut-offs to replace small turns
with straight flight. A changed heading with zero turn rate raises
`ValueError` in the legacy turn solver and returns `unsupported` from the
heading interface. An exact 180-degree difference selects a left turn;
differences on either side retain their direction, including near a heading
wrap. Legacy turn queries require finite headings and rates, including for
unchanged headings. Deviation bounds must be finite and nonnegative, with
relative turn-rate error below one; invalid declarations raise `ValueError`.
These declarations are checked even when vertical separation suffices.
Only the requirement for a nonzero rate on a changed heading is skipped
when the future turn is unused. Legacy vertical helpers require finite
levels, rates, times, gaps and rounding steps and raise `ValueError` otherwise.

Legacy nominal speeds, speed-uncertainty widths, thresholds and horizons
must be finite and nonnegative. Invalid values and lateral queries outside
the numerical domain return `False` with
unavailable distance/time diagnostics. Always inspect the decision before
using the diagnostics. For turning queries, the legacy distance/time pair
is not a guaranteed continuous-time minimum.

## Model and numerical details

The [motion-model reference](docs/motion-models.md) explains the speed-profile
enclosure, optional deviation bounds, diagnostics and numerical limits. It
also covers the existing constant-speed core and vertical filtering.

The [numerical validation and timing guide](paper/validation.md) gives the
independent reference assumptions, fixed inputs and package comparison
commands. The first call may compile Numba kernels; benchmark repeated calls
after compilation when measuring steady-state query cost.

## Development and tests

GitHub Actions runs formatting, linting, both Python type checkers, frontend
checks, the full Python suite and real Chromium tests on pull requests and
pushes to `main`. CI uses Python 3.13 on Ubuntu with locked dependencies.

From the repository root:

```bash
uv sync --locked --group apps --group browser
uv run --locked ruff format --check .
uv run --locked ruff check
uv run --locked ty check
uv run --locked pyright
uv run --locked pytest tests/ paper/tests/
```

CI runs most Python tests in four worker processes and reports the slowest
tests. The two modules that exercise multiprocessing pools run in a separate
serial job because forking from pytest-xdist's threaded workers can deadlock.
Both jobs run on every pull request. Each pytest worker has a fresh Numba cache;
audit subprocesses also start with fresh Python and Numba caches.

The `apps` group includes the web application and its integration tests, and
the `browser` group supplies the browser test tools. The library itself does
not require either group. Test functions are subject to the same
annotation checks as the library. Both `ty` and Pyright check Python code;
Pyright uses standard mode. Numerical tests include independently
integrated motion, arbitrary-precision references and exact vertical-gap
calculations. Audit guardrails are tested with invalid source and report data,
and with real production solvers, without replacing solver or worker functions.

For the visualiser's JavaScript type checks, install the locked development
tools with `npm ci --ignore-scripts`, then run `npm run check:frontend`. These
tools are not needed to serve the application.

The optional browser suite starts its own local server and uses real HTTP
responses and Chromium to check the form units, linked plots and position map:

```bash
uv sync --locked --group apps --group browser
uv run --locked playwright install chromium
uv run --locked pytest tests/browser/check_visualiser.py
```

Run the independent numerical audit with:

```bash
uv run --locked python -m paper.validate_performance_numerics \
    --output numerical_reference.json
```

## Relevant-aircraft visualiser

The interactive FastAPI application uses the geographic heading interfaces,
with optional minimum-rate vertical separation. It shows the required lateral
interval, remaining altitude bands, constant-speed reference geometry and the
extra allowance used to cover the declared speed profiles and errors. The
distance plot shows the same inflated threshold as the certificate. Its sampled
curve is explanatory; the decision comes from the continuous-time solver.

Choose an example encounter or enter independent speed bounds, headings and
positions through **Edit inputs**. Relative position error is entered in NM and
relative velocity error in knots. With vertical separation enabled, enter current
and selected flight levels and the minimum climb/descent rate. The time slider
links the plots; the plot-window control changes only the displayed interval.
The relative-geometry axes cover that whole displayed window and stay fixed
as the selected time changes. Altitude ticks use multiples of 10 FL.
The position map runs the same check at each cell and refreshes on request.
Results distinguish certified, unresolved and unsupported queries. Red marks
unresolved results; it does not establish a conflict.

The web endpoints accept nested `a` and `b` aircraft declarations and return
certificate results and display samples. Scripts using the older flat demo
payloads must adopt this schema. The browser converts position and velocity
error inputs to the endpoint's metres and metres per second; the library's
Python interfaces retain their documented units.

```bash
uv sync --locked --group apps
uv run --locked uvicorn app:app --reload
```

Then browse to <http://127.0.0.1:8000/>. Turn-aware presets include a single-turn
dodge, a single-turn cut-in, mutual opening turns and mutual closing turns.

## Benchmarks and paper results

For isolated lateral-solver timings:

```bash
uv run --locked python scripts/benchmark_relevant_aircraft.py --random-cases 5000
```

The `fixed_time_evals_py` column counts hull samples in the solver's Python
control flow, with compiled geometry helpers. It can differ from the compiled
path near floating-point decision boundaries. Decisions, diagnostics and timings
come from the compiled solver.

See [numerical validation and timing](paper/validation.md) for comparisons
between package versions, and [paper reproduction](paper/README.md) for the
figure and numerical-check scripts. An interactive geometry walkthrough is
also available in
[`relative_aircraft_projection_walkthrough.ipynb`](relative_aircraft_projection_walkthrough.ipynb).
