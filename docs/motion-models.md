# Motion models and numerical limits

The [README](../README.md) provides runnable examples and migration guidance.
This reference describes the motion assumptions behind the separation
decision and the meaning of its diagnostics.

## Prescribed heading motion

`HeadingMotion` describes a velocity direction that follows one immediate,
fixed-rate shortest turn and then stays at the target heading. Initial and
target headings are in degrees clockwise from north. Turn rate is a
nonnegative magnitude in degrees per second. The shortest signed heading
difference determines direction, with a left turn at an exact 180-degree tie.
Equal initial and target headings describe straight flight.

Each aircraft's horizontal speed can be any measurable profile between
`speed_min_mps` and `speed_max_mps`. These bounds must hold throughout the
requested horizon, including climb/descent changes and the straight
continuation. No acceleration law is required. The heading law is prescribed
independently of speed, so a turn whose rate changes with speed does not
satisfy this model.

The interface supports `turn_direction="shortest"`,
`heading_law="fixed_rate"` and `turn_delay_s=0.0`. Other declarations return
`unsupported`. Changing the target heading with zero turn rate also returns
`unsupported`. These fields let adapters describe an unsupported case
explicitly; they do not enable directed turns, route turns or response delays.

## Coordinate frames

The geographic entry points are `certify_heading_separation_from_latlon` and
`certify_heading_separation_with_vertical_from_latlon`. Both accept `a_lat`,
`a_lon`, `b_lat` and `b_lon` in degrees, with east-positive longitude, and use
one shared conversion. Positions and motion descriptions must refer to the
same starting time. `HeadingMotion` contains no position or frame origin, so
it can be reused when checking an aircraft against different neighbours.

Aircraft A is the local origin. The conversion uses `latlon_to_local_xy`,
with the shorter signed longitude difference from A to B. If latitudes and
that longitude difference are expressed in radians, B's coordinates are:

```text
east_B  = R cos(latitude_A) longitude_difference
north_B = R (latitude_B - latitude_A)
```

Here `R = 6378137 m`. A has coordinates `(0, 0)`, so the heading solver receives
`(-east_B, -north_B)`, the position of A minus the position of B. Both aircraft's
headings are used directly as directions clockwise from the common north
axis. The frame is fixed for the query. All distance and time diagnostics
refer to this planar calculation. Swapping A and B changes the reference
latitude and can change the approximation slightly.

Longitude wrapping handles pairs that straddle the date line, including
equivalent `-180` and `180` positions. Latitude must be strictly between
`-90` and `90` degrees, and longitude must be in `[-180, 180]`. Non-finite
coordinates, poles and out-of-range coordinates raise `ValueError`. An
exactly 180-degree longitude difference also raises `ValueError` because its
local direction is ambiguous. The combined function validates coordinates
even when vertical separation would suffice immediately.

The local approximation needs separate error bounds if a caller requires a
guarantee about physical separation. The conversion does not account for
Earth curvature or differences in local north between the aircraft, or
estimate the resulting error. Bounds on these differences must hold throughout
the checked interval and be included in the applicable
[deviation bounds](#deviation-bounds). A bound on the initial position alone
does not cover later projection or heading-frame error. The model is intended
for local encounters; valid coordinates do not establish its accuracy.

For callers that already have positions and headings in a common Cartesian
east/north frame, `certify_heading_separation` and
`certify_heading_separation_with_vertical` accept `relative_position_m`
directly, with A minus B and east before north, in metres. The geographic
functions perform the conversion and call these same solvers with the same
options. The returned types, status meanings, deviation envelopes and
vertical timing rules are identical.

## Speed-profile enclosure

The fixed-time position for a constant speed is linear in that speed. The
two independent speed intervals therefore give an exact relative-position
convex hull at each time for the constant-speed model. The turn-aware core
tests separation between those times using scalar and directional bounds,
subdividing intervals when necessary. An interval that remains unresolved at
the stopping width prevents acceptance.

To cover varying speed, the heading interface compares each profile at time
`t` with its mean speed over `[0, t]`. That mean lies inside the declared
interval, so the corresponding constant-speed position is already covered
by the hull. A deviation envelope bounds the difference between the two
positions.

For speed-interval width `rho`, turn-rate magnitude `omega` in radians per
second and elapsed turn angle `theta` in radians, the position envelope is

```text
E(t) = 4 rho / omega * (1 - cos(theta / 2))
```

The elapsed angle stops increasing when the turn ends. Subtracting the final
heading direction from the displacement integral makes the speed-variation
term vanish during the continuation. The envelope therefore stops growing
after the turn, even if speed continues to vary. It is zero for straight
flight, a zero-width speed interval or a zero-length horizon. The
implementation uses an equivalent expression that avoids cancellation for
small angles.

The interface adds both aircraft's envelopes to the separation threshold,
using their maximum over the requested horizon. For turning queries, it
also enlarges the narrower speed interval where necessary to use the core's
shared uncertainty width. Both steps preserve enclosure but can reject an
encounter that a tighter calculation would certify. Straight queries retain
the two independent speed intervals and use the exact-real straight-heading
geometry.

## Deviation bounds

The caller can declare additional errors through these parameters:

| Parameter | Meaning | Contribution to threshold inflation at horizon `T` |
| --- | --- | --- |
| `relative_position_error_m` on the query | Bound on the norm of the initial relative-position error | The supplied bound |
| `relative_velocity_error_mps` on the query | Pointwise bound on the norm of an additive relative-velocity error | The supplied bound times `T` |
| `heading_deviation_max_deg` on each `HeadingMotion` | Pointwise circular direction difference from that aircraft's prescribed heading law | `2 U T sin(delta / 2)`, with maximum speed `U` and angular bound `delta` in radians |

All contributions are added to the speed-profile envelopes.
`result.inflation_m` reports the resulting threshold increase. For an
unsupported declaration it is zero because no certificate is attempted.

The heading bound applies at every instant. A tolerance on the final heading
or on turn rate alone does not establish it. Bounds must describe the same
reference frame and velocity decomposition as the declared motion.

Common additive drift cancels from relative velocity. Differential wind can
be included through a valid relative-velocity bound, but that bound does not
establish that the executed heading law is unchanged. Any wind-induced
direction change must also be covered by the motion declaration or a valid
heading-deviation bound. The library does not derive these bounds from
weather, aircraft performance or simulator settings.

## Diagnostics

The decision concerns the declared motion enclosure over the complete
horizon. The optional `base_` diagnostics describe the underlying
constant-speed hull before accounting for the deviation envelope.

| `diagnostic_kind` | `base_min_distance_m` | Time information |
| --- | --- | --- |
| `straight_minimum` | Minimum from the straight-heading hull projection | `base_closest_time_s` identifies a model time attaining the minimum. |
| `sampled_minimum` | Smallest sampled turn-hull distance | `base_closest_time_s` is the sample time. |
| `interval_bound` | Lower bound on hull distance over an interval | `base_bound_interval_s` gives the endpoints; `base_closest_time_s` is `None`. |
| `None` | Unavailable | No distance/time diagnostics are supplied. |

A sampled minimum is not a continuous-time lower bound. Subtracting
`inflation_m` from it does not produce an executed-motion safety margin.
Neither a closest-approach time nor a bound interval identifies first
separation loss. Use `status` for acceptance; computing a conflict-onset time
requires a separate calculation.

## Numerical limits

The enclosure and geometric arguments use exact real arithmetic. The
implementation uses floating-point arithmetic with an empirical `1e-6 m`
guard. It does not provide validated interval arithmetic or a proved bound
on accumulated rounding error. A `certified_safe` status inherits this
numerical limitation as well as the caller's motion assumptions. The margin
applies to straight and turning checks, including zero-length horizons.

Hull construction uses a shared monotone-chain algorithm with filtered
orientation signs and an exact expansion fallback for ambiguous signs. If
coordinate scaling or product exponents prevent that fallback, it returns an
enclosing rectangle. This preserves containment at the cost of a looser
distance bound. These predicates do not bound the errors in trajectory
construction, distance calculations or time certification.

The numerical domain limits the initial relative distance plus maximum
total travel, and the inflated separation threshold, to about 8.8 million
metres. The shared limit is the guard divided by 512 times binary64 machine
epsilon. It keeps floating-point spacing small relative to the guard; the
factor is an engineering resolution policy. Separate speed and time limits
prevent overflow in intermediate calculations.

Queries outside the supported numerical domain return `unresolved` from
the heading interface. Legacy lateral functions return `False` with
unavailable distance/time diagnostics. These outcomes do not establish a
conflict. The [numerical validation guide](../paper/validation.md) describes
the independent checks and their scope.

## Existing lateral core

`catch_up_projection_interval` uses one convex projection for straight
headings. Its exact-real geometry also covers measurable speed variation
within the declared intervals, since each accumulated displacement depends
only on the mean speed over that interval.

`catch_up_projection_interval_with_turns` assumes constant speed over the
horizon unless one of its specific deviation envelopes is supplied. It uses
exact fixed-time reachable hulls and sufficient continuous-time interval
checks. A scalar distance bound handles easy intervals, and projected-motion
bounds along separating directions can certify additional intervals before
subdivision.

The optional `turn_speed_schedule_uncertainty_kt` covers its stated in-turn
deviation model. It should not be used to infer coverage of arbitrary speed
changes through the straight continuation. Use `certify_heading_separation`
for the full-horizon speed-profile contract.

## Vertical filtering

The package also provides an independent vertical cleared-band timing
check. It is an engineering extension and is not part of the lateral-only
paper experiments. `certify_heading_separation_with_vertical` applies it to
the heading interface and determines the required lateral horizon internally.
The existing combined functions use the same timing calculation.

`time_to_vertical_overlap_resolution` calculates when the two remaining
current-to-selected flight-level bands become separated by the required
vertical gap. Supply `vertical_rate_fpm` as a shared minimum climb/descent
rate magnitude. Direction is determined by current and selected levels.
Defaults are 1,000 ft/min and a gap of 10 flight levels. Negative rate inputs
are treated as magnitudes.

For a climb, the possible altitude band extends from the level reached at
the minimum rate up to the selected level. For a descent, it extends from
the selected level up to the level reached at the minimum rate. Faster
motion remains inside these bands. For example, a climb from FL100 to FL120
at no less than 1,000 ft/min puts the aircraft somewhere in [FL105, FL120]
after 30 seconds. Once the two bands have the required gap, their subsequent
shrinkage preserves that separation.

This argument permits different, time-varying rates for the two aircraft,
each no less than the supplied minimum until its selected level is reached.
Both must move towards their selected levels without overshoot and then
remain there. The calculation compares whole altitude bands, rather than
assuming that flying both aircraft at the minimum rate gives the smallest
vertical separation. If the aircraft have different guaranteed minimum
rates, use the smaller as the shared lower bound; this may postpone the
calculated resolution time.

Band edges are rounded towards one another to multiples of 10 flight levels
by default. Pass `rounded=0` to the standalone helper, or
`vertical_rounding_fl=0` to a combined function, to disable that rounding.
The standalone helpers and existing combined functions apply a tolerance of
`1e-9` flight levels to each compared edge at a grid boundary. An edge within
that tolerance can be treated as lying on the boundary. This can enlarge the
reported gap by up to `2e-9` flight levels, or `2e-7` ft, before accounting for
other floating-point error. The tolerance suppresses changes of grid cell
caused by tiny boundary differences; it is not an error bound on the motion
calculation.

The heading interface's combined check separately verifies the unrounded
gap using exact integer arithmetic on the supplied floating-point values.
The grid tolerance therefore cannot reduce the requested minimum gap for
this interface. If a proposed resolution time is early, the check moves it
later and verifies it again. The returned time is sufficient for separation
and need not be the first possible time.

For example, one aircraft remains at FL100 while the other climbs immediately
from FL100 to FL120 at no less than 1,000 ft/min:

```python
from geometric_safety import time_to_vertical_overlap_resolution

resolution_time_s = time_to_vertical_overlap_resolution(
    a_current_fl=100.0,
    a_selected_fl=100.0,
    b_current_fl=100.0,
    b_selected_fl=120.0,
    vertical_rate_fpm=1000.0,
    required_gap_fl=10.0,
    rounded=10,
)
print(resolution_time_s)
```

Output:

```text
60.0
```

`VERTICAL_OVERLAP_NEVER_RESOLVES_S` is the sentinel `-1.0` returned when
the bands never resolve under the model. A return value of zero means the
bands are already separated.

The combined functions use this calculation to shorten the lateral horizon.
`catch_up_projection_interval_with_vertical` uses straight headings, and
`catch_up_projection_interval_with_turns_and_vertical` includes turns:

```python
from geometric_safety import catch_up_projection_interval_with_turns_and_vertical

is_safe, _, _, resolution_time_s = catch_up_projection_interval_with_turns_and_vertical(
    a_lat=51.0,
    a_lon=-1.0,
    a_heading0_deg=0.0,
    a_target_heading_deg=45.0,
    a_speed_kt=300.0,
    a_turn_rate_deg_sec=1.5,
    a_current_fl=100.0,
    a_selected_fl=100.0,
    b_lat=51.15,
    b_lon=-0.7,
    b_heading0_deg=90.0,
    b_target_heading_deg=90.0,
    b_speed_kt=280.0,
    b_turn_rate_deg_sec=0.0,
    b_current_fl=100.0,
    b_selected_fl=120.0,
    separation_threshold_m=5.0 * 1852.0,
    speed_diff_kt=20.0,
    projection_time_s=120.0,
    vertical_rate_fpm=1000.0,
    required_vertical_gap_fl=10.0,
    vertical_rounding_fl=10,
)
print(bool(is_safe), resolution_time_s)
```

Output:

```text
True 60.0
```

If vertical bands are already resolved, the combined decision is safe and
the returned lateral diagnostics describe the geometry at time zero. If
they never resolve, the lateral solver checks the full requested horizon.
Otherwise, it only checks up to the vertical resolution time, capped at the
requested horizon.

The heading interface's combined function takes `minimum_vertical_rate_fpm`
explicitly. It returns a `CombinedHeadingAssuranceResult` with:

- `status` and `reason` for the whole requested horizon;
- `vertical_resolution_time_s`, or `None` if no usable resolution time was established;
- `lateral_horizon_s`, capped at the requested horizon;
- `lateral_result`, the heading solver's result over that lateral interval.

If the altitude bands are already separated, the combined result is
`certified_safe`, the lateral horizon is zero and `lateral_result` is `None`.
The horizontal law can be unsupported in this case, since vertical
separation alone establishes the condition. Malformed numeric inputs still
raise `ValueError`. Otherwise, a required lateral check retains its normal
`certified_safe`, `unresolved` or `unsupported` status. Its inflation, distance
and time diagnostics describe only the interval actually checked.

Unlike the existing combined functions, this interface rejects negative
minimum rates, negative gaps and negative or non-integer rounding steps.
A zero minimum guarantees no vertical progress. It can still certify bands
that are separated initially. Flight levels describe exact levels within
the model; there is no altitude-error allowance in this interface.

If the vertical calculation cannot supply a verified, representable time,
the combined function retains the full lateral horizon. The numerical guard
and exact verification are shared in `vertical.py`. When a proposed time is
early, the remaining gap deficit divided by the minimum rate gives a
conservative additional delay. At least one compared band edge must keep
moving at that rate until the required gap is reached, provided the selected
levels can supply it. The corrected time is rounded upward to a floating-point
value and checked again. The lateral solver retains its own
[numerical limits](#numerical-limits). See the
[combined example](../README.md#include-vertical-separation) for a complete call.

The minimum rate must bound progress from the start of the interval until
the selected level is reached. An eventual rate does not cover an initial
delay or ramp-up. The lateral speed and turn assumptions must also hold
throughout the lateral check, including during altitude changes.

A zero required vertical gap retains the mathematical nonnegative-gap
criterion in the library. It does not disable vertical checking and can make
coincident, zero-width altitude bands satisfy that criterion. The visualiser
requires a positive gap. When vertical checking is enabled, API requests
must explicitly declare both aircraft's current and selected levels and the
minimum climb/descent rate; omitted values cannot support a certificate.
