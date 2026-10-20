"""
Relative-aircraft catch-up projection using convex geometry.

Plain-language overview
-----------------------
This module asks a practical controller-style question:

    "Given where two aircraft are now, where they are pointed, how fast they might be
    going, and how they may turn, can they possibly come too close within the next
    few minutes?"

The key idea is not to simulate one nominal future. Instead, we construct the entire
envelope of futures allowed by the model and then ask whether any member of that
envelope enters the protected separation region.

There are two versions of the calculation:

1. Straight-heading method
   If both aircraft keep a fixed heading, the full set of possible relative motions can
   be collapsed into one convex 2-D shape. The problem then becomes: how close can that
   shape come to the origin? Under its assumptions, this version is exact.

2. Turn-aware method
   If one or both aircraft are turning at a fixed rate toward a target heading, the
   possible relative positions still form a simple convex shape at any one chosen time,
   but that shape changes over time. The method therefore:
   - computes the exact worst-case distance at selected times,
   - certifies whole time intervals using an interval-local bound on how fast the
     reachable hull can move,
   - and treats any interval that cannot be certified finely enough as unsafe.

In short:

    straight flight -> one exact geometric check,
    turning flight -> many exact geometric checks plus a conservative time certificate.

This means the turn-aware version is intentionally cautious: it should only say "safe"
when the whole modelled time horizon has been certified safe. The public API also
offers an optional additional robustness buffer for in-turn speed variation beyond the
core constant-speed turn model.

Formal statement
----------------
The remainder of this docstring recasts the same idea in a more formal style.

The underlying question is:

    Given two aircraft states now, bounded speed uncertainty, and a finite horizon
    [0, T], can any admissible motion bring the aircraft closer than a required
    separation distance D?

Both public methods answer that same decision problem, but under different kinematic
models:

1. `catch_up_projection_interval`
   Exact for fixed headings and bounded constant speeds.
2. `catch_up_projection_interval_with_turns`
   Exact in speed and conservative in time when each aircraft may execute one fixed-rate
   turn toward a target heading and then continue straight.

The sections below are intended to bridge implementation and exposition: they define the
objects being optimised, explain why the straight-line method reduces to a single convex
projection, and explain why the turning method requires a certified search in time.


Problem statement
-----------------
Fix a local east/north tangent plane. Let

    p_A(0), p_B(0) in R^2

be the current aircraft positions, and define the initial relative position

    r_0 = p_A(0) - p_B(0).

Let each aircraft speed lie in a closed interval

    s_A in [s_A^-, s_A^+],   s_B in [s_B^-, s_B^+],

with speeds treated as constant over the projection horizon once chosen. The core
quantity of interest is

    d_* = min ||p_A(t) - p_B(t)||
          over t in [0, T] and all admissible speeds.

The safety decision is then simply:

    safe  <=>  d_* >= D.

So the aim is not to predict one nominal closest approach, but to bound the worst case
over an admissible envelope of motion.


Method 1: Straight headings
---------------------------
For the straight-line model, each aircraft has a fixed heading. If `u_A` and `u_B` are
the corresponding unit heading vectors, then

    p_A(t) = p_A(0) + t s_A u_A,
    p_B(t) = p_B(0) + t s_B u_B.

Hence the relative motion is

    r(t) = r_0 + t (s_A u_A - s_B u_B).

The set of admissible relative velocities is the image of the speed rectangle under an
affine map:

    V = { s_A u_A - s_B u_B : s_A in [s_A^-, s_A^+], s_B in [s_B^-, s_B^+] }.

Because the speed intervals are one-dimensional, `V` is the convex hull of the four
corner velocities obtained from the speed extremes. Over the full horizon, the set of
all admissible relative displacements is

    S(T) = { t v : t in [0, T], v in V }.

Since `V` is convex and contains all admissible relative velocities,

    S(T) = conv({0} union {T v_i}),

where `v_i` are the four extreme relative velocities. This reduces the continuous-time
reachability problem to a single compact convex polygon in displacement space.

The worst-case separation becomes

    d_* = min_{d in S(T)} ||r_0 + d||.

Equivalently, one projects `-r_0` onto `S(T)`:

    d^* = argmin_{d in S(T)} ||d - (-r_0)||.

Then

    d_* = ||r_0 + d^*||.

Thus the straight-line method is exact under its model: one convex hull construction and
one Euclidean projection solve the full problem. The reported closest time is recovered
geometrically from the location of the projection on the displacement cone.


Method 2: One fixed-rate turn, then straight
--------------------------------------------
The turn-aware method assumes each aircraft follows a deterministic heading profile:

    - start at heading h_0,
    - turn at fixed rate `turn_rate_deg_sec` toward target heading h_T,
    - stop turning once the target heading is reached,
    - continue straight on h_T thereafter.

Let Delta h be the signed heading change and let

    tau = |Delta h| / |turn_rate_deg_sec|

be the turn duration when `turn_rate_deg_sec != 0`. Write the unit heading vector as

    u(h) = (sin h, cos h).

For a fixed speed s, the aircraft position can be written in the form

    p(t; s) = p(0) + s b(t),

where the basis vector `b(t)` depends only on the heading profile, not on speed:

    b(t) = integral_0^t u(h(xi)) d xi.

Because the heading profile is piecewise simple, this integral is available in closed
form. During the turn,

    h(t) = h_0 + sigma |turn_rate_deg_sec| t,

with `sigma` the turn direction, so `b(t)` is obtained by analytically integrating the
rotating unit vector. After the turn, one appends the straight segment

    (t - tau) u(h_T).

This closed-form basis is what `_turn_displacement_basis` computes.

The key structural fact is that, at any fixed time t, position is affine in speed.
Therefore the relative-position set at that time is

    R(t) = r_0 + { s_A b_A(t) - s_B b_B(t)
                   : s_A in [s_A^-, s_A^+], s_B in [s_B^-, s_B^+] }.

Again this is the image of a speed rectangle under an affine map, so `R(t)` is a small
convex polygon: a point, segment, or parallelogram described by the four speed-corner
combinations. The exact worst-case distance at time t is then

    d(t) = min_{r in R(t)} ||r||,

which is computed by projecting the origin onto that polygon.

Thus, unlike the straight-line method, the difficulty is no longer the speed envelope at
a fixed time; that part remains exact and convex. The difficulty is minimising `d(t)`
over continuous time:

    d_* = min_{t in [0, T]} d(t).


Certified adaptive time search
------------------------------
The implementation does not march over a fixed time grid. Instead it evaluates `d(t)`
exactly at selected times and certifies whole time intervals as safe whenever possible.

Let `R(t)` be the fixed-time reachable relative-position hull and let

    d(t) = dist(0, R(t)).

Each hull corner is a trajectory of the form

    r_i(t) = r_0 + s_A b_A(t) - s_B b_B(t),

so its instantaneous speed is

    ||r_i'(t)|| = ||s_A u_A(t) - s_B u_B(t)||.

A valid Lipschitz constant for `d(t)` on an interval `I = [t_0, t_1]` is therefore

    L_I = sup_{t in I} max_i ||r_i'(t)||,

because the convex hull cannot move faster, in Hausdorff distance, than its fastest
corner and distance-to-set is 1-Lipschitz with respect to that motion.

This is tighter than the simpler global bound

    L_global = s_A^+ + s_B^+,

which remains available as an opt-out for comparison. The local bound is the
default because it uses the known heading profiles on the interval instead of
assuming worst-case anti-parallel closing everywhere.

The practical reason this is fast is that each heading profile is piecewise linear in
time: turn at fixed rate until a known completion time, then hold the target heading.
Hence the relative heading on any subinterval between turn-completion events is linear.
For one speed corner pair,

    ||s_A u_A(t) - s_B u_B(t)||^2
        = s_A^2 + s_B^2 - 2 s_A s_B cos(delta(t)),

with `delta(t)` the relative heading. Over a linear-heading subinterval, the maximum is
attained at an endpoint or at an internal anti-parallel crossing

    delta(t) = (2k + 1) pi.

So the interval-local bound can be computed exactly with a constant amount of work:
check the interval endpoints, split only at turn-completion events if needed, and test
whether an anti-parallel crossing occurs on each linear piece.

Once a Lipschitz constant `L_I` is available, an interval [t_0, t_1] with endpoint
distances `d_0 = d(t_0)` and `d_1 = d(t_1)` satisfies the lower bound

    d(t) >= (d_0 + d_1 - L_I (t_1 - t_0)) / 2.

Therefore:

    if (d_0 + d_1 - L_I (t_1 - t_0)) / 2 >= D,
    the entire interval is certified safe.

When the scalar bound fails, the solver also tests fixed separating directions from
the endpoint hulls and their combined hull. Along each direction, bounds on the
projected velocities give two affine lower bounds for every speed corner, one from
each interval endpoint. Their maximum bounds that corner throughout the interval.
Taking the minimum over the four corners covers every admissible speed pair.

The algorithm starts with [0, T]. If neither certificate succeeds, it samples the
midpoint exactly, splits the interval in two, and repeats. Refinement continues until
each interval is either:

    - certified safe, or
    - smaller than a configured minimum interval width.

Intervals that remain uncertified at that minimum width are conservatively treated as
unsafe. This gives the central guarantee:

    the turn-aware method never returns "safe" unless all remaining time intervals have
    been certified safe under the model.

The reported `min_distance_m` and `closest_time_s` in the turning case are informative
diagnostics based on sampled points and interval certificates. The boolean safety result
is the primary mathematically guaranteed output.


Relationship between the two methods
------------------------------------
The straight-line method solves one global convex projection problem because the entire
reachable set over [0, T] collapses into a single displacement hull.

The turn-aware method solves a family of exact convex projection problems, one per
sampled time, because turning destroys the simple cone structure in time even though the
speed envelope remains convex at each fixed time.

So the distinction is:

    straight headings: exact in speed, exact in time,
    fixed-rate turns: exact in speed, conservative in time.


Structured hull optimization
----------------------------
Both methods share the same fixed-time geometric core:

    base + s_A a_basis - s_B b_basis,

with `s_A` and `s_B` drawn from closed speed intervals. The image of the speed
rectangle is therefore determined by just four speed-corner points.

This observation is now used explicitly in the implementation:

1. A shared helper constructs those four corner points for both
   - the straight-line displacement hull at `t = T`, and
   - the turn-aware relative-position hull at a fixed time `t`.
2. The hot-path hull construction for these tiny 4- and 5-point inputs uses a
   specialized small-set hull builder rather than the full generic monotone-chain
   implementation.
3. The generic `convex_hull` function is retained as a reference implementation and
   sanity oracle; tests compare the optimized structured path against it.

This optimization does not change the mathematics of the reachable sets. It only
exploits the fact that the relevant hulls in this module are tiny and highly
structured.


How this maps to functions
--------------------------
- `heading_to_unit_vector`: heading -> east/north direction.
- `latlon_to_local_xy`: latitude/longitude -> local east/north coordinates.
- `small_convex_hull`: optimized hull builder for the module's tiny structured inputs.
- `_speed_rectangle_corner_points`: shared affine speed-corner kernel.
- `compute_relative_velocity_hull`: straight-line reachable displacement hull over
  [0, T].
- `closest_point_on_convex_polygon`: Euclidean projection onto a convex polygon.
- `_closest_time_from_projection`: straight-line time recovery from the projected point.
- `catch_up_projection_interval`: exact straight-line safety test.
- `_turn_displacement_basis`: closed-form unit-speed displacement for a turn-then-straight
  heading law.
- `_interval_local_lipschitz_mps`: exact interval-local corner-speed supremum used as a
  turn-aware Lipschitz constant.
- `_relative_position_hull_at_time`: exact relative-position hull at one time.
- `_min_distance_to_relative_hull_at_time`: exact worst-case distance at one time.
- `_interval_distance_lower_bound`: Lipschitz certificate for a time interval.
- `catch_up_projection_interval_with_turns`: adaptive certified time search.


Safety interpretation
---------------------
- Straight-line method:
  - the full `(is_separated, min_distance_m, closest_time_s)` tuple is exact under the
    fixed-heading, bounded-constant-speed model.
- Turn-aware method:
  - exact in speed,
  - conservative in time,
  - and certified safe only when every remaining interval has been proved safe.
- The turn-aware API can optionally inflate the separation threshold by a conservative
  robustness buffer if one wants protection against speed variation during an active
  turn, beyond the core constant-speed model.
- If `is_separated` is `True` in the turn-aware method, the model certifies that no
  admissible speed realisation violates separation in the horizon considered.
- If `is_separated` is `False` in the turn-aware method, a separation loss is feasible
  in the modelled envelope or an interval could not be certified safe at the configured
  time resolution. With the optional robustness buffer enabled, `False` can also mean
  that the buffered threshold could not be certified even though the raw model remained
  clear of the nominal threshold.
- In the turn-aware method, `min_distance_m` and `closest_time_s` are diagnostics that
  summarize the most critical sampled point or interval certificate encountered.
- This is a deterministic reachability check, not a probability-of-conflict estimate.


Assumptions and limits
----------------------
- Horizontal 2-D kinematics only.
- Local tangent-plane projection; appropriate for local interactions, not long-haul
  geodesic fidelity.
- Symmetric bounded speed uncertainty, clipped at zero.
- No wind, acceleration, or stochastic dynamics.
- Straight-line mode assumes headings are constant over [0, T].
- Turn-aware mode assumes at most one fixed-rate turn per aircraft, from current
  heading to target heading, followed by straight flight.
- Turn-aware mode is conservative in time because it certifies intervals rather than
  solving the continuous-time minimum in closed form.
"""

import math
from typing import TypedDict

import numba
import numpy as np
from numpy.typing import NDArray

from geometric_safety.util import (
    DEG_TO_RAD,
    KT_TO_MPS,
    _project_point_on_convex_polygon,
    clip_value,
    closest_point_on_convex_polygon,
    convex_hull,
    heading_to_unit_vector,
    latlon_to_local_xy,
    shortest_turn_parameters,
    small_convex_hull,
)

# The adaptive certifier stops subdividing once an interval is this short. We use 3 s
# because radar updates arrive every 6 s in practice, so certifying below half a scan
# interval is usually a good speed/precision balance.
TURN_TIME_CERT_MIN_INTERVAL_S = 3.0

# Extra numerical margin on turn-certification comparisons. This is a precaution,
# not a proved bound on the complete floating-point evaluation chain. The mathematical
# safety theorem assumes exact arithmetic; numerical regression checks do not extend
# that theorem to every floating-point input.
TURN_CERT_FP_GUARD_M = 1e-6

# Supported numerical domain: retain at least 512 binary64 relative-resolution
# units within the comparison margin at the largest evaluated length scale.
# This engineering policy is not a proved error bound for the evaluation chain.
TURN_CERT_GUARD_RESOLUTION_FACTOR = 512.0
NUMERICAL_GEOMETRY_LIMIT_M = TURN_CERT_FP_GUARD_M / (TURN_CERT_GUARD_RESOLUTION_FACTOR * 2.0**-52)
NUMERICAL_SCALAR_LIMIT = math.sqrt(np.finfo(np.float64).max) / 64.0

DIAGNOSTIC_STRAIGHT_MINIMUM = 0
DIAGNOSTIC_SAMPLED_MINIMUM = 1
DIAGNOSTIC_INTERVAL_BOUND = 2
DIAGNOSTIC_UNAVAILABLE = 3


@numba.njit(cache=True, fastmath={"contract"})
def _geometry_within_numerical_domain(
    relative_position: NDArray[np.float64], travel_m: float, threshold_m: float
) -> bool:
    """Bound initial position plus total travel and the inflated threshold.

    Every evaluated relative position and displacement then has length at most
    NUMERICAL_GEOMETRY_LIMIT_M. Nonfinite or negative bounds are unsupported.
    """
    if not math.isfinite(relative_position[0]) or not math.isfinite(relative_position[1]):
        return False
    if not math.isfinite(travel_m) or travel_m < 0.0 or not math.isfinite(threshold_m) or threshold_m < 0.0:
        return False
    return (
        math.hypot(relative_position[0], relative_position[1]) + travel_m <= NUMERICAL_GEOMETRY_LIMIT_M
        and threshold_m <= NUMERICAL_GEOMETRY_LIMIT_M
    )


@numba.njit(cache=True, fastmath={"contract"})
def _scalar_within_numerical_range(value: float) -> bool:
    """Keep speed and time products away from overflow in the reused kernel."""
    return math.isfinite(value) and 0.0 <= value <= NUMERICAL_SCALAR_LIMIT


@numba.njit(cache=True)
def _record_solver_diagnostic(
    diagnostic_out: NDArray[np.float64] | None, kind: int, left_s: float, right_s: float
) -> None:
    """Fill optional provenance without changing the legacy solver return tuple."""
    if diagnostic_out is not None:
        diagnostic_out[0] = kind
        diagnostic_out[1] = left_s
        diagnostic_out[2] = right_s


# The turn-aware certifier defaults to an interval-local Lipschitz bound because it is
# less conservative than the simple global closing-rate bound while remaining easy to
# justify. Passing `use_interval_local_lipschitz=False` switches to the simpler global
# bound `a_max + b_max` for comparison or debugging.
TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT = True

# Optional robustness assumption for the turn-aware solver: during an active turn, an
# aircraft may vary its instantaneous speed within `nominal +/- this value` even though
# the core reachable-set model still uses one constant speed over the horizon. The code
# converts this to a conservative additive separation buffer and keeps it disabled by
# default.
TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT = 0.0


@numba.njit(cache=True, fastmath=True)
def _speed_rectangle_corner_points(
    base: np.ndarray,
    a_basis: np.ndarray,
    a_min_speed: float,
    a_max_speed: float,
    b_basis: np.ndarray,
    b_min_speed: float,
    b_max_speed: float,
) -> np.ndarray:
    """
    Map a 2-D speed rectangle to its four relative-space corner points.

    For both the straight and turn-aware methods, the reachable relative displacement or
    position set has the affine form

        base + s_A a_basis - s_B b_basis,

    with `s_A` and `s_B` drawn from closed speed intervals. The four speed-corner
    combinations therefore determine the entire convex set. Their order is
    (A minimum, B minimum), (A maximum, B minimum), (A maximum, B maximum),
    (A minimum, B maximum). Directional certification retains these identities
    when matching an endpoint position to its speed pair.
    """
    pts = np.empty((4, 2), dtype=np.float64)
    anchor_x = base[0] + a_min_speed * a_basis[0] - b_min_speed * b_basis[0]
    anchor_y = base[1] + a_min_speed * a_basis[1] - b_min_speed * b_basis[1]
    delta_a_x = (a_max_speed - a_min_speed) * a_basis[0]
    delta_a_y = (a_max_speed - a_min_speed) * a_basis[1]
    delta_b_x = -(b_max_speed - b_min_speed) * b_basis[0]
    delta_b_y = -(b_max_speed - b_min_speed) * b_basis[1]

    pts[0, 0] = anchor_x
    pts[0, 1] = anchor_y
    pts[1, 0] = anchor_x + delta_a_x
    pts[1, 1] = anchor_y + delta_a_y
    pts[2, 0] = anchor_x + delta_a_x + delta_b_x
    pts[2, 1] = anchor_y + delta_a_y + delta_b_y
    pts[3, 0] = anchor_x + delta_b_x
    pts[3, 1] = anchor_y + delta_b_y

    return pts


@numba.njit(cache=True, fastmath=False)
def _product_roundoff(a: float, b: float, product: float) -> float:
    """Recover a binary64 product's residual using Dekker splitting.

    Hull coordinates are bounded by the numerical geometry domain, so multiplying
    them by the binary64 splitter cannot overflow. Strict arithmetic preserves the
    separate products and subtractions required by the error-free transform.
    """
    splitter = 134217729.0  # 2**27 + 1 for a binary64 significand.
    split_a = splitter * a
    high_a = split_a - (split_a - a)
    low_a = a - high_a
    split_b = splitter * b
    high_b = split_b - (split_b - b)
    low_b = b - high_b
    return ((high_a * high_b - product) + high_a * low_b + low_a * high_b) + low_a * low_b


@numba.njit(cache=True, fastmath=False)
def _difference_of_products(a: float, b: float, c: float, d: float) -> float:
    """Evaluate a*b-c*d without losing a thin hull's determinant to cancellation."""
    left = a * b
    right = c * d
    # An exact product of two binary64 significands spans at most 106 bits.
    # Leave that margin above the minimum normal exponent (-1022), so the
    # low-product terms and cancellation residuals also avoid underflow.
    # Common coordinate scaling handles uniformly small hulls; more extreme
    # mixed scales remain unverified instead of appearing collinear.
    minimum_product = 2.0 ** (-1022 + 106)
    if (a != 0.0 and b != 0.0 and abs(left) < minimum_product) or (
        c != 0.0 and d != 0.0 and abs(right) < minimum_product
    ):
        return math.nan
    difference = left - right
    # Knuth TwoDiff, as in heading_diff, also retains the subtraction residual.
    right_virtual = left - difference
    residual = (left - (difference + right_virtual)) + (right_virtual - right)
    residual += _product_roundoff(a, b, left) - _product_roundoff(c, d, right)
    return difference + residual


@numba.njit(cache=True, fastmath=False)
def _closest_time_from_projection(closest_vec: NDArray[np.float64], hull: NDArray[np.float64], T: float) -> float:
    """
    Map a reachable straight-line displacement back to the earliest elapsed time.

    The straight reachable set is `S(T) = conv({0} U {T v_i})`. For a point `p in S(T)`,
    the earliest achievable time is the smallest `alpha in [0, 1]` such that

        p in alpha S(T).

    In 2-D, Caratheodory's theorem implies that this earliest representation can be
    expressed using the origin and at most two outer hull vertices, so the code searches
    all one- and two-vertex combinations and returns `t = alpha T`.

    Parameters
    ----------
    closest_vec : np.ndarray
        A point in the reachable displacement set (on or inside the hull).
    hull : np.ndarray
        Convex hull vertices of the reachable set.
    T : float
        Projection horizon in seconds.

    Returns
    -------
    float
        The time `t in [0, T]` at which this displacement is achieved in the straight
        constant-heading model, or NaN if no numerical representation is verified.
    """
    # The origin corresponds to zero elapsed time.
    if closest_vec[0] == 0.0 and closest_vec[1] == 0.0:
        return 0.0

    source_hull = np.asarray(hull, dtype=np.float64)
    n = source_hull.shape[0]
    if n == 0:
        return math.nan

    # A common power-of-two scale preserves each represented direction and all
    # barycentric coefficients while keeping uniformly tiny hulls out of underflow.
    scale = max(abs(closest_vec[0]), abs(closest_vec[1]))
    for vertex in source_hull:
        scale = max(scale, abs(vertex[0]), abs(vertex[1]))
    if not math.isfinite(scale):
        return math.nan
    _, exponent = math.frexp(scale)
    point = np.empty(2, dtype=np.float64)
    hull_arr = np.empty((n, 2), dtype=np.float64)
    for coordinate in range(2):
        point[coordinate] = math.ldexp(closest_vec[coordinate], -exponent)
        if closest_vec[coordinate] != 0.0 and point[coordinate] == 0.0:
            return math.nan
        for i in range(n):
            hull_arr[i, coordinate] = math.ldexp(source_hull[i, coordinate], -exponent)
            if source_hull[i, coordinate] != 0.0 and hull_arr[i, coordinate] == 0.0:
                return math.nan
    closest_vec = point

    # A truly one-dimensional hull has no competing nearby direction. Rounded
    # segment projections may use the coordinate reconstruction check there.
    one_dimensional = True
    direction_x, direction_y = 0.0, 0.0
    for vertex in hull_arr:
        if direction_x == 0.0 and direction_y == 0.0:
            direction_x, direction_y = vertex[0], vertex[1]
        elif _difference_of_products(direction_x, vertex[1], direction_y, vertex[0]) != 0.0:
            one_dimensional = False
            break

    # For p=u*a+v*b with nonnegative weights, the candidate time is T*(u+v).
    # The finite sentinel remains distinct under fastmath callers. All arithmetic
    # here is strict, including the compensated determinant calculations.
    best_alpha = 2.0
    # Each reconstructed coordinate uses at most two products, one addition and
    # one subtraction. gamma(8) also allows rounding in coefficient formation.
    # Bounds use those operation magnitudes, never an angular tolerance that could
    # collapse distinct rays. This checks reconstruction, not the preceding hull
    # and projection arithmetic or a validated interval enclosure.
    unit_roundoff = 2.0**-53
    reconstruction_factor = 8.0 * unit_roundoff / (1.0 - 8.0 * unit_roundoff)

    # Degenerate ray case: if `p` lies on the segment from the origin to a hull vertex,
    # then only one outer vertex is needed in the representation.
    for i in range(n):
        a = hull_arr[i]
        coordinate = 0 if abs(a[0]) >= abs(a[1]) else 1
        if a[coordinate] == 0.0:
            continue
        # Even a one-ulp transverse difference can change the earliest time
        # substantially on a thin hull. Reconstruction tolerance alone must not
        # turn a nearby ray into a collinear one.
        if not one_dimensional and _difference_of_products(a[0], closest_vec[1], a[1], closest_vec[0]) != 0.0:
            continue
        alpha = closest_vec[coordinate] / a[coordinate]
        if not 0.0 <= alpha <= 1.0:
            continue
        reconstructed_x = alpha * a[0]
        reconstructed_y = alpha * a[1]
        tolerance_x = reconstruction_factor * (abs(reconstructed_x) + abs(closest_vec[0]))
        tolerance_y = reconstruction_factor * (abs(reconstructed_y) + abs(closest_vec[1]))
        if abs(reconstructed_x - closest_vec[0]) > tolerance_x:
            continue
        if abs(reconstructed_y - closest_vec[1]) > tolerance_y:
            continue
        if alpha < best_alpha:
            best_alpha = alpha

    # General 2-D case: solve `closest_vec = u a + v b` for each vertex pair. On
    # nearly parallel vertex pairs the solve is ill-conditioned, so every candidate
    # must also reconstruct `closest_vec` before it is accepted; without this guard
    # a cancellation-corrupted solution can slip through the sign tests.
    for i in range(n):
        a = hull_arr[i]
        for j in range(i + 1, n):
            b = hull_arr[j]
            det = _difference_of_products(a[0], b[1], a[1], b[0])
            if not math.isfinite(det) or det == 0.0:
                continue
            u = _difference_of_products(closest_vec[0], b[1], closest_vec[1], b[0]) / det
            v = _difference_of_products(a[0], closest_vec[1], a[1], closest_vec[0]) / det
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            if u < 0.0 or v < 0.0:
                continue
            alpha = u + v
            if alpha > 1.0:
                continue
            term_ax, term_ay = u * a[0], u * a[1]
            term_bx, term_by = v * b[0], v * b[1]
            residual_x = term_ax + term_bx - closest_vec[0]
            residual_y = term_ay + term_by - closest_vec[1]
            tolerance_x = reconstruction_factor * (abs(term_ax) + abs(term_bx) + abs(closest_vec[0]))
            tolerance_y = reconstruction_factor * (abs(term_ay) + abs(term_by) + abs(closest_vec[1]))
            if abs(residual_x) > tolerance_x or abs(residual_y) > tolerance_y:
                continue
            if alpha < best_alpha:
                best_alpha = alpha

    # A support projection is not a radial intersection and cannot recover a valid
    # time when reconstruction fails. Keep diagnostics unavailable in that case.
    return T * best_alpha if best_alpha <= 1.0 else math.nan


@numba.njit(cache=True, fastmath=True)
def compute_relative_velocity_hull(
    a_heading: float,
    a_speed_kt: float,
    b_heading: float,
    b_speed_kt: float,
    speed_diff_kt: float,
    projection_time_s: float,
) -> np.ndarray:
    """
    Build the straight-line reachable relative-displacement hull over the projection window.

    Parameters
    ----------
    a_heading / b_heading: float
        Headings of aircraft A and B in degrees.
    a_speed_kt / b_speed_kt: float
        Nominal speeds in knots.
    speed_diff_kt: float
        Symmetric speed uncertainty used for both aircraft.
    projection_time_s: float
        Projection horizon in seconds.

    Returns
    -------
    np.ndarray
        Convex hull vertices spanning all admissible relative displacements in `[0, T]`.

    Notes
    -----
    The returned hull is `conv({0} U {T v_i})`, not just the outer velocity polygon
    scaled by `T`. Including the origin makes the hull represent all intermediate times
    in one object rather than only the exact-time slice at `t = T`.
    """
    a_min = max(a_speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS
    a_max = (a_speed_kt + speed_diff_kt) * KT_TO_MPS
    b_min = max(b_speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS
    b_max = (b_speed_kt + speed_diff_kt) * KT_TO_MPS

    dir_a = heading_to_unit_vector(a_heading)
    dir_b = heading_to_unit_vector(b_heading)
    a_basis = projection_time_s * dir_a
    b_basis = projection_time_s * dir_b

    # Append the origin so the hull covers the whole projection interval `[0, T]`, not
    # just the outer displacement slice at exactly `t = T`.
    reachable_points = np.empty((5, 2), dtype=np.float64)
    reachable_points[0, 0] = 0.0
    reachable_points[0, 1] = 0.0
    reachable_points[1:] = _speed_rectangle_corner_points(
        base=np.zeros(2, dtype=np.float64),
        a_basis=a_basis,
        a_min_speed=a_min,
        a_max_speed=a_max,
        b_basis=b_basis,
        b_min_speed=b_min,
        b_max_speed=b_max,
    )
    return small_convex_hull(reachable_points)


@numba.njit(cache=True, fastmath=False)
def _straight_hull_projection_interval(
    relative_position_m: NDArray[np.float64],
    displacement_hull: NDArray[np.float64],
    separation_threshold_m: float,
    projection_time_s: float,
) -> tuple[bool, float, float]:
    """Validate and project a Cartesian straight-flight displacement hull.

    The hull contains the origin and the horizon's reachable speed corners, in
    metres. The relative position is A minus B in the same frame. Both the legacy
    geographic adapter and independent-speed heading queries use this calculation.
    An unverified time makes the diagnostics unavailable and cannot certify safety.
    """
    if not _scalar_within_numerical_range(projection_time_s) or displacement_hull.shape[0] == 0:
        return False, math.nan, math.nan
    hull_travel_m = 0.0
    for vertex in displacement_hull:
        radius_m = math.hypot(vertex[0], vertex[1])
        if not math.isfinite(radius_m):
            return False, math.nan, math.nan
        hull_travel_m = max(hull_travel_m, radius_m)
    if not _geometry_within_numerical_domain(relative_position_m, hull_travel_m, separation_threshold_m):
        return False, math.nan, math.nan

    closest_vec, _, left, right, fraction = _project_point_on_convex_polygon(-relative_position_m, displacement_hull)
    rel_after = relative_position_m + closest_vec
    min_distance_m = float(np.linalg.norm(rel_after))
    closest_time_s = math.nan
    if left >= 0:
        a, b = displacement_hull[left], displacement_hull[right]
        a_is_origin = a[0] == 0.0 and a[1] == 0.0
        b_is_origin = b[0] == 0.0 and b[1] == 0.0
        # Keep the actual segment projection rather than reconstructing its ray
        # from rounded coordinates. A radial boundary gives its elapsed fraction;
        # a nonradial boundary consists of horizon-time speed combinations.
        if left == right:
            closest_time_s = 0.0 if a_is_origin else projection_time_s
        elif a_is_origin:
            closest_time_s = fraction * projection_time_s
        elif b_is_origin:
            closest_time_s = (1.0 - fraction) * projection_time_s
        else:
            _, exponent = math.frexp(max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])))
            cross = _difference_of_products(
                math.ldexp(a[0], -exponent),
                math.ldexp(b[1], -exponent),
                math.ldexp(a[1], -exponent),
                math.ldexp(b[0], -exponent),
            )
            if math.isfinite(cross) and cross != 0.0:
                closest_time_s = projection_time_s
    if not math.isfinite(closest_time_s):
        closest_time_s = _closest_time_from_projection(closest_vec, displacement_hull, projection_time_s)
    if not math.isfinite(min_distance_m) or not math.isfinite(closest_time_s):
        return False, math.nan, math.nan
    return min_distance_m >= separation_threshold_m, min_distance_m, closest_time_s


@numba.njit(cache=True, fastmath=False)
def catch_up_projection_interval(
    a_lat: float,
    a_lon: float,
    a_heading: float,
    a_speed_kt: float,
    b_lat: float,
    b_lon: float,
    b_heading: float,
    b_speed_kt: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    rel_pos_override: np.ndarray | None = None,
    velocity_hull: np.ndarray | None = None,
) -> tuple[bool, float, float]:
    """
    Evaluate straight-line horizontal separation over a finite projection horizon.

    Parameters
    ----------
    a_lat / a_lon: float
        Aircraft A latitude/longitude in degrees.
    a_heading: float
        Aircraft A heading in degrees.
    a_speed_kt: float
        Aircraft A nominal speed in knots.
    b_lat / b_lon: float
        Aircraft B latitude/longitude in degrees.
    b_heading: float
        Aircraft B heading in degrees.
    b_speed_kt: float
        Aircraft B nominal speed in knots.
    separation_threshold_m: float
        Required spacing (metres).
    speed_diff_kt: float
        Speed uncertainty (knots) applied symmetrically.
    projection_time_s: float
        Projection horizon in seconds.
    rel_pos_override: np.ndarray | None
        Optional pre-computed relative position (metres).
    velocity_hull: np.ndarray | None
        Optional pre-computed reachable hull.

    Returns
    -------
    tuple[bool, float, float]
        `(is_separated, min_distance_m, closest_time_s)`. Unsupported numerical
        geometry or unverified closest-time reconstruction returns `(False, NaN, NaN)`.

    Notes
    -----
    This is the exact fixed-heading method described in the module docstring. The whole
    speed-and-time envelope is reduced to one convex reachable-displacement hull, and
    the result follows from projecting `-rel_pos0` onto that hull.
    """
    if rel_pos_override is None:
        rel_pos0 = -latlon_to_local_xy(b_lat, b_lon, ref_lat=a_lat, ref_lon=a_lon)
    else:
        rel_pos0 = np.asarray(rel_pos_override, dtype=np.float64)

    if not _scalar_within_numerical_range(projection_time_s):
        return False, math.nan, math.nan
    if velocity_hull is None:
        a_max_speed_mps = (a_speed_kt + speed_diff_kt) * KT_TO_MPS
        b_max_speed_mps = (b_speed_kt + speed_diff_kt) * KT_TO_MPS
        if not _scalar_within_numerical_range(a_max_speed_mps + b_max_speed_mps):
            return False, math.nan, math.nan
        travel_m = (a_max_speed_mps + b_max_speed_mps) * projection_time_s
        if not _geometry_within_numerical_domain(rel_pos0, travel_m, separation_threshold_m):
            return False, math.nan, math.nan

    hull = (
        np.asarray(velocity_hull, dtype=np.float64)
        if velocity_hull is not None
        else compute_relative_velocity_hull(
            a_heading=a_heading,
            a_speed_kt=a_speed_kt,
            b_heading=b_heading,
            b_speed_kt=b_speed_kt,
            speed_diff_kt=speed_diff_kt,
            projection_time_s=projection_time_s,
        )
    )

    return _straight_hull_projection_interval(rel_pos0, hull, separation_threshold_m, projection_time_s)


@numba.njit(cache=True, fastmath=True)
def _turn_displacement_basis(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    t_s: float,
) -> np.ndarray:
    """
    Return the unit-speed displacement basis up to time `t_s`.

    The returned vector has units of seconds; multiplying by a speed in m/s gives a
    displacement in metres. Every nonzero instructed turn is retained. A changed
    target with a zero turn rate is invalid.

    Notes
    -----
    If the aircraft turns at fixed rate until the target heading is reached and then
    flies straight, its position can be written as

        p(t; s) = p(0) + s b(t),

    where `s` is the chosen constant speed and `b(t)` is this basis vector. This
    separation between speed and heading profile is what makes the fixed-time reachable
    set affine in speed.
    """
    out = np.empty(2, dtype=np.float64)
    turn_angle_deg, signed_turn_rate_deg_sec, turn_duration_s = shortest_turn_parameters(
        heading0_deg, target_heading_deg, turn_rate_deg_sec
    )

    if turn_angle_deg == 0.0:
        d = heading_to_unit_vector(target_heading_deg)
        out[0] = d[0] * t_s
        out[1] = d[1] * t_s
        return out

    # The sign of the heading difference chooses left vs. right turn; the magnitude of
    # the supplied turn rate gives the angular speed.
    signed_turn_rate_rad_per_s = signed_turn_rate_deg_sec * DEG_TO_RAD
    turn_time_s = clip_value(t_s, 0.0, turn_duration_s)
    heading0_rad = heading0_deg * DEG_TO_RAD
    half_sweep = 0.5 * signed_turn_rate_rad_per_s * turn_time_s
    # The chord has the midpoint heading and length t * sinc(half_sweep).
    # This avoids subtracting almost equal trigonometric values or dividing by rate.
    sinc = math.sin(half_sweep) / half_sweep if half_sweep != 0.0 else 1.0
    chord_length = turn_time_s * sinc
    out[0] = chord_length * math.sin(heading0_rad + half_sweep)
    out[1] = chord_length * math.cos(heading0_rad + half_sweep)

    # If the requested time extends past the turn, append the straight segment flown
    # on the target heading.
    if t_s > turn_duration_s:
        d = heading_to_unit_vector(target_heading_deg)
        straight_time_s = t_s - turn_duration_s
        out[0] += d[0] * straight_time_s
        out[1] += d[1] * straight_time_s

    return out


@numba.njit(cache=True, fastmath=True)
def _heading_at_time_from_turn_parameters(
    heading0_deg: float,
    target_heading_deg: float,
    signed_turn_rate_deg_sec: float,
    turn_duration_s: float,
    t_s: float,
) -> float:
    """Evaluate a heading using an already resolved shortest-turn law."""
    if signed_turn_rate_deg_sec == 0.0:
        return target_heading_deg
    return heading0_deg + signed_turn_rate_deg_sec * clip_value(t_s, 0.0, turn_duration_s)


@numba.njit(cache=True, fastmath=True)
def _unwrapped_heading_deg_at_time(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    t_s: float,
) -> float:
    """
    Return the deterministic heading profile value at time `t_s` without angle wrapping.

    The unwrapped representation is useful for interval-local Lipschitz bounds because
    relative heading then varies piecewise linearly in time. A wrapped heading would
    introduce artificial discontinuities at 0/360 deg.
    """
    _, signed_turn_rate_deg_sec, turn_duration_s = shortest_turn_parameters(
        heading0_deg, target_heading_deg, turn_rate_deg_sec
    )
    return _heading_at_time_from_turn_parameters(
        heading0_deg, target_heading_deg, signed_turn_rate_deg_sec, turn_duration_s, t_s
    )


@numba.njit(cache=True, fastmath=True)
def _relative_speed_mps(a_speed_mps: float, b_speed_mps: float, delta_heading_deg: float) -> float:
    """
    Return the instantaneous relative speed magnitude for one speed pair.

    If the current headings differ by `delta`, then

        ||s_A u_A - s_B u_B||^2 = s_A^2 + s_B^2 - 2 s_A s_B cos(delta).

    This quantity controls how quickly any one speed-corner trajectory can move inside
    the turn-aware reachable set.
    """
    half_sine = math.sin(0.5 * delta_heading_deg * DEG_TO_RAD)
    sq = (a_speed_mps - b_speed_mps) ** 2 + 4.0 * a_speed_mps * b_speed_mps * half_sine**2
    return math.sqrt(max(0.0, sq))


@numba.njit(cache=True, fastmath=True)
def _max_relative_speed_on_linear_heading_interval(
    a_speed_mps: float,
    b_speed_mps: float,
    delta_start_deg: float,
    delta_end_deg: float,
) -> float:
    """
    Maximise relative speed for one speed-corner pair over a linear heading interval.

    On any interval where both aircraft headings are linear in time, the relative
    heading is also linear in time. The relative speed depends only on `cos(delta)`, so
    the maximum over the interval occurs either:

    - at one endpoint, or
    - at an internal anti-parallel crossing `delta = (2k + 1) * 180 deg`,
      where the relative speed is exactly `a_speed_mps + b_speed_mps`.
    """
    best = max(
        _relative_speed_mps(a_speed_mps, b_speed_mps, delta_start_deg),
        _relative_speed_mps(a_speed_mps, b_speed_mps, delta_end_deg),
    )

    delta_lo = min(delta_start_deg, delta_end_deg)
    delta_hi = max(delta_start_deg, delta_end_deg)
    first_odd_multiple = 180.0 + 360.0 * math.ceil((delta_lo - 180.0) / 360.0)

    if first_odd_multiple <= delta_hi:
        best = max(best, a_speed_mps + b_speed_mps)

    return best


@numba.njit(cache=True, fastmath=True)
def _interval_local_lipschitz_mps(
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    t0_s: float,
    t1_s: float,
    global_lipschitz_mps: float,
) -> float:
    """
    Return the exact interval-local corner-speed supremum used as a Lipschitz constant.

    Motivation
    ----------
    A simple valid global bound is `a_max + b_max`, the largest possible closing rate
    across the whole horizon. That is always safe, but it is often too pessimistic
    because the two aircraft headings may be far from anti-parallel on most intervals.

    The distance-to-set function is 1-Lipschitz with respect to the Hausdorff motion of
    the reachable relative-position hull. Each hull corner moves with velocity

        v(t) = s_A u_A(t) - s_B u_B(t),

    so a valid interval-local Lipschitz constant for `d(t)` is

        sup_{t in [t0, t1]} max_corners ||v_corner(t)||.

    Because each heading profile is piecewise linear in time, the relative heading on
    each subinterval is linear. Therefore the supremum is attained at a subinterval
    endpoint or at an anti-parallel crossing, which makes the bound quick to compute and
    straightforward to justify.
    """
    if t1_s <= t0_s:
        return 0.0

    breakpoints = np.empty(4, dtype=np.float64)
    breakpoint_count = 0
    breakpoints[breakpoint_count] = t0_s
    breakpoint_count += 1

    a_turn_angle_deg, a_signed_rate_deg_sec, a_turn_end_s = shortest_turn_parameters(
        a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec
    )
    if a_turn_angle_deg != 0.0 and t0_s < a_turn_end_s < t1_s:
        breakpoints[breakpoint_count] = a_turn_end_s
        breakpoint_count += 1

    b_turn_angle_deg, b_signed_rate_deg_sec, b_turn_end_s = shortest_turn_parameters(
        b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec
    )
    if b_turn_angle_deg != 0.0 and t0_s < b_turn_end_s < t1_s:
        breakpoints[breakpoint_count] = b_turn_end_s
        breakpoint_count += 1

    breakpoints[breakpoint_count] = t1_s
    breakpoint_count += 1

    # Sort and deduplicate the small breakpoint list.
    for i in range(breakpoint_count):
        min_i = i
        for j in range(i + 1, breakpoint_count):
            if breakpoints[j] < breakpoints[min_i]:
                min_i = j
        tmp = breakpoints[i]
        breakpoints[i] = breakpoints[min_i]
        breakpoints[min_i] = tmp

    best = 0.0
    unique_breakpoints = np.empty(4, dtype=np.float64)
    unique_count = 0
    for i in range(breakpoint_count):
        # Every distinct endpoint matters, even when the interval is very short.
        if unique_count == 0 or breakpoints[i] != unique_breakpoints[unique_count - 1]:
            unique_breakpoints[unique_count] = breakpoints[i]
            unique_count += 1

    for i in range(unique_count - 1):
        left_t_s = unique_breakpoints[i]
        right_t_s = unique_breakpoints[i + 1]
        if right_t_s <= left_t_s:
            continue

        delta_left_deg = _heading_at_time_from_turn_parameters(
            a_heading0_deg, a_target_heading_deg, a_signed_rate_deg_sec, a_turn_end_s, left_t_s
        ) - _heading_at_time_from_turn_parameters(
            b_heading0_deg, b_target_heading_deg, b_signed_rate_deg_sec, b_turn_end_s, left_t_s
        )
        delta_right_deg = _heading_at_time_from_turn_parameters(
            a_heading0_deg, a_target_heading_deg, a_signed_rate_deg_sec, a_turn_end_s, right_t_s
        ) - _heading_at_time_from_turn_parameters(
            b_heading0_deg, b_target_heading_deg, b_signed_rate_deg_sec, b_turn_end_s, right_t_s
        )

        for a_speed_mps in (a_min_speed_mps, a_max_speed_mps):
            for b_speed_mps in (b_min_speed_mps, b_max_speed_mps):
                cand = _max_relative_speed_on_linear_heading_interval(
                    a_speed_mps,
                    b_speed_mps,
                    delta_left_deg,
                    delta_right_deg,
                )
                if cand > best:
                    best = cand
                    if best >= global_lipschitz_mps:
                        return global_lipschitz_mps

    return min(best, global_lipschitz_mps)


@numba.njit(cache=True, fastmath=True)
def _relative_position_corners_at_time(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    t_s: float,
) -> np.ndarray:
    """
    Build the four speed corners at a single time `t_s`, retaining their order.

    With fixed turn-rate profiles, each aircraft position is linear in its constant
    speed. The relative position set is therefore the affine image of a speed rectangle,
    i.e. a segment or parallelogram represented by up to four corner points.

    Returns
    -------
    np.ndarray
        Four speed corners, including duplicates or collinear points.
    """
    # For a fixed time, position = start + speed * basis(time). The speed bounds for A
    # and B therefore generate a four-corner rectangle in speed space, which maps to at
    # most four corner points in relative-position space.
    a_basis = _turn_displacement_basis(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec, t_s)
    b_basis = _turn_displacement_basis(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec, t_s)
    return _speed_rectangle_corner_points(
        base=rel_pos0,
        a_basis=a_basis,
        a_min_speed=a_min_speed_mps,
        a_max_speed=a_max_speed_mps,
        b_basis=b_basis,
        b_min_speed=b_min_speed_mps,
        b_max_speed=b_max_speed_mps,
    )


@numba.njit(cache=True, fastmath=True)
def _relative_position_hull_at_time(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    t_s: float,
) -> np.ndarray:
    """Exact fixed-time hull with duplicate and collinear corners removed."""
    return small_convex_hull(
        _relative_position_corners_at_time(
            rel_pos0,
            a_heading0_deg,
            a_target_heading_deg,
            a_turn_rate_deg_sec,
            a_min_speed_mps,
            a_max_speed_mps,
            b_heading0_deg,
            b_target_heading_deg,
            b_turn_rate_deg_sec,
            b_min_speed_mps,
            b_max_speed_mps,
            t_s,
        )
    )


@numba.njit(cache=True, fastmath=True)
def _sample_relative_hull_at_time(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    t_s: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Retain the corners and closest point for interval certification."""
    points = _relative_position_corners_at_time(
        rel_pos0,
        a_heading0_deg,
        a_target_heading_deg,
        a_turn_rate_deg_sec,
        a_min_speed_mps,
        a_max_speed_mps,
        b_heading0_deg,
        b_target_heading_deg,
        b_turn_rate_deg_sec,
        b_min_speed_mps,
        b_max_speed_mps,
        t_s,
    )
    closest, distance = closest_point_on_convex_polygon(np.zeros(2), small_convex_hull(points))
    return distance, points, closest


@numba.njit(cache=True, fastmath=True)
def _min_distance_to_relative_hull_at_time(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    t_s: float,
) -> float:
    """
    Return the exact worst-case relative distance at time `t_s`.

    This is the distance from the origin to the fixed-time reachable relative-position
    hull produced by `_relative_position_hull_at_time`.
    """
    # The exact worst-case distance at a fixed time is just the distance from the origin
    # to the reachable relative-position hull at that time.
    hull = _relative_position_hull_at_time(
        rel_pos0=rel_pos0,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        a_min_speed_mps=a_min_speed_mps,
        a_max_speed_mps=a_max_speed_mps,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        b_min_speed_mps=b_min_speed_mps,
        b_max_speed_mps=b_max_speed_mps,
        t_s=t_s,
    )
    origin = np.zeros(2, dtype=np.float64)
    _closest_point, min_dist_m = closest_point_on_convex_polygon(origin, hull)
    return min_dist_m


@numba.njit(cache=True, fastmath=True)
def _interval_distance_lower_bound(start_dist_m: float, end_dist_m: float, lipschitz_mps: float, dt_s: float) -> float:
    """
    Lower bound the minimum of an L-Lipschitz distance function over an interval.

    If `d(t)` is `L`-Lipschitz on `[t0, t1]`, then the minimum achievable value on the
    interval is at least `(d(t0) + d(t1) - L * (t1 - t0)) / 2`.

    This is the certificate used by the turn-aware adaptive search: if the bound is
    still above the separation threshold, the whole interval can be declared safe
    without any additional sampling inside it.
    """
    return max(0.0, 0.5 * (start_dist_m + end_dist_m - lipschitz_mps * dt_s))


@numba.njit(cache=True)
def _projected_heading_range(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    normal: np.ndarray,
    left_s: float,
    right_s: float,
) -> tuple[float, float]:
    """Range of the unit velocity projected on a fixed separating direction."""
    _, signed_rate_deg_sec, duration_s = shortest_turn_parameters(heading0_deg, target_heading_deg, turn_rate_deg_sec)
    left = (
        _heading_at_time_from_turn_parameters(heading0_deg, target_heading_deg, signed_rate_deg_sec, duration_s, left_s)
        * DEG_TO_RAD
    )
    right = (
        _heading_at_time_from_turn_parameters(
            heading0_deg, target_heading_deg, signed_rate_deg_sec, duration_s, right_s
        )
        * DEG_TO_RAD
    )
    lo, hi = min(left, right), max(left, right)
    v0 = normal[0] * math.sin(lo) + normal[1] * math.cos(lo)
    v1 = normal[0] * math.sin(hi) + normal[1] * math.cos(hi)
    lower, upper = min(v0, v1), max(v0, v1)
    phase = math.atan2(normal[0], normal[1])
    amplitude = math.hypot(normal[0], normal[1])
    # The prescribed heading is monotone and then constant. Its complete range
    # includes both endpoints and any stationary projections between them.
    if phase + 2.0 * math.pi * math.ceil((lo - phase) / (2.0 * math.pi)) <= hi:
        upper = max(upper, amplitude)
    minimum_phase = phase + math.pi
    if minimum_phase + 2.0 * math.pi * math.ceil((lo - minimum_phase) / (2.0 * math.pi)) <= hi:
        lower = min(lower, -amplitude)
    return lower, upper


@numba.njit(cache=True)
def _projected_motion_lower_bound(qa: float, qb: float, lower: float, upper: float, duration: float) -> float:
    """Minimise the two affine lower bounds obtained from the interval ends."""
    best = min(max(qa, qb - upper * duration), max(qa + lower * duration, qb))
    if upper > lower:
        crossing = (qa - qb + upper * duration) / (upper - lower)
        if 0.0 < crossing < duration:
            best = min(best, max(qa + lower * crossing, qb - upper * (duration - crossing)))
    return best


@numba.njit(cache=True)
def _interval_directional_lower_bound(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    left_s: float,
    right_s: float,
    required_m: float,
    include_hull: bool = True,
) -> float:
    """Certify a whole interval using fixed directions and projected velocities.

    Each speed corner has q(t) >= max(q(a)+m*(t-a), q(b)-M*(b-t)),
    where m and M bound its projected derivative throughout the interval.
    The minimum over corners also bounds every convex combination. Candidate
    directions come from the two endpoint hulls and their combined hull.
    Direction selection need not be exact: validity follows from evaluating
    all four corners along the same fixed direction, normalised to unit length.

    Stop once required_m is reached. Passing infinity evaluates all directions.
    The caller applies the deviation envelope and numerical comparison margin.
    """
    points = np.empty((2, 4, 2))
    for index, t_s in enumerate((left_s, right_s)):
        a_basis = _turn_displacement_basis(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec, t_s)
        b_basis = _turn_displacement_basis(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec, t_s)
        points[index] = _speed_rectangle_corner_points(
            rel_pos0,
            a_basis,
            a_min_speed_mps,
            a_max_speed_mps,
            b_basis,
            b_min_speed_mps,
            b_max_speed_mps,
        )
    closest_points = np.empty((2, 2))
    for index in range(2):
        closest_points[index], _ = closest_point_on_convex_polygon(np.zeros(2), small_convex_hull(points[index]))
    return _directional_lower_bound_from_samples(
        points,
        closest_points,
        a_heading0_deg,
        a_target_heading_deg,
        a_turn_rate_deg_sec,
        a_min_speed_mps,
        a_max_speed_mps,
        b_heading0_deg,
        b_target_heading_deg,
        b_turn_rate_deg_sec,
        b_min_speed_mps,
        b_max_speed_mps,
        left_s,
        right_s,
        required_m,
        include_hull,
    )


@numba.njit(cache=True)
def _directional_lower_bound_from_samples(
    points: np.ndarray,
    closest_points: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    left_s: float,
    right_s: float,
    required_m: float,
    include_hull: bool,
) -> float:
    """Evaluate projected-motion bounds using cached endpoint geometry."""
    best = 0.0
    for index in range(3 if include_hull else 2):
        if index < 2:
            closest = closest_points[index]
        else:
            closest, _ = closest_point_on_convex_polygon(np.zeros(2), convex_hull(points.reshape((8, 2))))
        length = math.hypot(closest[0], closest[1])
        if length == 0.0:
            continue
        normal = closest / length
        al, au = _projected_heading_range(
            a_heading0_deg,
            a_target_heading_deg,
            a_turn_rate_deg_sec,
            normal,
            left_s,
            right_s,
        )
        bl, bu = _projected_heading_range(
            b_heading0_deg,
            b_target_heading_deg,
            b_turn_rate_deg_sec,
            normal,
            left_s,
            right_s,
        )
        directional = math.inf
        for corner in range(4):
            # Match _speed_rectangle_corner_points: (min,min), (max,min),
            # (max,max), (min,max). Hull construction must not reorder these samples.
            sa = a_max_speed_mps if corner == 1 or corner == 2 else a_min_speed_mps
            sb = b_max_speed_mps if corner >= 2 else b_min_speed_mps
            qa = points[0, corner, 0] * normal[0] + points[0, corner, 1] * normal[1]
            qb = points[1, corner, 0] * normal[0] + points[1, corner, 1] * normal[1]
            directional = min(
                directional,
                _projected_motion_lower_bound(
                    qa,
                    qb,
                    sa * al - sb * bu,
                    sa * au - sb * bl,
                    right_s - left_s,
                ),
            )
        best = max(best, directional)
        if best >= required_m:
            break
    return best


@numba.njit(cache=True, fastmath=True)
def _turn_speed_schedule_envelope_m(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    t_s: float,
    schedule_uncertainty_kt: float,
    min_speed_mps: float,
    max_speed_mps: float,
) -> float:
    """
    Return the one-aircraft deviation envelope for bounded in-turn speed variation.

    The perturbed class holds the realised constant speed everywhere except during the
    active turn, where the instantaneous speed may follow any measurable speed profile inside
    `[s - delta, s + delta]` intersected with the admissible interval. The intersection
    is essential: excursions beyond the admissible interval cannot be absorbed along
    track and admit no envelope of this shape (see the paper's robustness section).

    For an instructed sweep `|dH| <= pi/2` the proven envelope is

        delta * t_active * (1 - cos(theta_a)) / theta_a,

    which never exceeds the coarser `delta * t_active * sin(theta_a / 2)`
    bound and is approached by extreme speed profiles in the long-horizon limit. For
    larger instructed sweeps the unconditional fallback

        min(2 * delta, width) * 4 * t_active * (1 - cos(theta_a / 2)) / theta_a

    applies, where `width` is the admissible interval width. The form is selected
    once per aircraft from the TOTAL instructed sweep, not from the sweep completed
    by the query time: a per-time switch could step downward at the pi/2 crossing
    when `width < ~0.854 * delta`, breaking the nondecreasing-envelope hypothesis
    of the inflated interval certificate. Each form on its own is nondecreasing in
    `t_s` (it grows with the completed sweep and freezes at turn completion), so
    the per-aircraft selection keeps the total envelope monotone.
    """
    if t_s <= 0.0 or schedule_uncertainty_kt <= 0.0:
        return 0.0

    signed_angle_deg, signed_rate_deg_sec, turn_duration_s = shortest_turn_parameters(
        heading0_deg, target_heading_deg, turn_rate_deg_sec
    )
    turn_angle_deg = abs(signed_angle_deg)
    if turn_angle_deg == 0.0:
        return 0.0

    active_turn_time_s = min(t_s, turn_duration_s)
    sweep_rad = abs(signed_rate_deg_sec) * DEG_TO_RAD * active_turn_time_s
    if active_turn_time_s <= 0.0 or sweep_rad == 0.0:
        return 0.0

    delta_mps = schedule_uncertainty_kt * KT_TO_MPS
    if turn_angle_deg <= 90.0:
        return delta_mps * active_turn_time_s * 2.0 * math.sin(0.5 * sweep_rad) ** 2 / sweep_rad

    width_mps = max(max_speed_mps - min_speed_mps, 0.0)
    amplitude_mps = min(2.0 * delta_mps, width_mps)
    return amplitude_mps * 8.0 * active_turn_time_s * math.sin(0.25 * sweep_rad) ** 2 / sweep_rad


@numba.njit(cache=True, fastmath=True)
def _turn_execution_envelope_m(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    t_s: float,
    max_speed_mps: float,
    turn_delay_max_s: float,
    turn_rate_error_max: float,
) -> float:
    """
    Return the one-aircraft deviation envelope for bounded turn execution error.

    The perturbed class delays turn initiation by up to `turn_delay_max_s` seconds and
    scales the turn rate by `(1 + e)` with `|e| <= turn_rate_error_max < 1`, ending at
    the instructed absolute target heading. The envelope integrates the worst-case
    heading-profile gap and multiplies by the maximum speed:

        s_max * [ min(|omega| t_d, |dH|) * min(t, tau + t_d)
                  + e * |dH| * min(t, tau / (1 - e)) ].

    The value saturates shortly after the instructed turn completes.
    """
    if t_s <= 0.0 or (turn_delay_max_s <= 0.0 and turn_rate_error_max <= 0.0):
        return 0.0

    signed_angle_deg, signed_rate_deg_sec, turn_duration_s = shortest_turn_parameters(
        heading0_deg, target_heading_deg, turn_rate_deg_sec
    )
    turn_angle_deg = abs(signed_angle_deg)
    if turn_angle_deg == 0.0:
        return 0.0

    omega_rad_s = abs(signed_rate_deg_sec) * DEG_TO_RAD
    turn_angle_rad = turn_angle_deg * DEG_TO_RAD

    phi_delay = 0.0
    if turn_delay_max_s > 0.0:
        heading_gap_rad = min(omega_rad_s * turn_delay_max_s, turn_angle_rad)
        phi_delay = heading_gap_rad * min(t_s, turn_duration_s + turn_delay_max_s)

    phi_rate = 0.0
    if 0.0 < turn_rate_error_max < 1.0:
        phi_rate = turn_rate_error_max * turn_angle_rad * min(t_s, turn_duration_s / (1.0 - turn_rate_error_max))

    return max_speed_mps * (phi_delay + phi_rate)


@numba.njit(cache=True, fastmath=True)
def _deviation_envelope_m(
    t_s: float,
    eta_const_m: float,
    eta_rate_mps: float,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    turn_speed_schedule_uncertainty_kt: float,
    turn_delay_max_s: float,
    turn_rate_error_max: float,
) -> float:
    """
    Return the total deviation envelope eta(t) for the configured perturbation classes.

    `eta_const_m` covers constant offsets such as relative-position estimation error.
    `eta_rate_mps` covers linear-in-time classes: a differential wind bound contributes
    its magnitude directly, and a heading-estimate bound of `eps_i` radians per aircraft
    contributes `s_i_max * eps_i`. The remaining terms are the closed-form in-turn
    schedule and turn-execution envelopes. The total is nondecreasing in `t_s`, which
    the inflated interval certificate relies on.
    """
    eta_m = eta_const_m + eta_rate_mps * max(t_s, 0.0)
    eta_m += _turn_speed_schedule_envelope_m(
        heading0_deg=a_heading0_deg,
        target_heading_deg=a_target_heading_deg,
        turn_rate_deg_sec=a_turn_rate_deg_sec,
        t_s=t_s,
        schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
        min_speed_mps=a_min_speed_mps,
        max_speed_mps=a_max_speed_mps,
    )
    eta_m += _turn_speed_schedule_envelope_m(
        heading0_deg=b_heading0_deg,
        target_heading_deg=b_target_heading_deg,
        turn_rate_deg_sec=b_turn_rate_deg_sec,
        t_s=t_s,
        schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
        min_speed_mps=b_min_speed_mps,
        max_speed_mps=b_max_speed_mps,
    )
    eta_m += _turn_execution_envelope_m(
        heading0_deg=a_heading0_deg,
        target_heading_deg=a_target_heading_deg,
        turn_rate_deg_sec=a_turn_rate_deg_sec,
        t_s=t_s,
        max_speed_mps=a_max_speed_mps,
        turn_delay_max_s=turn_delay_max_s,
        turn_rate_error_max=turn_rate_error_max,
    )
    eta_m += _turn_execution_envelope_m(
        heading0_deg=b_heading0_deg,
        target_heading_deg=b_target_heading_deg,
        turn_rate_deg_sec=b_turn_rate_deg_sec,
        t_s=t_s,
        max_speed_mps=b_max_speed_mps,
        turn_delay_max_s=turn_delay_max_s,
        turn_rate_error_max=turn_rate_error_max,
    )
    return eta_m


@numba.njit(cache=True, fastmath=True)
def _catch_up_projection_interval_with_turns_local(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    use_interval_local_lipschitz: bool = TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
    turn_speed_schedule_uncertainty_kt: float = TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
    min_cert_interval_s: float = TURN_TIME_CERT_MIN_INTERVAL_S,
    eta_const_m: float = 0.0,
    eta_rate_mps: float = 0.0,
    turn_delay_max_s: float = 0.0,
    turn_rate_error_max: float = 0.0,
    directional_mode: int = 2,
    diagnostic_out: NDArray[np.float64] | None = None,
) -> tuple[bool, float, float]:
    """Local solver; directional_mode 0 retains the scalar reference path.

    Mode 1 adds endpoint directions; mode 2 also tests the combined hull.
    These private switches support matched verification and timing.
    diagnostic_out optionally receives [kind, left_s, right_s]. Codes distinguish
    a straight minimum, a sampled minimum, an interval bound and unavailable
    diagnostics. The legacy return tuple remains unchanged.
    """
    if diagnostic_out is not None and diagnostic_out.size < 3:
        raise ValueError("diagnostic_out must contain at least three elements")
    _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_UNAVAILABLE, math.nan, math.nan)
    a_turn_angle_deg, _, _ = shortest_turn_parameters(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec)
    b_turn_angle_deg, _, _ = shortest_turn_parameters(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec)
    a_is_turning = a_turn_angle_deg != 0.0
    b_is_turning = b_turn_angle_deg != 0.0

    # Convert the speed envelope once; the remaining geometry uses metres and seconds.
    a_min_speed_mps = max((a_speed_kt - speed_diff_kt), 0.0) * KT_TO_MPS
    a_max_speed_mps = (a_speed_kt + speed_diff_kt) * KT_TO_MPS
    b_min_speed_mps = max((b_speed_kt - speed_diff_kt), 0.0) * KT_TO_MPS
    b_max_speed_mps = (b_speed_kt + speed_diff_kt) * KT_TO_MPS
    global_lipschitz_mps = a_max_speed_mps + b_max_speed_mps
    if not _scalar_within_numerical_range(projection_time_s) or not _scalar_within_numerical_range(
        global_lipschitz_mps
    ):
        return False, math.nan, math.nan
    if not _geometry_within_numerical_domain(
        rel_pos0, global_lipschitz_mps * projection_time_s, separation_threshold_m + eta_const_m
    ):
        return False, math.nan, math.nan

    # Constant and linear envelope terms apply to straight motion too, so the exact
    # straight delegation is only taken when no envelope is configured. The turn-only
    # envelope terms vanish without an instructed turn.
    if not a_is_turning and not b_is_turning and eta_const_m <= 0.0 and eta_rate_mps <= 0.0:
        hull = compute_relative_velocity_hull(
            a_target_heading_deg, a_speed_kt, b_target_heading_deg, b_speed_kt, speed_diff_kt, projection_time_s
        )
        result = _straight_hull_projection_interval(rel_pos0, hull, separation_threshold_m, projection_time_s)
        if _scalar_within_numerical_range(result[1]) and _scalar_within_numerical_range(result[2]):
            _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_STRAIGHT_MINIMUM, result[2], result[2])
        return result

    # A conservative global Lipschitz constant for distance is the maximum possible
    # closing rate, bounded by the sum of the two maximum speeds. The default path
    # tightens this per interval using the known heading profiles, but the simple global
    # bound is retained as an opt-out option for debugging and comparison.

    # Deviation envelopes inflate the required threshold per evaluated time. The
    # geometry and certification logic are unchanged, and the zero-envelope fast path
    # skips the envelope evaluation entirely.
    has_envelope = (
        eta_const_m > 0.0
        or eta_rate_mps > 0.0
        or turn_speed_schedule_uncertainty_kt > 0.0
        or turn_delay_max_s > 0.0
        or turn_rate_error_max > 0.0
    )

    if projection_time_s <= 0.0:
        min_dist_now = float(np.linalg.norm(rel_pos0))
        _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, 0.0, 0.0)
        return min_dist_now >= separation_threshold_m + eta_const_m, min_dist_now, 0.0

    if min_cert_interval_s <= 0.0:
        min_cert_interval_s = TURN_TIME_CERT_MIN_INTERVAL_S

    # Always check the horizon endpoints explicitly before doing any interval logic.
    # At t = 0 only the constant envelope term is active.
    start_dist_m = float(np.linalg.norm(rel_pos0))
    if start_dist_m < separation_threshold_m + eta_const_m:
        _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, 0.0, 0.0)
        return False, start_dist_m, 0.0

    end_dist_m, end_points, end_closest = _sample_relative_hull_at_time(
        rel_pos0=rel_pos0,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        a_min_speed_mps=a_min_speed_mps,
        a_max_speed_mps=a_max_speed_mps,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        b_min_speed_mps=b_min_speed_mps,
        b_max_speed_mps=b_max_speed_mps,
        t_s=projection_time_s,
    )

    # These track the smallest sampled distance we have seen. They are useful both for
    # diagnostics and for early conflict detection when a sampled midpoint is already bad.
    best_sample_dist_m = start_dist_m
    best_sample_time_s = 0.0
    if end_dist_m < best_sample_dist_m:
        best_sample_dist_m = end_dist_m
        best_sample_time_s = projection_time_s

    # If the whole horizon already certifies as safe from the endpoint distances, we can
    # return immediately without any subdivision.
    initial_lower_bound_m = _interval_distance_lower_bound(
        start_dist_m=start_dist_m,
        end_dist_m=end_dist_m,
        lipschitz_mps=(
            _interval_local_lipschitz_mps(
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=a_target_heading_deg,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=b_target_heading_deg,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                t0_s=0.0,
                t1_s=projection_time_s,
                global_lipschitz_mps=global_lipschitz_mps,
            )
            if use_interval_local_lipschitz
            else global_lipschitz_mps
        ),
        dt_s=projection_time_s,
    )
    eta_horizon_m = (
        _deviation_envelope_m(
            t_s=projection_time_s,
            eta_const_m=eta_const_m,
            eta_rate_mps=eta_rate_mps,
            a_heading0_deg=a_heading0_deg,
            a_target_heading_deg=a_target_heading_deg,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min_speed_mps,
            a_max_speed_mps=a_max_speed_mps,
            b_heading0_deg=b_heading0_deg,
            b_target_heading_deg=b_target_heading_deg,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min_speed_mps,
            b_max_speed_mps=b_max_speed_mps,
            turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
            turn_delay_max_s=turn_delay_max_s,
            turn_rate_error_max=turn_rate_error_max,
        )
        if has_envelope
        else 0.0
    )
    if not _geometry_within_numerical_domain(
        rel_pos0, global_lipschitz_mps * projection_time_s, separation_threshold_m + eta_horizon_m
    ):
        return False, math.nan, math.nan
    if initial_lower_bound_m >= separation_threshold_m + eta_horizon_m + TURN_CERT_FP_GUARD_M:
        _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, best_sample_time_s, best_sample_time_s)
        return True, best_sample_dist_m, best_sample_time_s

    # Depth-first interval subdivision uses an explicit stack because Numba handles
    # arrays well but not Python recursion.
    stack_capacity = 4
    span_s = projection_time_s
    while span_s > min_cert_interval_s:
        stack_capacity += 1
        span_s *= 0.5

    left_t_stack = np.empty(stack_capacity, dtype=np.float64)
    right_t_stack = np.empty(stack_capacity, dtype=np.float64)
    left_d_stack = np.empty(stack_capacity, dtype=np.float64)
    right_d_stack = np.empty(stack_capacity, dtype=np.float64)
    point_stack = np.empty((stack_capacity, 2, 4, 2))
    closest_stack = np.empty((stack_capacity, 2, 2))

    top = 0
    left_t_stack[top] = 0.0
    right_t_stack[top] = projection_time_s
    left_d_stack[top] = start_dist_m
    right_d_stack[top] = end_dist_m
    point_stack[top, 0] = rel_pos0
    point_stack[top, 1] = end_points
    closest_stack[top, 0] = rel_pos0
    closest_stack[top, 1] = end_closest
    top += 1

    # If we fail to certify a tiny interval, report the most critical lower bound we saw
    # rather than an arbitrary large sampled distance.
    most_critical_lower_bound_m = initial_lower_bound_m
    most_critical_time_s = 0.5 * projection_time_s
    most_critical_left_s = 0.0
    most_critical_right_s = projection_time_s

    while top > 0:
        top -= 1
        left_t_s = left_t_stack[top]
        right_t_s = right_t_stack[top]
        left_dist_m = left_d_stack[top]
        right_dist_m = right_d_stack[top]
        # The left child is stored above the parent before the right child reuses
        # its slot, so these endpoint views need no temporary copies.
        interval_points = point_stack[top]
        interval_closest = closest_stack[top]
        dt_s = right_t_s - left_t_s
        interval_lipschitz_mps = (
            _interval_local_lipschitz_mps(
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=a_target_heading_deg,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=b_target_heading_deg,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                t0_s=left_t_s,
                t1_s=right_t_s,
                global_lipschitz_mps=global_lipschitz_mps,
            )
            if use_interval_local_lipschitz
            else global_lipschitz_mps
        )

        interval_lower_bound_m = _interval_distance_lower_bound(
            start_dist_m=left_dist_m,
            end_dist_m=right_dist_m,
            lipschitz_mps=interval_lipschitz_mps,
            dt_s=dt_s,
        )
        if interval_lower_bound_m < most_critical_lower_bound_m:
            most_critical_lower_bound_m = interval_lower_bound_m
            most_critical_time_s = 0.5 * (left_t_s + right_t_s)
            most_critical_left_s = left_t_s
            most_critical_right_s = right_t_s

        # The envelope is nondecreasing, so evaluating it at the interval's right
        # endpoint bounds it over the whole interval, matching the inflated interval
        # certificate.
        interval_eta_m = (
            _deviation_envelope_m(
                t_s=right_t_s,
                eta_const_m=eta_const_m,
                eta_rate_mps=eta_rate_mps,
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=a_target_heading_deg,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=b_target_heading_deg,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
                turn_delay_max_s=turn_delay_max_s,
                turn_rate_error_max=turn_rate_error_max,
            )
            if has_envelope
            else 0.0
        )

        # This whole interval is certified safe, so there is no need to look inside it.
        if interval_lower_bound_m >= separation_threshold_m + interval_eta_m + TURN_CERT_FP_GUARD_M:
            continue

        if directional_mode:
            directional_lower_bound_m = _directional_lower_bound_from_samples(
                interval_points,
                interval_closest,
                a_heading0_deg,
                a_target_heading_deg,
                a_turn_rate_deg_sec,
                a_min_speed_mps,
                a_max_speed_mps,
                b_heading0_deg,
                b_target_heading_deg,
                b_turn_rate_deg_sec,
                b_min_speed_mps,
                b_max_speed_mps,
                left_t_s,
                right_t_s,
                separation_threshold_m + interval_eta_m + TURN_CERT_FP_GUARD_M,
                directional_mode >= 2,
            )
            if directional_lower_bound_m >= separation_threshold_m + interval_eta_m + TURN_CERT_FP_GUARD_M:
                continue

        # At the minimum certification width we stop refining. If the interval still
        # cannot be certified, we conservatively report it as potentially unsafe.
        if dt_s <= min_cert_interval_s:
            if best_sample_dist_m < separation_threshold_m:
                _record_solver_diagnostic(
                    diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, best_sample_time_s, best_sample_time_s
                )
                return False, best_sample_dist_m, best_sample_time_s
            _record_solver_diagnostic(
                diagnostic_out, DIAGNOSTIC_INTERVAL_BOUND, most_critical_left_s, most_critical_right_s
            )
            return False, most_critical_lower_bound_m, most_critical_time_s

        # Sample the midpoint to tighten both halves of the interval. If the midpoint
        # itself is already within the threshold, we have a direct witness of conflict.
        mid_t_s = 0.5 * (left_t_s + right_t_s)
        if not left_t_s < mid_t_s < right_t_s:
            # Timestamp spacing can exceed the stopping width. Repeating this split
            # would not shrink a child and could overrun the fixed-depth stack.
            _record_solver_diagnostic(
                diagnostic_out, DIAGNOSTIC_INTERVAL_BOUND, most_critical_left_s, most_critical_right_s
            )
            return False, most_critical_lower_bound_m, math.nan
        mid_dist_m, mid_points, mid_closest = _sample_relative_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=a_heading0_deg,
            a_target_heading_deg=a_target_heading_deg,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min_speed_mps,
            a_max_speed_mps=a_max_speed_mps,
            b_heading0_deg=b_heading0_deg,
            b_target_heading_deg=b_target_heading_deg,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min_speed_mps,
            b_max_speed_mps=b_max_speed_mps,
            t_s=mid_t_s,
        )
        if mid_dist_m < best_sample_dist_m:
            best_sample_dist_m = mid_dist_m
            best_sample_time_s = mid_t_s

        # A midpoint inside the inflated threshold can never be certified, so the
        # search can stop early. With no envelope this is the plain conflict witness.
        mid_eta_m = (
            _deviation_envelope_m(
                t_s=mid_t_s,
                eta_const_m=eta_const_m,
                eta_rate_mps=eta_rate_mps,
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=a_target_heading_deg,
                a_turn_rate_deg_sec=a_turn_rate_deg_sec,
                a_min_speed_mps=a_min_speed_mps,
                a_max_speed_mps=a_max_speed_mps,
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=b_target_heading_deg,
                b_turn_rate_deg_sec=b_turn_rate_deg_sec,
                b_min_speed_mps=b_min_speed_mps,
                b_max_speed_mps=b_max_speed_mps,
                turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
                turn_delay_max_s=turn_delay_max_s,
                turn_rate_error_max=turn_rate_error_max,
            )
            if has_envelope
            else 0.0
        )
        if mid_dist_m < separation_threshold_m + mid_eta_m:
            _record_solver_diagnostic(
                diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, best_sample_time_s, best_sample_time_s
            )
            return False, best_sample_dist_m, best_sample_time_s

        # Push the two child intervals. DFS keeps memory small and tends to find nearby
        # conflicts quickly once an interval starts looking critical.
        left_t_stack[top + 1] = left_t_s
        right_t_stack[top + 1] = mid_t_s
        left_d_stack[top + 1] = left_dist_m
        right_d_stack[top + 1] = mid_dist_m
        point_stack[top + 1, 0], point_stack[top + 1, 1] = interval_points[0], mid_points
        closest_stack[top + 1, 0], closest_stack[top + 1, 1] = interval_closest[0], mid_closest

        left_t_stack[top] = mid_t_s
        left_d_stack[top] = mid_dist_m
        point_stack[top, 0] = mid_points
        closest_stack[top, 0] = mid_closest
        # The right child's right endpoint already occupies the reused slot.
        top += 2

    _record_solver_diagnostic(diagnostic_out, DIAGNOSTIC_SAMPLED_MINIMUM, best_sample_time_s, best_sample_time_s)
    return True, best_sample_dist_m, best_sample_time_s


class TurnInputs(TypedDict):
    """Required keyword inputs to the geographic turn-aware projection."""

    a_lat: float
    a_lon: float
    a_heading0_deg: float
    a_target_heading_deg: float
    a_speed_kt: float
    a_turn_rate_deg_sec: float
    b_lat: float
    b_lon: float
    b_heading0_deg: float
    b_target_heading_deg: float
    b_speed_kt: float
    b_turn_rate_deg_sec: float
    separation_threshold_m: float
    speed_diff_kt: float
    projection_time_s: float


@numba.njit(cache=True, fastmath=True)
def catch_up_projection_interval_with_turns(
    a_lat: float,
    a_lon: float,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    b_lat: float,
    b_lon: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    use_interval_local_lipschitz: bool = TURN_USE_INTERVAL_LOCAL_LIPSCHITZ_DEFAULT,
    turn_speed_schedule_uncertainty_kt: float = TURN_SPEED_SCHEDULE_UNCERTAINTY_KT_DEFAULT,
    eta_const_m: float = 0.0,
    eta_rate_mps: float = 0.0,
    turn_delay_max_s: float = 0.0,
    turn_rate_error_max: float = 0.0,
) -> tuple[bool, float, float]:
    """
    Evaluate horizontal separation when one or both aircraft may execute one turn.

    For turning cases, this function is exact in speed and conservative in time:
    at each sampled time it computes the exact reachable relative-position hull for
    the constant-speed envelope, then uses scalar Lipschitz and directional
    projected-motion bounds to certify intervals as safe. Intervals that cannot be certified at the
    configured minimum width are treated as unsafe.

    When both target headings equal their initial headings, it delegates to the
    exact straight-line `catch_up_projection_interval`. A changed target heading
    requires a nonzero turn rate, however small the instructed turn.

    Parameters
    ----------
    a_lat, a_lon : float
        Aircraft A current position.
    a_heading0_deg : float
        Aircraft A current heading (start of turn, or straight heading).
    a_target_heading_deg : float
        Aircraft A heading after turn (== heading0 if straight).
    a_speed_kt : float
        Aircraft A nominal speed in knots.
    a_turn_rate_deg_sec : float
        Aircraft A rate of turn in deg/s; 0.0 if straight.
    b_lat, b_lon : float
        Aircraft B current position.
    b_heading0_deg : float
        Aircraft B current heading.
    b_target_heading_deg : float
        Aircraft B heading after turn.
    b_speed_kt : float
        Aircraft B nominal speed in knots.
    b_turn_rate_deg_sec : float
        Aircraft B rate of turn in deg/s; 0.0 if straight.
    separation_threshold_m : float
        Required spacing in metres.
    speed_diff_kt : float
        Symmetric speed uncertainty in knots.
    projection_time_s : float
        Total projection horizon in seconds.
    use_interval_local_lipschitz : bool, optional
        If True, use the tighter interval-local Lipschitz bound derived from the
        instructed heading profiles. If False, use the simpler global bound
        `a_max + b_max`. Directional certification is applied after either scalar
        bound fails.
    turn_speed_schedule_uncertainty_kt : float, optional
        Deviation-envelope bound for turning aircraft only. If positive, the decision
        covers instantaneous in-turn speed variation within
        `nominal +/- turn_speed_schedule_uncertainty_kt`, clipped to the admissible
        interval, by inflating the threshold with the proven time-varying envelope.
        The envelope is zero for straight aircraft and is disabled by default.
    eta_const_m : float, optional
        Constant deviation-envelope term in metres, covering a bounded
        relative-position estimation error.
    eta_rate_mps : float, optional
        Linear deviation-envelope rate in metres per second. A differential wind bound
        contributes its magnitude directly, and a heading-estimate bound of
        `eps_i` radians per aircraft contributes `s_i_max * eps_i`.
    turn_delay_max_s : float, optional
        Deviation-envelope bound on turn-initiation delay in seconds.
    turn_rate_error_max : float, optional
        Deviation-envelope bound on the relative turn-rate error, as a fraction
        strictly below one.

    Returns
    -------
    tuple[bool, float, float]
        `(is_separated, min_distance_m, closest_time_s)`.

        For turning cases, `min_distance_m` and `closest_time_s` are informative
        sampled/certified estimates. The primary contract is the boolean separation
        result, which is sound with respect to the modelled envelope. If any
        deviation-envelope parameter is positive, the boolean decision also covers
        the corresponding bounded perturbation classes, while the distance/time pair
        remains a raw diagnostic of the constant-speed geometry.

        If timestamp spacing prevents further refinement, the decision is False,
        the distance is an interval lower bound and the time is unavailable (NaN).

    Notes
    -----
    The algorithm is:

    1. Evaluate the exact reachable relative-position hull at the interval endpoints.
    2. Use the endpoint distances and a Lipschitz constant to test whether the whole
       interval is already certified safe. By default this Lipschitz constant is the
       exact interval-local supremum of the corner relative speeds; the simpler global
       bound `a_max + b_max` remains available as an option.
    3. If needed, test projected-motion bounds along fixed directions from the two
       endpoint hulls and their combined hull. All speed corners must satisfy the
       threshold along the same direction throughout the interval.
    4. If neither bound certifies the interval, split it at its midpoint, evaluate that fixed-time hull
       exactly, and continue with the two child intervals using an explicit DFS stack.
    5. Treat any interval at or below `TURN_TIME_CERT_MIN_INTERVAL_S` that still
       cannot be certified as potentially unsafe.

    So the boolean return value is a certificate about the whole horizon, while the
    distance/time pair is a compact summary of the most critical sampled or certified
    point encountered during the search. All deviation envelopes are applied by
    inflating the threshold per evaluated time, not by changing the reachable-set
    geometry.
    """
    rel_pos0 = -latlon_to_local_xy(b_lat, b_lon, ref_lat=a_lat, ref_lon=a_lon)
    return _catch_up_projection_interval_with_turns_local(
        rel_pos0=rel_pos0,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_speed_kt=a_speed_kt,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_speed_kt=b_speed_kt,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        separation_threshold_m=separation_threshold_m,
        speed_diff_kt=speed_diff_kt,
        projection_time_s=projection_time_s,
        use_interval_local_lipschitz=use_interval_local_lipschitz,
        turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
        min_cert_interval_s=TURN_TIME_CERT_MIN_INTERVAL_S,
        eta_const_m=eta_const_m,
        eta_rate_mps=eta_rate_mps,
        turn_delay_max_s=turn_delay_max_s,
        turn_rate_error_max=turn_rate_error_max,
    )
