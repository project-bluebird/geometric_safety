"""Attainable straight-flight times for thin reachable hulls."""

from fractions import Fraction
import math

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
import pytest

from geometric_safety import relevant_aircraft as ra
from geometric_safety.heading import HeadingMotion, certify_heading_separation


@pytest.mark.parametrize(
    ("a", "b", "c", "d"),
    [
        (134217729.0, 134217727.0, 134217728.0, 134217728.0),
        (134217728.0, 134217728.0, 134217729.0, 134217727.0),
        (1.0, 1.0, 1.0, math.nextafter(1.0, 0.0)),
    ],
)
def test_compensated_determinant_retains_exact_binary64_cancellation(a: float, b: float, c: float, d: float) -> None:
    expected = Fraction(a) * Fraction(b) - Fraction(c) * Fraction(d)
    assert ra._difference_of_products(a, b, c, d) == float(expected)


def test_distinct_nearby_rays_do_not_supply_a_false_closest_time() -> None:
    # The minimum y coordinate is attained only at the horizon. Longer vertices
    # lie within 1e-9 radians of this point but cannot attain its displacement early.
    point = np.array([1_000_000.0, -0.001])
    hull = np.array([[0.0, 0.0], [1_000_000.0, -0.001], [7_000_000.0, -0.001], [7_000_000.0, 0.0]])
    assert ra._closest_time_from_projection(point, hull, 1000.0) == pytest.approx(1000.0, abs=1e-10)
    separated, distance, time_s = ra._straight_hull_projection_interval(-point, hull, 1.0, 1000.0)
    assert not separated
    assert distance == 0.0
    assert time_s == pytest.approx(1000.0, abs=1e-10)


@pytest.mark.parametrize("extra_vertex", [False, True])
@pytest.mark.parametrize("exponent", [0, -600])
def test_one_ulp_transverse_difference_preserves_the_attainable_time(extra_vertex: bool, exponent: int) -> None:
    scale = math.ldexp(1.0, exponent)
    point = np.array([1.0, 1.0]) * scale
    vertices = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0 + 2.0**-51]]
    if extra_vertex:
        # A negative coefficient on this vertex must not be clamped to zero.
        vertices.append([2.0, 3.0])
    hull = np.array(vertices) * scale
    assert ra._closest_time_from_projection(point, hull, 1000.0) == 1000.0


def test_mixed_scale_determinant_underflow_does_not_manufacture_a_time() -> None:
    epsilon = 2.0**-52
    scale = 2.0**-508
    point = np.array([scale, scale * (1.0 - epsilon)])
    longer = np.array([2.0 * scale * (1.0 + epsilon), 2.0 * scale])
    hull = np.array([[0.0, 0.0], point, longer, [1.0, 2.0]])
    # Both rounded determinant products are normal, but their exact difference
    # is too small for binary64. The longer point lies strictly above this ray.
    exact_cross = Fraction(float(longer[0])) * Fraction(float(point[1])) - Fraction(float(longer[1])) * Fraction(
        float(point[0])
    )
    assert exact_cross < 0
    time_s = ra._closest_time_from_projection(point, hull, 1000.0)
    assert math.isnan(time_s) or time_s == 1000.0


def test_independent_speed_hull_reports_an_attainable_closest_time() -> None:
    b_heading_deg = 90.00000000572958
    a = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=100.0,
        speed_max_mps=500.0,
    )
    b = HeadingMotion(
        heading0_deg=b_heading_deg,
        target_heading_deg=b_heading_deg,
        turn_rate_deg_s=0.0,
        speed_min_mps=100.0,
        speed_max_mps=300.0,
    )
    position = (-200_000.0, -3.000001472867912e-5)
    result = certify_heading_separation(position, a, b, separation_threshold_m=1.0, horizon_s=1000.0)
    assert result.status == "unresolved"
    assert result.diagnostic_kind == "straight_minimum"
    assert result.base_min_distance_m == pytest.approx(0.0, abs=1e-10)
    assert result.base_closest_time_s is not None
    assert result.base_closest_time_s == pytest.approx(1000.0, abs=1e-7)

    # Check transverse reachability independently of the production hull and its
    # time-recovery calculation. At the erroneous 500 s time, the maximum progress
    # is only about 1.5e-5 m. Binary64 trigonometric/position rounding is far smaller.
    with mp.workdps(80):
        time_s = mp.mpf(result.base_closest_time_s)
        north_a = mp.cos(mp.pi / 2)
        north_b = mp.cos(mp.mpf(b_heading_deg) * mp.pi / 180)
        corners = [time_s * (sa * north_a - sb * north_b) for sa in (100, 500) for sb in (100, 300)]
        required_north = -mp.mpf(position[1])
        rounding_m = 64 * math.ulp(500_000.0)
        assert min(corners) - rounding_m <= required_north <= max(corners) + rounding_m


@pytest.mark.parametrize("nearby", [False, True])
def test_unrepresented_point_has_no_manufactured_time(nearby: bool) -> None:
    vertices = [[0.0, 0.0], [2.0, 2.0 + 2.0**-51], [2.0, 3.0]] if nearby else [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    hull = np.array(vertices)
    assert math.isnan(ra._closest_time_from_projection(np.array([1.0, 1.0]), hull, 1000.0))


@pytest.mark.parametrize("horizon_s", [-1.0, math.inf])
def test_cartesian_projection_refuses_an_unsupported_horizon(horizon_s: float) -> None:
    separated, distance, time_s = ra._straight_hull_projection_interval(
        np.array([10.0, 0.0]), np.zeros((1, 2)), 1.0, horizon_s
    )
    assert not separated
    assert math.isnan(distance)
    assert math.isnan(time_s)


def _exact_earliest_time(point: NDArray[np.float64], hull: NDArray[np.float64], horizon: float) -> float:
    """Enumerate feasible basic solutions of the displacement linear programme."""
    x, y = Fraction(float(point[0])), Fraction(float(point[1]))
    candidates: list[Fraction] = []
    vertices = [(Fraction(float(vertex[0])), Fraction(float(vertex[1]))) for vertex in hull]
    for ax, ay in vertices:
        if ax * y == ay * x and (ax or ay):
            alpha = x / ax if ax else y / ay
            if 0 <= alpha <= 1:
                candidates.append(alpha)
        for bx, by in vertices:
            determinant = ax * by - ay * bx
            if not determinant:
                continue
            u = (x * by - y * bx) / determinant
            v = (ax * y - ay * x) / determinant
            if u >= 0 and v >= 0 and u + v <= 1:
                candidates.append(u + v)
    return float(min(candidates) * Fraction(horizon)) if candidates else math.nan


@pytest.mark.parametrize("reciprocal", [False, True])
def test_parallel_conflicts_keep_attainable_diagnostics(reciprocal: bool) -> None:
    rng = np.random.default_rng(20261021)
    exact_interiors = 0
    for _ in range(80):
        heading = float(rng.uniform(0.0, 360.0))
        hull = ra.compute_relative_velocity_hull(heading, 282.54, heading + 180.0 * reciprocal, 327.29, 27.49, 738.9)
        # Select an interior point, then independently solve for its attainable time
        # using exact rational arithmetic on the represented hull coordinates.
        point = hull.mean(axis=0) * 0.7
        expected = _exact_earliest_time(point, hull, 738.9)
        safe, distance, time_s = ra._straight_hull_projection_interval(-point, hull, 9260.0, 738.9)
        assert not safe
        assert distance == pytest.approx(0.0, abs=1e-8)
        assert 0.0 <= time_s <= 738.9
        if math.isfinite(expected):
            exact_interiors += 1
            assert time_s == pytest.approx(expected, rel=2e-13, abs=1e-10)
    assert exact_interiors >= 30
