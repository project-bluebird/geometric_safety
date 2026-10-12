"""Regressions for thin reachable polygons and the heading caller."""

from fractions import Fraction
from itertools import combinations
import math

import mpmath as mp
import numba
import numpy as np
from numpy.typing import NDArray
import pytest

from geometric_safety.heading import HeadingMotion, certify_heading_separation
from geometric_safety.util import (
    _closest_point_on_segment,
    _point_in_convex_polygon,
    closest_point_on_convex_polygon,
    small_convex_hull,
)

type FloatArray = NDArray[np.float64]
type ExactPoint = tuple[Fraction, Fraction]

CAPTURED_CORNERS = np.array(
    [
        [605.6336307024169, 9240.183607778697],
        [605.6336307024175, 9240.183607778712],
        [605.6336307024171, 9240.183607778697],
        [605.6336307024166, 9240.183607778683],
    ]
)


@numba.njit(cache=True, fastmath=True)
def _compiled_distance(point: FloatArray, corners: FloatArray) -> float:
    return closest_point_on_convex_polygon(point, small_convex_hull(corners))[1]


@numba.njit(cache=True, fastmath=True)
def _compiled_contains(point: FloatArray, hull: FloatArray) -> bool:
    return _point_in_convex_polygon(point, hull)


def _cross(a: ExactPoint, b: ExactPoint, c: ExactPoint) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _exact_distance(point: FloatArray, corners: FloatArray) -> float:
    """Exact rational convex-combination and segment reference for binary64 inputs.

    Any point in a planar convex hull is in a triangle of its input points.
    Otherwise its closest point lies on a segment between two input points.
    This reference neither constructs the production hull nor uses its predicate.
    """
    query = (Fraction(float(point[0])), Fraction(float(point[1])))
    vertices = [(Fraction(float(x)), Fraction(float(y))) for x, y in corners]
    for a, b, c in combinations(vertices, 3):
        if _cross(a, b, c) == 0:
            continue
        sides = (_cross(a, b, query), _cross(b, c, query), _cross(c, a, query))
        if all(value >= 0 for value in sides) or all(value <= 0 for value in sides):
            return 0.0
    distances = [(x - query[0]) ** 2 + (y - query[1]) ** 2 for x, y in vertices]
    for a, b in combinations(vertices, 2):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx**2 + dy**2
        if length_squared == 0:
            continue
        ratio = ((query[0] - a[0]) * dx + (query[1] - a[1]) * dy) / length_squared
        ratio = min(Fraction(1), max(Fraction(0), ratio))
        distances.append((a[0] + ratio * dx - query[0]) ** 2 + (a[1] + ratio * dy - query[1]) ** 2)
    return math.sqrt(float(min(distances)))


@pytest.mark.parametrize("scale", [2.0**-30, 1.0, 2.0**30])
def test_segment_projection_matches_rational_reference(scale: float) -> None:
    rng = np.random.default_rng(2026101205)
    for i in range(250):
        a = rng.uniform(-10, 10, 2) * scale
        length = scale * (2.0 ** int(rng.integers(-45, 4)))
        b = a.copy() if i % 20 == 0 else a + rng.uniform(-1, 1, 2) * length
        point = a + rng.uniform(-2, 2, 2) * length
        projection, distance = _closest_point_on_segment(point, a, b)
        expected = _exact_distance(point, np.array([a, b]))
        assert distance == pytest.approx(expected, abs=8 * math.ulp(16.0 * scale), rel=1e-12)
        assert float(np.linalg.norm(point - projection)) == pytest.approx(distance, rel=3e-15, abs=0.0)
        assert min(a[0], b[0]) <= projection[0] <= max(a[0], b[0])
        assert min(a[1], b[1]) <= projection[1] <= max(a[1], b[1])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [2.0**-30, 1.0, 2.0**30])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (10.0, -20.0)])
def test_thin_polygon_matches_exact_reference(reverse: bool, scale: float, offset: tuple[float, float]) -> None:
    width = 2.0**-20
    corners = (np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0 + width]]) + offset) * scale
    if reverse:
        corners = corners[::-1].copy()
    # Inside, edge, vertex, outside but inside the bounding box, and far outside.
    for xy in ((0.5, 0.5 + width / 4), (0.5, 0.5), (1.0, 1.0 + width), (0.5, 0.5 + width), (-3.0, 4.0)):
        point = (np.array(xy) + offset) * scale
        expected = _exact_distance(point, corners)
        for contains in (_point_in_convex_polygon, _compiled_contains):
            assert bool(contains(point, corners)) == (expected == 0.0)
        for distance in (
            closest_point_on_convex_polygon(point, corners)[1],
            _compiled_distance(point, corners),
        ):
            if expected == 0.0:
                assert distance == 0.0
            else:
                assert distance > 0.0
                assert distance == pytest.approx(expected, abs=8 * math.ulp(32.0 * scale), rel=1e-12)


@pytest.mark.parametrize(
    "corners",
    [
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [1.0, 2.0]],
    ],
)
def test_collapsed_polygons_match_exact_reference(corners: list[list[float]]) -> None:
    vertices = np.array(corners)
    for xy in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (2.0, 2.0), (4.0, 5.0)):
        point = np.array(xy)
        expected = _exact_distance(point, vertices)
        assert closest_point_on_convex_polygon(point, vertices)[1] == pytest.approx(expected, abs=1e-14)
        assert _compiled_distance(point, vertices) == pytest.approx(expected, abs=1e-14)


def test_seeded_thin_polygons_and_boundary_points_match_exact_reference() -> None:
    rng = np.random.default_rng(2026101204)
    for _ in range(400):
        centre = rng.uniform(-10000, 10000, size=2)
        direction = rng.uniform(-1.0, 1.0, size=2)
        width = 2.0 ** int(rng.integers(-40, -4))
        normal = np.array([-direction[1], direction[0]]) * width
        corners = np.array(
            [centre - direction, centre + direction, centre + direction + normal, centre - direction + normal]
        )
        point = centre + float(rng.choice([0.0, 0.5, 1.0, 2.0])) * normal
        expected = _exact_distance(point, corners)
        hull = small_convex_hull(corners)
        distance = _compiled_distance(point, corners)
        assert distance == pytest.approx(expected, abs=8 * math.ulp(20000.0), rel=1e-12)
        if expected == 0.0:
            assert distance == 0.0
            if len(hull) >= 3:
                assert _point_in_convex_polygon(point, hull)
                assert _compiled_contains(point, hull)


@pytest.mark.parametrize("height", [1e-200, 1e200])
def test_segment_distance_retains_norm_scaling(height: float) -> None:
    point = np.array([0.0, height])
    projected, distance = _closest_point_on_segment(point, np.zeros(2), np.array([1.0, 0.0]))
    assert np.array_equal(projected, np.zeros(2))
    assert distance == pytest.approx(height, rel=2e-15, abs=0.0)


@pytest.mark.parametrize("radius", [9259.99, 9260.0])
def test_inside_or_at_clearance_is_not_certified(radius: float) -> None:
    rate = 0.03
    alpha = math.degrees(math.asin(radius * math.radians(rate) / 600.0))
    a = HeadingMotion(
        heading0_deg=alpha,
        target_heading_deg=alpha + 60.0,
        turn_rate_deg_s=rate,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    b = HeadingMotion(
        heading0_deg=-alpha,
        target_heading_deg=-alpha + 60.0,
        turn_rate_deg_s=rate,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    result = certify_heading_separation((0.0, radius), a, b, separation_threshold_m=9260.0, horizon_s=4000.0)
    assert result.status == "unresolved"


@pytest.mark.parametrize("compiled", [False, True])
def test_captured_tiny_reachable_polygon_is_far_from_origin(compiled: bool) -> None:
    # A supporting projection independently bounds the entire convex hull.
    with mp.workdps(80):
        first_x, first_y = map(mp.mpf, CAPTURED_CORNERS[0])
        norm = mp.sqrt(first_x**2 + first_y**2)
        lower = min((mp.mpf(x) * first_x + mp.mpf(y) * first_y) / norm for x, y in CAPTURED_CORNERS)
        upper = min(mp.sqrt(mp.mpf(x) ** 2 + mp.mpf(y) ** 2) for x, y in CAPTURED_CORNERS)
    assert lower > 9260.0
    origin = np.zeros(2)
    distance = (
        _compiled_distance(origin, CAPTURED_CORNERS)
        if compiled
        else closest_point_on_convex_polygon(origin, small_convex_hull(CAPTURED_CORNERS))[1]
    )
    assert float(lower) - 1e-9 <= distance <= float(upper) + 1e-9


@pytest.mark.parametrize("horizon_s", [1000.0, 4000.0])
def test_singleton_speed_circular_clearance_is_certified(horizon_s: float) -> None:
    radius = 9260.01
    rate = 0.03
    alpha = math.degrees(math.asin(radius * math.radians(rate) / 600.0))
    # The relative trajectory has radius 9260.01 m throughout the turn;
    # the common straight continuation is tangent and increases separation.
    a = HeadingMotion(
        heading0_deg=alpha,
        target_heading_deg=alpha + 60.0,
        turn_rate_deg_s=rate,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    b = HeadingMotion(
        heading0_deg=-alpha,
        target_heading_deg=-alpha + 60.0,
        turn_rate_deg_s=rate,
        speed_min_mps=300.0,
        speed_max_mps=300.0,
    )
    result = certify_heading_separation(
        (0.0, radius),
        a,
        b,
        separation_threshold_m=9260.0,
        horizon_s=horizon_s,
    )
    assert result.status == "certified_safe"
    assert result.base_min_distance_m == pytest.approx(radius, abs=1e-7)
