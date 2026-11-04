"""Independent checks of orientation signs and conservative exponent-range fallback."""

from fractions import Fraction
import math

import numba
import numpy as np
from numpy.typing import NDArray
import pytest

from geometric_safety._numeric import _orientation_sign
from geometric_safety.util import convex_hull


@numba.njit(fastmath=True)
def _fast_orientation(points: NDArray[np.float64]) -> float:
    return _orientation_sign(points[0], points[1], points[2])


def _exact_sign(points: NDArray[np.float64]) -> int:
    origin, a, b = [(Fraction(float(x)), Fraction(float(y))) for x, y in points]
    determinant = (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    return (determinant > 0) - (determinant < 0)


@pytest.mark.parametrize("exponent", [-900, -450, 0, 450, 900])
def test_orientation_matches_exact_sign_across_scales(exponent: int) -> None:
    rng = np.random.default_rng(202610202)
    for i in range(300):
        origin = rng.uniform(-1.0, 1.0, size=2)
        direction = rng.uniform(-1.0, 1.0, size=2)
        points = np.array([origin, origin + direction, origin + 0.25 * direction])
        if i % 3:
            points[2, i % 2] = math.nextafter(points[2, i % 2], math.inf)
        points = np.ldexp(points, exponent)
        expected = _exact_sign(points)
        assert _orientation_sign(points[0], points[1], points[2]) == expected
        assert _fast_orientation(points) == expected


def test_extreme_mixed_scales_keep_a_conservative_enclosure() -> None:
    points = np.array([[0.0, 0.0], [1e300, 1e300], [1e-300, 2e-300]])
    assert _orientation_sign(points[0], points[1], points[2]) == _exact_sign(points) == 1
    points[2, 1] = 1e-300
    assert math.isnan(_orientation_sign(points[0], points[1], points[2]))
    hull = convex_hull(points)
    assert np.array_equal(hull, np.array([[0.0, 0.0], [1e300, 0.0], [1e300, 1e300], [0.0, 1e300]]))
