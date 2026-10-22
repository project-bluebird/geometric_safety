"""Strict binary64 arithmetic for geometric decisions."""

import math

import numba
import numpy as np
from numpy.typing import NDArray


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
    difference, residual = _two_diff(left, right)
    residual += _product_roundoff(a, b, left) - _product_roundoff(c, d, right)
    return difference + residual


@numba.njit(cache=True, fastmath=False)
def _two_diff(a: float, b: float) -> tuple[float, float]:
    """Return the rounded difference and its exact subtraction residual."""
    difference = a - b
    b_virtual = a - difference
    residual = (a - (difference + b_virtual)) + (b_virtual - b)
    return difference, residual


@numba.njit(cache=True, fastmath=False)
def _grow_expansion(expansion: NDArray[np.float64], length: int, value: float) -> int:
    """Add one value exactly to a nonoverlapping expansion, eliminating zeros."""
    carry = value
    written = 0
    for index in range(length):
        term = expansion[index]
        total = carry + term
        virtual = total - carry
        residual = (carry - (total - virtual)) + (term - virtual)
        if residual != 0.0:
            expansion[written] = residual
            written += 1
        carry = total
    if carry != 0.0 or written == 0:
        expansion[written] = carry
        written += 1
    return written


@numba.njit(cache=True, fastmath=False)
def _orientation_sign(origin: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the determinant sign, or NaN when exponent range prevents verification.

    Shewchuk's first-stage orient2d filter handles well-separated determinants.
    Ambiguous cases sum the exact difference products with expansion arithmetic.
    Power-of-two scaling must preserve the inputs, and product residuals must
    remain normal. The hull caller supplies an outer rectangle if either fails.
    https://www.cs.cmu.edu/~quake/robust.html
    """
    ax, ay = a[0] - origin[0], a[1] - origin[1]
    bx, by = b[0] - origin[0], b[1] - origin[1]
    left, right = ax * by, ay * bx
    determinant = left - right
    minimum_product = 2.0 ** (-1022 + 106)
    unit_roundoff = 2.0**-53
    error = (3.0 + 16.0 * unit_roundoff) * unit_roundoff * (abs(left) + abs(right))
    products_in_range = (ax == 0.0 or by == 0.0 or abs(left) >= minimum_product) and (
        ay == 0.0 or bx == 0.0 or abs(right) >= minimum_product
    )
    if products_in_range and math.isfinite(determinant) and abs(determinant) > error:
        return 1.0 if determinant > 0.0 else -1.0

    coordinates = np.array([origin[0], origin[1], a[0], a[1], b[0], b[1]])
    largest = 0.0
    for value in coordinates:
        if not math.isfinite(value):
            return math.nan
        largest = max(largest, abs(value))
    _, exponent = math.frexp(largest)
    for index in range(6):
        scaled = math.ldexp(coordinates[index], -exponent)
        if math.ldexp(scaled, exponent) != coordinates[index]:
            return math.nan
        coordinates[index] = scaled
    ax, ax_tail = _two_diff(coordinates[2], coordinates[0])
    ay, ay_tail = _two_diff(coordinates[3], coordinates[1])
    bx, bx_tail = _two_diff(coordinates[4], coordinates[0])
    by, by_tail = _two_diff(coordinates[5], coordinates[1])
    expansion = np.empty(32, dtype=np.float64)
    expansion[0] = 0.0
    length = 1
    for x, y, sign in (
        (ax, by, 1.0),
        (ax, by_tail, 1.0),
        (ax_tail, by, 1.0),
        (ax_tail, by_tail, 1.0),
        (ay, bx, -1.0),
        (ay, bx_tail, -1.0),
        (ay_tail, bx, -1.0),
        (ay_tail, bx_tail, -1.0),
    ):
        product = x * y
        if x != 0.0 and y != 0.0 and abs(product) < minimum_product:
            return math.nan
        residual = _product_roundoff(x, y, product)
        length = _grow_expansion(expansion, length, sign * residual)
        length = _grow_expansion(expansion, length, sign * product)
    result = expansion[length - 1]
    return 1.0 if result > 0.0 else (-1.0 if result < 0.0 else 0.0)
