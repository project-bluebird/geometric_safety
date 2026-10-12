"""Shortest-turn semantics shared by displacement and certificate helpers."""

import math

import numba
import pytest

from geometric_safety.util import shortest_turn_parameters


@numba.njit(cache=True, fastmath=True)
def _fastmath_parameters(initial: float, target: float, rate: float) -> tuple[float, float, float]:
    return shortest_turn_parameters(initial, target, rate)


@pytest.mark.parametrize(
    ("initial", "target", "rate", "expected"),
    [
        (0.0, 360.0, 0.0, (0.0, 0.0, 0.0)),
        (30.0, 30.0, 3.0, (0.0, 0.0, 0.0)),
        (0.0, 90.0, -3.0, (90.0, 3.0, 30.0)),
        (90.0, 0.0, 3.0, (-90.0, -3.0, 30.0)),
        (350.0, 10.0, 2.0, (20.0, 2.0, 10.0)),
        (0.0, 180.0, 3.0, (-180.0, -3.0, 60.0)),
    ],
)
def test_shortest_turn_parameters_match_declared_semantics(
    initial: float, target: float, rate: float, expected: tuple[float, float, float]
) -> None:
    assert shortest_turn_parameters(initial, target, rate) == expected
    assert _fastmath_parameters(initial, target, rate) == expected


def test_neighbour_of_half_turn_preserves_positive_direction() -> None:
    target = math.nextafter(180.0, 0.0)
    angle, rate, duration = _fastmath_parameters(0.0, target, 3.0)
    assert 0.0 < angle < 180.0
    assert rate == 3.0
    assert duration < 60.0


@pytest.mark.parametrize("rate", [0.0, math.inf, math.nan])
def test_invalid_rate_cannot_define_a_changed_heading(rate: float) -> None:
    with pytest.raises(ValueError, match=r"finite|nonzero turn rate"):
        shortest_turn_parameters(0.0, 1.0, rate)
