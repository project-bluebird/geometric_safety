"""Analytical and attainable-profile controls for the independent study reference."""

from collections.abc import Iterator
from dataclasses import replace

import mpmath as mp
import pytest

from paper.heading_study_reference import (
    Budget,
    Model,
    Motion,
    Path,
    Query,
    ReachableSet,
    SpeedSegment,
    SupportPoint,
    Vector,
    Witness,
    aircraft_support,
    classify,
    closest_simplex,
    fixed_time_distance,
    integrate_witness,
    zero,
)


@pytest.fixture(autouse=True)
def precision() -> Iterator[None]:
    with mp.workdps(80):
        yield


def stationary() -> Motion:
    return Motion(0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("target", [-90.0, 90.0, 180.0])
def test_turn_antiderivative_matches_quadrature_across_completion(target: float) -> None:
    path = Path.from_motion(Motion(0.0, target, 90.0, 1.0, 1.0))
    for end in (path.duration / 2, path.duration, path.duration + 2):
        actual = path.integral(mp.mpf(0), end)
        breaks = sorted({mp.mpf(0), min(end, path.duration), end})
        east = mp.quad(lambda t: mp.sin(path.heading(t)), breaks)
        north = mp.quad(lambda t: mp.cos(path.heading(t)), breaks)
        assert isinstance(east, mp.mpf)
        assert isinstance(north, mp.mpf)
        expected = Vector(east, north)
        assert (actual - expected).norm() < mp.mpf("1e-65")


def test_varying_speed_support_exceeds_constant_speed_support() -> None:
    path = Path.from_motion(Motion(0.0, 90.0, 90.0, 0.0, 2.0))
    normal = Vector(mp.mpf(1), mp.mpf(-1))
    variable, segments = aircraft_support(path, normal, mp.mpf(1), "V")
    constant, _ = aircraft_support(path, normal, mp.mpf(1), "C")
    assert abs(constant.dot(normal)) < mp.mpf("1e-65")
    assert abs(variable.dot(normal) - 4 * (mp.sqrt(2) - 1) / mp.pi) < mp.mpf("1e-65")
    assert len(segments) == 2
    assert abs(segments[0].right - mp.mpf("0.5")) < mp.mpf("1e-65")
    assert tuple(segment.speed for segment in segments) == (0, 2)


def test_varying_speed_support_includes_straight_continuation() -> None:
    path = Path.from_motion(Motion(0.0, 90.0, 90.0, 0.0, 2.0))
    normal = Vector(mp.mpf(1), mp.mpf(-1))
    point, segments = aircraft_support(path, normal, mp.mpf(3), "V")
    expected = 4 * (mp.sqrt(2) - 1) / mp.pi + 4
    assert abs(point.dot(normal) - expected) < mp.mpf("1e-65")
    assert segments[-1] == SpeedSegment(mp.mpf(1), mp.mpf(3), mp.mpf(2))


def test_two_moving_aircraft_have_correct_relative_support() -> None:
    reach = ReachableSet(
        Vector(mp.mpf(10), mp.mpf(20)),
        Path.from_motion(Motion(90.0, 90.0, 0.0, 1.0, 3.0)),
        Path.from_motion(Motion(0.0, 0.0, 0.0, 2.0, 4.0)),
        "V",
        Budget(),
    )
    # At 5 seconds the relative rectangle is [15, 25] x [0, 10].
    bracket = fixed_time_distance(reach, mp.mpf(5), mp.mpf("1e-50"))
    assert abs(bracket.lower - 15) < mp.mpf("1e-48")
    assert abs(bracket.upper - 15) < mp.mpf("1e-48")


def test_time_bound_accounts_for_both_aircraft() -> None:
    # Initial, final and midpoint samples all exceed the threshold. The pair
    # meets at 2.5 seconds, so midpoint sampling needs both aircraft's speed.
    query = Query(
        (-50.0, 0.0),
        Motion(90.0, 90.0, 0.0, 10.0, 10.0),
        Motion(270.0, 270.0, 0.0, 10.0, 10.0),
        20.0,
        30.0,
    )
    result = classify(query, "V", margin_m=1.0)
    assert result.classification == "conflict"
    assert result.upper_m < 29


@pytest.mark.parametrize("model", ["C", "V"])
def test_straight_distance_matches_exact_segment_projection(model: Model) -> None:
    reach = ReachableSet(
        Vector(mp.mpf(-100), mp.mpf(40)),
        Path.from_motion(Motion(90.0, 90.0, 0.0, 5.0, 10.0)),
        Path.from_motion(stationary()),
        model,
        Budget(),
    )
    for time, expected in (
        (0, mp.sqrt(11600)),
        (5, mp.sqrt(4100)),
        (10, mp.mpf(40)),
        (15, mp.mpf(40)),
        (30, mp.sqrt(4100)),
    ):
        bracket = fixed_time_distance(reach, mp.mpf(time), mp.mpf("1e-50"))
        assert abs(bracket.lower - expected) < mp.mpf("1e-48")
        assert abs(bracket.upper - expected) < mp.mpf("1e-48")
        assert abs(integrate_witness(reach, bracket.witness).norm() - expected) < mp.mpf("1e-48")


@pytest.mark.parametrize("model", ["C", "V"])
@pytest.mark.parametrize(
    ("threshold", "classification"), [(30.0, "separated"), (50.0, "conflict"), (40.0, "unresolved")]
)
def test_continuous_time_straight_control(model: Model, threshold: float, classification: str) -> None:
    query = Query((-100.0, 40.0), Motion(90.0, 90.0, 0.0, 5.0, 10.0), stationary(), 30.0, threshold)
    outcome = classify(query, model, margin_m=1.0)
    assert outcome.classification == classification
    if classification == "separated":
        assert 31 < outcome.lower_m <= 40
    elif classification == "conflict":
        assert outcome.witness is not None
        assert outcome.upper_m < 49


def test_variable_profile_conflict_has_no_constant_speed_counterpart() -> None:
    # During a north-to-east quarter-turn, every constant-speed point has x <= y.
    # Flying only the second half reaches a point strictly beyond that half-plane.
    x = 2 * mp.sqrt(2) / mp.pi
    y = 2 * (2 - mp.sqrt(2)) / mp.pi
    assert (x - y) / mp.sqrt(2) > mp.mpf("0.37")
    query = Query((-float(x), -float(y)), Motion(0.0, 90.0, 90.0, 0.0, 2.0), stationary(), 1.0, 0.1)
    for digits in (80, 120):
        constant = classify(query, "C", digits=digits, margin_m=0.005)
        variable = classify(query, "V", digits=digits, margin_m=0.005)
        assert constant.classification == "separated"
        assert variable.classification == "conflict"
        assert variable.witness is not None
        assert variable.upper_m < mp.mpf("0.095")


@pytest.mark.parametrize("target", [45.0, -70.0])
def test_support_points_are_attainable_after_turn_completion(target: float) -> None:
    reach = ReachableSet(
        Vector(mp.mpf(10), mp.mpf(20)),
        Path.from_motion(Motion(0.0, target, 3.0, 80.0, 120.0)),
        Path.from_motion(Motion(270.0, 230.0, 2.0, 60.0, 90.0)),
        "V",
        Budget(),
    )
    for direction in (Vector(mp.mpf(1), mp.mpf(0)), Vector(mp.mpf(-2), mp.mpf(3)), Vector(mp.mpf(0), mp.mpf(-1))):
        sample = reach.support(direction, mp.mpf(60))
        integrated = integrate_witness(reach, Witness(mp.mpf(60), (sample,), (mp.mpf(1),)))
        assert (integrated - sample.point).norm() < mp.mpf("1e-55")
        opposite = reach.support(direction.scale(mp.mpf(-1)), mp.mpf(60))
        assert direction.dot(sample.point) >= direction.dot(opposite.point)


def test_zero_width_turning_sets_are_identical() -> None:
    a = Path.from_motion(Motion(20.0, 100.0, 2.0, 110.0, 110.0))
    b = Path.from_motion(Motion(300.0, 220.0, 3.0, 90.0, 90.0))
    position = Vector(mp.mpf(-25000), mp.mpf(12000))
    for time in (mp.mpf(5), mp.mpf(40), mp.mpf(200)):
        constant = fixed_time_distance(ReachableSet(position, a, b, "C", Budget()), time, mp.mpf("1e-45"))
        variable = fixed_time_distance(ReachableSet(position, a, b, "V", Budget()), time, mp.mpf("1e-45"))
        assert abs(constant.upper - variable.upper) < mp.mpf("1e-40")


def test_mirrored_turns_have_identical_distance_brackets() -> None:
    a = Motion(30.0, 120.0, 2.0, 80.0, 120.0)
    b = Motion(260.0, 220.0, 3.0, 60.0, 100.0)
    mirrored_a = replace(a, heading_deg=-a.heading_deg, target_deg=-a.target_deg)
    mirrored_b = replace(b, heading_deg=-b.heading_deg, target_deg=-b.target_deg)
    original = ReachableSet(
        Vector(mp.mpf(15000), mp.mpf(-8000)), Path.from_motion(a), Path.from_motion(b), "V", Budget()
    )
    mirrored = ReachableSet(
        Vector(mp.mpf(-15000), mp.mpf(-8000)), Path.from_motion(mirrored_a), Path.from_motion(mirrored_b), "V", Budget()
    )
    first = fixed_time_distance(original, mp.mpf(100), mp.mpf("0.01"))
    second = fixed_time_distance(mirrored, mp.mpf(100), mp.mpf("0.01"))
    assert abs(first.lower - second.lower) < mp.mpf("1e-50")
    assert abs(first.upper - second.upper) < mp.mpf("1e-50")


def test_origin_inside_triangle_returns_attainable_convex_weights() -> None:
    points = tuple(SupportPoint(Vector(mp.mpf(x), mp.mpf(y)), (), ()) for x, y in ((-2, -1), (3, -1), (0, 4)))
    point, vertices, weights = closest_simplex(points)
    assert point == zero()
    reconstructed = zero()
    for vertex, weight in zip(vertices, weights, strict=True):
        assert weight >= 0
        reconstructed = reconstructed + vertex.point.scale(weight)
    assert abs(sum(weights) - 1) < mp.mpf("1e-65")
    assert reconstructed.norm() < mp.mpf("1e-65")


def test_invalid_witness_is_rejected() -> None:
    path = Path.from_motion(Motion(0.0, 0.0, 0.0, 1.0, 2.0))
    reach = ReachableSet(zero(), path, Path.from_motion(stationary()), "V", Budget())
    sample = SupportPoint(
        zero(), (SpeedSegment(mp.mpf(0), mp.mpf(1), mp.mpf(3)),), (SpeedSegment(mp.mpf(0), mp.mpf(1), mp.mpf(0)),)
    )
    with pytest.raises(ValueError, match="inadmissible"):
        integrate_witness(reach, Witness(mp.mpf(1), (sample,), (mp.mpf(1),)))


@pytest.mark.parametrize(("support", "cells", "reason"), [(0, 100, "support_budget"), (100, 0, "time_budget")])
def test_exhaustion_remains_unresolved(support: int, cells: int, reason: str) -> None:
    query = Query((1000.0, 0.0), stationary(), stationary(), 10.0, 100.0)
    result = classify(query, "V", max_support=support, max_cells=cells)
    assert result.classification == "unresolved"
    assert result.reason == reason
