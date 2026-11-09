"""Independent high-precision reference for prescribed headings and bounded speeds.

Support points maximise a linear functional over the measurable speed profiles.
Closest points in their convex hull give attainable upper bounds; supporting
lines give lower bounds. A speed-based Lipschitz bound covers intervals in time.
The arithmetic is high precision, not validated outward interval arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations, pairwise
from typing import Literal

import mpmath as mp

type Model = Literal["C", "V"]
type Classification = Literal["separated", "conflict", "unresolved"]


@dataclass(frozen=True)
class Vector:
    x: mp.mpf
    y: mp.mpf

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    def scale(self, factor: mp.mpf) -> Vector:
        return Vector(self.x * factor, self.y * factor)

    def dot(self, other: Vector) -> mp.mpf:
        return self.x * other.x + self.y * other.y

    def norm(self) -> mp.mpf:
        result = mp.sqrt(self.dot(self))
        if not isinstance(result, mp.mpf):
            raise ArithmeticError("A real squared norm must have a real square root")
        return result


def zero() -> Vector:
    return Vector(mp.mpf(0), mp.mpf(0))


@dataclass(frozen=True)
class Motion:
    heading_deg: float
    target_deg: float
    rate_deg_s: float
    lower_mps: float
    upper_mps: float


@dataclass(frozen=True)
class Query:
    relative_position_m: tuple[float, float]
    a: Motion
    b: Motion
    horizon_s: float
    threshold_m: float


@dataclass(frozen=True)
class Path:
    initial: mp.mpf
    omega: mp.mpf
    duration: mp.mpf
    lower: mp.mpf
    upper: mp.mpf

    @classmethod
    def from_motion(cls, motion: Motion) -> Path:
        heading = mp.mpf(motion.heading_deg)
        angle = (mp.mpf(motion.target_deg) - heading + 180) % 360 - 180
        rate = mp.mpf(motion.rate_deg_s)
        if motion.lower_mps < 0 or motion.upper_mps < motion.lower_mps:
            raise ValueError("Reference speed interval must be nonnegative and ordered")
        if rate < 0 or (angle and rate == 0):
            raise ValueError("A nonzero turn requires a positive rate")
        return cls(
            heading * mp.pi / 180,
            mp.sign(angle) * rate * mp.pi / 180,
            abs(angle / rate) if angle else mp.mpf(0),
            mp.mpf(motion.lower_mps),
            mp.mpf(motion.upper_mps),
        )

    def heading(self, time: mp.mpf) -> mp.mpf:
        return self.initial + self.omega * min(time, self.duration)

    def direction(self, time: mp.mpf) -> Vector:
        heading = self.heading(time)
        return Vector(mp.sin(heading), mp.cos(heading))

    def integral(self, left: mp.mpf, right: mp.mpf) -> Vector:
        """Integrate unit direction, splitting a possible turn/straight transition."""
        active_end = min(right, self.duration)
        result = zero()
        if left < active_end:
            start, end = self.heading(left), self.heading(active_end)
            result = Vector((mp.cos(start) - mp.cos(end)) / self.omega, (mp.sin(end) - mp.sin(start)) / self.omega)
        tail_start = max(left, self.duration)
        if tail_start < right:
            result = result + self.direction(right).scale(right - tail_start)
        return result

    def numerical_integral(self, left: mp.mpf, right: mp.mpf) -> Vector:
        times = [left, right]
        if left < self.duration < right:
            times.insert(1, self.duration)
        east = mp.quad(lambda time: mp.sin(self.heading(time)), times)
        north = mp.quad(lambda time: mp.cos(self.heading(time)), times)
        if not isinstance(east, mp.mpf) or not isinstance(north, mp.mpf):
            raise ArithmeticError("Real direction quadrature must return real values")
        return Vector(east, north)


@dataclass(frozen=True)
class SpeedSegment:
    left: mp.mpf
    right: mp.mpf
    speed: mp.mpf


@dataclass(frozen=True)
class SupportPoint:
    point: Vector
    a_segments: tuple[SpeedSegment, ...]
    b_segments: tuple[SpeedSegment, ...]


@dataclass(frozen=True)
class Witness:
    time: mp.mpf
    vertices: tuple[SupportPoint, ...]
    weights: tuple[mp.mpf, ...]


@dataclass(frozen=True)
class DistanceBracket:
    lower: mp.mpf
    upper: mp.mpf
    witness: Witness


class BudgetExceeded(Exception):
    """The reference must retain an unresolved result on budget exhaustion."""


@dataclass
class Budget:
    max_support: int = 50_000
    max_cells: int = 16_384
    support: int = 0
    cells: int = 0

    def use_support(self) -> None:
        if self.support >= self.max_support:
            raise BudgetExceeded("support_budget")
        self.support += 1

    def use_cell(self) -> None:
        if self.cells >= self.max_cells:
            raise BudgetExceeded("time_budget")
        self.cells += 1


def aircraft_support(
    path: Path, direction: Vector, time: mp.mpf, model: Model
) -> tuple[Vector, tuple[SpeedSegment, ...]]:
    if time == 0:
        return zero(), ()
    if model == "C":
        basis = path.integral(mp.mpf(0), time)
        speed = path.upper if basis.dot(direction) >= 0 else path.lower
        return basis.scale(speed), (SpeedSegment(mp.mpf(0), time, speed),)
    breaks = {mp.mpf(0), time}
    if 0 < path.duration < time:
        breaks.add(path.duration)
    active = min(time, path.duration)
    if active > 0 and direction.norm() > 0:
        offset = mp.atan2(direction.x, direction.y) + mp.pi / 2
        first, last = sorted((path.initial, path.heading(active)))
        for index in range(int(mp.ceil((first - offset) / mp.pi)), int(mp.floor((last - offset) / mp.pi)) + 1):
            crossing = (offset + index * mp.pi - path.initial) / path.omega
            if 0 < crossing < active:
                breaks.add(crossing)
    times = sorted(breaks)
    segments: list[SpeedSegment] = []
    result = zero()
    for left, right in pairwise(times):
        speed = path.upper if path.direction((left + right) / 2).dot(direction) >= 0 else path.lower
        segments.append(SpeedSegment(left, right, speed))
        result = result + path.integral(left, right).scale(speed)
    return result, tuple(segments)


@dataclass
class ReachableSet:
    position: Vector
    a: Path
    b: Path
    model: Model
    budget: Budget

    def support(self, direction: Vector, time: mp.mpf) -> SupportPoint:
        self.budget.use_support()
        a_position, a_segments = aircraft_support(self.a, direction, time, self.model)
        b_position, b_segments = aircraft_support(self.b, direction.scale(mp.mpf(-1)), time, self.model)
        return SupportPoint(self.position + a_position - b_position, a_segments, b_segments)


def closest_simplex(points: tuple[SupportPoint, ...]) -> tuple[Vector, tuple[SupportPoint, ...], tuple[mp.mpf, ...]]:
    """Exhaustive point/segment/triangle projection for at most three support points."""
    for a, b, c in combinations(points, 3):
        u, v = b.point - a.point, c.point - a.point
        determinant = u.x * v.y - u.y * v.x
        if determinant:
            wb = (-a.point.x * v.y + a.point.y * v.x) / determinant
            wc = (-u.x * a.point.y + u.y * a.point.x) / determinant
            wa = 1 - wb - wc
            if min(wa, wb, wc) >= 0:
                return zero(), (a, b, c), (wa, wb, wc)
    best = min(points, key=lambda sample: sample.point.dot(sample.point))
    projection, vertices, weights = best.point, (best,), (mp.mpf(1),)
    distance_squared = projection.dot(projection)
    for a, b in combinations(points, 2):
        edge = b.point - a.point
        denominator = edge.dot(edge)
        if denominator == 0:
            continue
        fraction = min(mp.mpf(1), max(mp.mpf(0), -a.point.dot(edge) / denominator))
        candidate = a.point + edge.scale(fraction)
        if candidate.dot(candidate) < distance_squared:
            projection, vertices, weights = candidate, (a, b), (1 - fraction, fraction)
            distance_squared = candidate.dot(candidate)
    return projection, vertices, weights


def fixed_time_distance(reachable: ReachableSet, time: mp.mpf, tolerance_m: mp.mpf) -> DistanceBracket:
    direction = (
        reachable.position.scale(-1 / reachable.position.norm())
        if reachable.position.norm()
        else Vector(mp.mpf(1), mp.mpf(0))
    )
    sample = reachable.support(direction, time)
    vertices, weights = (sample,), (mp.mpf(1),)
    projection = sample.point
    lower = mp.mpf(0)
    while True:
        upper = projection.norm()
        witness = Witness(time, vertices, weights)
        if upper <= tolerance_m:
            return DistanceBracket(lower, upper, witness)
        unit = projection.scale(1 / upper)
        next_point = reachable.support(unit.scale(mp.mpf(-1)), time)
        lower = max(lower, mp.mpf(0), unit.dot(next_point.point))
        if upper - lower <= tolerance_m or any(next_point.point == vertex.point for vertex in vertices):
            return DistanceBracket(lower, upper, witness)
        projection, vertices, weights = closest_simplex((*vertices, next_point))


def integrate_witness(reachable: ReachableSet, witness: Witness) -> Vector:
    """Quadrature of the attainable profiles, independent of the support antiderivative."""
    result = reachable.position
    total_weight = sum(witness.weights, mp.mpf(0))
    if total_weight <= 0 or any(weight < 0 for weight in witness.weights):
        raise ValueError("Witness weights must form a convex combination")
    for sample, weight in zip(witness.vertices, witness.weights, strict=True):
        for path, segments, sign in ((reachable.a, sample.a_segments, 1), (reachable.b, sample.b_segments, -1)):
            if witness.time > 0 and (not segments or segments[0].left != 0 or segments[-1].right != witness.time):
                raise ValueError("Witness does not cover the requested interval")
            previous = mp.mpf(0)
            for segment in segments:
                if (
                    segment.left != previous
                    or segment.right <= segment.left
                    or not path.lower <= segment.speed <= path.upper
                ):
                    raise ValueError("Witness has an inadmissible speed segment")
                displacement = path.numerical_integral(segment.left, segment.right)
                result = result + displacement.scale(sign * segment.speed * weight / total_weight)
                previous = segment.right
    return result


@dataclass(frozen=True)
class ReferenceResult:
    classification: Classification
    reason: str
    lower_m: mp.mpf
    upper_m: mp.mpf
    witness: Witness | None
    support_evaluations: int
    time_cells: int


def classify(
    query: Query,
    model: Model,
    *,
    digits: int = 80,
    margin_m: float = 25.0,
    max_support: int = 50_000,
    max_cells: int = 16_384,
) -> ReferenceResult:
    """Classify using numerical distance brackets outside a stated exclusion band."""
    with mp.workdps(digits):
        budget = Budget(max_support, max_cells)
        reachable = ReachableSet(
            Vector(*(mp.mpf(value) for value in query.relative_position_m)),
            Path.from_motion(query.a),
            Path.from_motion(query.b),
            model,
            budget,
        )
        horizon, threshold, margin = mp.mpf(query.horizon_s), mp.mpf(query.threshold_m), mp.mpf(margin_m)
        if horizon < 0 or threshold < 0 or margin <= 0:
            raise ValueError("Nonnegative horizon/threshold and positive reference margin required")
        lipschitz = reachable.a.upper + reachable.b.upper
        best_upper = mp.inf
        best_witness: Witness | None = None
        certified_lower = mp.inf
        cells: list[tuple[mp.mpf, int, mp.mpf, mp.mpf]] = []

        def result(status: Classification, reason: str, lower: mp.mpf) -> ReferenceResult:
            return ReferenceResult(status, reason, lower, best_upper, best_witness, budget.support, budget.cells)

        def evaluate(time: mp.mpf) -> DistanceBracket:
            nonlocal best_upper, best_witness
            bracket = fixed_time_distance(reachable, time, margin / 8)
            if bracket.upper < best_upper:
                best_upper, best_witness = bracket.upper, bracket.witness
            return bracket

        def witness_conflicts() -> bool:
            if best_upper < threshold - margin and best_witness is not None:
                checked = integrate_witness(reachable, best_witness).norm()
                if checked >= threshold - margin or abs(checked - best_upper) > mp.mpf("1e-20"):
                    raise ArithmeticError("Independent witness integration disagrees with the support hull")
                return True
            return False

        def add_cell(left: mp.mpf, right: mp.mpf) -> None:
            nonlocal certified_lower
            budget.use_cell()
            bracket = evaluate((left + right) / 2)
            lower = max(mp.mpf(0), bracket.lower - lipschitz * (right - left) / 2)
            if lower > threshold + margin:
                certified_lower = min(certified_lower, lower)
            else:
                heapq.heappush(cells, (lower, budget.cells, left, right))

        try:
            events = sorted(
                {mp.mpf(0), horizon, min(horizon, reachable.a.duration), min(horizon, reachable.b.duration)}
            )
            initial_bracket = evaluate(mp.mpf(0))
            if witness_conflicts():
                return result("conflict", "attainable_profile", mp.mpf(0))
            if horizon == 0:
                return result(
                    "separated" if initial_bracket.lower > threshold + margin else "unresolved",
                    "initial_distance",
                    initial_bracket.lower,
                )
            for time in events[1:]:
                evaluate(time)
                if witness_conflicts():
                    return result("conflict", "attainable_profile", mp.mpf(0))
            for left, right in pairwise(events):
                add_cell(left, right)
                if witness_conflicts():
                    return result("conflict", "attainable_profile", mp.mpf(0))
            while cells:
                lower, _, left, right = heapq.heappop(cells)
                if min(lower, certified_lower) > threshold - margin and best_upper <= threshold + margin:
                    return result("unresolved", "boundary_band", min(lower, certified_lower))
                middle = (left + right) / 2
                add_cell(left, middle)
                if witness_conflicts():
                    return result("conflict", "attainable_profile", mp.mpf(0))
                add_cell(middle, right)
                if witness_conflicts():
                    return result("conflict", "attainable_profile", mp.mpf(0))
            return result("separated", "whole_horizon_bound", certified_lower)
        except BudgetExceeded as error:
            return result("unresolved", str(error), mp.mpf(0))
