"""Independent high-precision oracle for the straight-heading safety test.

This module shares no geometry code with ``geometric_safety.relevant_aircraft``:
it re-derives the straight-heading answer from the manuscript's definitions in
arbitrary-precision arithmetic (mpmath). The turn-aware solver delegates its
zero-turn cases to the production closed form, so validating that closed form
against itself proves nothing; agreement with this implementation is evidence
about the code, not a code path compared with itself.

The construction follows the manuscript directly: corner relative velocities
over the speed rectangle, the displacement hull ``S(T) = conv({0} u {T v_i})``,
the minimum separation as the distance from ``-r_0`` to that hull, and the
earliest closest-approach time as ``T`` times the gauge of the projection
point with respect to ``S(T)``.

Points are plain ``(x, y)`` tuples of ``mp.mpf`` scalars throughout.
"""

from __future__ import annotations

import mpmath as mp

Scalar = mp.mpf
Point = tuple[Scalar, Scalar]

DEFAULT_DPS = 30
_GAUGE_BISECTIONS = 80

# Definitional, not imported from the production code: one knot is 1852 m per hour.
_KT_TO_MPS_NUM = 1852
_KT_TO_MPS_DEN = 3600


def _cross(o: Point, a: Point, b: Point) -> Scalar:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[Point]) -> list[Point]:
    """Monotone chain; returns the hull counter-clockwise without repetition."""
    pts: list[Point] = sorted({(p[0], p[1]) for p in points})
    if len(pts) <= 2:
        return pts
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _point_in_convex(hull: list[Point], q: Point, tol: Scalar) -> bool:
    """Sign test for a genuinely two-dimensional counter-clockwise hull."""
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        edge_len = mp.hypot(b[0] - a[0], b[1] - a[1])
        if edge_len <= tol:
            continue
        # Normalised to a perpendicular distance so short edges do not blur
        # the test.
        if _cross(a, b, q) / edge_len < -tol:
            return False
    return True


def _hull_is_degenerate(hull: list[Point], tol: Scalar) -> bool:
    """True when the hull carries no usable area (a point or a sliver)."""
    if len(hull) < 3:
        return True
    area2 = mp.mpf(0)
    perimeter = mp.mpf(0)
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        area2 += a[0] * b[1] - a[1] * b[0]
        perimeter += mp.hypot(b[0] - a[0], b[1] - a[1])
    return abs(area2) <= tol * (perimeter + 1)


def _farthest_pair(points: list[Point]) -> tuple[tuple[Point, Point], Scalar]:
    best = (points[0], points[0])
    best_dist = mp.mpf(-1)
    for i in range(len(points)):
        for j in range(i, len(points)):
            dist = mp.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
            if dist > best_dist:
                best_dist = dist
                best = (points[i], points[j])
    return best, best_dist


def _point_segment(q: Point, a: Point, b: Point) -> tuple[Scalar, Point]:
    """Distance from q to segment ab and the closest point on it."""
    ab = (b[0] - a[0], b[1] - a[1])
    denom = ab[0] * ab[0] + ab[1] * ab[1]
    if denom == 0:
        return mp.hypot(q[0] - a[0], q[1] - a[1]), a
    t = ((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / denom
    t = max(mp.mpf(0), min(mp.mpf(1), t))
    p: Point = (a[0] + t * ab[0], a[1] + t * ab[1])
    return mp.hypot(q[0] - p[0], q[1] - p[1]), p


def _distance_to_hull(hull: list[Point], q: Point, tol: Scalar) -> tuple[Scalar, Point]:
    """Distance from q to conv(hull points) and the projection point."""
    if _point_in_convex(hull, q, tol):
        return mp.mpf(0), q
    if len(hull) == 1:
        return mp.hypot(q[0] - hull[0][0], q[1] - hull[0][1]), hull[0]
    best_dist = mp.hypot(q[0] - hull[0][0], q[1] - hull[0][1])
    best_point = hull[0]
    n = len(hull)
    for i in range(n):
        for j in range(i + 1, n):
            dist, point = _point_segment(q, hull[i], hull[j])
            if dist < best_dist:
                best_dist = dist
                best_point = point
    return best_dist, best_point


def _gauge(hull: list[Point], p: Point, tol: Scalar) -> Scalar:
    """inf{lambda in [0,1] : p in lambda * conv(hull)}, with 0 a hull point."""
    if mp.hypot(p[0], p[1]) <= tol:
        return mp.mpf(0)
    if not _point_in_convex(hull, p, tol):
        return mp.mpf(1)
    lo, hi = mp.mpf(0), mp.mpf(1)
    for _ in range(_GAUGE_BISECTIONS):
        mid = (lo + hi) / 2
        if mid == 0:
            break
        scaled: Point = (p[0] / mid, p[1] / mid)
        if _point_in_convex(hull, scaled, tol):
            hi = mid
        else:
            lo = mid
    return hi


def independent_straight_outcome(
    *,
    rel_east_m: float,
    rel_north_m: float,
    a_heading_deg: float,
    a_speed_kt: float,
    b_heading_deg: float,
    b_speed_kt: float,
    speed_diff_kt: float,
    horizon_s: float,
    threshold_m: float,
    dps: int = DEFAULT_DPS,
) -> tuple[bool, float, float]:
    """Return (is_safe, min_distance_m, earliest_closest_time_s) at high precision."""
    with mp.workdps(dps):
        kt = mp.mpf(_KT_TO_MPS_NUM) / mp.mpf(_KT_TO_MPS_DEN)
        h_a = mp.radians(mp.mpf(a_heading_deg))
        h_b = mp.radians(mp.mpf(b_heading_deg))
        u_a = (mp.sin(h_a), mp.cos(h_a))
        u_b = (mp.sin(h_b), mp.cos(h_b))
        diff = mp.mpf(speed_diff_kt) * kt
        s_a = mp.mpf(a_speed_kt) * kt
        s_b = mp.mpf(b_speed_kt) * kt
        speeds_a = (max(s_a - diff, mp.mpf(0)), s_a + diff)
        speeds_b = (max(s_b - diff, mp.mpf(0)), s_b + diff)
        horizon = mp.mpf(horizon_s)

        points: list[Point] = [(mp.mpf(0), mp.mpf(0))]
        for sa in speeds_a:
            for sb in speeds_b:
                vx = sa * u_a[0] - sb * u_b[0]
                vy = sa * u_a[1] - sb * u_b[1]
                points.append((horizon * vx, horizon * vy))

        scale = max(mp.hypot(px, py) for px, py in points)
        scale = max(scale, mp.hypot(mp.mpf(rel_east_m), mp.mpf(rel_north_m)), mp.mpf(1))
        tol = scale * mp.mpf(10) ** (-(dps - 8))

        hull = _convex_hull(points)
        q: Point = (-mp.mpf(rel_east_m), -mp.mpf(rel_north_m))

        if _hull_is_degenerate(hull, tol):
            # The reachable set collapses to (nearly) a line through the
            # origin: use its principal segment, and compute the gauge as a
            # one-dimensional coordinate ratio, which stays exact where the
            # sliver's sign tests would not.
            (seg_a, seg_b), seg_len = _farthest_pair(hull)
            if seg_len <= tol:
                distance = mp.hypot(q[0] - seg_a[0], q[1] - seg_a[1])
                closest_time = mp.mpf(0)
            else:
                distance, projection = _point_segment(q, seg_a, seg_b)
                direction = ((seg_b[0] - seg_a[0]) / seg_len, (seg_b[1] - seg_a[1]) / seg_len)
                coords = [pt[0] * direction[0] + pt[1] * direction[1] for pt in hull]
                s_min = min(coords)
                s_max = max(coords)
                s_p = projection[0] * direction[0] + projection[1] * direction[1]
                if abs(s_p) <= tol:
                    lam = mp.mpf(0)
                elif s_p > 0:
                    lam = s_p / s_max
                else:
                    lam = s_p / s_min
                closest_time = horizon * max(mp.mpf(0), min(mp.mpf(1), lam))
        else:
            distance, projection = _distance_to_hull(hull, q, tol)
            closest_time = horizon * _gauge(hull, projection, tol)

        return bool(distance >= mp.mpf(threshold_m)), float(distance), float(closest_time)
