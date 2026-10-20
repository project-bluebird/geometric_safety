"""Generic geometry primitives and aviation/geospatial utilities.

Nothing in this module knows about the reachability solver or its specific
models. Functions here are pure 2-D computational geometry (convex hulls,
projections, point-in-polygon) plus a handful of aviation coordinate helpers
(heading vectors, local tangent-plane projection, heading arithmetic).
"""

import math

import numba
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Unit-conversion and physical constants
# ---------------------------------------------------------------------------

KT_TO_MPS = 1852.0 / 3600.0
NMI_TO_M = 1852.0
M_TO_NMI = 1.0 / NMI_TO_M
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
EARTH_RADIUS_IN_METERS = 6378137.0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


@numba.jit(nopython=True, fastmath=True, cache=True)
def clip_value(x: float, min_val: float, max_val: float) -> float:
    """Return `x` clipped to the closed interval `[min_val, max_val]`."""
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x


# ---------------------------------------------------------------------------
# Aviation / geospatial helpers
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def heading_to_unit_vector(heading_deg: float) -> np.ndarray:
    """
    Convert an aviation heading into a local east/north unit vector.

    Parameters
    ----------
    heading_deg: float
        Heading in degrees.

    Returns
    -------
    np.ndarray
        Two-element unit vector `(east, north)`.

    Notes
    -----
    The module uses the aviation convention "clockwise from north", so the components
    are

        east  = sin(heading),
        north = cos(heading).

    This differs from the standard mathematical angle convention.
    """
    heading_rad = heading_deg * DEG_TO_RAD
    out = np.empty(2, dtype=np.float64)
    out[0] = np.sin(heading_rad)
    out[1] = np.cos(heading_rad)
    return out


@numba.njit(cache=True, fastmath=True)
def latlon_to_local_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> np.ndarray:
    """
    Project latitude/longitude into a local east/north tangent-plane frame.

    Parameters
    ----------
    lat / lon: float
        Point to project, in degrees.
    ref_lat / ref_lon: float
        Reference position used as the local origin, in degrees.

    Returns
    -------
    np.ndarray
        East/north offset in metres from `(ref_lat, ref_lon)` to `(lat, lon)`.

    Notes
    -----
    This is the local projection used throughout the module. It is the first-order
    tangent-plane approximation:

        east  ≈ R cos(ref_lat) Δlon,
        north ≈ R Δlat.

    That is appropriate for local conflict checks, but not intended for long-range
    geodesic fidelity.
    """
    d_lat = (lat - ref_lat) * DEG_TO_RAD
    d_lon = (lon - ref_lon) * DEG_TO_RAD
    ref_lat_rad = ref_lat * DEG_TO_RAD
    east = EARTH_RADIUS_IN_METERS * d_lon * np.cos(ref_lat_rad)
    north = EARTH_RADIUS_IN_METERS * d_lat
    out = np.empty(2, dtype=np.float64)
    out[0] = east
    out[1] = north
    return out


# TwoDiff has no products, so contraction alone leaves its arithmetic strict.
# Explicit flags prevent a fastmath caller from adding reassociation flags.
@numba.jit(nopython=True, fastmath={"contract"}, cache=True)
def heading_diff(h1: float, h2: float, abs_diff: bool = False) -> float:
    """
    Return the shortest signed difference of the supplied binary64 headings.

    The result is in [-180, 180), or its absolute value when abs_diff=True.
    Exact half-turn ties use -180. A compensated difference preserves the side
    of a half-turn or wrap even when ordinary subtraction rounds onto it.
    Headings are reduced separately to avoid overflow of their difference.

    The output remains binary64. If a positive difference just below 180 rounds
    to 180, return the adjacent float below 180 to retain its positive direction.
    Other results use normal rounding. Non-finite inputs return NaN.
    """
    if not math.isfinite(h1) or not math.isfinite(h2):
        return math.nan

    a = np.fmod(h2, 360.0)
    b = np.fmod(h1, 360.0)
    difference = a - b
    # Knuth TwoDiff: difference + residual is the exact difference of a and b.
    b_virtual = a - difference
    a_virtual = difference + b_virtual
    b_roundoff = b_virtual - b
    a_roundoff = a - a_virtual
    residual = a_roundoff + b_roundoff

    remainder = np.fmod(difference, 360.0)
    if remainder > 180.0 or (remainder == 180.0 and residual >= 0.0):
        remainder -= 360.0
    elif remainder < -180.0 or (remainder == -180.0 and residual < 0.0):
        remainder += 360.0
    result = remainder + residual
    if result == 180.0:
        result = np.nextafter(180.0, 0.0)
    return abs(result) if abs_diff else result


@numba.njit(cache=True, fastmath={"contract"})
def shortest_turn_parameters(
    heading0_deg: float, target_heading_deg: float, turn_rate_deg_sec: float
) -> tuple[float, float, float]:
    """Return signed angle, signed rate and duration for a shortest fixed-rate turn.

    The angle and rate are degrees and degrees per second; duration is seconds.
    Exact half-turn ties turn left. Equal headings have zero rate and duration.
    A changed heading requires a finite, nonzero supplied rate magnitude.
    """
    angle_deg = heading_diff(heading0_deg, target_heading_deg)
    if not math.isfinite(angle_deg) or not math.isfinite(turn_rate_deg_sec):
        raise ValueError("Headings and turn rate must be finite")
    if angle_deg == 0.0:
        return 0.0, 0.0, 0.0
    if turn_rate_deg_sec == 0.0:
        raise ValueError("A changed target heading requires a nonzero turn rate")
    signed_rate_deg_sec = math.copysign(abs(turn_rate_deg_sec), angle_deg)
    return angle_deg, signed_rate_deg_sec, abs(angle_deg) / abs(turn_rate_deg_sec)


# ---------------------------------------------------------------------------
# 2-D computational geometry primitives
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the signed 2-D cross product of the vectors `OA` and `OB`.

    Parameters
    ----------
    o: np.ndarray
        Origin of the vectors.
    a / b: np.ndarray
        Vector endpoints.

    Returns
    -------
    float
        Signed area proportional to the parallelogram area. Positive values indicate
        that `O -> A -> B` is a counter-clockwise turn.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@numba.njit(cache=True, fastmath=True)
def small_convex_hull(pts: np.ndarray) -> np.ndarray:
    """
    Compute a convex hull for tiny point sets without the generic monotone-chain path.

    This helper is used only for the structured 4- and 5-point hulls that arise from
    the speed-rectangle geometry in this module. The full `convex_hull` implementation
    is retained below as the generic reference path and sanity oracle.
    """
    n = pts.shape[0]

    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Deduplicate in-place; these structured point sets are tiny enough that an O(n^2)
    # pass is quicker than setting up a more general sorting pipeline.
    unique_pts = np.empty((n, 2), dtype=np.float64)
    unique_len = 0
    for i in range(n):
        x = pts[i, 0]
        y = pts[i, 1]
        duplicate = False
        for j in range(unique_len):
            if unique_pts[j, 0] == x and unique_pts[j, 1] == y:
                duplicate = True
                break
        if not duplicate:
            unique_pts[unique_len, 0] = x
            unique_pts[unique_len, 1] = y
            unique_len += 1

    if unique_len == 0:
        return np.empty((0, 2), dtype=np.float64)

    if unique_len == 1:
        out = np.empty((1, 2), dtype=np.float64)
        out[0, :] = unique_pts[0]
        return out

    if unique_len == 2:
        out = np.empty((2, 2), dtype=np.float64)
        if unique_pts[1, 0] < unique_pts[0, 0] or (
            unique_pts[1, 0] == unique_pts[0, 0] and unique_pts[1, 1] < unique_pts[0, 1]
        ):
            out[0, :] = unique_pts[1]
            out[1, :] = unique_pts[0]
        else:
            out[0, :] = unique_pts[0]
            out[1, :] = unique_pts[1]
        return out

    # Gift-wrap the hull. For <= 5 points this is simpler and faster than invoking the
    # fully generic hull builder.
    start = 0
    for i in range(1, unique_len):
        if unique_pts[i, 0] < unique_pts[start, 0] or (
            unique_pts[i, 0] == unique_pts[start, 0] and unique_pts[i, 1] < unique_pts[start, 1]
        ):
            start = i

    hull_idx = np.empty(unique_len, dtype=np.int64)
    hull_len = 0
    p = start

    while True:
        hull_idx[hull_len] = p
        hull_len += 1

        q = -1
        for r in range(unique_len):
            if r == p:
                continue
            if q == -1:
                q = r
                continue

            turn = cross_2d(unique_pts[p], unique_pts[q], unique_pts[r])
            if turn > 0.0:
                q = r
            elif turn == 0.0:
                dx_q = unique_pts[q, 0] - unique_pts[p, 0]
                dy_q = unique_pts[q, 1] - unique_pts[p, 1]
                dx_r = unique_pts[r, 0] - unique_pts[p, 0]
                dy_r = unique_pts[r, 1] - unique_pts[p, 1]
                if dx_r * dx_r + dy_r * dy_r > dx_q * dx_q + dy_q * dy_q:
                    q = r

        if q == start:
            break

        p = q

        if hull_len >= unique_len:
            break

    hull = np.empty((hull_len, 2), dtype=np.float64)
    for i in range(hull_len):
        hull[i, :] = unique_pts[hull_idx[i]]

    return hull


@numba.njit(cache=True, fastmath=True)
def convex_hull(pts: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a set of 2-D points using Andrew's monotone chain.

    Parameters
    ----------
    pts: np.ndarray
        Array of (x, y) coordinates with shape (N, 2).

    Returns
    -------
    np.ndarray
        Hull vertices ordered counter-clockwise, with duplicate and collinear interior
        points removed.
    """
    n = pts.shape[0]

    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Selection-sort indices by (x, y) because `np.lexsort` is not available in Numba's
    # nopython mode.
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i

    for i in range(n):
        min_i = i
        for j in range(i + 1, n):
            pi = pts[idx[min_i]]
            pj = pts[idx[j]]
            if pj[0] < pi[0] or (pj[0] == pi[0] and pj[1] < pi[1]):
                min_i = j
        tmp = idx[i]
        idx[i] = idx[min_i]
        idx[min_i] = tmp

    # Deduplicate after sorting so the hull construction only sees distinct points.
    uniq_idx = np.empty(n, dtype=np.int64)
    uniq_len = 0
    prev_x = np.inf
    prev_y = np.inf
    for k in range(n):
        i_idx = idx[k]
        x = pts[i_idx, 0]
        y = pts[i_idx, 1]
        if k == 0 or x != prev_x or y != prev_y:
            uniq_idx[uniq_len] = i_idx
            uniq_len += 1
            prev_x = x
            prev_y = y

    m = uniq_len
    if m == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts_unique = np.empty((m, 2), dtype=np.float64)
    for i in range(m):
        pts_unique[i, :] = pts[uniq_idx[i]]

    if m == 1:
        return pts_unique

    # Build the lower chain of the monotone hull, removing non-left turns so only the
    # outer envelope remains.
    lower = np.empty(m, dtype=np.int64)
    lower_len = 0
    for i in range(m):
        while lower_len >= 2:
            p1 = pts_unique[lower[lower_len - 2]]
            p2 = pts_unique[lower[lower_len - 1]]
            p3 = pts_unique[i]
            if cross_2d(p1, p2, p3) <= 0.0:
                lower_len -= 1
            else:
                break
        lower[lower_len] = i
        lower_len += 1

    # Build the upper chain by scanning the sorted points in reverse.
    upper = np.empty(m, dtype=np.int64)
    upper_len = 0
    for i in range(m - 1, -1, -1):
        while upper_len >= 2:
            p1 = pts_unique[upper[upper_len - 2]]
            p2 = pts_unique[upper[upper_len - 1]]
            p3 = pts_unique[i]
            if cross_2d(p1, p2, p3) <= 0.0:
                upper_len -= 1
            else:
                break
        upper[upper_len] = i
        upper_len += 1

    hull_len = lower_len + upper_len - 2
    hull = np.empty((hull_len, 2), dtype=np.float64)

    h = 0
    for i in range(lower_len):
        hull[h, :] = pts_unique[lower[i]]
        h += 1
    for j in range(1, upper_len - 1):
        hull[h, :] = pts_unique[upper[j]]
        h += 1

    return hull


@numba.njit(cache=True, fastmath=False)
def _point_in_convex_polygon(pt: np.ndarray, hull: np.ndarray) -> bool:
    """
    Check whether a point lies inside or on the boundary of a convex polygon.

    Parameters
    ----------
    pt: np.ndarray
        Query point.
    hull: np.ndarray
        Convex polygon vertices in order.

    Returns
    -------
    bool
        `True` if the point is inside or its edge orientations are unresolved
        at floating-point precision.
    """
    hull_len = hull.shape[0]
    if hull_len < 3:
        return False

    # These comparisons also reject points beyond a collapsed hull's endpoints.
    min_x = max_x = hull[0, 0]
    min_y = max_y = hull[0, 1]
    for i in range(1, hull_len):
        min_x = min(min_x, hull[i, 0])
        max_x = max(max_x, hull[i, 0])
        min_y = min(min_y, hull[i, 1])
        max_y = max(max_y, hull[i, 1])
    if pt[0] < min_x or pt[0] > max_x or pt[1] < min_y or pt[1] > max_y:
        return False

    sign = 0
    # Shewchuk's orient2d first-stage error filter, with binary64 unit roundoff.
    # https://www.cs.cmu.edu/~quake/robust.html
    # Uncertain orientations count as boundary; no exact-predicate claim is made.
    # Keep fastmath disabled so reassociation cannot invalidate the filter.
    unit_roundoff = 2.0**-53
    orientation_error_factor = (3.0 + 16.0 * unit_roundoff) * unit_roundoff

    for i in range(hull_len):
        a = hull[i]
        b = hull[(i + 1) % hull_len]
        left = (b[0] - a[0]) * (pt[1] - a[1])
        right = (b[1] - a[1]) * (pt[0] - a[0])
        cross = left - right
        error = orientation_error_factor * (abs(left) + abs(right))

        if abs(cross) <= error:
            continue

        # For a convex polygon with consistent winding, the cross-product sign must not
        # change as we walk around the boundary.
        if sign == 0:
            sign = 1 if cross > 0.0 else -1
        elif (cross > 0.0 and sign == -1) or (cross < 0.0 and sign == 1):
            return False

    return True


@numba.njit(cache=True, fastmath=True, inline="always")
def _segment_projection_coordinates(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float, float]:
    """Return projected coordinates, distance and the segment fraction."""
    ab_x = bx - ax
    ab_y = by - ay
    denom = ab_x * ab_x + ab_y * ab_y
    x, y = ax, ay
    t = 0.0
    if denom != 0.0:
        t = ((px - ax) * ab_x + (py - ay) * ab_y) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        x = ax + t * ab_x
        y = ay + t * ab_y
    return x, y, math.hypot(px - x, py - y), t


@numba.njit(cache=True, fastmath=True)
def _closest_point_on_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Project a point onto the closed line segment `AB`.

    Parameters
    ----------
    pt: np.ndarray
        Query point.
    a / b: np.ndarray
        Segment endpoints.

    Returns
    -------
    tuple[np.ndarray, float]
        Closest point on the segment and its Euclidean distance from `pt`.
    """
    x, y, distance, _ = _segment_projection_coordinates(pt[0], pt[1], a[0], a[1], b[0], b[1])
    proj = np.empty(2, dtype=np.float64)
    proj[0], proj[1] = x, y
    return proj, distance


@numba.njit(cache=True, fastmath=True)
def _project_point_on_convex_polygon(
    pt: NDArray[np.float64], hull: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, int, int, float]:
    """Retain the selected segment's indices and fraction with its projection.

    Equal indices identify a vertex. Indices of -1 identify an interior query,
    for which no boundary segment was selected. Keeping the selected segment
    avoids trying to infer it again from rounded projection coordinates.
    """
    hull_len = hull.shape[0]

    if hull_len == 0:
        raise ValueError("Hull must contain at least one point.")

    best_point = hull[0]
    best_dist = float(np.linalg.norm(pt - best_point))

    if hull_len == 1:
        return best_point, best_dist, 0, 0, 0.0

    if hull_len == 2:
        x, y, distance, fraction = _segment_projection_coordinates(
            pt[0], pt[1], hull[0, 0], hull[0, 1], hull[1, 0], hull[1, 1]
        )
        projection = np.empty(2, dtype=np.float64)
        projection[0], projection[1] = x, y
        return projection, distance, 0, 1, fraction

    # Interior points project to themselves.
    if _point_in_convex_polygon(pt, hull):
        return pt, 0.0, -1, -1, 0.0

    # Otherwise the projection must lie on one of the polygon edges.
    best_x, best_y = best_point[0], best_point[1]
    best_left, best_right, best_fraction = 0, 0, 0.0
    for i in range(hull_len):
        a = hull[i]
        b = hull[(i + 1) % hull_len]

        x, y, cand_dist, fraction = _segment_projection_coordinates(pt[0], pt[1], a[0], a[1], b[0], b[1])

        if cand_dist < best_dist:
            best_dist = cand_dist
            best_x, best_y = x, y
            best_left, best_right, best_fraction = i, (i + 1) % hull_len, fraction

    best_point = np.empty(2, dtype=np.float64)
    best_point[0], best_point[1] = best_x, best_y
    return best_point, best_dist, best_left, best_right, best_fraction


@numba.njit(cache=True, fastmath=True)
def closest_point_on_convex_polygon(pt: np.ndarray, hull: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Find the Euclidean projection of a point onto a convex polygon.

    Parameters
    ----------
    pt: np.ndarray
        Query point.
    hull: np.ndarray
        Convex polygon vertices.

    Returns
    -------
    tuple[np.ndarray, float]
        Projected point and the corresponding Euclidean distance.
    """
    projection, distance, _, _, _ = _project_point_on_convex_polygon(pt, hull)
    return projection, distance
