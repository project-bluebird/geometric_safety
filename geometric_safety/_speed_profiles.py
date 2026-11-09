"""Directional interval certificate for independently bounded speed profiles.

For a unit normal n, the minimum relative projection is
q(n,t) = n.r0 + m_A I_A - w_A J_A - m_B I_B - w_B J_B,
where I_i integrates n.u_i and J_i integrates its absolute value.
The speed choices are independent at every time. Thus q bounds every
admissible relative position without an isotropic speed-profile allowance.

Over a time interval, q' is bounded using the existing projected-heading
ranges. The existing two-affine-bound calculation then bounds q throughout
that interval. Candidate directions come from constant-speed geometry and
perpendiculars to both final tracks. Every candidate uses the same interval
bound; an imperfect direction can prevent acceptance but cannot justify it.
The caller applies displacement-error allowances and a numerical margin to
the required distance, and validates the supported motion and numerical domain.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from geometric_safety.relevant_aircraft import (
    _projected_heading_range,
    _projected_motion_lower_bound,
    _sample_relative_hull_at_time,
)
from geometric_safety.util import heading_to_unit_vector, shortest_turn_parameters


class Parameters(NamedTuple):
    heading: float
    target: float
    rate: float
    lower: float
    upper: float
    initial: float
    omega: float
    duration: float


def prepare_motion(
    heading_deg: float, target_deg: float, rate_deg_s: float, lower_mps: float, upper_mps: float
) -> Parameters:
    _, rate, duration = shortest_turn_parameters(heading_deg, target_deg, rate_deg_s)
    return Parameters(
        float(heading_deg),
        float(target_deg),
        float(rate_deg_s),
        float(lower_mps),
        float(upper_mps),
        math.radians(math.fmod(heading_deg, 360.0)),
        math.radians(rate),
        duration,
    )


@numba.njit(cache=True, fastmath=False)
def projected_integral(p: Parameters, normal: NDArray[np.float64], left: float, right: float) -> float:
    half_sweep = 0.5 * p.omega * (right - left)
    middle = p.initial + p.omega * (left + 0.5 * (right - left))
    sinc = math.sin(half_sweep) / half_sweep if half_sweep != 0 else 1.0
    return (right - left) * sinc * (normal[0] * math.sin(middle) + normal[1] * math.cos(middle))


@numba.njit(cache=True, fastmath=False)
def projection_integrals(p: Parameters, normal: NDArray[np.float64], time: float) -> tuple[float, float]:
    active = min(time, p.duration)
    signed = projected_integral(p, normal, 0.0, active)
    absolute = 0.0
    previous = 0.0
    if active > 0.0 and p.omega != 0.0:
        offset = math.atan2(normal[0], normal[1]) + 0.5 * math.pi
        first, last = min(p.initial, p.initial + p.omega * active), max(p.initial, p.initial + p.omega * active)
        lo = math.ceil((first - offset) / math.pi)
        hi = math.floor((last - offset) / math.pi)
        # A shortest turn spans at most pi, hence at most one interior zero.
        # Traversing in motion order also handles roots at both endpoints.
        for item in range(lo, hi + 1):
            index = item if p.omega > 0 else hi - (item - lo)
            crossing = (offset + index * math.pi - p.initial) / p.omega
            if 0.0 < crossing < active:
                absolute += abs(projected_integral(p, normal, previous, crossing))
                previous = crossing
    absolute += abs(projected_integral(p, normal, previous, active))
    if time <= p.duration:
        return signed, absolute
    final = p.initial + p.omega * p.duration
    tail = (time - active) * (normal[0] * math.sin(final) + normal[1] * math.cos(final))
    return signed + tail, absolute + abs(tail)


@numba.njit(cache=True, fastmath=False)
def minimum_projection(
    position: NDArray[np.float64], a: Parameters, b: Parameters, normal: NDArray[np.float64], time: float
) -> float:
    ai, aj = projection_integrals(a, normal, time)
    bi, bj = projection_integrals(b, normal, time)
    return (
        position[0] * normal[0]
        + position[1] * normal[1]
        + 0.5 * (a.lower + a.upper) * ai
        - 0.5 * (a.upper - a.lower) * aj
        - 0.5 * (b.lower + b.upper) * bi
        - 0.5 * (b.upper - b.lower) * bj
    )


@numba.njit(cache=True, fastmath=False)
def directional_interval(
    position: NDArray[np.float64], a: Parameters, b: Parameters, normal: NDArray[np.float64], left: float, right: float
) -> float:
    qa = minimum_projection(position, a, b, normal, left)
    qb = minimum_projection(position, a, b, normal, right)
    al, au = _projected_heading_range(a.heading, a.target, a.rate, normal, left, right)
    bl, bu = _projected_heading_range(b.heading, b.target, b.rate, normal, left, right)
    lower = min(a.lower * al, a.upper * al) - max(b.lower * bu, b.upper * bu)
    upper = min(a.lower * au, a.upper * au) - max(b.lower * bl, b.upper * bl)
    return _projected_motion_lower_bound(qa, qb, lower, upper, right - left)


@numba.njit(cache=True, fastmath=False)
def interval_certificate(
    position: NDArray[np.float64],
    a: Parameters,
    b: Parameters,
    final_normals: NDArray[np.float64],
    left: float,
    right: float,
    required: float,
) -> float:
    best = -math.inf
    for time in (left, right, 0.5 * (left + right)):
        distance, _, closest = _sample_relative_hull_at_time(
            position, a.heading, a.target, a.rate, a.lower, a.upper, b.heading, b.target, b.rate, b.lower, b.upper, time
        )
        # The constant-speed set is a subset of the varying-speed set.
        if distance < required:
            return best
        length = math.hypot(closest[0], closest[1])
        if length == 0.0:
            continue
        normal = closest / length
        bound = directional_interval(position, a, b, normal, left, right)
        best = max(best, bound)
        if best >= required:
            return best
    # A final-track normal removes that aircraft's speed uncertainty from the
    # straight continuation. Check both sides, including the preceding turn.
    for perpendicular in final_normals:
        for sign in (1.0, -1.0):
            bound = directional_interval(position, a, b, sign * perpendicular, left, right)
            best = max(best, bound)
            if best >= required:
                return best
    return best


@numba.njit(cache=True, fastmath=False)
def certify_speed_profiles(
    position: NDArray[np.float64],
    a: Parameters,
    b: Parameters,
    horizon: float,
    required: float,
    minimum_interval: float,
    max_cells: int,
) -> tuple[bool, float, float, float]:
    """Return acceptance and the weakest accepted interval's bound and endpoints.

    Every accepted interval covers every admissible speed profile. Bounds from
    incomplete searches are unavailable; exhaustion cannot produce acceptance.
    """
    final_normals = np.empty((2, 2))
    for index, motion in enumerate((a, b)):
        track = heading_to_unit_vector(motion.target)
        final_normals[index, 0], final_normals[index, 1] = -track[1], track[0]
    stack = np.empty((64, 2))
    stack[0, 0], stack[0, 1] = 0.0, horizon
    top, evaluated = 1, 0
    weakest, weakest_left, weakest_right = math.inf, 0.0, horizon
    while top > 0:
        top -= 1
        left, right = stack[top, 0], stack[top, 1]
        evaluated += 1
        if evaluated > max_cells:
            return False, math.nan, math.nan, math.nan
        bound = interval_certificate(position, a, b, final_normals, left, right, required)
        if math.isfinite(bound) and bound >= required:
            if bound < weakest:
                weakest, weakest_left, weakest_right = bound, left, right
            continue
        if right - left <= minimum_interval or top + 2 > len(stack):
            return False, math.nan, math.nan, math.nan
        middle = left + 0.5 * (right - left)
        if not left < middle < right:
            return False, math.nan, math.nan, math.nan
        stack[top, 0], stack[top, 1] = left, middle
        stack[top + 1, 0], stack[top + 1, 1] = middle, right
        top += 2
    return True, weakest, weakest_left, weakest_right
