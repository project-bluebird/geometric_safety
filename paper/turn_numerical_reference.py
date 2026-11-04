"""Independent arbitrary-precision reference for turn certificate calculations.

Inputs are the exact values of the supplied binary64 numbers. Unit conversion
uses 1852/3600 and angles use mpmath's pi. No production geometry is imported.
This is a numerical reference, not validated interval arithmetic.
"""

from __future__ import annotations

import itertools

import mpmath as mp

from paper.numerical_validation import ReferenceCertificate, SerialisedTurnCase, TurnCase

type ReferenceAircraft = tuple[mp.mpf, mp.mpf, mp.mpf, tuple[mp.mpf, mp.mpf], mp.mpf, mp.mpf]


def turn(h: mp.mpf, target: mp.mpf, rate: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    angle = (target - h + 180) % 360 - 180
    return angle, abs(angle / rate) if angle else mp.mpf(0)


def basis(h: mp.mpf, target: mp.mpf, rate: mp.mpf, t: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    angle, duration = turn(h, target, rate)
    if not angle:
        return t * mp.sin(mp.radians(target)), t * mp.cos(mp.radians(target))
    omega = mp.sign(angle) * mp.radians(abs(rate))
    active = min(t, duration)
    initial = mp.radians(h)
    final = initial + omega * active
    # Direct integration, deliberately different from the production chord formula.
    x = (mp.cos(initial) - mp.cos(final)) / omega
    y = (mp.sin(final) - mp.sin(initial)) / omega
    straight = max(t - duration, mp.mpf(0))
    return x + straight * mp.sin(mp.radians(target)), y + straight * mp.cos(mp.radians(target))


def distance(points: list[tuple[mp.mpf, mp.mpf]]) -> mp.mpf:
    """Distance to a convex hull via all triangles and all line segments.

    With four corners, exhaustive barycentric membership avoids reproducing
    the production hull construction and its collinearity tolerances.
    """
    for a, b, c in itertools.combinations(points, 3):
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - a[0], c[1] - a[1]
        det = ux * vy - uy * vx
        if det:
            u = (-a[0] * vy + a[1] * vx) / det
            v = (-ux * a[1] + uy * a[0]) / det
            if u >= 0 and v >= 0 and u + v <= 1:
                return mp.mpf(0)
    best = min(mp.hypot(*p) for p in points)
    for a, b in itertools.combinations(points, 2):
        dx, dy = b[0] - a[0], b[1] - a[1]
        den = dx * dx + dy * dy
        if den:
            u = max(mp.mpf(0), min(mp.mpf(1), -(a[0] * dx + a[1] * dy) / den))
            best = min(best, mp.hypot(a[0] + u * dx, a[1] + u * dy))
    return best


def certificate(case: TurnCase | SerialisedTurnCase, t0: float, t1: float, dps: int = 80) -> ReferenceCertificate:
    """Reference lower bound, envelope and residual before a numerical guard."""
    with mp.workdps(dps):
        q: dict[str, mp.mpf] = {k: mp.mpf(v) for k, v in case.items() if isinstance(v, (float, int))}
        left, right = mp.mpf(t0), mp.mpf(t1)
        knot = mp.mpf(1852) / 3600
        aircraft: list[ReferenceAircraft] = []
        for prefix in ("a", "b"):
            h, target, rate = (
                q[f"{prefix}_{key}"] for key in ("heading0_deg", "target_heading_deg", "turn_rate_deg_sec")
            )
            speed, width = q[f"{prefix}_speed_kt"], q["speed_diff_kt"]
            speeds = (max(speed - width, mp.mpf(0)) * knot, (speed + width) * knot)
            angle, duration = turn(h, target, rate)
            aircraft.append((h, target, rate, speeds, angle, duration))
        position = tuple(mp.mpf(float(v)) for v in case["rel_pos0"])
        distances: list[mp.mpf] = []
        for t in (left, right):
            a, b = aircraft
            ba, bb = basis(a[0], a[1], a[2], t), basis(b[0], b[1], b[2], t)
            points = [
                (position[0] + sa * ba[0] - sb * bb[0], position[1] + sa * ba[1] - sb * bb[1])
                for sa, sb in itertools.product(a[3], b[3])
            ]
            distances.append(distance(points))
        breaks = sorted({left, right} | {a[5] for a in aircraft if left < a[5] < right})

        def heading(a: ReferenceAircraft, t: mp.mpf) -> mp.mpf:
            h, target, rate, _, angle, duration = a
            return h + mp.sign(angle) * abs(rate) * min(t, duration) if angle else target

        lipschitz = mp.mpf(0)
        for lft, rgt in itertools.pairwise(breaks):
            a, b = aircraft
            dl = heading(a, lft) - heading(b, lft)
            dr = heading(a, rgt) - heading(b, rgt)
            anti = 180 + 360 * mp.ceil((min(dl, dr) - 180) / 360) <= max(dl, dr)
            for sa, sb in itertools.product(a[3], b[3]):
                candidates: list[mp.mpf] = []
                for d in (dl, dr):
                    squared = max(mp.mpf(0), sa * sa + sb * sb - 2 * sa * sb * mp.cos(mp.radians(d)))
                    speed = mp.sqrt(squared)
                    # mpmath also supports complex roots; this reference requires real speeds.
                    if not isinstance(speed, mp.mpf):
                        raise ArithmeticError("Nonnegative squared speed produced a nonreal root")
                    candidates.append(speed)
                if anti:
                    candidates.append(sa + sb)
                lipschitz = max([lipschitz, *candidates])
        lower = max(mp.mpf(0), (sum(distances, start=mp.mpf(0)) - lipschitz * (right - left)) / 2)
        eta = q["eta_const_m"] + q["eta_rate_mps"] * right
        for _h, _target, rate, speeds, angle, duration in aircraft:
            if not angle:
                continue
            active = min(right, duration)
            sweep = mp.radians(abs(rate)) * active
            delta = q["turn_speed_schedule_uncertainty_kt"] * knot
            if sweep and delta:
                if abs(angle) <= 90:
                    eta += delta * active * (1 - mp.cos(sweep)) / sweep
                else:
                    eta += min(2 * delta, speeds[1] - speeds[0]) * 4 * active * (1 - mp.cos(sweep / 2)) / sweep
            delay, error = q["turn_delay_max_s"], q["turn_rate_error_max"]
            gap = min(mp.radians(abs(rate)) * delay, mp.radians(abs(angle)))
            eta += speeds[1] * (
                gap * min(right, duration + delay) + error * mp.radians(abs(angle)) * min(right, duration / (1 - error))
            )
        return {
            "lower": lower,
            "eta": eta,
            "residual": lower - eta - q["separation_threshold_m"],
            "left_distance": distances[0],
            "right_distance": distances[1],
            "lipschitz": lipschitz,
        }
