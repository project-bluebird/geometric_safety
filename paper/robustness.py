"""Perturbed-truth simulation and deviation envelopes for the robustness study.

The manuscript's robustness section derives, for each bounded out-of-model
perturbation class, a deviation envelope ``eta(t)`` such that certifying against the
inflated threshold ``D + eta(t)`` restores the one-sided safety guarantee. This module
provides the paper-pipeline side of that story: converting perturbation bounds into
solver envelope parameters, simulating perturbed truth trajectories, and measuring
whether certified-safe decisions ever coincide with a true loss of separation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety.util import DEG_TO_RAD, KT_TO_MPS, heading_diff
from paper.evaluation import Encounter

TRUTH_COARSE_DT_S = 0.25
TRUTH_REFINED_DT_S = 0.05
TRUTH_REFINE_WINDOW_S = 6.0
SCHEDULE_SEGMENTS = 12
OU_SPEED_STEP_S = 2.0


@dataclass(frozen=True)
class TruthModelConfig:
    """Configuration of the truth simulator beyond the stated perturbation classes.

    With all fields at their defaults the truth simulator draws from exactly the
    classes the envelopes cover, so certified decisions cannot be violated and the
    experiment verifies the implementation. The remaining fields move the truth
    outside every stated class, which is where validation happens.

    ``exceedance_factor`` scales the truth perturbation magnitudes relative to the
    bounds the certificate inflates for. In-turn speed schedules stay clipped to the
    admissible interval regardless, since excursions beyond the interval belong to
    the interval-widening remedy rather than the schedule class.

    ``ou_speed_sigma_kt`` enables mean-reverting speed drift over the whole horizon,
    clipped to the admissible interval. This violates the constant-speed assumption
    everywhere, not only during turns. ``roll_time_s`` replaces the instant-rate turn
    with a trapezoidal rate profile that ramps in and out over the given time.
    ``heading_jitter_deg`` adds piecewise-constant track-keeping error, redrawn every
    ``heading_jitter_period_s`` seconds.
    """

    exceedance_factor: float = 1.0
    ou_speed_sigma_kt: float = 0.0
    ou_speed_timescale_s: float = 60.0
    roll_time_s: float = 0.0
    heading_jitter_deg: float = 0.0
    heading_jitter_period_s: float = 20.0

    def is_in_class(self) -> bool:
        return (
            self.exceedance_factor <= 1.0
            and self.ou_speed_sigma_kt <= 0.0
            and self.roll_time_s <= 0.0
            and self.heading_jitter_deg <= 0.0
        )


@dataclass(frozen=True)
class PerturbationBounds:
    """Bounds defining the perturbation classes active in one robustness experiment.

    Each field matches one class in the manuscript's robustness section. A zero field
    disables its class. ``heading_error_deg`` applies per aircraft with the
    rotated-profile convention, and ``turn_delay_max_s`` / ``turn_rate_error_max``
    apply per aircraft with the absolute-target convention.
    """

    differential_wind_kt: float = 0.0
    position_error_m: float = 0.0
    heading_error_deg: float = 0.0
    turn_delay_max_s: float = 0.0
    turn_rate_error_max: float = 0.0
    schedule_uncertainty_kt: float = 0.0

    def is_null(self) -> bool:
        return all(value <= 0.0 for value in asdict(self).values())

    def active_classes(self) -> list[str]:
        return [name for name, value in asdict(self).items() if value > 0.0]


def solver_envelope_kwargs(encounter: Encounter, bounds: PerturbationBounds) -> dict[str, float]:
    """Convert perturbation bounds into the solver's deviation-envelope parameters.

    The wind bound contributes its magnitude to the linear rate term directly. A
    heading-estimate bound of ``eps`` radians per aircraft contributes
    ``s_max * eps`` per aircraft, using each aircraft's maximum admissible speed.
    """
    a_max_speed_mps = (encounter.a_speed_kt + encounter.speed_diff_kt) * KT_TO_MPS
    b_max_speed_mps = (encounter.b_speed_kt + encounter.speed_diff_kt) * KT_TO_MPS
    heading_error_rad = bounds.heading_error_deg * DEG_TO_RAD
    eta_rate_mps = bounds.differential_wind_kt * KT_TO_MPS + heading_error_rad * (
        a_max_speed_mps + b_max_speed_mps
    )
    return {
        "eta_const_m": bounds.position_error_m,
        "eta_rate_mps": eta_rate_mps,
        "turn_delay_max_s": bounds.turn_delay_max_s,
        "turn_rate_error_max": bounds.turn_rate_error_max,
        "turn_speed_schedule_uncertainty_kt": bounds.schedule_uncertainty_kt,
    }


def deviation_envelope_m(encounter: Encounter, bounds: PerturbationBounds, t_s: float) -> float:
    """Evaluate the total deviation envelope eta(t) for the configured bounds."""
    params = solver_envelope_kwargs(encounter, bounds)
    return float(
        ra._deviation_envelope_m(
            t_s=t_s,
            eta_const_m=params["eta_const_m"],
            eta_rate_mps=params["eta_rate_mps"],
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            a_min_speed_mps=max(encounter.a_speed_kt - encounter.speed_diff_kt, 0.0) * KT_TO_MPS,
            a_max_speed_mps=(encounter.a_speed_kt + encounter.speed_diff_kt) * KT_TO_MPS,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            b_min_speed_mps=max(encounter.b_speed_kt - encounter.speed_diff_kt, 0.0) * KT_TO_MPS,
            b_max_speed_mps=(encounter.b_speed_kt + encounter.speed_diff_kt) * KT_TO_MPS,
            turn_speed_schedule_uncertainty_kt=params["turn_speed_schedule_uncertainty_kt"],
            turn_delay_max_s=params["turn_delay_max_s"],
            turn_rate_error_max=params["turn_rate_error_max"],
        )
    )


@dataclass(frozen=True)
class _AircraftDraw:
    """One sampled perturbed realisation of a single aircraft."""

    constant_speed_mps: float
    heading_error_rad: float
    turn_delay_s: float
    turn_rate_scale: float
    schedule_times_s: np.ndarray
    schedule_speeds_mps: np.ndarray
    roll_time_s: float
    jitter_times_s: np.ndarray
    jitter_offsets_rad: np.ndarray


def _ou_speed_path_mps(
    rng: np.random.Generator,
    *,
    constant_speed_mps: float,
    min_speed_mps: float,
    max_speed_mps: float,
    sigma_kt: float,
    timescale_s: float,
    horizon_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a mean-reverting speed path over the whole horizon on a coarse grid."""
    n_steps = max(round(horizon_s / OU_SPEED_STEP_S), 1)
    edges = np.linspace(0.0, horizon_s, n_steps + 1)
    sigma_mps = sigma_kt * KT_TO_MPS
    theta = 1.0 / max(timescale_s, 1e-6)
    values = np.empty(n_steps, dtype=np.float64)
    level = constant_speed_mps
    for idx in range(n_steps):
        drift = theta * (constant_speed_mps - level) * OU_SPEED_STEP_S
        diffusion = sigma_mps * math.sqrt(2.0 * theta * OU_SPEED_STEP_S) * float(rng.standard_normal())
        level = level + drift + diffusion
        level = min(max(level, min_speed_mps), max_speed_mps)
        values[idx] = level
    return edges, values


def _sample_aircraft_draw(
    rng: np.random.Generator,
    *,
    speed_kt: float,
    speed_diff_kt: float,
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    bounds: PerturbationBounds,
    adversarial: bool,
    truth: TruthModelConfig,
    horizon_s: float,
) -> _AircraftDraw:
    """Sample one aircraft's realised speed and perturbation values.

    The exceedance factor scales the truth magnitudes relative to the stated bounds.
    In-turn schedules stay clipped to the admissible interval at any factor.
    """
    factor = truth.exceedance_factor
    min_speed_mps = max(speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS
    max_speed_mps = (speed_kt + speed_diff_kt) * KT_TO_MPS
    if adversarial:
        constant_speed_mps = float(rng.choice(np.asarray([min_speed_mps, max_speed_mps])))
    else:
        constant_speed_mps = float(rng.uniform(min_speed_mps, max_speed_mps))

    heading_error_rad = 0.0
    if bounds.heading_error_deg > 0.0:
        bound_rad = bounds.heading_error_deg * DEG_TO_RAD * factor
        if adversarial:
            heading_error_rad = float(rng.choice(np.asarray([-bound_rad, bound_rad])))
        else:
            heading_error_rad = float(rng.uniform(-bound_rad, bound_rad))

    turn_delay_s = 0.0
    if bounds.turn_delay_max_s > 0.0:
        limit_s = bounds.turn_delay_max_s * factor
        turn_delay_s = float(limit_s if adversarial else rng.uniform(0.0, limit_s))

    turn_rate_scale = 1.0
    if bounds.turn_rate_error_max > 0.0:
        bound = min(bounds.turn_rate_error_max * factor, 0.95)
        if adversarial:
            turn_rate_scale = float(1.0 + rng.choice(np.asarray([-bound, bound])))
        else:
            turn_rate_scale = float(1.0 + rng.uniform(-bound, bound))

    turn_angle_deg = abs(heading_diff(heading0_deg, target_heading_deg))
    is_turning = turn_angle_deg >= ra.TURN_HEADING_EPS_DEG and abs(turn_rate_deg_sec) > ra.TURN_RATE_EPS_DEG_PER_S

    if truth.ou_speed_sigma_kt > 0.0:
        schedule_times_s, schedule_speeds_mps = _ou_speed_path_mps(
            rng,
            constant_speed_mps=constant_speed_mps,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            sigma_kt=truth.ou_speed_sigma_kt,
            timescale_s=truth.ou_speed_timescale_s,
            horizon_s=horizon_s,
        )
    elif bounds.schedule_uncertainty_kt > 0.0 and is_turning:
        executed_rate = abs(turn_rate_deg_sec) * turn_rate_scale
        turn_duration_s = turn_angle_deg / executed_rate
        edges = np.linspace(turn_delay_s, turn_delay_s + turn_duration_s, SCHEDULE_SEGMENTS + 1)
        delta_mps = bounds.schedule_uncertainty_kt * KT_TO_MPS * factor
        low = max(constant_speed_mps - delta_mps, min_speed_mps)
        high = min(constant_speed_mps + delta_mps, max_speed_mps)
        if adversarial:
            values = rng.choice(np.asarray([low, high]), size=SCHEDULE_SEGMENTS)
        else:
            values = rng.uniform(low, high, size=SCHEDULE_SEGMENTS)
        schedule_times_s = edges
        schedule_speeds_mps = np.asarray(values, dtype=np.float64)
    else:
        schedule_times_s = np.empty(0, dtype=np.float64)
        schedule_speeds_mps = np.empty(0, dtype=np.float64)

    if truth.heading_jitter_deg > 0.0:
        n_segments = max(round(horizon_s / max(truth.heading_jitter_period_s, 1.0)), 1)
        jitter_times_s = np.linspace(0.0, horizon_s, n_segments + 1)
        jitter_bound_rad = truth.heading_jitter_deg * DEG_TO_RAD
        jitter_offsets_rad = rng.uniform(-jitter_bound_rad, jitter_bound_rad, size=n_segments)
    else:
        jitter_times_s = np.empty(0, dtype=np.float64)
        jitter_offsets_rad = np.empty(0, dtype=np.float64)

    return _AircraftDraw(
        constant_speed_mps=constant_speed_mps,
        heading_error_rad=heading_error_rad,
        turn_delay_s=turn_delay_s,
        turn_rate_scale=turn_rate_scale,
        schedule_times_s=schedule_times_s,
        schedule_speeds_mps=schedule_speeds_mps,
        roll_time_s=truth.roll_time_s if is_turning else 0.0,
        jitter_times_s=jitter_times_s,
        jitter_offsets_rad=jitter_offsets_rad,
    )


def _perturbed_displacement(
    times_s: np.ndarray,
    *,
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    draw: _AircraftDraw,
) -> np.ndarray:
    """Integrate one aircraft's perturbed displacement on a uniform time grid.

    The executed heading profile applies the sampled delay and rate scale toward the
    commanded absolute target, the sampled speed schedule applies inside the executed
    turn window, and the whole path is rotated by the sampled heading error. Midpoint
    integration on the grid keeps the discretisation error far below envelope scale.
    """
    dt_s = float(times_s[1] - times_s[0]) if times_s.size > 1 else 0.0
    mid_times = times_s[:-1] + 0.5 * dt_s

    turn_angle_deg = heading_diff(heading0_deg, target_heading_deg)
    is_turning = abs(turn_angle_deg) >= ra.TURN_HEADING_EPS_DEG and abs(turn_rate_deg_sec) > ra.TURN_RATE_EPS_DEG_PER_S
    if is_turning:
        executed_rate_deg_s = abs(turn_rate_deg_sec) * draw.turn_rate_scale
        if draw.roll_time_s > 0.0:
            # Trapezoidal rate transient: the rate ramps in over roll_time_s and the
            # cumulative angle is clamped at the commanded turn angle. The ramp-in lag
            # is the out-of-class element; the clamp approximates the roll-out.
            ramp = np.clip((mid_times - draw.turn_delay_s) / draw.roll_time_s, 0.0, 1.0)
            cumulative_angle_deg = np.minimum(
                np.cumsum(executed_rate_deg_s * ramp * dt_s), abs(turn_angle_deg)
            )
            headings_deg = heading0_deg + math.copysign(1.0, turn_angle_deg) * cumulative_angle_deg
        else:
            turn_duration_s = abs(turn_angle_deg) / executed_rate_deg_s
            progress_s = np.clip(mid_times - draw.turn_delay_s, 0.0, turn_duration_s)
            headings_deg = heading0_deg + math.copysign(1.0, turn_angle_deg) * executed_rate_deg_s * progress_s
    else:
        headings_deg = np.full(mid_times.shape, heading0_deg)

    headings_rad = headings_deg * DEG_TO_RAD + draw.heading_error_rad
    if draw.jitter_times_s.size:
        segment_idx = np.clip(
            np.searchsorted(draw.jitter_times_s, mid_times, side="right") - 1,
            0,
            draw.jitter_offsets_rad.size - 1,
        )
        headings_rad = headings_rad + draw.jitter_offsets_rad[segment_idx]

    speeds_mps = np.full(mid_times.shape, draw.constant_speed_mps)
    if draw.schedule_times_s.size:
        in_window = (mid_times >= draw.schedule_times_s[0]) & (mid_times < draw.schedule_times_s[-1])
        segment_idx = np.clip(
            np.searchsorted(draw.schedule_times_s, mid_times[in_window], side="right") - 1,
            0,
            draw.schedule_speeds_mps.size - 1,
        )
        speeds_mps[in_window] = draw.schedule_speeds_mps[segment_idx]

    step_east = speeds_mps * np.sin(headings_rad) * dt_s
    step_north = speeds_mps * np.cos(headings_rad) * dt_s
    east = np.concatenate(([0.0], np.cumsum(step_east)))
    north = np.concatenate(([0.0], np.cumsum(step_north)))
    return np.stack([east, north], axis=1)


def simulate_truth_min_separation(
    encounter: Encounter,
    bounds: PerturbationBounds,
    rng: np.random.Generator,
    *,
    adversarial: bool = True,
    truth: TruthModelConfig | None = None,
) -> float:
    """Simulate one perturbed truth realisation and return its minimum separation.

    The relative trajectory composes the sampled per-aircraft executed paths, a
    constant differential wind at the full bound in a random direction, and an
    initial relative-position offset at the full bound in a random direction. A
    coarse pass locates the minimum and a refined pass sharpens it. The optional
    truth-model configuration moves the simulation outside the stated classes.
    """
    if truth is None:
        truth = TruthModelConfig()
    a_draw = _sample_aircraft_draw(
        rng,
        speed_kt=encounter.a_speed_kt,
        speed_diff_kt=encounter.speed_diff_kt,
        heading0_deg=encounter.a_heading0_deg,
        target_heading_deg=encounter.a_target_heading_deg,
        turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        bounds=bounds,
        adversarial=adversarial,
        truth=truth,
        horizon_s=encounter.projection_time_s,
    )
    b_draw = _sample_aircraft_draw(
        rng,
        speed_kt=encounter.b_speed_kt,
        speed_diff_kt=encounter.speed_diff_kt,
        heading0_deg=encounter.b_heading0_deg,
        target_heading_deg=encounter.b_target_heading_deg,
        turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        bounds=bounds,
        adversarial=adversarial,
        truth=truth,
        horizon_s=encounter.projection_time_s,
    )

    wind_vector_mps = np.zeros(2)
    if bounds.differential_wind_kt > 0.0:
        wind_dir = rng.uniform(0.0, 2.0 * math.pi)
        magnitude = bounds.differential_wind_kt * KT_TO_MPS * truth.exceedance_factor
        if not adversarial:
            magnitude *= float(rng.uniform(0.0, 1.0))
        wind_vector_mps = magnitude * np.array([math.sin(wind_dir), math.cos(wind_dir)])

    offset_m = np.zeros(2)
    if bounds.position_error_m > 0.0:
        offset_dir = rng.uniform(0.0, 2.0 * math.pi)
        magnitude = bounds.position_error_m * truth.exceedance_factor
        if not adversarial:
            magnitude *= float(rng.uniform(0.0, 1.0))
        offset_m = magnitude * np.array([math.sin(offset_dir), math.cos(offset_dir)])

    def min_separation_on(times_s: np.ndarray) -> tuple[float, float]:
        a_disp = _perturbed_displacement(
            times_s,
            heading0_deg=encounter.a_heading0_deg,
            target_heading_deg=encounter.a_target_heading_deg,
            turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            draw=a_draw,
        )
        b_disp = _perturbed_displacement(
            times_s,
            heading0_deg=encounter.b_heading0_deg,
            target_heading_deg=encounter.b_target_heading_deg,
            turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            draw=b_draw,
        )
        rel = encounter.rel_pos0()[None, :] + offset_m[None, :] + a_disp - b_disp
        rel += wind_vector_mps[None, :] * times_s[:, None]
        distances = np.hypot(rel[:, 0], rel[:, 1])
        idx = int(np.argmin(distances))
        return float(distances[idx]), float(times_s[idx])

    coarse_n = max(round(encounter.projection_time_s / TRUTH_COARSE_DT_S), 1)
    coarse_times = np.linspace(0.0, encounter.projection_time_s, coarse_n + 1)
    coarse_min, coarse_argmin_s = min_separation_on(coarse_times)

    window_lo = max(coarse_argmin_s - TRUTH_REFINE_WINDOW_S, 0.0)
    window_hi = min(coarse_argmin_s + TRUTH_REFINE_WINDOW_S, encounter.projection_time_s)
    refined_n = max(round((window_hi - window_lo) / TRUTH_REFINED_DT_S), 1)
    refined_times = np.linspace(window_lo, window_hi, refined_n + 1)
    refined_min, _ = min_separation_on(refined_times)

    return min(coarse_min, refined_min)


@dataclass(frozen=True)
class RobustnessCaseResult:
    """Violation statistics for one encounter under one perturbation configuration."""

    certified_safe: bool
    inflated_certified_safe: bool
    n_sims: int
    truth_min_separation_m: float
    n_below_threshold: int
    n_violations: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_robustness_case(
    encounter: Encounter,
    bounds: PerturbationBounds,
    *,
    n_sims: int,
    rng: np.random.Generator,
    ref_lat: float,
    ref_lon: float,
    adversarial: bool = True,
    truth: TruthModelConfig | None = None,
) -> RobustnessCaseResult:
    """Evaluate one encounter: certify with and without inflation, then attack it.

    A violation is a truth realisation whose minimum separation drops below the
    threshold on an encounter the inflated certificate declared safe. For truth drawn
    from the stated classes the manuscript's propositions require a zero count, so
    that configuration verifies the implementation. Out-of-class or exceedance truth
    configurations can produce genuine violations, which is where validation happens.
    """
    from paper.evaluation import encounter_to_turn_kwargs

    kwargs = encounter_to_turn_kwargs(encounter, ref_lat, ref_lon)
    plain = ra.catch_up_projection_interval_with_turns(**kwargs)
    inflated = ra.catch_up_projection_interval_with_turns(**kwargs, **solver_envelope_kwargs(encounter, bounds))

    truth_min = math.inf
    n_below = 0
    for _ in range(n_sims):
        sim_min = simulate_truth_min_separation(encounter, bounds, rng, adversarial=adversarial, truth=truth)
        truth_min = min(truth_min, sim_min)
        if sim_min < encounter.separation_threshold_m:
            n_below += 1

    n_violations = n_below if bool(inflated[0]) else 0
    return RobustnessCaseResult(
        certified_safe=bool(plain[0]),
        inflated_certified_safe=bool(inflated[0]),
        n_sims=n_sims,
        truth_min_separation_m=truth_min,
        n_below_threshold=n_below,
        n_violations=n_violations,
    )
