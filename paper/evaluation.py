"""Internal experiment helpers for the manuscript evaluation package."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Any

import numba
import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety.util import DEG_TO_RAD, EARTH_RADIUS_IN_METERS, KT_TO_MPS, NMI_TO_M, RAD_TO_DEG

DEFAULT_REF_LAT = 51.0
DEFAULT_REF_LON = -1.0
DEFAULT_THRESHOLD_M = 5.0 * NMI_TO_M
DEFAULT_HORIZONS_S = (300.0, 600.0, 900.0, 1200.0)
DEFAULT_DT_MIN_VALUES_S = (1.0, 3.0, 6.0, 12.0)
COARSE_REFERENCE_DT_S = 0.25
REFINED_REFERENCE_DT_S = 0.05
COARSE_PROXY_DT_S = 6.0
AMBIGUITY_BAND_M = 0.05 * NMI_TO_M
REFINE_NEAR_THRESHOLD_BAND_M = 0.10 * NMI_TO_M
NEAR_THRESHOLD_SAFE_MARGIN_MIN_M = 0.05 * NMI_TO_M
NEAR_THRESHOLD_SAFE_MARGIN_MAX_M = 1.50 * NMI_TO_M
MARGIN_BINS_NMI = (0.05, 0.25, 0.50, 0.75, 1.00, 1.50)
TURN_ANGLE_BINS_DEG = (0.0, 10.0, 20.0, 35.0, 50.0, 60.0)
SPEED_UNCERTAINTY_BINS_KT = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
TURN_ZERO_PROBABILITY = 0.35


@dataclass(frozen=True)
class Encounter:
    """Synthetic encounter in local east/north coordinates."""

    rel_east_m: float
    rel_north_m: float
    a_heading0_deg: float
    a_target_heading_deg: float
    a_speed_kt: float
    a_turn_rate_deg_sec: float
    b_heading0_deg: float
    b_target_heading_deg: float
    b_speed_kt: float
    b_turn_rate_deg_sec: float
    speed_diff_kt: float
    projection_time_s: float
    separation_threshold_m: float = DEFAULT_THRESHOLD_M

    def rel_pos0(self) -> np.ndarray:
        return np.array([self.rel_east_m, self.rel_north_m], dtype=np.float64)

    def max_turn_angle_deg(self) -> float:
        return max(
            abs(_signed_heading_delta_deg(self.a_heading0_deg, self.a_target_heading_deg)),
            abs(_signed_heading_delta_deg(self.b_heading0_deg, self.b_target_heading_deg)),
        )

    def turn_count(self) -> int:
        count = 0
        if abs(_signed_heading_delta_deg(self.a_heading0_deg, self.a_target_heading_deg)) > 1e-9:
            count += 1
        if abs(_signed_heading_delta_deg(self.b_heading0_deg, self.b_target_heading_deg)) > 1e-9:
            count += 1
        return count


@dataclass(frozen=True)
class MethodOutcome:
    """Result for one method on one encounter."""

    is_safe: bool
    min_distance_m: float
    closest_time_s: float
    margin_m: float

    @classmethod
    def from_tuple(cls, result: tuple[bool, float, float], threshold_m: float) -> MethodOutcome:
        return cls(
            is_safe=bool(result[0]),
            min_distance_m=float(result[1]),
            closest_time_s=float(result[2]),
            margin_m=float(result[1]) - threshold_m,
        )


def outcomes_equivalent(
    lhs: MethodOutcome,
    rhs: MethodOutcome,
    *,
    distance_abs_tol_m: float = 1e-6,
    time_abs_tol_s: float = 1e-6,
) -> bool:
    """Return whether two method outcomes agree up to negligible roundoff."""
    return (
        lhs.is_safe == rhs.is_safe
        and math.isclose(lhs.min_distance_m, rhs.min_distance_m, rel_tol=1e-12, abs_tol=distance_abs_tol_m)
        and math.isclose(lhs.margin_m, rhs.margin_m, rel_tol=1e-12, abs_tol=distance_abs_tol_m)
        and math.isclose(lhs.closest_time_s, rhs.closest_time_s, rel_tol=1e-12, abs_tol=time_abs_tol_s)
    )


@dataclass(frozen=True)
class ReferencedEncounter:
    """Encounter bundled with its dense-reference outcome."""

    encounter: Encounter
    reference: MethodOutcome


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level configuration for the paper evaluation script."""

    straight_n: int = 50_000
    mixed_turn_n: int = 20_000
    near_threshold_n: int = 5_000
    seed: int = 20260325
    ref_lat: float = DEFAULT_REF_LAT
    ref_lon: float = DEFAULT_REF_LON
    coarse_reference_dt_s: float = COARSE_REFERENCE_DT_S
    refined_reference_dt_s: float = REFINED_REFERENCE_DT_S
    coarse_proxy_dt_s: float = COARSE_PROXY_DT_S
    ambiguity_band_m: float = AMBIGUITY_BAND_M
    refine_near_threshold_band_m: float = REFINE_NEAR_THRESHOLD_BAND_M
    near_threshold_margin_min_m: float = NEAR_THRESHOLD_SAFE_MARGIN_MIN_M
    near_threshold_margin_max_m: float = NEAR_THRESHOLD_SAFE_MARGIN_MAX_M
    dt_min_values_s: tuple[float, ...] = DEFAULT_DT_MIN_VALUES_S
    turn_zero_probability: float = TURN_ZERO_PROBABILITY
    runtimes_rounds: int = 5
    n_jobs: int = 1
    quick: bool = False

    @classmethod
    def quick_defaults(cls) -> EvaluationConfig:
        return cls(
            straight_n=200,
            mixed_turn_n=120,
            near_threshold_n=24,
            seed=20260325,
            runtimes_rounds=2,
            quick=True,
        )


def local_xy_to_latlon(east_m: float, north_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert local east/north coordinates back to latitude/longitude."""
    ref_lat_rad = ref_lat * DEG_TO_RAD
    lat = ref_lat + (north_m / EARTH_RADIUS_IN_METERS) * RAD_TO_DEG
    lon = ref_lon + (east_m / (EARTH_RADIUS_IN_METERS * math.cos(ref_lat_rad))) * RAD_TO_DEG
    return lat, lon


def encounter_to_turn_kwargs(encounter: Encounter, ref_lat: float, ref_lon: float) -> dict[str, float]:
    """Convert an encounter to the public turn-aware API inputs."""
    b_lat, b_lon = local_xy_to_latlon(-encounter.rel_east_m, -encounter.rel_north_m, ref_lat, ref_lon)
    return {
        "a_lat": ref_lat,
        "a_lon": ref_lon,
        "a_heading0_deg": encounter.a_heading0_deg,
        "a_target_heading_deg": encounter.a_target_heading_deg,
        "a_speed_kt": encounter.a_speed_kt,
        "a_turn_rate_deg_sec": encounter.a_turn_rate_deg_sec,
        "b_lat": b_lat,
        "b_lon": b_lon,
        "b_heading0_deg": encounter.b_heading0_deg,
        "b_target_heading_deg": encounter.b_target_heading_deg,
        "b_speed_kt": encounter.b_speed_kt,
        "b_turn_rate_deg_sec": encounter.b_turn_rate_deg_sec,
        "separation_threshold_m": encounter.separation_threshold_m,
        "speed_diff_kt": encounter.speed_diff_kt,
        "projection_time_s": encounter.projection_time_s,
    }


def encounter_to_row(encounter: Encounter) -> dict[str, Any]:
    """Serialize an encounter into a flat row."""
    row = asdict(encounter)
    row["max_turn_angle_deg"] = encounter.max_turn_angle_deg()
    row["turn_count"] = encounter.turn_count()
    row["initial_range_nmi"] = math.hypot(encounter.rel_east_m, encounter.rel_north_m) / NMI_TO_M
    return row


def generate_straight_suite(n: int, seed: int, threshold_m: float = DEFAULT_THRESHOLD_M) -> list[Encounter]:
    """Generate the straight-heading exactness suite."""
    rng = np.random.default_rng(seed)
    encounters = []
    for _ in range(n):
        rel_east_m, rel_north_m = _sample_relative_position_m(rng, 6.0, 40.0)
        projection_time_s = float(rng.choice(np.asarray(DEFAULT_HORIZONS_S)))
        a_heading0_deg = float(rng.uniform(0.0, 360.0))
        b_heading0_deg = float(rng.uniform(0.0, 360.0))
        encounters.append(
            Encounter(
                rel_east_m=rel_east_m,
                rel_north_m=rel_north_m,
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=a_heading0_deg,
                a_speed_kt=float(rng.uniform(220.0, 450.0)),
                a_turn_rate_deg_sec=0.0,
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=b_heading0_deg,
                b_speed_kt=float(rng.uniform(220.0, 450.0)),
                b_turn_rate_deg_sec=0.0,
                speed_diff_kt=float(rng.uniform(5.0, 30.0)),
                projection_time_s=projection_time_s,
                separation_threshold_m=threshold_m,
            )
        )
    return encounters


def generate_mixed_turn_suite(
    n: int,
    seed: int,
    threshold_m: float = DEFAULT_THRESHOLD_M,
    turn_zero_probability: float = TURN_ZERO_PROBABILITY,
) -> list[Encounter]:
    """Generate the broad mixed-turn suite."""
    rng = np.random.default_rng(seed)
    encounters = []
    for _ in range(n):
        rel_east_m, rel_north_m = _sample_relative_position_m(rng, 6.0, 40.0)
        a_heading0_deg = float(rng.uniform(0.0, 360.0))
        b_heading0_deg = float(rng.uniform(0.0, 360.0))
        a_turn_angle_deg = _sample_turn_angle_deg(rng, turn_zero_probability)
        b_turn_angle_deg = _sample_turn_angle_deg(rng, turn_zero_probability)
        encounters.append(
            Encounter(
                rel_east_m=rel_east_m,
                rel_north_m=rel_north_m,
                a_heading0_deg=a_heading0_deg,
                a_target_heading_deg=(a_heading0_deg + a_turn_angle_deg) % 360.0,
                a_speed_kt=float(rng.uniform(220.0, 450.0)),
                a_turn_rate_deg_sec=0.0 if abs(a_turn_angle_deg) < 1e-9 else float(rng.uniform(1.0, 3.5)),
                b_heading0_deg=b_heading0_deg,
                b_target_heading_deg=(b_heading0_deg + b_turn_angle_deg) % 360.0,
                b_speed_kt=float(rng.uniform(220.0, 450.0)),
                b_turn_rate_deg_sec=0.0 if abs(b_turn_angle_deg) < 1e-9 else float(rng.uniform(1.0, 3.5)),
                speed_diff_kt=float(rng.uniform(5.0, 30.0)),
                projection_time_s=float(rng.choice(np.asarray(DEFAULT_HORIZONS_S))),
                separation_threshold_m=threshold_m,
            )
        )
    return encounters


def generate_near_threshold_suite(
    n: int,
    seed: int,
    config: EvaluationConfig,
) -> list[ReferencedEncounter]:
    """Generate safe turning encounters whose dense-reference margin stays near threshold."""
    if config.n_jobs <= 1 or n <= 1:
        return _generate_near_threshold_chunk(n, seed, config)

    worker_count = min(max(1, int(config.n_jobs)), n)
    worker_targets = _split_counts(n, worker_count)
    rng = np.random.default_rng(seed)
    worker_seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=worker_count)]

    accepted: list[ReferencedEncounter] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_generate_near_threshold_chunk, target_n, worker_seed, config)
            for target_n, worker_seed in zip(worker_targets, worker_seeds, strict=True)
            if target_n > 0
        ]
        for future in futures:
            accepted.extend(future.result())
    return accepted[:n]


def _generate_near_threshold_chunk(
    n: int,
    seed: int,
    config: EvaluationConfig,
) -> list[ReferencedEncounter]:
    """Generate one accepted shard of the near-threshold suite."""
    rng = np.random.default_rng(seed)
    accepted: list[ReferencedEncounter] = []
    batch_size = 64 if config.quick else 512
    while len(accepted) < n:
        batch = generate_mixed_turn_suite(
            n=batch_size,
            seed=int(rng.integers(0, 2**31 - 1)),
            threshold_m=DEFAULT_THRESHOLD_M,
            turn_zero_probability=config.turn_zero_probability,
        )
        for encounter in batch:
            if encounter.turn_count() == 0:
                continue
            reference = evaluate_dense_reference(
                encounter,
                coarse_dt_s=config.coarse_reference_dt_s,
                refined_dt_s=config.refined_reference_dt_s,
                near_threshold_band_m=config.refine_near_threshold_band_m,
            )
            if (
                reference.is_safe
                and config.near_threshold_margin_min_m <= reference.margin_m <= config.near_threshold_margin_max_m
            ):
                accepted.append(ReferencedEncounter(encounter=encounter, reference=reference))
                if len(accepted) >= n:
                    break
    return accepted


def _split_counts(total: int, buckets: int) -> list[int]:
    """Split a target count into nearly equal positive chunks."""
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _split_into_chunks(items: list[Any], n_chunks: int) -> list[list[Any]]:
    """Split a list into *n_chunks* roughly equal sublists."""
    sizes = _split_counts(len(items), n_chunks)
    chunks: list[list[Any]] = []
    offset = 0
    for size in sizes:
        if size > 0:
            chunks.append(items[offset : offset + size])
        offset += size
    return chunks


def sampled_proxy_counterexample() -> Encounter:
    """Return a fixed encounter showing that 6 s uniform sampling remains heuristic."""
    return Encounter(
        rel_east_m=-16375.684015836852,
        rel_north_m=24264.177302164444,
        a_heading0_deg=58.17630503744534,
        a_target_heading_deg=221.1584928588891,
        a_speed_kt=368.8709301454392,
        a_turn_rate_deg_sec=1.821894717845222,
        b_heading0_deg=309.25330925753775,
        b_target_heading_deg=80.2711455347613,
        b_speed_kt=277.9560718689216,
        b_turn_rate_deg_sec=1.2463926724122139,
        speed_diff_kt=17.688910234868644,
        projection_time_s=1200.0,
        separation_threshold_m=DEFAULT_THRESHOLD_M,
    )


def build_sampled_proxy_counterexample_row(
    config: EvaluationConfig,
    *,
    sample_dt_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed sampling-miss counterexample under the current protocol."""
    encounter = sampled_proxy_counterexample()
    coarse_dt_s = config.coarse_proxy_dt_s if sample_dt_s is None else float(sample_dt_s)
    reference = evaluate_dense_reference(
        encounter,
        coarse_dt_s=config.coarse_reference_dt_s,
        refined_dt_s=config.refined_reference_dt_s,
        near_threshold_band_m=config.refine_near_threshold_band_m,
    )
    proposed = evaluate_proposed(encounter)
    nominal = evaluate_nominal_proxy(
        encounter,
        coarse_dt_s=config.coarse_reference_dt_s,
        refined_dt_s=config.refined_reference_dt_s,
        near_threshold_band_m=config.refine_near_threshold_band_m,
    )
    coarse = evaluate_coarse_turn_proxy(encounter, sample_dt_s=coarse_dt_s)
    row = encounter_to_row(encounter)
    row.update(outcome_to_row("reference", reference))
    row.update(outcome_to_row("proposed", proposed))
    row.update(outcome_to_row("nominal_proxy", nominal))
    row.update(outcome_to_row("coarse_proxy", coarse))
    row["coarse_proxy_dt_s"] = coarse_dt_s
    row["eligible_for_boolean_tallies"] = int(abs(reference.margin_m) >= config.ambiguity_band_m)
    return row


def evaluate_straight_oracle(encounter: Encounter) -> MethodOutcome:
    """Evaluate the exact straight-heading oracle."""
    result = ra.catch_up_projection_interval(
        a_lat=0.0,
        a_lon=0.0,
        a_heading=encounter.a_heading0_deg,
        a_speed_kt=encounter.a_speed_kt,
        b_lat=0.0,
        b_lon=0.0,
        b_heading=encounter.b_heading0_deg,
        b_speed_kt=encounter.b_speed_kt,
        separation_threshold_m=encounter.separation_threshold_m,
        speed_diff_kt=encounter.speed_diff_kt,
        projection_time_s=encounter.projection_time_s,
        rel_pos_override=encounter.rel_pos0(),
    )
    return MethodOutcome.from_tuple(result, encounter.separation_threshold_m)


def evaluate_proposed(
    encounter: Encounter,
    *,
    dt_min_s: float = ra.TURN_TIME_CERT_MIN_INTERVAL_S,
    use_interval_local_lipschitz: bool = True,
    turn_speed_schedule_uncertainty_kt: float = 0.0,
    ref_lat: float = DEFAULT_REF_LAT,
    ref_lon: float = DEFAULT_REF_LON,
    use_public_api: bool = False,
) -> MethodOutcome:
    """Evaluate the proposed method either through the public API or the local core."""
    if use_public_api:
        result = ra.catch_up_projection_interval_with_turns(
            **encounter_to_turn_kwargs(encounter, ref_lat, ref_lon),
            use_interval_local_lipschitz=use_interval_local_lipschitz,
            turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
        )
    else:
        result = ra._catch_up_projection_interval_with_turns_local(
            rel_pos0=encounter.rel_pos0(),
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_speed_kt=encounter.a_speed_kt,
            a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_speed_kt=encounter.b_speed_kt,
            b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            separation_threshold_m=encounter.separation_threshold_m,
            speed_diff_kt=encounter.speed_diff_kt,
            projection_time_s=encounter.projection_time_s,
            use_interval_local_lipschitz=use_interval_local_lipschitz,
            turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
            min_cert_interval_s=dt_min_s,
        )
    return MethodOutcome.from_tuple(result, encounter.separation_threshold_m)


def evaluate_dense_reference(
    encounter: Encounter,
    *,
    coarse_dt_s: float = COARSE_REFERENCE_DT_S,
    refined_dt_s: float = REFINED_REFERENCE_DT_S,
    near_threshold_band_m: float = REFINE_NEAR_THRESHOLD_BAND_M,
) -> MethodOutcome:
    """Evaluate the dense bounded-speed reference for the mixed-turn model."""
    coarse_times = build_uniform_times(encounter.projection_time_s, coarse_dt_s)
    coarse_distances = _bounded_distances_at_times(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        speed_diff_kt=encounter.speed_diff_kt,
        times_s=coarse_times,
    )
    best_index = int(np.argmin(coarse_distances))
    best_distance_m = float(coarse_distances[best_index])
    best_time_s = float(coarse_times[best_index])
    refined_times = build_refinement_times(
        coarse_times=coarse_times,
        coarse_distances=coarse_distances,
        projection_time_s=encounter.projection_time_s,
        threshold_m=encounter.separation_threshold_m,
        coarse_dt_s=coarse_dt_s,
        refined_dt_s=refined_dt_s,
        near_threshold_band_m=near_threshold_band_m,
    )
    if refined_times.size > 0:
        refined_distances = _bounded_distances_at_times(
            rel_pos0=encounter.rel_pos0(),
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_speed_kt=encounter.a_speed_kt,
            a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_speed_kt=encounter.b_speed_kt,
            b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            speed_diff_kt=encounter.speed_diff_kt,
            times_s=refined_times,
        )
        refined_index = int(np.argmin(refined_distances))
        refined_distance_m = float(refined_distances[refined_index])
        if refined_distance_m < best_distance_m:
            best_distance_m = refined_distance_m
            best_time_s = float(refined_times[refined_index])
    return MethodOutcome(
        is_safe=best_distance_m >= encounter.separation_threshold_m,
        min_distance_m=best_distance_m,
        closest_time_s=best_time_s,
        margin_m=best_distance_m - encounter.separation_threshold_m,
    )


def evaluate_nominal_proxy(
    encounter: Encounter,
    *,
    coarse_dt_s: float = COARSE_REFERENCE_DT_S,
    refined_dt_s: float = REFINED_REFERENCE_DT_S,
    near_threshold_band_m: float = REFINE_NEAR_THRESHOLD_BAND_M,
) -> MethodOutcome:
    """Evaluate the nominal-speed, no-envelope proxy baseline."""
    if encounter.turn_count() == 0:
        result = ra.catch_up_projection_interval(
            a_lat=0.0,
            a_lon=0.0,
            a_heading=encounter.a_heading0_deg,
            a_speed_kt=encounter.a_speed_kt,
            b_lat=0.0,
            b_lon=0.0,
            b_heading=encounter.b_heading0_deg,
            b_speed_kt=encounter.b_speed_kt,
            separation_threshold_m=encounter.separation_threshold_m,
            speed_diff_kt=0.0,
            projection_time_s=encounter.projection_time_s,
            rel_pos_override=encounter.rel_pos0(),
        )
        return MethodOutcome.from_tuple(result, encounter.separation_threshold_m)

    coarse_times = build_uniform_times(encounter.projection_time_s, coarse_dt_s)
    coarse_distances = _nominal_distances_at_times(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        times_s=coarse_times,
    )
    best_index = int(np.argmin(coarse_distances))
    best_distance_m = float(coarse_distances[best_index])
    best_time_s = float(coarse_times[best_index])
    refined_times = build_refinement_times(
        coarse_times=coarse_times,
        coarse_distances=coarse_distances,
        projection_time_s=encounter.projection_time_s,
        threshold_m=encounter.separation_threshold_m,
        coarse_dt_s=coarse_dt_s,
        refined_dt_s=refined_dt_s,
        near_threshold_band_m=near_threshold_band_m,
    )
    if refined_times.size > 0:
        refined_distances = _nominal_distances_at_times(
            rel_pos0=encounter.rel_pos0(),
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_speed_kt=encounter.a_speed_kt,
            a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_speed_kt=encounter.b_speed_kt,
            b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            times_s=refined_times,
        )
        refined_index = int(np.argmin(refined_distances))
        refined_distance_m = float(refined_distances[refined_index])
        if refined_distance_m < best_distance_m:
            best_distance_m = refined_distance_m
            best_time_s = float(refined_times[refined_index])
    return MethodOutcome(
        is_safe=best_distance_m >= encounter.separation_threshold_m,
        min_distance_m=best_distance_m,
        closest_time_s=best_time_s,
        margin_m=best_distance_m - encounter.separation_threshold_m,
    )


def evaluate_coarse_turn_proxy(encounter: Encounter, *, sample_dt_s: float = COARSE_PROXY_DT_S) -> MethodOutcome:
    """Evaluate the uniformly sampled turn-aware heuristic."""
    times_s = build_uniform_times(encounter.projection_time_s, sample_dt_s)
    distances = _bounded_distances_at_times(
        rel_pos0=encounter.rel_pos0(),
        a_heading0_deg=encounter.a_heading0_deg,
        a_target_heading_deg=encounter.a_target_heading_deg,
        a_speed_kt=encounter.a_speed_kt,
        a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
        b_heading0_deg=encounter.b_heading0_deg,
        b_target_heading_deg=encounter.b_target_heading_deg,
        b_speed_kt=encounter.b_speed_kt,
        b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
        speed_diff_kt=encounter.speed_diff_kt,
        times_s=times_s,
    )
    best_index = int(np.argmin(distances))
    best_distance_m = float(distances[best_index])
    best_time_s = float(times_s[best_index])
    return MethodOutcome(
        is_safe=bool(np.all(distances >= encounter.separation_threshold_m)),
        min_distance_m=best_distance_m,
        closest_time_s=best_time_s,
        margin_m=best_distance_m - encounter.separation_threshold_m,
    )


def count_fixed_time_evals(
    encounter: Encounter,
    *,
    dt_min_s: float,
    use_interval_local_lipschitz: bool,
) -> tuple[int, MethodOutcome]:
    """Count fixed-time hull evaluations using the Python shadow of the local core."""
    py_func = ra._catch_up_projection_interval_with_turns_local.py_func
    globals_dict = py_func.__globals__
    original = globals_dict["_min_distance_to_relative_hull_at_time"]
    counter = {"n": 0}

    def wrapped(*args: object, **kwargs: object) -> object:
        counter["n"] += 1
        return original(*args, **kwargs)

    globals_dict["_min_distance_to_relative_hull_at_time"] = wrapped
    try:
        result = py_func(
            rel_pos0=encounter.rel_pos0(),
            a_heading0_deg=encounter.a_heading0_deg,
            a_target_heading_deg=encounter.a_target_heading_deg,
            a_speed_kt=encounter.a_speed_kt,
            a_turn_rate_deg_sec=encounter.a_turn_rate_deg_sec,
            b_heading0_deg=encounter.b_heading0_deg,
            b_target_heading_deg=encounter.b_target_heading_deg,
            b_speed_kt=encounter.b_speed_kt,
            b_turn_rate_deg_sec=encounter.b_turn_rate_deg_sec,
            separation_threshold_m=encounter.separation_threshold_m,
            speed_diff_kt=encounter.speed_diff_kt,
            projection_time_s=encounter.projection_time_s,
            use_interval_local_lipschitz=use_interval_local_lipschitz,
            turn_speed_schedule_uncertainty_kt=0.0,
            min_cert_interval_s=dt_min_s,
        )
    finally:
        globals_dict["_min_distance_to_relative_hull_at_time"] = original
    return counter["n"], MethodOutcome.from_tuple(result, encounter.separation_threshold_m)


def warm_up_numba(config: EvaluationConfig) -> None:
    """Warm up the main compiled kernels before timing."""
    encounter = generate_mixed_turn_suite(1, config.seed, turn_zero_probability=config.turn_zero_probability)[0]
    evaluate_proposed(encounter, use_public_api=True, ref_lat=config.ref_lat, ref_lon=config.ref_lon)
    evaluate_proposed(encounter, dt_min_s=config.dt_min_values_s[0], use_public_api=False)
    evaluate_dense_reference(encounter)
    evaluate_nominal_proxy(encounter)
    evaluate_coarse_turn_proxy(encounter, sample_dt_s=config.coarse_proxy_dt_s)


def benchmark_fixed_time_kernel(rounds: int = 7, reps: int = 30_000) -> dict[str, float]:
    """Benchmark the fixed-time hull kernel in microseconds."""
    kwargs = {
        "rel_pos0": np.array([5000.0, -7000.0], dtype=np.float64),
        "a_heading0_deg": 10.0,
        "a_target_heading_deg": 60.0,
        "a_turn_rate_deg_sec": 2.5,
        "a_min_speed_mps": 150.0,
        "a_max_speed_mps": 180.0,
        "b_heading0_deg": 240.0,
        "b_target_heading_deg": 210.0,
        "b_turn_rate_deg_sec": 1.5,
        "b_min_speed_mps": 120.0,
        "b_max_speed_mps": 160.0,
        "t_s": 45.0,
    }
    samples_us = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            ra._min_distance_to_relative_hull_at_time(**kwargs)
        samples_us.append((time.perf_counter() - start) * 1e6 / reps)
    return summarize_runtime_samples(samples_us)


def benchmark_solver_runtimes(
    encounters: list[Encounter],
    *,
    ref_lat: float,
    ref_lon: float,
) -> list[float]:
    """Measure compiled solver runtimes for the public API in microseconds."""
    samples_us = []
    for encounter in encounters:
        kwargs = encounter_to_turn_kwargs(encounter, ref_lat, ref_lon)
        start = time.perf_counter()
        ra.catch_up_projection_interval_with_turns(**kwargs)
        samples_us.append((time.perf_counter() - start) * 1e6)
    return samples_us


def _evaluate_straight_chunk(encounters: list[Encounter]) -> list[dict[str, Any]]:
    """Evaluate a chunk of straight encounters, returning row dicts.

    The independent oracle is a separate high-precision implementation with no
    shared geometry code. The turn-aware solver delegates zero-turn cases to
    the production closed form, so the closed form cannot referee itself; the
    independent columns are the ones that validate the implementation.
    """
    from paper.independent_oracle import independent_straight_outcome

    rows = []
    for encounter in encounters:
        oracle = evaluate_straight_oracle(encounter)
        proposed = evaluate_proposed(encounter)
        nominal = evaluate_nominal_proxy(encounter)
        indep_safe, indep_distance_m, indep_time_s = independent_straight_outcome(
            rel_east_m=encounter.rel_east_m,
            rel_north_m=encounter.rel_north_m,
            a_heading_deg=encounter.a_heading0_deg,
            a_speed_kt=encounter.a_speed_kt,
            b_heading_deg=encounter.b_heading0_deg,
            b_speed_kt=encounter.b_speed_kt,
            speed_diff_kt=encounter.speed_diff_kt,
            horizon_s=encounter.projection_time_s,
            threshold_m=encounter.separation_threshold_m,
        )
        row = encounter_to_row(encounter)
        row.update(outcome_to_row("oracle", oracle))
        row.update(outcome_to_row("proposed", proposed))
        row.update(outcome_to_row("nominal_proxy", nominal))
        row["independent_is_safe"] = int(indep_safe)
        row["independent_min_distance_m"] = indep_distance_m
        row["independent_closest_time_s"] = indep_time_s
        rows.append(row)
    return rows


def run_straight_suite(encounters: list[Encounter], *, n_jobs: int = 1) -> dict[str, Any]:
    """Evaluate the straight exactness suite."""
    if n_jobs <= 1:
        rows = _evaluate_straight_chunk(encounters)
    else:
        chunks = _split_into_chunks(encounters, n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            rows = [row for chunk_rows in executor.map(_evaluate_straight_chunk, chunks) for row in chunk_rows]

    oracle_safe = [bool(row["oracle_is_safe"]) for row in rows]
    proposed_safe = [bool(row["proposed_is_safe"]) for row in rows]
    nominal_safe = [bool(row["nominal_proxy_is_safe"]) for row in rows]
    independent_safe = [bool(row["independent_is_safe"]) for row in rows]
    distance_errors = [abs(row["proposed_min_distance_m"] - row["oracle_min_distance_m"]) for row in rows]
    time_errors = [abs(row["proposed_closest_time_s"] - row["oracle_closest_time_s"]) for row in rows]
    indep_distance_errors = [abs(row["proposed_min_distance_m"] - row["independent_min_distance_m"]) for row in rows]
    indep_time_errors = [abs(row["proposed_closest_time_s"] - row["independent_closest_time_s"]) for row in rows]
    oracle_safe_arr = np.asarray(oracle_safe, dtype=bool)
    proposed_safe_arr = np.asarray(proposed_safe, dtype=bool)
    nominal_safe_arr = np.asarray(nominal_safe, dtype=bool)
    independent_safe_arr = np.asarray(independent_safe, dtype=bool)
    summary = {
        "n": len(rows),
        "oracle_vs_proposed_agreement_rate": float(np.mean(oracle_safe_arr == proposed_safe_arr)),
        "max_distance_error_m": float(np.max(distance_errors) if distance_errors else 0.0),
        "max_time_error_s": float(np.max(time_errors) if time_errors else 0.0),
        "independent_vs_proposed_agreement_rate": float(np.mean(independent_safe_arr == proposed_safe_arr)),
        "max_distance_error_vs_independent_m": float(np.max(indep_distance_errors) if indep_distance_errors else 0.0),
        "max_time_error_vs_independent_s": float(np.max(indep_time_errors) if indep_time_errors else 0.0),
        "nominal_proxy": boolean_confusion_summary(
            reference_safe=oracle_safe_arr,
            method_safe=nominal_safe_arr,
            eligible_mask=np.ones_like(oracle_safe_arr, dtype=bool),
        ),
    }
    return {"rows": rows, "summary": summary}


def _evaluate_mixed_turn_chunk(
    args: tuple[list[Encounter], EvaluationConfig],
) -> list[dict[str, Any]]:
    """Evaluate a chunk of mixed-turn encounters, returning row dicts."""
    encounters, config = args
    rows = []
    for encounter in encounters:
        reference = evaluate_dense_reference(
            encounter,
            coarse_dt_s=config.coarse_reference_dt_s,
            refined_dt_s=config.refined_reference_dt_s,
            near_threshold_band_m=config.refine_near_threshold_band_m,
        )
        proposed = evaluate_proposed(encounter)
        nominal = evaluate_nominal_proxy(
            encounter,
            coarse_dt_s=config.coarse_reference_dt_s,
            refined_dt_s=config.refined_reference_dt_s,
            near_threshold_band_m=config.refine_near_threshold_band_m,
        )
        coarse = evaluate_coarse_turn_proxy(encounter, sample_dt_s=config.coarse_proxy_dt_s)
        eligible_case = abs(reference.margin_m) >= config.ambiguity_band_m
        row = encounter_to_row(encounter)
        row.update(outcome_to_row("reference", reference))
        row.update(outcome_to_row("proposed", proposed))
        row.update(outcome_to_row("nominal_proxy", nominal))
        row.update(outcome_to_row("coarse_proxy", coarse))
        row["eligible_for_boolean_tallies"] = int(eligible_case)
        rows.append(row)
    return rows


def run_mixed_turn_suite(
    encounters: list[Encounter],
    *,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate the broad mixed-turn suite."""
    runtime_samples_us = benchmark_solver_runtimes(encounters, ref_lat=config.ref_lat, ref_lon=config.ref_lon)

    if config.n_jobs <= 1:
        rows = _evaluate_mixed_turn_chunk((encounters, config))
    else:
        chunks = _split_into_chunks(encounters, config.n_jobs)
        chunk_args = [(chunk, config) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
            rows = [row for chunk_rows in executor.map(_evaluate_mixed_turn_chunk, chunk_args) for row in chunk_rows]

    for row, runtime_us in zip(rows, runtime_samples_us, strict=True):
        row["runtime_us"] = float(runtime_us)
    reference_safe_arr = np.asarray([bool(row["reference_is_safe"]) for row in rows], dtype=bool)
    proposed_safe_arr = np.asarray([bool(row["proposed_is_safe"]) for row in rows], dtype=bool)
    nominal_safe_arr = np.asarray([bool(row["nominal_proxy_is_safe"]) for row in rows], dtype=bool)
    coarse_safe_arr = np.asarray([bool(row["coarse_proxy_is_safe"]) for row in rows], dtype=bool)
    eligible_arr = np.asarray([bool(row["eligible_for_boolean_tallies"]) for row in rows], dtype=bool)
    dense_safe_mask = eligible_arr & reference_safe_arr
    summary = {
        "n": len(rows),
        "eligible_n": int(np.sum(eligible_arr)),
        "proposed": boolean_confusion_summary(reference_safe_arr, proposed_safe_arr, eligible_arr),
        "nominal_proxy": boolean_confusion_summary(reference_safe_arr, nominal_safe_arr, eligible_arr),
        "coarse_proxy": boolean_confusion_summary(reference_safe_arr, coarse_safe_arr, eligible_arr),
        "certification_rate_given_reference_safe": (
            float(np.mean(proposed_safe_arr[dense_safe_mask])) if np.any(dense_safe_mask) else 0.0
        ),
        "runtime": summarize_runtime_samples(runtime_samples_us),
    }
    return {"rows": rows, "summary": summary}


def _evaluate_near_threshold_chunk(
    args: tuple[list[ReferencedEncounter], EvaluationConfig],
) -> list[dict[str, Any]]:
    """Evaluate a chunk of near-threshold encounters across all ablation settings."""
    referenced_encounters, config = args
    rows: list[dict[str, Any]] = []
    for referenced in referenced_encounters:
        encounter = referenced.encounter
        reference = referenced.reference
        base_row = encounter_to_row(encounter)
        base_row.update(outcome_to_row("reference", reference))
        margin_nmi = reference.margin_m / NMI_TO_M
        for use_local in (True, False):
            for dt_min_s in config.dt_min_values_s:
                start = time.perf_counter()
                outcome = evaluate_proposed(
                    encounter,
                    dt_min_s=dt_min_s,
                    use_interval_local_lipschitz=use_local,
                    use_public_api=False,
                )
                runtime_us = (time.perf_counter() - start) * 1e6
                fixed_time_evals, counted_outcome = count_fixed_time_evals(
                    encounter,
                    dt_min_s=dt_min_s,
                    use_interval_local_lipschitz=use_local,
                )
                if not outcomes_equivalent(outcome, counted_outcome):
                    raise RuntimeError("Instrumented and compiled ablation outcomes diverged.")
                row = dict(base_row)
                row.update(outcome_to_row("proposed", outcome))
                row["lipschitz_mode"] = "local" if use_local else "global"
                row["dt_min_s"] = float(dt_min_s)
                row["fixed_time_evals"] = int(fixed_time_evals)
                row["runtime_us"] = float(runtime_us)
                row["margin_bin_nmi"] = margin_bin_label(margin_nmi, MARGIN_BINS_NMI)
                rows.append(row)
    return rows


def run_near_threshold_suite(
    referenced_encounters: list[ReferencedEncounter],
    *,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate ablations on the near-threshold safe suite."""
    if config.n_jobs <= 1:
        rows = _evaluate_near_threshold_chunk((referenced_encounters, config))
    else:
        chunks = _split_into_chunks(referenced_encounters, config.n_jobs)
        chunk_args = [(chunk, config) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
            rows = [
                row for chunk_rows in executor.map(_evaluate_near_threshold_chunk, chunk_args) for row in chunk_rows
            ]

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["lipschitz_mode"], float(row["dt_min_s"])), []).append(row)
    summary_rows = []
    for (lipschitz_mode, dt_min_s), group_rows in sorted(grouped.items()):
        certification_flags = np.asarray([bool(row["proposed_is_safe"]) for row in group_rows], dtype=bool)
        fixed_time_counts = [int(row["fixed_time_evals"]) for row in group_rows]
        runtimes_us = [float(row["runtime_us"]) for row in group_rows]
        summary_rows.append(
            {
                "lipschitz_mode": lipschitz_mode,
                "dt_min_s": float(dt_min_s),
                "n": len(group_rows),
                "certification_rate": float(np.mean(certification_flags)),
                "median_fixed_time_evals": float(statistics.median(fixed_time_counts)),
                "p95_fixed_time_evals": float(np.percentile(np.asarray(fixed_time_counts, dtype=np.float64), 95.0)),
                "median_runtime_us": float(statistics.median(runtimes_us)),
            }
        )
    default_rows = filter_default_near_threshold_rows(rows)
    margin_summary = summarize_by_margin_bin(rows)
    default_margin_summary = summarize_by_margin_bin(default_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "margin_summary": margin_summary,
        "default_margin_summary": default_margin_summary,
    }


def select_narrative_scenarios(
    straight_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    near_threshold_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Choose representative scenarios for the paper figures."""
    crossing_candidates = [
        row
        for row in straight_rows
        if 70.0 <= abs(_wrapped_angle_difference_deg(row["a_heading0_deg"], row["b_heading0_deg"])) <= 110.0
        and 0.5 <= row["oracle_closest_time_s"] / row["projection_time_s"] <= 0.95
    ]
    safe_crossing_candidates = [
        row
        for row in crossing_candidates
        if row["oracle_is_safe"] == 1 and 0.25 <= row["oracle_margin_m"] / NMI_TO_M <= 2.5
    ]
    if safe_crossing_candidates:
        crossing = min(
            safe_crossing_candidates,
            key=lambda item: abs(item["oracle_margin_m"] / NMI_TO_M - 0.75),
        )
    else:
        crossing = next(
            iter(sorted(crossing_candidates, key=lambda item: item["oracle_margin_m"])),
            straight_rows[0],
        )
    proxy_miss = next(
        (
            row
            for row in mixed_rows
            if row["turn_count"] >= 1 and row["nominal_proxy_is_safe"] == 1 and row["reference_is_safe"] == 0
        ),
        mixed_rows[0],
    )
    dt_sensitive = next(
        (
            row
            for row in near_threshold_rows
            if row["lipschitz_mode"] == "local"
            and row["dt_min_s"] == 12.0
            and row["reference_is_safe"] == 1
            and row["proposed_is_safe"] == 0
        ),
        near_threshold_rows[0],
    )
    return {
        "crossing": crossing,
        "proxy_miss": proxy_miss,
        "dt_sensitive": dt_sensitive,
    }


def boolean_confusion_summary(
    reference_safe: np.ndarray,
    method_safe: np.ndarray,
    eligible_mask: np.ndarray,
) -> dict[str, float]:
    """Summarize boolean agreement against a reference classifier."""
    eligible_reference = reference_safe[eligible_mask]
    eligible_method = method_safe[eligible_mask]
    if eligible_reference.size == 0:
        return {
            "false_safe_rate": 0.0,
            "false_unsafe_rate": 0.0,
            "overall_disagreement_rate": 0.0,
            "eligible_n": 0.0,
        }
    safe_mask = eligible_reference
    unsafe_mask = ~eligible_reference
    false_safe_rate = float(np.mean(eligible_method[unsafe_mask])) if np.any(unsafe_mask) else 0.0
    false_unsafe_rate = float(np.mean(~eligible_method[safe_mask])) if np.any(safe_mask) else 0.0
    return {
        "false_safe_rate": false_safe_rate,
        "false_unsafe_rate": false_unsafe_rate,
        "overall_disagreement_rate": float(np.mean(eligible_reference != eligible_method)),
        "eligible_n": float(eligible_reference.size),
    }


def summarize_runtime_samples(samples_us: list[float]) -> dict[str, float]:
    """Summarize runtime samples in microseconds."""
    values = np.asarray(samples_us, dtype=np.float64)
    if values.size == 0:
        return {"mean_us": 0.0, "median_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    return {
        "mean_us": float(np.mean(values)),
        "median_us": float(np.median(values)),
        "p50_us": float(np.percentile(values, 50.0)),
        "p95_us": float(np.percentile(values, 95.0)),
        "p99_us": float(np.percentile(values, 99.0)),
        "max_us": float(np.max(values)),
    }


def summarize_by_margin_bin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize certification by dense-reference margin bins."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["margin_bin_nmi"]), []).append(row)
    summary = []
    for margin_bin, group_rows in grouped.items():
        certification = np.asarray([bool(row["proposed_is_safe"]) for row in group_rows], dtype=bool)
        summary.append(
            {
                "margin_bin_nmi": margin_bin,
                "n": len(group_rows),
                "certification_rate": float(np.mean(certification)),
            }
        )
    return sorted(summary, key=lambda row: row["margin_bin_nmi"])


def filter_default_near_threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows for the default near-threshold certification setting."""
    return [row for row in rows if row["lipschitz_mode"] == "local" and abs(float(row["dt_min_s"]) - 3.0) <= 1e-12]


def build_uniform_times(projection_time_s: float, dt_s: float) -> np.ndarray:
    """Build a uniform time grid that always includes the final horizon."""
    if projection_time_s <= 0.0:
        return np.asarray([0.0], dtype=np.float64)
    steps = math.ceil(projection_time_s / dt_s)
    times_s = np.empty(steps + 1, dtype=np.float64)
    for i in range(steps):
        times_s[i] = i * dt_s
    times_s[steps] = projection_time_s
    return times_s


def build_refinement_times(
    *,
    coarse_times: np.ndarray,
    coarse_distances: np.ndarray,
    projection_time_s: float,
    threshold_m: float,
    coarse_dt_s: float,
    refined_dt_s: float,
    near_threshold_band_m: float,
) -> np.ndarray:
    """Build refinement times around the coarse minimum and near-threshold neighborhoods."""
    if projection_time_s <= 0.0:
        return np.empty(0, dtype=np.float64)
    best_index = int(np.argmin(coarse_distances))
    marked_indices = {best_index}
    near_threshold = np.flatnonzero(np.abs(coarse_distances - threshold_m) <= near_threshold_band_m)
    for index in near_threshold.tolist():
        marked_indices.add(int(index))
    refined_slots: set[int] = set()
    for index in marked_indices:
        center_t_s = float(coarse_times[index])
        left_t_s = max(0.0, center_t_s - coarse_dt_s)
        right_t_s = min(projection_time_s, center_t_s + coarse_dt_s)
        slot = math.floor(left_t_s / refined_dt_s)
        last_slot = math.ceil(right_t_s / refined_dt_s)
        for refine_slot in range(slot, last_slot + 1):
            refined_slots.add(refine_slot)
    coarse_slots = {round(float(t_s) / refined_dt_s) for t_s in coarse_times.tolist()}
    output = sorted(slot for slot in refined_slots if slot not in coarse_slots)
    if not output:
        return np.empty(0, dtype=np.float64)
    refined_times = np.empty(len(output), dtype=np.float64)
    for index, slot in enumerate(output):
        refined_times[index] = min(projection_time_s, slot * refined_dt_s)
    return refined_times


def outcome_to_row(prefix: str, outcome: MethodOutcome) -> dict[str, Any]:
    """Flatten a method outcome into a row."""
    return {
        f"{prefix}_is_safe": int(outcome.is_safe),
        f"{prefix}_min_distance_m": float(outcome.min_distance_m),
        f"{prefix}_closest_time_s": float(outcome.closest_time_s),
        f"{prefix}_margin_m": float(outcome.margin_m),
    }


def margin_bin_label(margin_nmi: float, bin_edges_nmi: tuple[float, ...]) -> str:
    """Return a printable margin-bin label."""
    left = 0.0
    for right in bin_edges_nmi:
        if left <= margin_nmi < right:
            return f"[{left:.2f}, {right:.2f})"
        left = right
    return f"[{bin_edges_nmi[-1]:.2f}, +inf)"


@numba.njit(cache=True, fastmath=True)
def _bounded_distances_at_times(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    speed_diff_kt: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Evaluate the exact bounded-speed hull distance on a fixed time grid."""
    a_min_speed_mps = max(a_speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS
    a_max_speed_mps = (a_speed_kt + speed_diff_kt) * KT_TO_MPS
    b_min_speed_mps = max(b_speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS
    b_max_speed_mps = (b_speed_kt + speed_diff_kt) * KT_TO_MPS
    output = np.empty(times_s.shape[0], dtype=np.float64)
    for i in range(times_s.shape[0]):
        output[i] = ra._min_distance_to_relative_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=a_heading0_deg,
            a_target_heading_deg=a_target_heading_deg,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min_speed_mps,
            a_max_speed_mps=a_max_speed_mps,
            b_heading0_deg=b_heading0_deg,
            b_target_heading_deg=b_target_heading_deg,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min_speed_mps,
            b_max_speed_mps=b_max_speed_mps,
            t_s=times_s[i],
        )
    return output


@numba.njit(cache=True, fastmath=True)
def _nominal_distances_at_times(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Evaluate the nominal-speed point-trajectory distance on a fixed time grid."""
    a_speed_mps = a_speed_kt * KT_TO_MPS
    b_speed_mps = b_speed_kt * KT_TO_MPS
    output = np.empty(times_s.shape[0], dtype=np.float64)
    for i in range(times_s.shape[0]):
        a_disp = a_speed_mps * ra._turn_displacement_basis(
            a_heading0_deg,
            a_target_heading_deg,
            a_turn_rate_deg_sec,
            times_s[i],
        )
        b_disp = b_speed_mps * ra._turn_displacement_basis(
            b_heading0_deg,
            b_target_heading_deg,
            b_turn_rate_deg_sec,
            times_s[i],
        )
        rel = rel_pos0 + a_disp - b_disp
        output[i] = math.hypot(rel[0], rel[1])
    return output


def _sample_relative_position_m(
    rng: np.random.Generator,
    min_range_nmi: float,
    max_range_nmi: float,
) -> tuple[float, float]:
    range_m = rng.uniform(min_range_nmi, max_range_nmi) * NMI_TO_M
    bearing_rad = rng.uniform(0.0, 2.0 * math.pi)
    b_east_m = range_m * math.sin(bearing_rad)
    b_north_m = range_m * math.cos(bearing_rad)
    return -float(b_east_m), -float(b_north_m)


def _sample_turn_angle_deg(rng: np.random.Generator, zero_probability: float) -> float:
    if float(rng.random()) < zero_probability:
        return 0.0
    return math.copysign(float(rng.uniform(10.0, 60.0)), -1.0 if float(rng.random()) < 0.5 else 1.0)


def _signed_heading_delta_deg(heading0_deg: float, target_heading_deg: float) -> float:
    return ((target_heading_deg - heading0_deg + 540.0) % 360.0) - 180.0


def _wrapped_angle_difference_deg(a_heading_deg: float, b_heading_deg: float) -> float:
    return ((a_heading_deg - b_heading_deg + 540.0) % 360.0) - 180.0
