#!/usr/bin/env python3
"""
Benchmark isolated lateral-solver timings.

Measures warm-path latency of `catch_up_projection_interval_with_turns` across a few
representative scenarios and a random mixed-turn sweep.

Run with:
    uv run --locked python scripts/benchmark_relevant_aircraft.py
    uv run --locked python scripts/benchmark_relevant_aircraft.py --random-cases 5000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import statistics
import time
import typing

import numpy as np

from geometric_safety import relevant_aircraft as ra
from geometric_safety._instrumentation import copy_python_function, counted_python_function
from geometric_safety.util import DEG_TO_RAD, EARTH_RADIUS_IN_METERS, NMI_TO_M, RAD_TO_DEG


@dataclass(frozen=True)
class Scenario:
    """One isolated lateral-solver benchmark case."""

    name: str
    kwargs: TurnAwareKwargs
    reps: int


class TurnAwareKwargs(typing.TypedDict):
    """Keyword arguments for `catch_up_projection_interval_with_turns`."""

    a_lat: float
    a_lon: float
    a_heading0_deg: float
    a_target_heading_deg: float
    a_speed_kt: float
    a_turn_rate_deg_sec: float
    b_lat: float
    b_lon: float
    b_heading0_deg: float
    b_target_heading_deg: float
    b_speed_kt: float
    b_turn_rate_deg_sec: float
    separation_threshold_m: float
    speed_diff_kt: float
    projection_time_s: float


def local_xy_to_latlon(east_m: float, north_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert a local east/north offset back into latitude/longitude."""
    ref_lat_rad = ref_lat * DEG_TO_RAD
    lat = ref_lat + (north_m / EARTH_RADIUS_IN_METERS) * RAD_TO_DEG
    lon = ref_lon + (east_m / (EARTH_RADIUS_IN_METERS * math.cos(ref_lat_rad))) * RAD_TO_DEG
    return lat, lon


def run_turn_aware(kwargs: TurnAwareKwargs) -> tuple[bool, float, float]:
    """Call the compiled turn-aware lateral solver with typed kwargs."""
    return ra.catch_up_projection_interval_with_turns(**kwargs)


def benchmark_microseconds(
    fn: typing.Callable[..., object],
    kwargs: typing.Mapping[str, object],
    reps: int,
    rounds: int = 7,
) -> tuple[float, float, float]:
    """Return mean/median/max time in microseconds for repeated warm calls."""
    samples_us = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn(**kwargs)
        elapsed = time.perf_counter() - start
        samples_us.append(elapsed * 1e6 / reps)
    return statistics.mean(samples_us), statistics.median(samples_us), max(samples_us)


def count_fixed_time_evals(kwargs: TurnAwareKwargs) -> tuple[int, tuple[bool, float, float]]:
    """Count fixed-time samples in the local solver's Python control flow.

    Geometry helpers remain compiled. Counts may differ from the fully compiled
    path near floating-point decision boundaries; benchmark decisions, diagnostics
    and timings are obtained separately from the compiled public solver.
    """
    local_core, counter = counted_python_function(
        ra._catch_up_projection_interval_with_turns_local, "_sample_relative_hull_at_time"
    )
    public = copy_python_function(
        ra.catch_up_projection_interval_with_turns, {"_catch_up_projection_interval_with_turns_local": local_core}
    )
    # FunctionType preserves the production code and defaults but has a dynamic
    # return type. This copy has the same result contract as the public solver.
    result = typing.cast(tuple[bool, float, float], public(**kwargs))
    return counter.calls, result


def build_isolated_scenarios() -> list[Scenario]:
    """Construct representative isolated lateral-solver cases."""
    scenarios: list[Scenario] = [
        Scenario(
            name="straight_delegate",
            reps=20000,
            kwargs={
                "a_lat": 51.0,
                "a_lon": -1.0,
                "a_heading0_deg": 45.0,
                "a_target_heading_deg": 45.0,
                "a_speed_kt": 350.0,
                "a_turn_rate_deg_sec": 0.0,
                "b_lat": 51.03,
                "b_lon": -0.97,
                "b_heading0_deg": 135.0,
                "b_target_heading_deg": 135.0,
                "b_speed_kt": 300.0,
                "b_turn_rate_deg_sec": 0.0,
                "separation_threshold_m": 5.0 * NMI_TO_M,
                "speed_diff_kt": 25.0,
                "projection_time_s": 900.0,
            },
        ),
        Scenario(
            name="one_turn_easy",
            reps=10000,
            kwargs={
                "a_lat": 51.0,
                "a_lon": -1.0,
                "a_heading0_deg": 0.0,
                "a_target_heading_deg": 355.0,
                "a_speed_kt": 350.0,
                "a_turn_rate_deg_sec": 3.0,
                "b_lat": 51.0,
                "b_lon": -0.70,
                "b_heading0_deg": 0.0,
                "b_target_heading_deg": 0.0,
                "b_speed_kt": 350.0,
                "b_turn_rate_deg_sec": 0.0,
                "separation_threshold_m": 5.0 * NMI_TO_M,
                "speed_diff_kt": 15.0,
                "projection_time_s": 600.0,
            },
        ),
    ]

    b_lat, b_lon = local_xy_to_latlon(20240.274442110793, -8855.712362224745, 51.0, -1.0)
    scenarios.append(
        Scenario(
            name="one_turn_near_threshold",
            reps=4000,
            kwargs={
                "a_lat": 51.0,
                "a_lon": -1.0,
                "a_heading0_deg": 180.0,
                "a_target_heading_deg": 180.0,
                "a_speed_kt": 280.0,
                "a_turn_rate_deg_sec": 0.0,
                "b_lat": b_lat,
                "b_lon": b_lon,
                "b_heading0_deg": 315.0,
                "b_target_heading_deg": 270.0,
                "b_speed_kt": 260.0,
                "b_turn_rate_deg_sec": 1.0,
                "separation_threshold_m": 5.0 * NMI_TO_M,
                "speed_diff_kt": 25.0,
                "projection_time_s": 1200.0,
            },
        )
    )

    b_lat, b_lon = local_xy_to_latlon(9003.823495577744, -18483.59249536298, 51.0, -1.0)
    scenarios.append(
        Scenario(
            name="both_turns_near_threshold",
            reps=4000,
            kwargs={
                "a_lat": 51.0,
                "a_lon": -1.0,
                "a_heading0_deg": 60.0,
                "a_target_heading_deg": 90.0,
                "a_speed_kt": 380.0,
                "a_turn_rate_deg_sec": 3.0,
                "b_lat": b_lat,
                "b_lon": b_lon,
                "b_heading0_deg": 0.0,
                "b_target_heading_deg": 315.0,
                "b_speed_kt": 420.0,
                "b_turn_rate_deg_sec": 0.75,
                "separation_threshold_m": 5.0 * NMI_TO_M,
                "speed_diff_kt": 25.0,
                "projection_time_s": 1200.0,
            },
        )
    )

    return scenarios


def run_isolated_benchmarks(random_cases: int) -> None:
    """Print repeatable isolated lateral-solver timing numbers."""
    scenarios = build_isolated_scenarios()

    # Warm the compiled path once before timing.
    for scenario in scenarios:
        run_turn_aware(scenario.kwargs)

    ra._min_distance_to_relative_hull_at_time(
        rel_pos0=np.array([1000.0, -2000.0], dtype=np.float64),
        a_heading0_deg=0.0,
        a_target_heading_deg=30.0,
        a_turn_rate_deg_sec=3.0,
        a_min_speed_mps=150.0,
        a_max_speed_mps=180.0,
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_turn_rate_deg_sec=0.0,
        b_min_speed_mps=140.0,
        b_max_speed_mps=170.0,
        t_s=30.0,
    )

    print("=" * 88)
    print("Isolated Lateral-Solver Benchmarks")
    print("=" * 88)
    print("Representative warm-path timings")
    print("fixed_time_evals_py counts Python-control-flow samples; timings and results use the compiled solver.")
    print("name, safe, min_dist_nmi, closest_time_s, fixed_time_evals_py, mean_us, median_us, max_us")

    for scenario in scenarios:
        fixed_time_evals, _ = count_fixed_time_evals(scenario.kwargs)
        result = run_turn_aware(scenario.kwargs)
        mean_us, median_us, max_us = benchmark_microseconds(
            typing.cast(typing.Callable[..., object], ra.catch_up_projection_interval_with_turns),
            scenario.kwargs,
            scenario.reps,
        )
        print(
            f"{scenario.name}, {result[0]}, {result[1] / NMI_TO_M:.2f}, {result[2]:.1f}, "
            f"{fixed_time_evals}, {mean_us:.2f}, {median_us:.2f}, {max_us:.2f}"
        )

    inner_kwargs: dict[str, object] = {
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
    mean_us, median_us, max_us = benchmark_microseconds(ra._min_distance_to_relative_hull_at_time, inner_kwargs, 30000)
    print("\nInner fixed-time distance evaluation")
    print(f"mean_us={mean_us:.2f}, median_us={median_us:.2f}, max_us={max_us:.2f}")

    rng = np.random.default_rng(12345)
    random_timings_us: list[float] = []
    random_results = {"safe": 0, "unsafe": 0}
    for _ in range(random_cases):
        a_lat = 51.0
        a_lon = -1.0
        east = rng.uniform(-45000.0, 45000.0)
        north = rng.uniform(-45000.0, 45000.0)
        b_lat, b_lon = local_xy_to_latlon(east, north, a_lat, a_lon)

        a_h0 = rng.uniform(0.0, 360.0)
        b_h0 = rng.uniform(0.0, 360.0)
        a_turn = rng.uniform(-60.0, 60.0)
        b_turn = rng.uniform(-60.0, 60.0)
        if rng.random() < 0.35:
            a_turn = 0.0
        if rng.random() < 0.35:
            b_turn = 0.0

        kwargs = {
            "a_lat": a_lat,
            "a_lon": a_lon,
            "a_heading0_deg": a_h0,
            "a_target_heading_deg": (a_h0 + a_turn) % 360.0,
            "a_speed_kt": rng.uniform(220.0, 450.0),
            "a_turn_rate_deg_sec": 0.0 if abs(a_turn) < 1e-9 else rng.uniform(1.0, 3.5),
            "b_lat": b_lat,
            "b_lon": b_lon,
            "b_heading0_deg": b_h0,
            "b_target_heading_deg": (b_h0 + b_turn) % 360.0,
            "b_speed_kt": rng.uniform(220.0, 450.0),
            "b_turn_rate_deg_sec": 0.0 if abs(b_turn) < 1e-9 else rng.uniform(1.0, 3.5),
            "separation_threshold_m": 5.0 * NMI_TO_M,
            "speed_diff_kt": rng.uniform(10.0, 30.0),
            "projection_time_s": float(rng.choice(np.array([300.0, 600.0, 900.0, 1200.0]))),
        }
        start = time.perf_counter()
        is_safe, *_ = run_turn_aware(typing.cast(TurnAwareKwargs, kwargs))
        random_timings_us.append((time.perf_counter() - start) * 1e6)
        random_results["safe" if is_safe else "unsafe"] += 1

    random_timings_us.sort()
    print(f"\nRandom mixed-turn sweep ({random_cases} cases)")
    print(
        f"mean_us={statistics.mean(random_timings_us):.2f}, "
        f"p50_us={random_timings_us[len(random_timings_us) // 2]:.2f}, "
        f"p95_us={random_timings_us[int(0.95 * len(random_timings_us))]:.2f}, "
        f"p99_us={random_timings_us[int(0.99 * len(random_timings_us))]:.2f}, "
        f"max_us={random_timings_us[-1]:.2f}, "
        f"safe={random_results['safe']}, unsafe={random_results['unsafe']}"
    )

    print("\nWorst-case fixed-time evaluations implied by dt_min = 3 s")
    for horizon_s in (300.0, 600.0, 900.0, 1200.0):
        depth = math.ceil(math.log2(horizon_s / ra.TURN_TIME_CERT_MIN_INTERVAL_S))
        print(f"T={horizon_s:.0f}s -> depth={depth}, worst_case_fixed_time_evals={2**depth}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark isolated lateral separation checks.")
    parser.add_argument(
        "--random-cases",
        type=int,
        default=1000,
        help="Number of random mixed-turn cases for the isolated sweep.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    run_isolated_benchmarks(random_cases=args.random_cases)


if __name__ == "__main__":
    main()
