"""Check candidate heading instructions against a simultaneous traffic snapshot."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math

from geometric_safety import HeadingAssuranceResult, HeadingMotion, certify_heading_separation_from_latlon
from geometric_safety.util import KT_TO_MPS

CANDIDATE_HEADINGS_DEG = tuple(range(0, 91, 10))


@dataclass(frozen=True)
class AircraftSnapshot:
    """A position and declared motion starting at one simulation time."""

    timestamp_s: float
    latitude_deg: float
    longitude_deg: float
    motion: HeadingMotion


def assess_traffic(
    ownship: AircraftSnapshot,
    traffic: Mapping[str, AircraftSnapshot],
    *,
    separation_threshold_m: float,
    horizon_s: float,
) -> dict[str, HeadingAssuranceResult]:
    """Keep a result for every supplied pair; reject unaligned snapshots."""
    if not math.isfinite(ownship.timestamp_s) or any(
        aircraft.timestamp_s != ownship.timestamp_s for aircraft in traffic.values()
    ):
        raise ValueError("All aircraft must refer to the same finite starting time")
    return {
        identifier: certify_heading_separation_from_latlon(
            ownship.motion,
            aircraft.motion,
            a_lat=ownship.latitude_deg,
            a_lon=ownship.longitude_deg,
            b_lat=aircraft.latitude_deg,
            b_lon=aircraft.longitude_deg,
            separation_threshold_m=separation_threshold_m,
            horizon_s=horizon_s,
        )
        for identifier, aircraft in traffic.items()
    }


def passing_candidates(
    assessments: Mapping[float, Mapping[str, HeadingAssuranceResult]],
) -> tuple[float, ...]:
    """List candidates with pair checks that all permit acceptance."""
    return tuple(
        heading
        for heading, pairs in assessments.items()
        if pairs and all(result.status == "certified_safe" for result in pairs.values())
    )


def example_snapshot() -> tuple[AircraftSnapshot, dict[str, AircraftSnapshot]]:
    """Construct synthetic positions and bounds, without measurement errors."""
    east = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=3.0,
        speed_min_mps=190.0 * KT_TO_MPS,
        speed_max_mps=210.0 * KT_TO_MPS,
    )
    ownship = AircraftSnapshot(1000.0, 51.0, -1.40, east)
    traffic = {
        "opposing": AircraftSnapshot(1000.0, 51.0, -1.0, replace(east, heading0_deg=270.0, target_heading_deg=270.0)),
        "parallel": AircraftSnapshot(1000.0, 51.35, -1.40, east),
    }
    return ownship, traffic


def assess_candidates(
    ownship: AircraftSnapshot,
    traffic: Mapping[str, AircraftSnapshot],
    *,
    target_headings_deg: Iterable[float],
    separation_threshold_m: float,
    horizon_s: float,
) -> dict[float, dict[str, HeadingAssuranceResult]]:
    """Evaluate each alternative independently against the same traffic."""
    return {
        heading: assess_traffic(
            replace(ownship, motion=replace(ownship.motion, target_heading_deg=heading)),
            traffic,
            separation_threshold_m=separation_threshold_m,
            horizon_s=horizon_s,
        )
        for heading in target_headings_deg
    }


def run_example() -> dict[float, dict[str, HeadingAssuranceResult]]:
    """Check the demonstration snapshot over the next three minutes."""
    ownship, traffic = example_snapshot()
    return assess_candidates(
        ownship,
        traffic,
        target_headings_deg=CANDIDATE_HEADINGS_DEG,
        separation_threshold_m=5.0 * 1852.0,
        horizon_s=180.0,
    )


def main() -> None:
    assessments = run_example()
    for heading, pairs in assessments.items():
        statuses = ", ".join(f"{identifier}={result.status}" for identifier, result in pairs.items())
        print(f"{heading:03.0f}°: {statuses}")
    print("Passing headings:", ", ".join(f"{heading:03.0f}°" for heading in passing_candidates(assessments)))


if __name__ == "__main__":
    main()
