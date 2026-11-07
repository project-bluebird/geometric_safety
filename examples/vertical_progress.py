"""Use vertical shortening only when a minimum progress rate is established."""

from geometric_safety import (
    CombinedHeadingAssuranceResult,
    HeadingAssuranceResult,
    HeadingMotion,
    certify_heading_separation,
    certify_heading_separation_with_vertical,
)
from geometric_safety.util import KT_TO_MPS

SPEED_MIN_KT = 190.0
SPEED_MAX_KT = 210.0


def assess_encounter(
    minimum_vertical_rate_fpm: float | None,
) -> CombinedHeadingAssuranceResult | HeadingAssuranceResult:
    """Use None when the caller cannot guarantee immediate vertical progress."""
    a = HeadingMotion(
        heading0_deg=270.0,
        target_heading_deg=270.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=SPEED_MIN_KT * KT_TO_MPS,
        speed_max_mps=SPEED_MAX_KT * KT_TO_MPS,
    )
    b = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=SPEED_MIN_KT * KT_TO_MPS,
        speed_max_mps=SPEED_MAX_KT * KT_TO_MPS,
    )
    if minimum_vertical_rate_fpm is None:
        return certify_heading_separation(
            relative_position_m=(60000.0, 0.0),
            a=a,
            b=b,
            separation_threshold_m=5.0 * 1852.0,
            horizon_s=600.0,
        )
    return certify_heading_separation_with_vertical(
        relative_position_m=(60000.0, 0.0),
        a=a,
        b=b,
        a_current_fl=100.0,
        a_selected_fl=100.0,
        b_current_fl=100.0,
        b_selected_fl=120.0,
        minimum_vertical_rate_fpm=minimum_vertical_rate_fpm,
        separation_threshold_m=5.0 * 1852.0,
        horizon_s=600.0,
        required_vertical_gap_fl=10.0,
    )


def main() -> None:
    for label, minimum_rate in (("guaranteed_progress", 1000.0), ("no_progress_guarantee", None)):
        result = assess_encounter(minimum_rate)
        horizon_s = result.lateral_horizon_s if isinstance(result, CombinedHeadingAssuranceResult) else 600.0
        print(f"{label}: {result.status}, lateral horizon={horizon_s:.0f} s")


if __name__ == "__main__":
    main()
