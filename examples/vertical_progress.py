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
HORIZON_S = 600.0
A_LEVEL_FL = 100.0
B_CURRENT_FL = 100.0
B_SELECTED_FL = 120.0
GUARANTEED_RATE_FPM = 1000.0
REQUIRED_VERTICAL_GAP_FL = 10.0


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
            horizon_s=HORIZON_S,
        )
    return certify_heading_separation_with_vertical(
        relative_position_m=(60000.0, 0.0),
        a=a,
        b=b,
        a_current_fl=A_LEVEL_FL,
        a_selected_fl=A_LEVEL_FL,
        b_current_fl=B_CURRENT_FL,
        b_selected_fl=B_SELECTED_FL,
        minimum_vertical_rate_fpm=minimum_vertical_rate_fpm,
        separation_threshold_m=5.0 * 1852.0,
        horizon_s=HORIZON_S,
        required_vertical_gap_fl=REQUIRED_VERTICAL_GAP_FL,
    )


def main() -> None:
    for label, minimum_rate in (("guaranteed_progress", GUARANTEED_RATE_FPM), ("no_progress_guarantee", None)):
        result = assess_encounter(minimum_rate)
        horizon_s = result.lateral_horizon_s if isinstance(result, CombinedHeadingAssuranceResult) else HORIZON_S
        print(f"{label}: {result.status}, lateral horizon={horizon_s:.0f} s")


if __name__ == "__main__":
    main()
