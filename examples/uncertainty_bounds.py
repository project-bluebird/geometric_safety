"""Translate per-aircraft error bounds into a relative separation query."""

from geometric_safety import HeadingAssuranceResult, HeadingMotion, certify_heading_separation
from geometric_safety.util import KT_TO_MPS


def run_example() -> dict[str, HeadingAssuranceResult]:
    """Compare two declared position-error budgets for parallel aircraft."""
    a = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=190.0 * KT_TO_MPS,
        speed_max_mps=210.0 * KT_TO_MPS,
    )
    b = HeadingMotion(
        heading0_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=0.0,
        speed_min_mps=195.0 * KT_TO_MPS,
        speed_max_mps=215.0 * KT_TO_MPS,
    )
    # These illustrative norm bounds are assumed to hold, not estimated here.
    position_budgets_m = {"bounded_errors": 80.0 + 120.0, "wider_position_bounds": 400.0 + 400.0}
    residual_velocity_bound_mps = (1.0 + 1.0) * KT_TO_MPS
    return {
        name: certify_heading_separation(
            relative_position_m=(0.0, 10000.0),  # A is 10 km north of B.
            a=a,
            b=b,
            separation_threshold_m=5.0 * 1852.0,
            horizon_s=120.0,
            relative_position_error_m=position_bound_m,
            relative_velocity_error_mps=residual_velocity_bound_mps,
        )
        for name, position_bound_m in position_budgets_m.items()
    }


def main() -> None:
    for name, result in run_example().items():
        print(f"{name}: {result.status}, inflation={result.inflation_m:.2f} m")


if __name__ == "__main__":
    main()
