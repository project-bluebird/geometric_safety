"""Translate per-aircraft error bounds into a relative separation query."""

from geometric_safety import HeadingAssuranceResult, HeadingMotion, certify_heading_separation
from geometric_safety.util import KT_TO_MPS

HORIZON_S = 120.0
# Illustrative per-aircraft norm bounds for (A, B), assumed to hold rather than estimated here.
POSITION_ERROR_BOUNDS_M = {"bounded_errors": (80.0, 120.0), "wider_position_bounds": (400.0, 400.0)}
VELOCITY_ERROR_BOUNDS_KT = (1.0, 1.0)
RESIDUAL_VELOCITY_BOUND_MPS = sum(VELOCITY_ERROR_BOUNDS_KT) * KT_TO_MPS


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
    return {
        name: certify_heading_separation(
            relative_position_m=(0.0, 10000.0),  # A is 10 km north of B.
            a=a,
            b=b,
            separation_threshold_m=5.0 * 1852.0,
            horizon_s=HORIZON_S,
            relative_position_error_m=sum(position_bounds_m),
            relative_velocity_error_mps=RESIDUAL_VELOCITY_BOUND_MPS,
        )
        for name, position_bounds_m in POSITION_ERROR_BOUNDS_M.items()
    }


def main() -> None:
    for name, result in run_example().items():
        print(f"{name}: {result.status}, inflation={result.inflation_m:.2f} m")


if __name__ == "__main__":
    main()
