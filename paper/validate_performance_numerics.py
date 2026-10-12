"""Recheck optimised candidate directions using the preserved independent reference."""

import argparse
import json
from pathlib import Path

from geometric_safety import relevant_aircraft as ra
from paper import numerical_validation, turn_numerical_reference, validate_directional_production
from paper.benchmark_library_versions import digest
from paper.validate_directional_production import NumericalResult


class PerformanceNumericalResult(NumericalResult):
    source_sha256: dict[str, str]
    guard_m: float
    scope: str
    driver_sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = validate_directional_production.numerical_checks(512, 2026092031)
    report = PerformanceNumericalResult(
        **metrics,
        source_sha256={
            str(path.relative_to(path.parent.parent)): digest(path)
            for path in (
                Path(ra.__file__),
                Path(ra.__file__).with_name("util.py"),
                Path(turn_numerical_reference.__file__),
                Path(validate_directional_production.__file__),
                Path(numerical_validation.__file__),
            )
        },
        guard_m=ra.TURN_CERT_FP_GUARD_M,
        scope="Empirical comparison against an 80-digit reference, with selected 120-digit rechecks.",
        driver_sha256=digest(Path(__file__)),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
