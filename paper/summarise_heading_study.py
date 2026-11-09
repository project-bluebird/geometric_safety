"""Summarise every frozen query, retaining numerical-reference uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from paper.benchmark_contracts import boolean, digest, integer, object_json, object_value, parse_list, string
from paper.run_heading_study import StudyCase, load_protocol


@dataclass(frozen=True)
class Outcome:
    case: StudyCase
    nominal: bool
    constant: bool
    variable: bool
    constant_reference: str
    variable_reference: str


def read_outcomes(directory: Path) -> list[Outcome]:
    complete = object_json((directory / "complete.json").read_text())
    if digest(directory / "results.jsonl") != string(complete["results_sha256"]):
        raise ValueError("Results do not match the completed study hash")
    cases, _ = load_protocol(directory / "protocol.json")
    lookup = {case.identifier: case for case in cases}
    outcomes: list[Outcome] = []
    seen: set[str] = set()
    for line in (directory / "results.jsonl").read_text().splitlines():
        row = object_json(line)
        stored_case = object_value(row["case"])
        identifier = string(stored_case["identifier"])
        if identifier in seen:
            raise ValueError("Duplicate study case")
        seen.add(identifier)
        case = lookup[identifier]
        # JSON represents the coordinate tuple as a list.
        if stored_case != object_json(json.dumps(asdict(case))):
            raise ValueError("Saved inputs differ from the protocol")
        statuses = [
            string(object_value(row[key])["classification"]) for key in ("constant_reference", "variable_reference")
        ]
        if any(status not in {"separated", "conflict", "unresolved"} for status in statuses):
            raise ValueError("Unknown reference classification")
        variable_status = string(row["variable_status"])
        if variable_status not in {"certified_safe", "unresolved"}:
            raise ValueError("Unexpected library status for a supported study case")
        outcomes.append(
            Outcome(
                case,
                boolean(row["nominal_accepted"]),
                boolean(row["constant_accepted"]),
                variable_status == "certified_safe",
                statuses[0],
                statuses[1],
            )
        )
    if seen != set(lookup) or len(outcomes) != integer(complete["count"]):
        raise ValueError("Incomplete study results")
    return outcomes


@dataclass(frozen=True)
class WidthSummary:
    half_width_kt: float
    queries: int
    nominal_acceptances: int
    constant_acceptances: int
    variable_acceptances: int
    constant_reference_separated: int
    variable_reference_separated: int
    constant_reference_unresolved: int
    variable_reference_unresolved: int
    constant_separated_variable_conflict: int
    variable_separated_library_withheld: int


@dataclass(frozen=True)
class Menu:
    scene: str
    horizon_s: float
    half_width_kt: float
    nominal_headings: list[float]
    constant_headings: list[float]
    variable_headings: list[float]
    reference_separated_headings: list[float]
    reference_conflicting_headings: list[float]
    reference_unresolved_headings: list[float]


def menus(directory: Path, outcomes: list[Outcome]) -> list[Menu]:
    protocol = object_json((directory / "protocol.json").read_text())
    result: list[Menu] = []
    for scene in parse_list(protocol["multi_traffic_scenes"], object_value):
        pairs = set(parse_list(scene["pairs"], string))
        for horizon in sorted({item.case.query.horizon_s for item in outcomes}):
            for width in sorted({item.case.width_kt for item in outcomes}):
                rows = [
                    item
                    for item in outcomes
                    if item.case.scene in pairs and item.case.query.horizon_s == horizon and item.case.width_kt == width
                ]
                nominal: list[float] = []
                constant: list[float] = []
                variable: list[float] = []
                separated: list[float] = []
                conflicting: list[float] = []
                unresolved: list[float] = []
                for heading in sorted({item.case.target_deg for item in rows}):
                    subset = [item for item in rows if item.case.target_deg == heading]
                    if {item.case.scene for item in subset} != pairs:
                        raise ValueError("Incomplete traffic menu")
                    if all(item.nominal for item in subset):
                        nominal.append(heading)
                    if all(item.constant for item in subset):
                        constant.append(heading)
                    if all(item.variable for item in subset):
                        variable.append(heading)
                    if all(item.variable_reference == "separated" for item in subset):
                        separated.append(heading)
                    elif any(item.variable_reference == "conflict" for item in subset):
                        conflicting.append(heading)
                    else:
                        unresolved.append(heading)
                result.append(
                    Menu(
                        string(scene["id"]),
                        horizon,
                        width,
                        nominal,
                        constant,
                        variable,
                        separated,
                        conflicting,
                        unresolved,
                    )
                )
    return result


def summarise(directory: Path) -> dict[str, WidthSummary]:
    outcomes = read_outcomes(directory)
    by_width: dict[str, WidthSummary] = {}
    for width in sorted({item.case.width_kt for item in outcomes}):
        rows = [item for item in outcomes if item.case.width_kt == width]
        by_width[f"{width:g}"] = WidthSummary(
            width,
            len(rows),
            sum(item.nominal for item in rows),
            sum(item.constant for item in rows),
            sum(item.variable for item in rows),
            sum(item.constant_reference == "separated" for item in rows),
            sum(item.variable_reference == "separated" for item in rows),
            sum(item.constant_reference == "unresolved" for item in rows),
            sum(item.variable_reference == "unresolved" for item in rows),
            sum(item.constant_reference == "separated" and item.variable_reference == "conflict" for item in rows),
            sum(item.variable_reference == "separated" and not item.variable for item in rows),
        )
    if any(item.variable and item.variable_reference == "conflict" for item in outcomes):
        raise ValueError("An accepted varying-speed query has a conflicting reference")
    if any(item.constant_reference == "conflict" and item.variable_reference == "separated" for item in outcomes):
        raise ValueError("Reference classifications contradict model containment")
    data = {
        "by_speed_half_width": {key: asdict(value) for key, value in by_width.items()},
        "menus": [asdict(menu) for menu in menus(directory, outcomes)],
        "speed_variation_effect_cases": [
            item.case.identifier
            for item in outcomes
            if item.constant_reference == "separated" and item.variable_reference == "conflict"
        ],
        "conservative_withholding_cases": [
            item.case.identifier for item in outcomes if item.variable_reference == "separated" and not item.variable
        ],
        "reference_unresolved_cases": [
            item.case.identifier
            for item in outcomes
            if "unresolved" in (item.constant_reference, item.variable_reference)
        ],
    }
    (directory / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Fixed heading-screening study",
        "",
        "All 912 predeclared candidate queries were evaluated. Counts are per query, "
        "not independent operational encounters. Reference classifications require a 25 m exclusion band "
        "and decisive outcomes are checked at 80 and 120 digits. "
        "The reference is numerical, not validated interval arithmetic.",
        "",
        "| Speed half-width (kt) | Queries | Nominal accepted | Constant accepted | Varying accepted | "
        "Varying reference separated | Varying reference unresolved | Constant separated / varying conflict | "
        "Reference separated / library withheld |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.half_width_kt:g} | {row.queries} | {row.nominal_acceptances} | {row.constant_acceptances} | "
        f"{row.variable_acceptances} | {row.variable_reference_separated} | {row.variable_reference_unresolved} | "
        f"{row.constant_separated_variable_conflict} | {row.variable_separated_library_withheld} |"
        for row in by_width.values()
    )
    lines.extend(
        [
            "",
            "The last two columns distinguish witnessed execution-class differences from library conservatism. "
            "Unresolved references establish neither. Complete traffic menus and all case identifiers are in "
            "summary.json; results.jsonl retains the individual brackets and attainable profiles.",
            "",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines))
    return by_width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = summarise(args.directory)
    print(json.dumps({key: asdict(value) for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
