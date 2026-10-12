"""Generate a matched benchmark report from its recorded inputs and outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import TypedDict

from paper.benchmark_library_versions import (
    digest,
    list_value,
    number,
    object_json,
    object_value,
    read_cases,
    statistics_us,
)


class Group(TypedDict):
    count: int
    methods: dict[str, dict[str, float]]
    mean_speedup: dict[str, float]


def summarise(results: Path, cases_path: Path, output: Path) -> None:
    report = object_json(results.read_text())
    if report["cases_sha256"] != digest(cases_path):
        raise ValueError("Benchmark case file hash does not match the recorded input")
    cases = read_cases(cases_path)
    raw = object_value(report["raw_wall_us"])
    medians = {
        method: [statistics.median(number(list_value(row)[i]) for row in list_value(rows)) for i in range(len(cases))]
        for method, rows in raw.items()
    }
    outcomes = {
        method: [bool(v) for v in list_value(values)]
        for worker in object_value(report["workers"]).values()
        for method, values in object_value(object_value(worker)["outcomes"]).items()
    }
    groups: dict[str, Group] = {}
    for suite in ("broad", "targeted", "straight", "circular_clearance", "all"):
        indices = [i for i, case in enumerate(cases) if suite == "all" or case.suite == suite]
        methods = {
            name: {
                **statistics_us([values[i] for i in indices]),
                "accepted": float(sum(outcomes[name][i] for i in indices)),
            }
            for name, values in medians.items()
        }
        groups[suite] = {
            "count": len(indices),
            "methods": methods,
            "mean_speedup": {
                m: methods["before_" + m]["mean"] / methods[m]["mean"]
                for m in ("core_python", "core_compiled", "interface_prebuilt", "interface_construct")
            },
        }
    slow_indices = [int(number(v)) for v in list_value(report["slow_indices"])]
    slow_wall, slow_cpu = object_value(report["slow_raw_wall_us"]), object_value(report["slow_raw_cpu_us"])
    slow_cases = {}
    for column, index in enumerate(slow_indices):
        slow_cases[cases[index].id] = {
            method: {
                **statistics_us([number(list_value(row)[column]) for row in list_value(rows)]),
                "median_cpu": statistics.median(
                    number(list_value(row)[column]) for row in list_value(slow_cpu[method])
                ),
            }
            for method, rows in slow_wall.items()
        }
    compact = {
        "protocol": report["protocol"],
        "groups": groups,
        "checks": report["checks"],
        "slow_summary": report["slow_summary"],
        "slow_case_count": len(slow_indices),
        "slow_cases": slow_cases,
        "results_sha256": digest(results),
        "cases_sha256": digest(cases_path),
        "summariser_sha256": digest(Path(__file__)),
    }
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    protocol = object_value(report["protocol"])
    decision_changes = {
        method: sum(left != right for left, right in zip(outcomes["before_" + method], outcomes[method], strict=True))
        for method in ("core_python", "interface_prebuilt")
    }
    checks = object_value(report["checks"])
    check_lines = []
    for name in (
        "known_collision_repairs_verified",
        "known_separated_controls_verified",
        "no_current_acceptance_of_saved_unsafe_cases",
        "interface_matches_effective_core",
    ):
        verdict = checks.get(name)
        status = "passed" if verdict is True else "failed" if verdict is False else "not recorded"
        check_lines.append(f"{name}: {status}.")
    host = protocol.get("host", "not recorded")
    started_at = protocol.get("started_at_utc", "not recorded")
    thread_settings = json.dumps(protocol.get("single_thread_environment", {}), sort_keys=True)
    labels = {
        "broad": "Broad",
        "targeted": "Targeted",
        "straight": "Straight",
        "circular_clearance": "Near-threshold circles",
        "all": "All inputs",
    }
    lines = [
        "### Matched library benchmark",
        "",
        f"Host: {host}. Started (UTC): {started_at}.",
        f"Results SHA-256: `{digest(results)}`. Cases SHA-256: `{digest(cases_path)}`.",
        "",
        "The table compares the original scalar library, the supplied baseline and",
        "the candidate. Exact package identities and execution details are recorded",
        "in the input results file and copied to summary.json.",
        "",
        f"CPU: {protocol['cpu_model']}. Pinned logical CPU: {protocol.get('cpu', 'not recorded')}.",
        f"Numerical-library thread settings: `{thread_settings}`.",
        str(protocol.get("timed_boundary", "The timed call boundary was not recorded in this input.")),
        "",
        f"Core decisions changed on {decision_changes['core_python']} of {len(cases)} inputs.",
        f"Interface decisions changed on {decision_changes['interface_prebuilt']} of {len(cases)} inputs.",
        *check_lines,
        "",
        "Each table entry is the mean of the per-input median batch times, in",
        f"microseconds per call, using {protocol['rounds']} rounds of {protocol['calls_per_batch']} calls.",
        "",
        "| Inputs | Count | Original scalar | Core before | Core after | Interface before | Interface after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in labels.items():
        group = groups[name]
        values = [
            group["methods"][method]["mean"]
            for method in (
                "old_python",
                "before_core_python",
                "core_python",
                "before_interface_prebuilt",
                "interface_prebuilt",
            )
        ]
        lines.append(f"| {label} | {group['count']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines.extend(
        [
            "",
            "The separate compiled-wrapper call path gives:",
            "",
            "| Inputs | Original scalar | Core before | Core after |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, label in labels.items():
        values = [
            groups[name]["methods"][method]["mean"]
            for method in ("old_compiled", "before_core_compiled", "core_compiled")
        ]
        lines.append(f"| {label} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines.extend(
        [
            "",
            f"The union of each method's ten slowest inputs contained {len(slow_indices)} cases.",
            f"Each received {protocol['tail_rounds']} further individual-call measurements. These are observed",
            "tails on that selected set, in milliseconds, rather than runtime bounds.",
            "",
            "| Method | 95th percentile | 99th percentile | Maximum |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method, label in (
        ("old_python", "Original scalar"),
        ("before_core_python", "Core before"),
        ("core_python", "Core after"),
        ("before_interface_prebuilt", "Interface before"),
        ("interface_prebuilt", "Interface after"),
        ("interface_construct", "Interface after, including objects"),
    ):
        values = object_value(object_value(report["slow_summary"])[method])
        lines.append(
            f"| {label} | " + " | ".join(f"{number(values[k]) / 1000:.3f}" for k in ("p95", "p99", "max")) + " |"
        )
    lines.extend(
        [
            "",
            str(protocol.get("semantics", "Motion assumptions were not recorded in this input.")),
            "",
            "These are synthetic inputs on one shared host. CPU affinity does not",
            "reserve a core, and low observed load does not establish a runtime guarantee.",
            "Both wall and thread-CPU samples and package identities are retained",
            "in the input results file. This report does not establish a runtime guarantee.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "speedup": {name: group["mean_speedup"] for name, group in groups.items()},
                "slow_summary": report["slow_summary"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarise(args.results, args.cases, args.output)


if __name__ == "__main__":
    main()
