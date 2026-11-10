"""Plot complete heading menus and report their decision and latency outcomes."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from geometric_safety.util import heading_diff
from paper.benchmark_contracts import finite_number, integer, object_json, object_value, parse_list, string
from paper.run_heading_study import load_protocol
from paper.summarise_heading_study import menus, read_outcomes


def plot(directory: Path) -> None:
    plt.switch_backend("Agg")
    outcomes = read_outcomes(directory)
    menu_rows = menus(directory, outcomes)
    protocol = object_json((directory / "protocol.json").read_text())
    _, reference_settings = load_protocol(directory / "protocol.json")
    groups = {
        string(group["id"]): parse_list(group["pairs"], string)
        for group in parse_list(protocol["multi_traffic_scenes"], object_value)
    }
    palette = ["#cf666b", "#d6d9de", "#edbb64", "#348878"]
    labels = ["Reference conflict", "Reference unresolved", "Reference separated; withheld", "Library accepted"]
    widths = sorted({row.half_width_kt for row in menu_rows})
    headings = sorted({row.case.target_deg for row in outcomes})
    horizon_values = sorted({row.horizon_s for row in menu_rows})
    scenes = sorted({row.scene for row in menu_rows})
    if len(horizon_values) != 2 or len(scenes) != 2:
        raise ValueError("This figure layout requires the two scenes and horizons in the study protocol")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True, sharey=True, layout="constrained")
    for row_index, horizon in enumerate(horizon_values):
        for column, scene in enumerate(scenes):
            ax = axes[row_index, column]
            grid = np.zeros((len(widths), len(headings)), dtype=np.int64)
            for width_index, width in enumerate(widths):
                menu = next(
                    item
                    for item in menu_rows
                    if item.scene == scene and item.horizon_s == horizon and item.half_width_kt == width
                )
                for index, heading in enumerate(headings):
                    if heading in menu.variable_headings:
                        if heading not in menu.reference_separated_headings:
                            raise ValueError("An accepted heading lacks separated reference coverage")
                        category = 3
                    elif heading in menu.reference_separated_headings:
                        category = 2
                    elif heading in menu.reference_unresolved_headings:
                        category = 1
                    else:
                        category = 0
                    grid[width_index, index] = category
            ax.pcolormesh(
                np.arange(len(headings) + 1) - 0.5,
                np.arange(len(widths) + 1) - 0.5,
                grid,
                cmap=ListedColormap(palette),
                vmin=0,
                vmax=3,
                edgecolors="white",
                linewidth=1.5,
            )
            turning_traffic = any(
                heading_diff(item.case.query.b.heading_deg, item.case.query.b.target_deg) != 0
                for item in outcomes
                if item.case.scene in groups[scene]
            )
            title = "Includes turning traffic" if turning_traffic else "Traffic holds heading"
            ax.set_title(f"{scene}: {title} | {horizon:g} s", loc="left", fontsize=12, pad=12)
            ax.set_xticks(range(0, len(headings), 2), [f"{heading:03.0f}" for heading in headings[::2]])
            ax.set_yticks(range(len(widths)), [f"{width:g}" for width in widths])
            if row_index == 0 and column == 0:
                ax.invert_yaxis()
            ax.tick_params(length=0, labelsize=10, pad=6)
            ax.spines[:].set_visible(False)
            if column == 0:
                ax.set_ylabel("Speed half-width (kt)", labelpad=10)
            if row_index == 1:
                ax.set_xlabel("A target heading (deg)", labelpad=10)
    fig.suptitle("Heading choices against the supplied traffic", fontsize=16)
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(palette, labels, strict=True)],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / f"heading-menus.{extension}", dpi=180)
    plt.close(fig)

    lines = [
        (directory / "report.md").read_text().rstrip(),
        "",
        "## Complete heading menus",
        "",
        f"Each menu contains {len(headings)} candidate headings checked against its listed traffic aircraft. "
        "A candidate is accepted only when every pair check passes. "
        "The figure distinguishes a witnessed conflict from a library decision to withhold an instruction.",
        "",
        *(
            f"- {scene}: {', '.join(pairs)} ({len(headings) * len(pairs)} pair checks)"
            for scene, pairs in groups.items()
        ),
        "",
        "![Complete heading menus](heading-menus.png)",
        "",
        "| Scene | Horizon (s) | Half-width (kt) | Nominal accepted | Constant accepted | Varying accepted | "
        "Varying reference separated | Reference unresolved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {menu.scene} | {menu.horizon_s:g} | {menu.half_width_kt:g} | {len(menu.nominal_headings)} | "
        f"{len(menu.constant_headings)} | {len(menu.variable_headings)} | "
        f"{len(menu.reference_separated_headings)} | {len(menu.reference_unresolved_headings)} |"
        for menu in menu_rows
    )
    affected = [menu for menu in menu_rows if set(menu.reference_separated_headings) - set(menu.variable_headings)]
    empty = [menu for menu in affected if not menu.variable_headings]
    lines.extend(
        [
            "",
            f"The library withholds reference-separated headings in {len(affected)} of the {len(menu_rows)} menus. "
            f"In {len(empty)} menus this leaves no accepted heading "
            "despite at least one reference-separated candidate.",
            "",
            "## Effects of the execution model",
            "",
            "| Half-width (kt) | Nominal accepted / constant reference conflict | "
            "Nominal accepted / varying reference conflict |",
            "| --- | --- | --- |",
        ]
    )
    for width in widths:
        selected = [row for row in outcomes if row.case.width_kt == width]
        lines.append(
            f"| {width:g} | {sum(row.nominal and row.constant_reference == 'conflict' for row in selected)} | "
            f"{sum(row.nominal and row.variable_reference == 'conflict' for row in selected)} |"
        )
    effect_scenes = {
        row.case.identifier: row.case.scene
        for row in outcomes
        if row.constant_reference == "separated" and row.variable_reference == "conflict"
    }
    lines.append("")
    if effect_scenes:
        lines.extend(
            [
                "The following queries are separated for every admissible constant speed, "
                "but have an attainable varying-speed conflict. "
                "The distances below are bounds and witnessed distances, not exact minima.",
                "",
                "| Scene | Query | Constant lower bound (NM) | Varying witnessed distance (NM) | Witness time (s) |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for line in (directory / "results.jsonl").read_text().splitlines():
            record = object_json(line)
            identifier = string(object_value(record["case"])["identifier"])
            if identifier in effect_scenes:
                constant = object_value(object_value(record["constant_reference"])["recheck"])
                variable = object_value(object_value(record["variable_reference"])["recheck"])
                witness = object_value(variable["witness"])
                lines.append(
                    f"| {effect_scenes[identifier]} | {identifier} | {float(string(constant['lower_m'])) / 1852:.5f} | "
                    f"{float(string(variable['upper_m'])) / 1852:.5f} | {float(string(witness['time_s'])):.3f} |"
                )
    else:
        lines.append("No query has constant-speed separation and an attainable varying-speed conflict.")
    timing = object_json((directory / "timings.json").read_text())
    timing_rows = parse_list(timing["menus"], object_value)
    cold = object_json((directory / "cold-timing.json").read_text())
    affinity = ", ".join(str(int(value)) for value in parse_list(timing["cpu_affinity"], finite_number))
    machine = string(timing["machine"]) if "machine" in timing else "the recorded study host"
    timing_settings = object_value(protocol["timing"])
    rounds = integer(timing_settings["rounds"])
    repetitions = rounds * integer(timing_settings["individually_timed_repetitions_per_round"])
    if any(len(parse_list(row["samples_ms"], finite_number)) != repetitions for row in timing_rows):
        raise ValueError("Timing sample count differs from the protocol")
    lines.extend(
        [
            "",
            "## Cost and vertical progress",
            "",
            "Timing covers every public-API call and result handling per complete menu, with no early exit. "
            "Reusable aircraft declarations are prepared before timing. "
            f"Each menu has {repetitions} individually timed repetitions in {rounds} rounds; menu order is seeded. "
            f"Measurements use CPU affinity {affinity} on {machine}.",
            "",
            f"Menu medians span {min(finite_number(row['median_ms']) for row in timing_rows):.3f} to "
            f"{max(finite_number(row['median_ms']) for row in timing_rows):.3f} ms. "
            "The largest per-menu 95th percentile is "
            f"{max(finite_number(row['p95_ms']) for row in timing_rows):.3f} ms. "
            f"The first menu with an empty compilation cache takes "
            f"{finite_number(cold['first_complete_menu_ms']) / 1000:.2f} s, excluding imports and preparation. "
            "Individual samples, machine details and system load are retained alongside the results.",
            "",
            "| Minimum vertical rate (ft/min) | Lateral horizon (s) | Combined decision |",
            "| --- | --- | --- |",
        ]
    )
    control = object_value(object_value(protocol["vertical_example"])["analytical_control"])
    distance_nm = string(control["minimum_lateral_distance_at_resolution_nm"])
    distance_m = float(Fraction(distance_nm)) * 1852
    lines.extend(
        f"| {finite_number(row['minimum_rate_fpm']):g} | {finite_number(row['lateral_horizon_s']):g} | "
        f"{string(row['status'])} |"
        for row in parse_list(json.loads((directory / "vertical-example.json").read_text()), object_value)
    )
    lines.extend(
        [
            "",
            f"At {finite_number(control['rate_fpm']):g} ft/min, the analytical control develops vertical separation "
            f"after {finite_number(control['vertical_resolution_time_s']):g} s. Its minimum lateral distance up to "
            f"that time is {distance_nm} NM ({distance_m:,.6f} m).",
            "",
            "## Scope",
            "",
            "These are fixed synthetic geometries, not a sample of operational traffic. "
            "Counts across headings, widths and horizons are correlated. Varying speeds have no acceleration "
            "constraint and both turns follow prescribed angular rates. Reference separation uses continuous-time "
            "bounds; conflict uses actual admissible profiles checked by quadrature. "
            f"Decisive classifications agree at {reference_settings.digits} and {reference_settings.recheck_digits} "
            "digits, but this is not a formal floating-point proof.",
            "",
        ]
    )
    (directory / "study-results.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot(args.directory)


if __name__ == "__main__":
    main()
