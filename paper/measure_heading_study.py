"""Measure complete public-API menus and the protocol's vertical example."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

import geometric_safety
from geometric_safety import HeadingMotion, certify_heading_separation, certify_heading_separation_with_vertical
from paper.benchmark_contracts import (
    finite_number,
    integer,
    module_hashes,
    object_json,
    object_value,
    parse_list,
    string,
)
from paper.run_heading_study import StudyCase, heading_motion, load_protocol


@dataclass(frozen=True)
class PreparedPair:
    case: StudyCase
    a: HeadingMotion
    b: HeadingMotion


@dataclass(frozen=True)
class PreparedMenu:
    name: str
    pairs: tuple[PreparedPair, ...]
    expected_headings: tuple[float, ...]


def prepare(directory: Path) -> list[PreparedMenu]:
    cases, _ = load_protocol(directory / "protocol.json")
    protocol = object_json((directory / "protocol.json").read_text())
    summary = object_json((directory / "summary.json").read_text())
    groups = {
        string(group["id"]): set(parse_list(group["pairs"], string))
        for group in parse_list(protocol["multi_traffic_scenes"], object_value)
    }
    menus: list[PreparedMenu] = []
    for row in parse_list(summary["menus"], object_value):
        scene, horizon, width = (
            string(row["scene"]),
            finite_number(row["horizon_s"]),
            finite_number(row["half_width_kt"]),
        )
        selected = sorted(
            (
                case
                for case in cases
                if case.scene in groups[scene] and case.query.horizon_s == horizon and case.width_kt == width
            ),
            key=lambda case: (case.target_deg, case.scene),
        )
        if len(selected) != integer(protocol["pair_checks_per_complete_menu"]):
            raise ValueError("Unexpected complete-menu workload")
        pairs = tuple(
            PreparedPair(case, heading_motion(case.query.a), heading_motion(case.query.b)) for case in selected
        )
        menus.append(
            PreparedMenu(
                f"{scene}_t{horizon:g}_w{width:g}", pairs, tuple(parse_list(row["variable_headings"], finite_number))
            )
        )
    if len(menus) != integer(protocol["complete_menu_count"]):
        raise ValueError("Unexpected menu count")
    return menus


def evaluate_menu(menu: PreparedMenu) -> tuple[float, ...]:
    accepted: dict[float, bool] = {}
    for pair in menu.pairs:
        query = pair.case.query
        status = certify_heading_separation(
            query.relative_position_m,
            pair.a,
            pair.b,
            separation_threshold_m=query.threshold_m,
            horizon_s=query.horizon_s,
        ).status
        heading = pair.case.target_deg
        accepted[heading] = accepted.get(heading, True) and status == "certified_safe"
    return tuple(heading for heading, permitted in accepted.items() if permitted)


def vertical_example(directory: Path) -> list[dict[str, float | str | None]]:
    protocol = object_json((directory / "protocol.json").read_text())
    case = object_value(protocol["vertical_example"])
    own = parse_list(case["ownship_position_nm"], finite_number)
    traffic = parse_list(case["traffic_position_nm"], finite_number)
    position = ((own[0] - traffic[0]) * 1852, (own[1] - traffic[1]) * 1852)
    speed, width = finite_number(case["nominal_speed_kt"]), finite_number(case["speed_half_width_kt"])
    motions = [
        HeadingMotion(
            heading0_deg=finite_number(case[key]),
            target_heading_deg=finite_number(case[key]),
            turn_rate_deg_s=0.0,
            speed_min_mps=(speed - width) * (1852 / 3600),
            speed_max_mps=(speed + width) * (1852 / 3600),
        )
        for key in ("ownship_heading_deg", "traffic_heading_deg")
    ]
    rows: list[dict[str, float | str | None]] = []
    for rate in parse_list(case["minimum_vertical_rates_fpm"], finite_number):
        result = certify_heading_separation_with_vertical(
            position,
            motions[0],
            motions[1],
            a_current_fl=finite_number(case["ownship_current_fl"]),
            a_selected_fl=finite_number(case["ownship_selected_fl"]),
            b_current_fl=finite_number(case["traffic_current_fl"]),
            b_selected_fl=finite_number(case["traffic_selected_fl"]),
            minimum_vertical_rate_fpm=rate,
            separation_threshold_m=1852 * finite_number(case["separation_threshold_nm"]),
            horizon_s=finite_number(case["horizon_s"]),
            required_vertical_gap_fl=finite_number(case["required_vertical_gap_fl"]),
            vertical_rounding_fl=integer(case["vertical_rounding_step_fl"]),
        )
        rows.append(
            {
                "minimum_rate_fpm": rate,
                "status": result.status,
                "vertical_resolution_s": result.vertical_resolution_time_s,
                "lateral_horizon_s": result.lateral_horizon_s,
                "base_min_distance_m": None
                if result.lateral_result is None
                else result.lateral_result.base_min_distance_m,
            }
        )
    control = object_value(case["analytical_control"])
    row = next(row for row in rows if row["minimum_rate_fpm"] == finite_number(control["rate_fpm"]))
    expected_distance = float(Fraction(string(control["minimum_lateral_distance_at_resolution_nm"]))) * 1852
    if row["status"] != "certified_safe" or row["lateral_horizon_s"] != finite_number(
        control["vertical_resolution_time_s"]
    ):
        raise AssertionError("The vertical analytical control failed")
    distance = row["base_min_distance_m"]
    if not isinstance(distance, float) or abs(distance - expected_distance) > 1e-6:
        raise AssertionError("The vertical example differs from its analytical distance")
    return rows


@dataclass(frozen=True)
class TimingRow:
    menu: str
    median_ms: float
    p95_ms: float
    samples_ms: list[float]


def measure(directory: Path, cold: bool) -> None:
    package = Path(geometric_safety.__file__).resolve().parent
    manifest = object_json((directory / "manifest.json").read_text())
    expected_package = object_value(manifest["package_sha256"])
    if module_hashes(package) != expected_package:
        raise ValueError("Timing package differs from the evaluated study package")
    menus = prepare(directory)
    if cold:
        cache = os.environ.get("NUMBA_CACHE_DIR")
        if cache is None or any(Path(cache).rglob("*.nb[ci]")):
            raise ValueError("Cold timing requires a fresh Numba cache; invoke with --cold")
        start = time.perf_counter_ns()
        result = evaluate_menu(menus[0])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if result != menus[0].expected_headings:
            raise AssertionError("Cold-menu result differs from the fixed study")
        if module_hashes(package) != expected_package:
            raise RuntimeError("Package changed during cold timing")
        (directory / "cold-timing.json").write_text(
            json.dumps(
                {
                    "first_complete_menu_ms": elapsed,
                    "menu": menus[0].name,
                    "scope": "First API menu with an empty Numba cache; excludes imports and input preparation",
                    "machine": platform.node(),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "package_sha256": expected_package,
                },
                indent=2,
            )
            + "\n"
        )
        return
    protocol = object_json((directory / "protocol.json").read_text())
    settings = object_value(protocol["timing"])
    for menu in menus:
        if evaluate_menu(menu) != menu.expected_headings:
            raise AssertionError("Warmed menu differs from the fixed study")
    timings: dict[str, list[float]] = {menu.name: [] for menu in menus}
    rng = random.Random(integer(settings["order_seed"]))
    before = os.getloadavg()
    for _ in range(integer(settings["rounds"])):
        order = list(menus)
        rng.shuffle(order)
        for menu in order:
            for _ in range(integer(settings["individually_timed_repetitions_per_round"])):
                start = time.perf_counter_ns()
                result = evaluate_menu(menu)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if result != menu.expected_headings:
                    raise AssertionError("Timed menu differs from the fixed study")
                timings[menu.name].append(elapsed)
    rows = [
        TimingRow(name, float(np.median(values)), float(np.quantile(values, 0.95)), values)
        for name, values in timings.items()
    ]
    if module_hashes(package) != expected_package:
        raise RuntimeError("Package changed during timing")
    (directory / "timings.json").write_text(
        json.dumps(
            {
                "menus": [asdict(row) for row in rows],
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "load_before": before,
                "load_after": os.getloadavg(),
                "machine": platform.node(),
                "package_sha256": expected_package,
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "vertical-example.json").write_text(json.dumps(vertical_example(directory), indent=2) + "\n")
    print(
        json.dumps(
            {
                "minimum_menu_median_ms": min(row.median_ms for row in rows),
                "maximum_menu_median_ms": max(row.median_ms for row in rows),
                "maximum_menu_p95_ms": max(row.p95_ms for row in rows),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--cold-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold:
        with TemporaryDirectory(prefix="heading-study-cold-") as cache:
            environment = os.environ.copy()
            environment["NUMBA_CACHE_DIR"] = cache
            subprocess.run(
                [sys.executable, "-m", "paper.measure_heading_study", str(args.directory.resolve()), "--cold-worker"],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=True,
            )
    else:
        measure(args.directory, args.cold_worker)


if __name__ == "__main__":
    main()
