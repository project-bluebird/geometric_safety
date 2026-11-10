"""Time the exact nine-instruction sets used in the ten-minute paper study."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
from typing import TypedDict

import geometric_safety
from geometric_safety import HeadingMotion, certify_heading_separation
from paper.aviation_2027_abstract.compare_models import (
    DATA,
    REPOSITORY,
    Comparison,
    Geometry,
    geometry,
    heading,
    query_from,
)

type Prepared = tuple[tuple[float, float], HeadingMotion, HeadingMotion, float, float]
type SetKey = tuple[Geometry, int]


class Timing(TypedDict):
    geometry: Geometry
    scale: int
    headings: list[int]
    expected_statuses: list[str]
    elapsed_ms: list[float]
    cpu_ms: list[float]
    median_elapsed_ms: float
    median_cpu_ms: float


def evaluate(cases: list[Prepared]) -> tuple[str, ...]:
    return tuple(
        certify_heading_separation(position, a, b, separation_threshold_m=threshold, horizon_s=horizon).status
        for position, a, b, threshold, horizon in cases
    )


def hashes(package: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}


def main() -> None:
    package = Path(geometric_safety.__file__).resolve().parent
    expected_package = REPOSITORY / "geometric_safety"
    assert package == expected_package
    source_hashes = hashes(package)
    available = os.sched_getaffinity(0)
    cpu = 2 if 2 in available else min(available)
    os.sched_setaffinity(0, {cpu})
    rows: list[Comparison] = [json.loads(line) for line in (DATA / "model-comparison.jsonl").read_text().splitlines()]
    grouped: defaultdict[SetKey, list[Comparison]] = defaultdict(list)
    for row in rows:
        grouped[geometry(row["case"]), row["case"]["scale"]].append(row)
    prepared: dict[SetKey, list[Prepared]] = {}
    expectations: dict[SetKey, tuple[str, ...]] = {}
    for identity, group in sorted(grouped.items()):
        group.sort(key=lambda row: row["case"]["turn_deg"])
        assert [row["case"]["turn_deg"] for row in group] == list(range(-20, 21, 5))
        prepared[identity] = []
        for row in group:
            query = query_from(row["case"]["query"])
            prepared[identity].append(
                (
                    query.relative_position_m,
                    heading(query.a),
                    heading(query.b),
                    query.threshold_m,
                    query.horizon_s,
                )
            )
        expectations[identity] = tuple(row["library"]["status"] for row in group)
        assert evaluate(prepared[identity]) == expectations[identity]
    assert len(prepared) == 69
    samples: dict[SetKey, tuple[list[float], list[float]]] = {key: ([], []) for key in prepared}
    order = list(prepared)
    rng = random.Random(20261109)
    load_before = os.getloadavg()
    for _ in range(100):
        rng.shuffle(order)
        for key in order:
            cpu_start = time.thread_time_ns()
            started = time.perf_counter_ns()
            decisions = evaluate(prepared[key])
            elapsed = (time.perf_counter_ns() - started) / 1e6
            cpu_elapsed = (time.thread_time_ns() - cpu_start) / 1e6
            assert decisions == expectations[key]
            samples[key][0].append(elapsed)
            samples[key][1].append(cpu_elapsed)
    timings: list[Timing] = [
        {
            "geometry": key[0],
            "scale": key[1],
            "headings": list(range(-20, 21, 5)),
            "expected_statuses": list(expectations[key]),
            "elapsed_ms": values[0],
            "cpu_ms": values[1],
            "median_elapsed_ms": statistics.median(values[0]),
            "median_cpu_ms": statistics.median(values[1]),
        }
        for key, values in sorted(samples.items())
    ]
    assert hashes(package) == source_hashes
    medians = [row["median_elapsed_ms"] for row in timings]
    summary = {
        "min_median_ms": min(medians),
        "median_of_medians_ms": statistics.median(medians),
        "max_median_ms": max(medians),
    }
    result = {
        "scope": (
            "Nine local-plane public API calls and result handling per set; "
            "aircraft declarations prepared outside timing"
        ),
        "warmup": "Every set evaluated before timing; compilation excluded",
        "repetitions_per_set": 100,
        "sets": 69,
        "queries": 621,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "package_sha256": source_hashes,
        "summary": summary,
        "timings": timings,
    }
    (DATA / "paper-timings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
