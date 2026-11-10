"""Verify the paper's distinct-query counts against the evaluated library."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from geometric_safety import certify_heading_separation
from paper.aviation_2027_abstract.compare_models import (
    DATA,
    Comparison,
    Geometry,
    Wider,
    geometry,
    heading,
    query_from,
    source_hashes,
)
from paper.aviation_2027_abstract.study_data import CaseInput, LibraryDecision, ReferenceRow


class Baseline(TypedDict):
    reference: ReferenceRow
    library: LibraryDecision


def summarise() -> dict[str, object]:
    """Check every saved decision against the installed package and return the study counts."""
    original: list[Comparison] = [json.loads(s) for s in (DATA / "model-comparison.jsonl").read_text().splitlines()]
    doubled: list[Wider] = [json.loads(s) for s in (DATA / "doubled-wider.jsonl").read_text().splitlines()]
    base: list[Baseline] = [json.loads(s) for s in (DATA / "wider-results.jsonl").read_text().splitlines()]
    cases: list[tuple[CaseInput, str, LibraryDecision]] = [
        (r["case"], r["variable_reference"], r["library"]) for r in original
    ]
    cases.extend((r["case"], r["reference"]["status"], r["library"]) for r in doubled)
    cases.extend((r["reference"]["case"], r["reference"]["status"], r["library"]) for r in base)
    assert len(cases) == 873
    assert len({(geometry(c), c["scale"], c["turn_deg"]) for c, _, _ in cases}) == len(cases)
    counts: Counter[str] = Counter()
    accepted: defaultdict[tuple[Geometry, int], list[int]] = defaultdict(list)
    for case, reference, saved in cases:
        query = query_from(case["query"])
        result = certify_heading_separation(
            query.relative_position_m,
            heading(query.a),
            heading(query.b),
            separation_threshold_m=query.threshold_m,
            horizon_s=query.horizon_s,
        )
        assert result.status == saved["status"], case
        assert (result.status == "certified_safe") == (reference == "separated"), case
        counts[reference] += 1
        if result.status == "certified_safe":
            accepted[geometry(case), case["scale"]].append(case["turn_deg"])
    assert counts == {"separated": 396, "conflict": 476, "unresolved": 1}
    assert all(r["corners"] == r["constant"]["status"] == r["variable_reference"] for r in original)
    geometries = sorted({geometry(c) for c, _, _ in cases})
    assert len(geometries) == 23
    increases: dict[str, dict[str, int]] = {}
    for scale, expected in (
        (1, {"0": 4, "5": 18, "10": 1}),
        (2, {"0": 1, "5": 6, "10": 12, "15": 3, "none_up_to_90": 1}),
    ):
        changes: Counter[str] = Counter()
        for g in geometries:
            nominal = min(map(abs, accepted[g, 0]))
            turns = accepted[g, scale]
            changes[str(min(map(abs, turns)) - nominal) if turns else "none_up_to_90"] += 1
        assert changes == expected, changes
        increases[str(scale)] = dict(changes)
    # On the common assessed range, widening the bounds cannot add a retained turn.
    for g in geometries:
        common = [{t for t in accepted[g, s] if abs(t) <= 20} for s in (0, 1, 2)]
        assert common[2] <= common[1] <= common[0], g
    return {
        "distinct_geometries": len(geometries),
        "queries": len(cases),
        "reference_classifications": dict(counts),
        "constant_corner_comparison_queries": len(original),
        "minimum_turn_increases_deg": increases,
        "live_library_decisions_match_saved": True,
    }


def main() -> None:
    """Write the checked counts, confirming that no source changed during the run."""
    before = source_hashes()
    summary = summarise()
    assert source_hashes() == before
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (DATA / "results-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
