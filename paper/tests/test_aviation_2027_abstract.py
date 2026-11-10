"""Check the saved AVIATION 2027 abstract study against the current package."""

import json

from paper.aviation_2027_abstract.compare_models import DATA
from paper.aviation_2027_abstract.summarise_results import summarise


def test_current_package_reproduces_saved_abstract_decisions_and_counts() -> None:
    saved = json.loads((DATA / "results-summary.json").read_text())
    del saved["script_sha256"]

    # summarise() compares all 873 saved decisions with fresh calls before counting.
    assert summarise() == saved
