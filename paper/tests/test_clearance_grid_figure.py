"""The paper's constant-speed grid must not depend on the web form schema."""

import sys

import pytest

from geometric_safety.demo_utils import TURN_DEMO_SCENARIOS
from paper.render_clearance_grid_figure import REPO_ROOT, load_grid_payload


def test_clearance_grid_source_is_independent_of_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "app", None)
    _, params, payload = load_grid_payload(REPO_ROOT, 3)
    scenario = next(item for item in TURN_DEMO_SCENARIOS if item.preset_id == "single_turn_dodge")
    assert vars(params) == scenario.to_visual_params()
    assert payload["model_mode"] == "turn_aware"
    assert len(payload["safe"]) == 3
    assert all(len(row) == 3 for row in payload["safe"])
