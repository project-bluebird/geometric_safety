"""The paper grid uses real constant-speed checks and a portable source format."""

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

from geometric_safety.relevant_aircraft import catch_up_projection_interval_with_turns
from geometric_safety.util import NMI_TO_M
from paper.render_clearance_grid_figure import (
    REPO_ROOT,
    load_grid_payload,
    read_source_payload,
    render_figure,
    write_source_payload,
)


def test_clearance_grid_does_not_import_the_app() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from paper.render_clearance_grid_figure import REPO_ROOT, load_grid_payload
payload = load_grid_payload(REPO_ROOT, 3)
assert payload["model_mode"] == "turn_aware"
assert not any(name == "app" or name.startswith("app.") for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_grid_source_round_trip_and_render(tmp_path: Path, suffix: str) -> None:
    payload = load_grid_payload(REPO_ROOT, 3)
    path = tmp_path / ("grid" + suffix)
    write_source_payload(path, payload)
    restored = read_source_payload(path)
    assert restored == payload
    a, b = restored["a"], restored["b"]
    for row, latitude in zip(restored["safe"], restored["lats"], strict=True):
        for safe, longitude in zip(row, restored["lons"], strict=True):
            expected, _, _ = catch_up_projection_interval_with_turns(
                a_lat=a["lat"],
                a_lon=a["lon"],
                a_heading0_deg=a["heading"],
                a_target_heading_deg=a["target_heading_deg"],
                a_speed_kt=a["speed_kt"],
                a_turn_rate_deg_sec=a["turn_rate_deg_sec"],
                b_lat=latitude,
                b_lon=longitude,
                b_heading0_deg=b["heading"],
                b_target_heading_deg=b["target_heading_deg"],
                b_speed_kt=b["speed_kt"],
                b_turn_rate_deg_sec=b["turn_rate_deg_sec"],
                separation_threshold_m=restored["separation_threshold_nm"] * NMI_TO_M,
                speed_diff_kt=restored["speed_diff_kt"],
                projection_time_s=restored["projection_time_min"] * 60,
            )
            assert safe == expected
    figure = render_figure(restored)
    try:
        figure.savefig(tmp_path / "grid.pdf")
        assert len(figure.axes) == 1
        assert figure.axes[0].get_xlabel() == "Initial east offset of aircraft B (NMI)"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("density", [0, 1])
def test_grid_requires_two_coordinates(density: int) -> None:
    with pytest.raises(ValueError, match="At least two"):
        load_grid_payload(REPO_ROOT, density)
