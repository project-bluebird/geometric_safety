"""The app must display the same model, threshold and decision as the public interface."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app import app
from app.visualisation import VisualParams, presets
from geometric_safety import (
    certify_heading_separation_from_latlon,
    certify_heading_separation_with_vertical_from_latlon,
)
from geometric_safety.demo_utils import TURN_DEMO_SCENARIOS, build_turn_debug_payload
from geometric_safety.relevant_aircraft import catch_up_projection_interval_with_turns
from geometric_safety.util import KT_TO_MPS, NMI_TO_M


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def example(preset_id: str = "vertical_clearance") -> VisualParams:
    return next(p.params for p in presets() if p.id == preset_id).model_copy(deep=True)


@pytest.mark.parametrize("filename", ["relevant_aircraft.js", "relevant_aircraft.css"])
def test_page_refreshes_changed_assets(tmp_path: Path, filename: str) -> None:
    # Use a complete app copy to change real served assets without altering the
    # source checkout or substituting application behaviour.
    shutil.copytree(Path(__file__).resolve().parents[1] / "app", tmp_path / "app")
    script = textwrap.dedent("""
        from pathlib import Path
        import re
        import sys
        from fastapi.testclient import TestClient
        import app

        assert Path(app.__file__).parent == Path.cwd() / "app"
        filename = sys.argv[1]
        client = TestClient(app.app)
        pattern = rf"(/static/{re.escape(filename)}\\?v=[a-f0-9]+)"
        before = client.get("/")
        original = re.search(pattern, before.text)
        assert original is not None
        unchanged = re.search(pattern, client.get("/").text)
        assert unchanged is not None
        assert unchanged.group(1) == original.group(1)
        asset = Path("app/static") / filename
        original_bytes = asset.read_bytes()
        assert client.get(original.group(1)).content == original_bytes
        asset.write_bytes(original_bytes + b"\\n/* asset version regression */\\n")
        after = client.get("/")
        revised = re.search(pattern, after.text)
        assert revised is not None
        assert revised.group(1) != original.group(1)
        assert after.headers["cache-control"] == "no-cache"
        assert client.get(revised.group(1)).content == asset.read_bytes()
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, filename], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("preset_id", [p.id for p in presets()])
def test_detail_matches_public_geographic_certificate(client: TestClient, preset_id: str) -> None:
    params = example(preset_id)
    response = client.post("/api/time_sweep", json=params.model_dump())
    assert response.status_code == 200
    payload = response.json()
    a, b = params.a.motion(), params.b.motion()
    if params.vertical_enabled:
        direct = certify_heading_separation_with_vertical_from_latlon(
            a,
            b,
            a_lat=params.a.lat,
            a_lon=params.a.lon,
            b_lat=params.b.lat,
            b_lon=params.b.lon,
            a_current_fl=params.a.current_fl,
            a_selected_fl=params.a.selected_fl,
            b_current_fl=params.b.current_fl,
            b_selected_fl=params.b.selected_fl,
            minimum_vertical_rate_fpm=params.minimum_vertical_rate_fpm,
            required_vertical_gap_fl=params.required_vertical_gap_fl,
            vertical_rounding_fl=params.vertical_rounding_fl,
            separation_threshold_m=params.separation_threshold_nm * NMI_TO_M,
            horizon_s=params.horizon_s,
        )
        assert payload["certificate"]["status"] == direct.status
        assert payload["certificate"]["lateral_horizon_s"] == direct.lateral_horizon_s
        assert payload["certificate"]["vertical_resolution_time_s"] == direct.vertical_resolution_time_s
        lateral = direct.lateral_result
    else:
        lateral = certify_heading_separation_from_latlon(
            a,
            b,
            a_lat=params.a.lat,
            a_lon=params.a.lon,
            b_lat=params.b.lat,
            b_lon=params.b.lon,
            separation_threshold_m=params.separation_threshold_nm * NMI_TO_M,
            horizon_s=params.horizon_s,
        )
        assert payload["certificate"]["status"] == lateral.status
        assert payload["certificate"]["lateral_horizon_s"] == params.horizon_s
    assert lateral is not None
    assert payload["certificate"]["lateral"] == json.loads(json.dumps(asdict(lateral)))
    assert payload["geometry"]["effective_threshold_m"] == pytest.approx(
        params.separation_threshold_nm * NMI_TO_M + lateral.inflation_m
    )
    assert payload["times_s"][0] == 0.0
    assert payload["times_s"][-1] == params.horizon_s
    assert len(payload["geometry"]["frames"]) == len(payload["times_s"])


def test_vertical_cutoff_and_unrounded_bands(client: TestClient) -> None:
    params = example()
    data = client.post("/api/time_sweep", json=params.model_dump()).json()
    assert data["certificate"]["status"] == "certified_safe"
    assert data["certificate"]["lateral_horizon_s"] == pytest.approx(60.0)
    idx = data["times_s"].index(data["certificate"]["vertical_resolution_time_s"])
    assert data["vertical"][idx]["a_band_fl"] == pytest.approx([330.0, 330.0])
    assert data["vertical"][idx]["b_band_fl"] == pytest.approx([340.0, 350.0])
    assert data["vertical"][idx]["gap_fl"] == pytest.approx(10.0)
    # A later lateral overlap must not change what the shortened check asserts.
    assert min(frame["base_distance_m"] for frame in data["geometry"]["frames"]) < 5 * NMI_TO_M


@pytest.mark.parametrize("variant", ["zero_rate", "same_selected_level", "beyond_horizon"])
def test_no_shortening_when_vertical_separation_is_unavailable(client: TestClient, variant: str) -> None:
    params = example()
    if variant == "zero_rate":
        params.minimum_vertical_rate_fpm = 0.0
    elif variant == "same_selected_level":
        params.b.selected_fl = params.a.selected_fl
    else:
        params.minimum_vertical_rate_fpm = 10.0
    data = client.post("/api/time_sweep", json=params.model_dump()).json()
    assert data["certificate"]["lateral_horizon_s"] == params.horizon_s
    assert data["certificate"]["status"] == "unresolved"


def test_unsupported_turn_is_a_result_not_a_server_error(client: TestClient) -> None:
    params = example("single_turn_dodge")
    params.a.turn_rate_deg_s = 0.0
    response = client.post("/api/time_sweep", json=params.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["certificate"]["status"] == "unsupported"
    assert "zero turn rate" in data["certificate"]["reason"]
    assert data["geometry"] is None
    assert data["certificate"]["lateral"]["diagnostic_kind"] is None


def test_vertical_only_certificate_does_not_invent_horizontal_geometry(client: TestClient) -> None:
    params = example()
    params.b.current_fl = params.b.selected_fl
    params.a.target_heading_deg = 180.0
    response = client.post("/api/time_sweep", json=params.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["certificate"]["status"] == "certified_safe"
    assert data["certificate"]["lateral"] is None
    assert data["certificate"]["lateral_horizon_s"] == 0.0
    assert data["geometry"] is None
    assert data["vertical"]


def test_plot_threshold_includes_all_declared_errors(client: TestClient) -> None:
    params = example("single_turn_dodge")
    params.a.heading_deviation_deg = 2.0
    params.relative_position_error_m = 120.0
    params.relative_velocity_error_mps = 1.0
    data = client.post("/api/time_sweep", json=params.model_dump()).json()
    direct = certify_heading_separation_from_latlon(
        params.a.motion(),
        params.b.motion(),
        a_lat=params.a.lat,
        a_lon=params.a.lon,
        b_lat=params.b.lat,
        b_lon=params.b.lon,
        separation_threshold_m=5 * NMI_TO_M,
        horizon_s=params.horizon_s,
        relative_position_error_m=120.0,
        relative_velocity_error_mps=1.0,
    )
    assert data["certificate"]["lateral"]["inflation_m"] == direct.inflation_m
    assert data["geometry"]["effective_threshold_m"] == pytest.approx(5 * NMI_TO_M + direct.inflation_m)
    assert direct.inflation_m > 120 + params.horizon_s


def test_turn_display_uses_the_enlarged_kernel_speed_hull(client: TestClient) -> None:
    params = example("single_turn_dodge")
    params.b.speed_min_kt, params.b.speed_max_kt = 335.0, 345.0
    response = client.post("/api/time_sweep", json=params.model_dump())
    assert response.status_code == 200
    data = response.json()
    geometry = data["geometry"]
    # Both intervals are centred on 340 kt; A's 25 kt half-width also encloses B.
    bounds = (315.0 * KT_TO_MPS, 365.0 * KT_TO_MPS)
    assert geometry["a_hull_speed_bounds_mps"] == pytest.approx(bounds)
    assert geometry["b_hull_speed_bounds_mps"] == pytest.approx(bounds)
    origin_x, origin_y = geometry["frames"][0]["relative_hull"][0]
    # At 90 s, A has turned north to west at 1 deg/s, with displacement
    # (-v/omega, v/omega); B has travelled east by 90*v.
    omega = math.pi / 180.0
    expected_hull = [
        (origin_x - a_speed / omega - 90.0 * b_speed, origin_y + a_speed / omega)
        for a_speed in bounds
        for b_speed in bounds
    ]
    frame = geometry["frames"][data["times_s"].index(90.0)]
    assert len(frame["relative_hull"]) == len(expected_hull)
    for actual, expected in zip(sorted(frame["relative_hull"]), sorted(expected_hull), strict=True):
        assert actual == pytest.approx(expected, rel=0.0, abs=1e-7)
    # Here x < 0 < y, so increasing either speed moves farther from the origin.
    closest = expected_hull[0]
    assert closest[0] < 0 < closest[1]
    assert frame["closest_point"] == pytest.approx(closest, rel=0.0, abs=1e-7)
    assert frame["base_distance_m"] == pytest.approx(math.hypot(*closest), rel=0.0, abs=1e-7)


def test_zero_horizon_is_serialisable_and_has_one_frame(client: TestClient) -> None:
    params = example("lateral_only")
    params.horizon_s = 0.0
    data = client.post("/api/time_sweep", json=params.model_dump()).json()
    assert data["times_s"] == [0.0]
    assert len(data["geometry"]["frames"]) == 1


def test_date_line_positions_use_the_same_local_frame(client: TestClient) -> None:
    params = example("lateral_only")
    params.a.lat = params.b.lat = 0.0
    params.a.lon, params.b.lon = 179.9, -179.9
    data = client.post("/api/time_sweep", json=params.model_dump()).json()
    assert 20_000 < data["geometry"]["b"]["midpoint"][0][0] < 25_000
    assert data["geometry"]["frames"][0]["base_distance_m"] == pytest.approx(data["geometry"]["b"]["midpoint"][0][0])


@pytest.mark.parametrize("invalid", ["speed_order", "pole", "antipodal", "nonfinite", "negative_rate", "unknown"])
def test_invalid_inputs_are_explained_with_422(client: TestClient, invalid: str) -> None:
    body = example().model_dump()
    if invalid == "speed_order":
        body["a"]["speed_min_kt"] = 400.0
    elif invalid == "pole":
        body["a"]["lat"] = 90.0
    elif invalid == "antipodal":
        body["a"]["lon"], body["b"]["lon"] = 0.0, 180.0
    elif invalid == "nonfinite":
        body["a"]["lat"] = "NaN"
    elif invalid == "negative_rate":
        body["minimum_vertical_rate_fpm"] = -1.0
    else:
        body["turn_speed_schedule_uncertainty_kt"] = 20.0
    response = client.post("/api/time_sweep", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_grid_cells_match_selected_pair_decisions(client: TestClient) -> None:
    params = example("single_turn_dodge")
    params.grid_density = 15
    response = client.post("/api/grid", json=params.model_dump())
    assert response.status_code == 200
    grid = response.json()
    assert sum(grid["counts"].values()) == 225
    # Keep an asymmetric pair so swapping latitude and longitude indices fails.
    assert grid["statuses"][0][7] != grid["statuses"][7][0]
    for i, j in ((0, 0), (7, 7), (14, 14), (0, 7), (7, 0)):
        params.b.lat, params.b.lon = grid["lats"][i], grid["lons"][j]
        response = client.post("/api/time_sweep", json=params.model_dump())
        assert response.status_code == 200
        detail = response.json()
        assert grid["statuses"][i][j] == detail["certificate"]["status"]


def test_polar_grid_span_returns_validation_error(client: TestClient) -> None:
    params = example("lateral_only")
    params.a.lat = params.b.lat = 89.99
    assert client.post("/api/grid", json=params.model_dump()).status_code == 422


def test_page_embeds_valid_presets_and_responsive_assets(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    match = re.search(r'<script id="presetData" type="application/json">(.*?)</script>', response.text, re.DOTALL)
    assert match is not None
    choices = json.loads(match.group(1))
    assert len(choices) == 6
    for choice in choices:
        VisualParams.model_validate(choice["params"])
    assert 'name="viewport"' in response.text
    assert client.get("/static/relevant_aircraft.js").status_code == 200
    assert client.get("/static/relevant_aircraft.css").status_code == 200


def test_legacy_standalone_demo_still_matches_its_core_model() -> None:
    scenario = TURN_DEMO_SCENARIOS[0]
    b_lat, b_lon = scenario.b_latlon()
    payload = build_turn_debug_payload(
        a_lat=scenario.a_lat,
        a_lon=scenario.a_lon,
        a_heading0_deg=scenario.a_heading_deg,
        a_target_heading_deg=scenario.a_target_heading_deg,
        a_speed_kt=scenario.a_speed_kt,
        a_turn_rate_deg_sec=scenario.a_turn_rate_deg_sec,
        b_lat=b_lat,
        b_lon=b_lon,
        b_heading0_deg=scenario.b_heading_deg,
        b_target_heading_deg=scenario.b_target_heading_deg,
        b_speed_kt=scenario.b_speed_kt,
        b_turn_rate_deg_sec=scenario.b_turn_rate_deg_sec,
        separation_threshold_m=scenario.separation_threshold_nm * NMI_TO_M,
        speed_diff_kt=scenario.speed_diff_kt,
        projection_time_s=scenario.projection_time_min * 60.0,
    )
    direct = catch_up_projection_interval_with_turns(
        scenario.a_lat,
        scenario.a_lon,
        scenario.a_heading_deg,
        scenario.a_target_heading_deg,
        scenario.a_speed_kt,
        scenario.a_turn_rate_deg_sec,
        b_lat,
        b_lon,
        scenario.b_heading_deg,
        scenario.b_target_heading_deg,
        scenario.b_speed_kt,
        scenario.b_turn_rate_deg_sec,
        scenario.separation_threshold_nm * NMI_TO_M,
        scenario.speed_diff_kt,
        scenario.projection_time_min * 60.0,
    )
    assert payload["summary"]["is_separated"] == direct[0]
