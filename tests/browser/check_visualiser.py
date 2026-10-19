"""Real-browser regressions, run explicitly with the apps and browser groups."""

from collections.abc import Iterator
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from playwright.sync_api import Browser, Error, Page, expect, sync_playwright
from pydantic import TypeAdapter
import pytest

from app.visualisation import DetailPayload, GridPayload, VisualParams

ROOT = Path(__file__).resolve().parents[2]
DETAIL = TypeAdapter(DetailPayload)
GRID = TypeAdapter(GridPayload)
NUMBERS = TypeAdapter(list[float])
BOX = TypeAdapter(tuple[float, float, float, float])

# Observe the real drawing calls and forward every call unchanged to Chromium.
OBSERVE_CANVAS = """() => {
  window.canvasDraws = {};
  const proto = CanvasRenderingContext2D.prototype;
  for (const method of ['setTransform', 'fillRect', 'fillText', 'clip', 'moveTo', 'lineTo', 'arc', 'fill']) {
    const original = proto[method];
    proto[method] = function(...args) {
      const id = this.canvas.id;
      if (method === 'setTransform')
        window.canvasDraws[id] = {labels: [], points: [], circles: [], fills: [], clipped: false};
      const draw = window.canvasDraws[id];
      if (draw) {
        if (method === 'fillRect' && !draw.box) draw.box = args;
        if (method === 'fillText') draw.labels.push({text: String(args[0]), font: this.font, align: this.textAlign});
        if (method === 'clip') draw.clipped = true;
        if (draw.clipped && (method === 'moveTo' || method === 'lineTo')) draw.points.push(args);
        if (draw.clipped && method === 'arc') draw.circles.push(args.slice(0, 3));
        if (draw.clipped && method === 'fill') draw.fills.push(this.fillStyle);
      }
      return original.apply(this, args);
    };
  }
}"""

CHECK_HULL = """() => {
  const slider = document.getElementById('timeSlider');
  let first = null;
  for (let i = 0; i <= Number(slider.max); i++) {
    slider.value = i;
    slider.dispatchEvent(new Event('input', {bubbles: true}));
    const draw = window.canvasDraws.hullCanvas;
    const axes = JSON.stringify(draw.labels.filter(l => l.font.startsWith('10px') && /^-?[\\d.]+$/.test(l.text)));
    if (first === null) first = axes;
    if (axes !== first) throw new Error(`Axes changed at slider index ${i}`);
    const [x, y, w, h] = draw.box;
    const within = (px, py) => px >= x - 1e-6 && px <= x + w + 1e-6 && py >= y - 1e-6 && py <= y + h + 1e-6;
    if (draw.points.some(([px, py]) => !within(px, py))) throw new Error(`Hull clipped at index ${i}`);
    if (draw.circles.some(([cx, cy, r]) => !within(cx-r, cy-r) || !within(cx+r, cy+r)))
      throw new Error(`Circle clipped at index ${i}`);
  }
  return Number(slider.max) + 1;
}"""


@pytest.fixture(scope="module")
def live_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    log = tmp_path_factory.mktemp("visualiser-server") / "uvicorn.log"
    environment = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    with log.open("w", encoding="utf-8") as output:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "0"],
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                contents = log.read_text(encoding="utf-8")
                match = re.search(r"Uvicorn running on (http://127\.0\.0\.1:\d+)", contents)
                if match:
                    yield match.group(1)
                    break
                if server.poll() is not None:
                    raise RuntimeError(f"Visualiser server exited:\n{contents}")
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Visualiser server did not start:\n{log.read_text(encoding='utf-8')}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: Browser, live_url: str) -> Iterator[Page]:
    instance = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    instance.set_default_timeout(30_000)
    errors: list[str] = []

    def record_error(error: Error) -> None:
        errors.append(str(error))

    instance.on("pageerror", record_error)
    instance.add_init_script(f"({OBSERVE_CANVAS})()")
    try:
        instance.goto(live_url, wait_until="domcontentloaded")
        expect(instance.locator("#resultCard")).to_have_attribute("data-status", "certified_safe")
        yield instance
        assert not errors, errors
    finally:
        instance.close()


def _preset(page: Page, preset_id: str) -> DetailPayload:
    with page.expect_response(lambda response: response.url.endswith("/api/time_sweep")) as pending:
        page.locator("#preset").select_option(preset_id)
    response = pending.value
    assert response.ok, response.text()
    data = DETAIL.validate_python(response.json())
    expect(page.locator("#statusBadge")).not_to_have_text("Working")
    expect(page.locator("#resultCard")).to_have_attribute("data-status", data.certificate.status)
    return data


def test_presets_unsupported_motion_and_real_server_validation(page: Page) -> None:
    statuses: set[str] = set()
    for option in page.locator("#preset option").all():
        preset_id = option.get_attribute("value")
        assert preset_id is not None
        if preset_id != "custom":
            statuses.add(_preset(page, preset_id).certificate.status)
    assert {"certified_safe", "unresolved"} <= statuses

    _preset(page, "single_turn_dodge")
    assert "does not establish a conflict" in page.locator("#resultReason").inner_text()
    assert page.locator("#resultCard").evaluate("el => getComputedStyle(el).borderLeftColor") == "rgb(197, 106, 100)"
    page.locator("#editInputs").click()
    page.locator("#a_turn_rate_deg_s").fill("0")
    page.locator("#evaluateBtn").click()
    expect(page.locator("#resultCard")).to_have_attribute("data-status", "unsupported")
    expect(page.locator("#plots")).to_be_hidden()
    expect(page.locator("#geometryNotice")).to_be_visible()

    _preset(page, "vertical_clearance")
    page.locator("#a_speed_min_kt").fill("400")
    page.locator("#a_speed_max_kt").fill("300")
    with page.expect_response(lambda response: response.url.endswith("/api/time_sweep")) as pending:
        page.locator("#evaluateBtn").click()
    assert pending.value.status == 422
    expect(page.locator("#resultCard")).to_have_attribute("data-status", "error")
    expect(page.locator("#resultReason")).to_contain_text("maximum speed must be at least minimum speed")


def test_error_units_api_parity_and_map_selection(page: Page, live_url: str) -> None:
    page.locator("#editInputs").click()
    page.get_by_text("Positions and additional bounds", exact=True).click()
    page.locator("#relative_position_error_m").fill("0.5")
    page.locator("#relative_velocity_error_mps").fill("10")
    with page.expect_response(lambda response: response.url.endswith("/api/time_sweep")) as pending:
        page.locator("#evaluateBtn").click()
    response = pending.value
    assert response.ok, response.text()
    sent = VisualParams.model_validate(response.request.post_data_json)
    assert sent.relative_position_error_m == 926.0
    assert sent.relative_velocity_error_mps == pytest.approx(10.0 * 1852.0 / 3600.0)
    expected = sent.model_copy(
        update={"relative_position_error_m": 926.0, "relative_velocity_error_mps": 10 * 1852 / 3600}
    )
    direct = page.request.post(f"{live_url}/api/time_sweep", data=expected.model_dump())
    assert direct.ok, direct.text()
    assert DETAIL.validate_python(response.json()).certificate == DETAIL.validate_python(direct.json()).certificate

    page.locator("#mapSection > summary").click()
    page.locator("#grid_density").fill("15")
    with page.expect_response(lambda response: response.url.endswith("/api/grid")) as pending:
        page.locator("#refreshGridBtn").click()
    response = pending.value
    assert response.ok, response.text()
    grid = GRID.validate_python(response.json())
    sent = VisualParams.model_validate(response.request.post_data_json)
    assert sent.relative_position_error_m == 926.0
    assert sent.relative_velocity_error_mps == pytest.approx(10.0 * 1852.0 / 3600.0)
    direct_grid = page.request.post(f"{live_url}/api/grid", data=sent.model_dump())
    assert direct_grid.ok, direct_grid.text()
    assert grid.statuses == GRID.validate_python(direct_grid.json()).statuses
    expect(page.locator("#gridStatus")).to_contain_text("certified ·")
    x, y, width, height = BOX.validate_python(page.evaluate("window.canvasDraws.gridCanvas.box"))
    with page.expect_response(lambda response: response.url.endswith("/api/time_sweep")) as pending:
        page.locator("#gridCanvas").click(position={"x": x + width / 2, "y": y + height / 2})
    selected = VisualParams.model_validate(pending.value.request.post_data_json)
    assert selected.b.lat == grid.lats[len(grid.lats) // 2]
    assert selected.b.lon == grid.lons[len(grid.lons) // 2]
    expect(page.locator("#preset")).to_have_value("custom")


def test_fixed_axes_altitude_ticks_and_responsive_layout(page: Page, tmp_path: Path) -> None:
    ticks = NUMBERS.validate_python(
        page.evaluate(
            "window.canvasDraws.verticalCanvas.labels.filter(l => l.font.startsWith('10px') && l.align === 'right'"
            " && Number.isFinite(Number(l.text))).map(l => Number(l.text))"
        )
    )
    assert ticks == [320, 330, 340, 350, 360]
    assert page.evaluate("window.canvasDraws.spatialCanvas.fills.includes('#285fba')")
    assert page.evaluate("window.canvasDraws.spatialCanvas.fills.includes('#946c00')")
    for preset_id in ("vertical_clearance", "single_turn_dodge"):
        _preset(page, preset_id)
        for window in ("180", "300", "full"):
            page.locator("#plotWindow").select_option(window)
            assert TypeAdapter(int).validate_python(page.evaluate(CHECK_HULL)) > 1

    _preset(page, "vertical_clearance")
    page.locator("#editInputs").click()
    fields = ("a_current_fl", "a_selected_fl", "b_current_fl", "b_selected_fl")
    for levels in ((333, 333, 333, 357), (-20, -20, -10, 20), (-20, -20, -20, 1000)):
        for field, value in zip(fields, levels, strict=True):
            page.locator(f"#{field}").fill(str(value))
        page.locator("#evaluateBtn").click()
        expect(page.locator("#statusBadge")).not_to_have_text("Working")
        ticks = NUMBERS.validate_python(
            page.evaluate(
                "window.canvasDraws.verticalCanvas.labels.filter(l => l.font.startsWith('10px') && l.align === 'right'"
                " && Number.isFinite(Number(l.text))).map(l => Number(l.text))"
            )
        )
        assert 2 <= len(ticks) <= 8
        assert all(tick % 10 == 0 for tick in ticks)
    for editor_open in (True, False):
        if not editor_open:
            page.locator("#editInputs").click()
        for width in (1440, 960, 800, 390, 320):
            page.set_viewport_size({"width": width, "height": 1100})
            assert page.evaluate("document.documentElement.scrollWidth") <= width
    page.screenshot(path=str(tmp_path / "visualiser-mobile.png"), full_page=True)
