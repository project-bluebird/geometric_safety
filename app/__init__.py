"""Interactive heading-certificate explorer."""

from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.visualisation import DetailPayload, GridPayload, VisualParams, build_detail, build_grid, presets
from geometric_safety.util import KT_TO_MPS, NMI_TO_M

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app = FastAPI(title="Heading separation explorer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="visualiser-static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    asset_versions = {
        name: sha256((BASE_DIR / "static" / name).read_bytes()).hexdigest()[:12]
        for name in ("relevant_aircraft.js", "relevant_aircraft.css")
    }
    return templates.TemplateResponse(
        request,
        "relevant_aircraft.html",
        {
            "presets": [p.model_dump() for p in presets()],
            "asset_versions": asset_versions,
            "unit_scales": {"NM": NMI_TO_M, "kt": KT_TO_MPS},
        },
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/time_sweep")
def time_sweep_visual(params: VisualParams) -> DetailPayload:
    try:
        return build_detail(params)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/grid")
def grid_visual(params: VisualParams) -> GridPayload:
    try:
        return build_grid(params)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def get_app() -> FastAPI:
    return app
