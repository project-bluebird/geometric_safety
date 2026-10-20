"""Prepare display samples around the library's heading certificates.

Decisions, geographic conversion, speed enclosure and vertical bands all use
the library. Samples explain the result; they never determine acceptance.
"""

from dataclasses import dataclass
import math
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geometric_safety import (
    HeadingAssuranceResult,
    HeadingMotion,
    certify_heading_separation_from_latlon,
    certify_heading_separation_with_vertical_from_latlon,
)
from geometric_safety.demo_utils import (
    TURN_DEMO_SCENARIOS,
    local_xy_to_latlon,
    sample_trajectory_xy,
    speed_bounds_mps,
)
from geometric_safety.geographic import _relative_position_from_latlon
from geometric_safety.heading import _enclosing_common_speed_bounds_kt, _turn_angle_deg, _unsupported_reason
from geometric_safety.relevant_aircraft import _relative_position_hull_at_time
from geometric_safety.util import KT_TO_MPS, NMI_TO_M, closest_point_on_convex_polygon, shortest_turn_parameters
from geometric_safety.vertical import _time_to_selected_s, _vertical_band_at_time_fl, vertical_band_gap_fl

Status = Literal["certified_safe", "unresolved", "unsupported"]
Point = tuple[float, float]


class AircraftInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    heading_deg: float = Field(ge=0.0, le=360.0)
    target_heading_deg: float = Field(ge=0.0, le=360.0)
    turn_rate_deg_s: float = Field(ge=0.0, le=10.0)
    speed_min_kt: float = Field(ge=0.0, le=1000.0)
    speed_max_kt: float = Field(ge=0.0, le=1000.0)
    heading_deviation_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    current_fl: float = Field(default=330.0, ge=-20.0, le=1000.0)
    selected_fl: float = Field(default=330.0, ge=-20.0, le=1000.0)

    @model_validator(mode="after")
    def ordered_speeds(self) -> Self:
        if self.speed_max_kt < self.speed_min_kt:
            raise ValueError("maximum speed must be at least minimum speed")
        return self

    def motion(self) -> HeadingMotion:
        return HeadingMotion(
            heading0_deg=self.heading_deg,
            target_heading_deg=self.target_heading_deg,
            turn_rate_deg_s=self.turn_rate_deg_s,
            speed_min_mps=self.speed_min_kt * KT_TO_MPS,
            speed_max_mps=self.speed_max_kt * KT_TO_MPS,
            heading_deviation_max_deg=self.heading_deviation_deg,
        )


class VisualParams(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    a: AircraftInput
    b: AircraftInput
    separation_threshold_nm: float = Field(default=5.0, ge=0.1, le=50.0)
    horizon_s: float = Field(default=600.0, ge=0.0, le=1800.0)
    relative_position_error_m: float = Field(default=0.0, ge=0.0, le=100_000.0)
    relative_velocity_error_mps: float = Field(default=0.0, ge=0.0, le=500.0)
    vertical_enabled: bool = False
    minimum_vertical_rate_fpm: float = Field(default=1000.0, ge=0.0, le=10_000.0)
    required_vertical_gap_fl: float = Field(default=10.0, ge=0.0, le=100.0)
    vertical_rounding_fl: int = Field(default=10, ge=0, le=100)
    grid_density: int = Field(default=41, ge=15, le=101)
    view_half_extent_nm: float = Field(default=25.0, ge=5.0, le=100.0)


@dataclass(frozen=True)
class Certificate:
    status: Status
    reason: str
    horizon_s: float
    lateral_horizon_s: float
    vertical_resolution_time_s: float | None
    lateral: HeadingAssuranceResult | None


@dataclass(frozen=True)
class Paths:
    minimum: list[Point]
    midpoint: list[Point]
    maximum: list[Point]


@dataclass(frozen=True)
class GeometryFrame:
    relative_hull: list[Point]
    closest_point: Point
    base_distance_m: float
    midpoint_distance_m: float


@dataclass(frozen=True)
class Geometry:
    a: Paths
    b: Paths
    a_hull_speed_bounds_mps: tuple[float, float]
    b_hull_speed_bounds_mps: tuple[float, float]
    frames: list[GeometryFrame]
    effective_threshold_m: float | None


@dataclass(frozen=True)
class VerticalFrame:
    a_band_fl: tuple[float, float]
    b_band_fl: tuple[float, float]
    gap_fl: float


@dataclass(frozen=True)
class DetailPayload:
    certificate: Certificate
    times_s: list[float]
    geometry: Geometry | None
    geometry_note: str | None
    vertical: list[VerticalFrame] | None


@dataclass(frozen=True)
class GridPayload:
    east_m: list[float]
    north_m: list[float]
    lats: list[float]
    lons: list[float]
    statuses: list[list[Status]]
    counts: dict[Status, int]
    b_position_m: Point


def _certify(params: VisualParams, a: HeadingMotion, b: HeadingMotion, b_lat: float, b_lon: float) -> Certificate:
    if params.vertical_enabled:
        result = certify_heading_separation_with_vertical_from_latlon(
            a,
            b,
            a_lat=params.a.lat,
            a_lon=params.a.lon,
            b_lat=b_lat,
            b_lon=b_lon,
            a_current_fl=params.a.current_fl,
            a_selected_fl=params.a.selected_fl,
            b_current_fl=params.b.current_fl,
            b_selected_fl=params.b.selected_fl,
            minimum_vertical_rate_fpm=params.minimum_vertical_rate_fpm,
            required_vertical_gap_fl=params.required_vertical_gap_fl,
            vertical_rounding_fl=params.vertical_rounding_fl,
            separation_threshold_m=params.separation_threshold_nm * NMI_TO_M,
            horizon_s=params.horizon_s,
            relative_position_error_m=params.relative_position_error_m,
            relative_velocity_error_mps=params.relative_velocity_error_mps,
        )
        return Certificate(
            result.status,
            result.reason,
            params.horizon_s,
            result.lateral_horizon_s,
            result.vertical_resolution_time_s,
            result.lateral_result,
        )
    lateral = certify_heading_separation_from_latlon(
        a,
        b,
        a_lat=params.a.lat,
        a_lon=params.a.lon,
        b_lat=b_lat,
        b_lon=b_lon,
        separation_threshold_m=params.separation_threshold_nm * NMI_TO_M,
        horizon_s=params.horizon_s,
        relative_position_error_m=params.relative_position_error_m,
        relative_velocity_error_mps=params.relative_velocity_error_mps,
    )
    return Certificate(lateral.status, lateral.reason, params.horizon_s, params.horizon_s, None, lateral)


def _display_times(
    params: VisualParams, certificate: Certificate, a: HeadingMotion, b: HeadingMotion
) -> NDArray[np.float64]:
    events = [0.0, params.horizon_s, certificate.lateral_horizon_s]
    if certificate.vertical_resolution_time_s is not None:
        events.append(certificate.vertical_resolution_time_s)
    if certificate.lateral is not None and certificate.lateral.base_closest_time_s is not None:
        events.append(certificate.lateral.base_closest_time_s)
    for aircraft, motion in ((params.a, a), (params.b, b)):
        if _unsupported_reason(motion) is None:
            events.append(
                shortest_turn_parameters(motion.heading0_deg, motion.target_heading_deg, motion.turn_rate_deg_s)[2]
            )
        if params.vertical_enabled:
            events.append(
                _time_to_selected_s(aircraft.current_fl, aircraft.selected_fl, params.minimum_vertical_rate_fpm)
            )
    events = [event for event in events if math.isfinite(event) and 0.0 <= event <= params.horizon_s]
    return np.unique(np.concatenate((np.linspace(0.0, params.horizon_s, 361), events)))


def _geometry(
    params: VisualParams, certificate: Certificate, a: HeadingMotion, b: HeadingMotion, times: NDArray[np.float64]
) -> tuple[Geometry | None, str | None]:
    for label, motion in (("A", a), ("B", b)):
        reason = _unsupported_reason(motion)
        if reason is not None:
            return None, f"Horizontal geometry is unavailable for aircraft {label}: {reason}."
    lateral = certificate.lateral
    if lateral is not None and lateral.diagnostic_kind is None:
        return None, f"Horizontal geometry is unavailable: {lateral.reason}."

    a_bounds = (a.speed_min_mps, a.speed_max_mps)
    b_bounds = (b.speed_min_mps, b.speed_max_mps)
    if _turn_angle_deg(a) != 0.0 or _turn_angle_deg(b) != 0.0:
        shared = _enclosing_common_speed_bounds_kt(a, b)
        if shared is None:
            return None, "The declared speed bounds could not be enclosed for display."
        a_bounds = speed_bounds_mps(shared[0], shared[2])
        b_bounds = speed_bounds_mps(shared[1], shared[2])

    relative = np.asarray(_relative_position_from_latlon(params.a.lat, params.a.lon, params.b.lat, params.b.lon))
    a_origin, b_origin = np.zeros(2), -relative
    paths: list[Paths] = []
    for origin, motion, bounds in ((a_origin, a, a_bounds), (b_origin, b, b_bounds)):
        samples = [
            sample_trajectory_xy(
                origin, motion.heading0_deg, motion.target_heading_deg, speed, motion.turn_rate_deg_s, times
            )
            for speed in (bounds[0], 0.5 * (bounds[0] + bounds[1]), bounds[1])
        ]
        paths.append(Paths(*[[(float(x), float(y)) for x, y in sample] for sample in samples]))

    frames: list[GeometryFrame] = []
    for idx, time_s in enumerate(times):
        hull = _relative_position_hull_at_time(
            rel_pos0=relative,
            a_heading0_deg=a.heading0_deg,
            a_target_heading_deg=a.target_heading_deg,
            a_turn_rate_deg_sec=a.turn_rate_deg_s,
            a_min_speed_mps=a_bounds[0],
            a_max_speed_mps=a_bounds[1],
            b_heading0_deg=b.heading0_deg,
            b_target_heading_deg=b.target_heading_deg,
            b_turn_rate_deg_sec=b.turn_rate_deg_s,
            b_min_speed_mps=b_bounds[0],
            b_max_speed_mps=b_bounds[1],
            t_s=float(time_s),
        )
        closest, _ = closest_point_on_convex_polygon(np.zeros(2), hull)
        midpoint_relative = np.asarray(paths[0].midpoint[idx]) - np.asarray(paths[1].midpoint[idx])
        frames.append(
            GeometryFrame(
                [(float(x), float(y)) for x, y in hull],
                (float(closest[0]), float(closest[1])),
                float(np.linalg.norm(closest)),
                float(np.linalg.norm(midpoint_relative)),
            )
        )
    threshold = None if lateral is None else params.separation_threshold_nm * NMI_TO_M + lateral.inflation_m
    return Geometry(paths[0], paths[1], a_bounds, b_bounds, frames, threshold), None


def build_detail(params: VisualParams) -> DetailPayload:
    a, b = params.a.motion(), params.b.motion()
    certificate = _certify(params, a, b, params.b.lat, params.b.lon)
    times = _display_times(params, certificate, a, b)
    geometry, note = _geometry(params, certificate, a, b, times)
    vertical = None
    if params.vertical_enabled:
        vertical = [
            VerticalFrame(
                _vertical_band_at_time_fl(
                    params.a.current_fl, params.a.selected_fl, params.minimum_vertical_rate_fpm, float(time_s)
                ),
                _vertical_band_at_time_fl(
                    params.b.current_fl, params.b.selected_fl, params.minimum_vertical_rate_fpm, float(time_s)
                ),
                float(
                    vertical_band_gap_fl(
                        params.a.current_fl,
                        params.a.selected_fl,
                        params.b.current_fl,
                        params.b.selected_fl,
                        params.minimum_vertical_rate_fpm,
                        float(time_s),
                        0,
                    )
                ),
            )
            for time_s in times
        ]
    return DetailPayload(certificate, [float(value) for value in times], geometry, note, vertical)


def build_grid(params: VisualParams) -> GridPayload:
    a, b = params.a.motion(), params.b.motion()
    # Validate the selected pair even if the displayed grid does not include B.
    _certify(params, a, b, params.b.lat, params.b.lon)
    extent = params.view_half_extent_nm * NMI_TO_M
    axis = [float(value) for value in np.linspace(-extent, extent, params.grid_density)]
    lats = [local_xy_to_latlon(0.0, north, params.a.lat, params.a.lon)[0] for north in axis]
    unwrapped_lons = [local_xy_to_latlon(east, 0.0, params.a.lat, params.a.lon)[1] for east in axis]
    if any(not -90.0 < lat < 90.0 for lat in lats) or any(abs(lon - params.a.lon) >= 180.0 for lon in unwrapped_lons):
        raise ValueError("This map span crosses a pole or the local longitude limit. Reduce the map span.")
    lons = [math.remainder(lon, 360.0) for lon in unwrapped_lons]
    statuses: list[list[Status]] = [[_certify(params, a, b, lat, lon).status for lon in lons] for lat in lats]
    counts: dict[Status, int] = {"certified_safe": 0, "unresolved": 0, "unsupported": 0}
    for row in statuses:
        for status in row:
            counts[status] += 1
    relative = _relative_position_from_latlon(params.a.lat, params.a.lon, params.b.lat, params.b.lon)
    return GridPayload(axis, axis, lats, lons, statuses, counts, (-relative[0], -relative[1]))


class Preset(BaseModel):
    id: str
    title: str
    description: str
    params: VisualParams


def presets() -> list[Preset]:
    a = AircraftInput(
        lat=51.0,
        lon=-1.0,
        heading_deg=90.0,
        target_heading_deg=90.0,
        turn_rate_deg_s=0.0,
        speed_min_kt=315.0,
        speed_max_kt=365.0,
    )
    b_lat, b_lon = local_xy_to_latlon(20.0 * NMI_TO_M, 0.0, a.lat, a.lon)
    b = AircraftInput(
        lat=b_lat,
        lon=b_lon,
        heading_deg=270.0,
        target_heading_deg=270.0,
        turn_rate_deg_s=0.0,
        speed_min_kt=315.0,
        speed_max_kt=365.0,
        selected_fl=350.0,
    )
    clearance = VisualParams(a=a, b=b, vertical_enabled=True)
    options = [
        Preset(
            id="vertical_clearance",
            title="Climb before a later crossing",
            description=(
                "B climbs from FL330 to FL350. The minimum rate determines how long lateral separation is needed."
            ),
            params=clearance,
        )
    ]
    options.append(
        Preset(
            id="lateral_only",
            title="The same encounter, lateral only",
            description="The full horizon requires lateral separation when vertical progress is not declared.",
            params=clearance.model_copy(update={"vertical_enabled": False}),
        )
    )
    for scenario in TURN_DEMO_SCENARIOS:
        b_lat, b_lon = scenario.b_latlon()
        inputs = []
        for lat, lon, heading, target, rate, speed in (
            (
                scenario.a_lat,
                scenario.a_lon,
                scenario.a_heading_deg,
                scenario.a_target_heading_deg,
                scenario.a_turn_rate_deg_sec,
                scenario.a_speed_kt,
            ),
            (
                b_lat,
                b_lon,
                scenario.b_heading_deg,
                scenario.b_target_heading_deg,
                scenario.b_turn_rate_deg_sec,
                scenario.b_speed_kt,
            ),
        ):
            inputs.append(
                AircraftInput(
                    lat=lat,
                    lon=lon,
                    heading_deg=heading,
                    target_heading_deg=target,
                    turn_rate_deg_s=rate,
                    speed_min_kt=max(0.0, speed - scenario.speed_diff_kt),
                    speed_max_kt=speed + scenario.speed_diff_kt,
                )
            )
        options.append(
            Preset(
                id=scenario.preset_id,
                title=scenario.title,
                description=(
                    "Speeds may vary throughout the turn and its straight continuation. "
                    "The allowance encloses that variation."
                ),
                params=VisualParams(
                    a=inputs[0],
                    b=inputs[1],
                    horizon_s=scenario.projection_time_min * 60.0,
                    separation_threshold_nm=scenario.separation_threshold_nm,
                    view_half_extent_nm=scenario.view_half_extent_nm,
                ),
            )
        )
    return options
