"""Demo helpers, canned scenarios, and app-support utilities.

Everything in this module is auxiliary support for the interactive FastAPI
visualiser and the standalone ``python -m geometric_safety.plotting`` demos.
The core solver in ``geometric_safety.relevant_aircraft`` does not depend on
any of these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from geometric_safety.relevant_aircraft import (
    _min_distance_to_relative_hull_at_time,
    _relative_position_hull_at_time,
    _turn_displacement_basis,
    catch_up_projection_interval_with_turns,
)
from geometric_safety.util import (
    DEG_TO_RAD,
    EARTH_RADIUS_IN_METERS,
    KT_TO_MPS,
    M_TO_NMI,
    RAD_TO_DEG,
    closest_point_on_convex_polygon,
    heading_diff,
    latlon_to_local_xy,
)

DEMO_REF_LAT = 51.0
DEMO_REF_LON = -1.0
CUSTOM_TURN_PRESET_ID = "custom"


@dataclass(frozen=True, slots=True)
class TurnDemoScenario:
    """Serializable canned scenario used by the app and the standalone turn demo."""

    preset_id: str
    title: str
    a_lat: float
    a_lon: float
    a_heading_deg: float
    a_target_heading_deg: float
    a_speed_kt: float
    a_turn_rate_deg_sec: float
    b_east_m: float
    b_north_m: float
    b_heading_deg: float
    b_target_heading_deg: float
    b_speed_kt: float
    b_turn_rate_deg_sec: float
    speed_diff_kt: float
    separation_threshold_nm: float
    projection_time_min: float
    view_half_extent_nm: float
    grid_density: int = 75

    def b_latlon(self) -> tuple[float, float]:
        return local_xy_to_latlon(self.b_east_m, self.b_north_m, self.a_lat, self.a_lon)

    def to_visual_params(self) -> dict[str, float | int | str]:
        b_lat, b_lon = self.b_latlon()
        return {
            "model_mode": "turn_aware",
            "turn_preset_id": self.preset_id,
            "separation_threshold_nm": self.separation_threshold_nm,
            "speed_diff_kt": self.speed_diff_kt,
            "projection_time_min": self.projection_time_min,
            "grid_density": self.grid_density,
            "view_half_extent_nm": self.view_half_extent_nm,
            "a_lat": self.a_lat,
            "a_lon": self.a_lon,
            "a_heading": self.a_heading_deg,
            "a_target_heading_deg": self.a_target_heading_deg,
            "a_speed_kt": self.a_speed_kt,
            "a_turn_rate_deg_sec": self.a_turn_rate_deg_sec,
            "b_lat": b_lat,
            "b_lon": b_lon,
            "b_heading": self.b_heading_deg,
            "b_target_heading_deg": self.b_target_heading_deg,
            "b_speed_kt": self.b_speed_kt,
            "b_turn_rate_deg_sec": self.b_turn_rate_deg_sec,
            "turn_speed_schedule_uncertainty_kt": 0.0,
        }


TURN_DEMO_SCENARIOS: tuple[TurnDemoScenario, ...] = (
    TurnDemoScenario(
        preset_id="single_turn_dodge",
        title="Single-turn dodge",
        a_lat=DEMO_REF_LAT,
        a_lon=DEMO_REF_LON,
        a_heading_deg=0.0,
        a_target_heading_deg=270.0,
        a_speed_kt=340.0,
        a_turn_rate_deg_sec=1.0,
        b_east_m=-12000.0,
        b_north_m=-4000.0,
        b_heading_deg=90.0,
        b_target_heading_deg=90.0,
        b_speed_kt=340.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=25.0,
        separation_threshold_nm=5.0,
        projection_time_min=15.0,
        view_half_extent_nm=18.0,
    ),
    TurnDemoScenario(
        preset_id="single_turn_cut_in",
        title="Single-turn cut-in",
        a_lat=DEMO_REF_LAT,
        a_lon=DEMO_REF_LON,
        a_heading_deg=135.0,
        a_target_heading_deg=135.0,
        a_speed_kt=340.0,
        a_turn_rate_deg_sec=0.0,
        b_east_m=8000.0,
        b_north_m=-8000.0,
        b_heading_deg=180.0,
        b_target_heading_deg=225.0,
        b_speed_kt=360.0,
        b_turn_rate_deg_sec=1.0,
        speed_diff_kt=25.0,
        separation_threshold_nm=5.0,
        projection_time_min=10.0,
        view_half_extent_nm=18.0,
    ),
    TurnDemoScenario(
        preset_id="mutual_opening_turns",
        title="Mutual opening turns",
        a_lat=DEMO_REF_LAT,
        a_lon=DEMO_REF_LON,
        a_heading_deg=180.0,
        a_target_heading_deg=135.0,
        a_speed_kt=380.0,
        a_turn_rate_deg_sec=1.0,
        b_east_m=4000.0,
        b_north_m=-12000.0,
        b_heading_deg=225.0,
        b_target_heading_deg=270.0,
        b_speed_kt=320.0,
        b_turn_rate_deg_sec=1.0,
        speed_diff_kt=25.0,
        separation_threshold_nm=5.0,
        projection_time_min=15.0,
        view_half_extent_nm=20.0,
    ),
    TurnDemoScenario(
        preset_id="mutual_closing_turns",
        title="Mutual closing turns",
        a_lat=DEMO_REF_LAT,
        a_lon=DEMO_REF_LON,
        a_heading_deg=180.0,
        a_target_heading_deg=135.0,
        a_speed_kt=360.0,
        a_turn_rate_deg_sec=1.0,
        b_east_m=8000.0,
        b_north_m=8000.0,
        b_heading_deg=270.0,
        b_target_heading_deg=225.0,
        b_speed_kt=360.0,
        b_turn_rate_deg_sec=1.0,
        speed_diff_kt=25.0,
        separation_threshold_nm=5.0,
        projection_time_min=10.0,
        view_half_extent_nm=18.0,
    ),
)


def get_turn_preset_options() -> list[dict[str, Any]]:
    """Return canned turn scenarios in the UI payload format used by the app."""
    options: list[dict[str, Any]] = [{"id": CUSTOM_TURN_PRESET_ID, "title": "Custom", "params": None}]
    options.extend(
        {"id": scenario.preset_id, "title": scenario.title, "params": scenario.to_visual_params()}
        for scenario in TURN_DEMO_SCENARIOS
    )
    return options


def speed_bounds_mps(speed_kt: float, speed_diff_kt: float) -> tuple[float, float]:
    """Return the symmetric min/max speed envelope for a nominal speed."""
    return max(speed_kt - speed_diff_kt, 0.0) * KT_TO_MPS, (speed_kt + speed_diff_kt) * KT_TO_MPS


def local_xy_to_latlon(east_m: float, north_m: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Approximate inverse of `latlon_to_local_xy` for demo scenario setup."""
    ref_lat_rad = ref_lat * DEG_TO_RAD
    lat = ref_lat + (north_m / EARTH_RADIUS_IN_METERS) * RAD_TO_DEG
    lon = ref_lon + (east_m / (EARTH_RADIUS_IN_METERS * math.cos(ref_lat_rad))) * RAD_TO_DEG
    return lat, lon


def turn_duration_s(heading0_deg: float, target_heading_deg: float, turn_rate_deg_sec: float) -> float:
    """Return the deterministic duration of an instructed constant-rate turn."""
    if turn_rate_deg_sec == 0.0 or heading0_deg == target_heading_deg:
        return 0.0
    return abs(heading_diff(heading0_deg, target_heading_deg, abs_diff=True)) / abs(turn_rate_deg_sec)


def heading_summary(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    speed_kt: float,
) -> str:
    """Format one aircraft's lateral plan for debug text overlays and tooltips."""
    if heading0_deg == target_heading_deg:
        return f"{heading0_deg:.0f} deg straight @ {speed_kt:.0f} kt"
    if turn_rate_deg_sec == 0.0:
        return f"instant {heading0_deg:.0f}->{target_heading_deg:.0f} deg @ {speed_kt:.0f} kt"
    return f"{heading0_deg:.0f}->{target_heading_deg:.0f} deg @ {turn_rate_deg_sec:.1f} deg/s, {speed_kt:.0f} kt"


def sample_trajectory_xy(
    start_xy: np.ndarray,
    heading0_deg: float,
    target_heading_deg: float,
    speed_mps: float,
    turn_rate_deg_sec: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Sample one exact turn-then-straight trajectory in local XY."""
    xy = np.empty((times_s.shape[0], 2), dtype=np.float64)
    for idx in range(times_s.shape[0]):
        basis = _turn_displacement_basis(heading0_deg, target_heading_deg, turn_rate_deg_sec, float(times_s[idx]))
        xy[idx, 0] = start_xy[0] + speed_mps * basis[0]
        xy[idx, 1] = start_xy[1] + speed_mps * basis[1]
    return xy


def speed_segment_at_time(
    start_xy: np.ndarray,
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    nominal_speed_mps: float,
    min_speed_mps: float,
    max_speed_mps: float,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the min/nominal/max positions for one aircraft at time `t_s`."""
    basis = _turn_displacement_basis(heading0_deg, target_heading_deg, turn_rate_deg_sec, t_s)
    pos_min = start_xy + min_speed_mps * basis  # pyright: ignore[reportOperatorIssue]
    pos_nom = start_xy + nominal_speed_mps * basis  # pyright: ignore[reportOperatorIssue]
    pos_max = start_xy + max_speed_mps * basis  # pyright: ignore[reportOperatorIssue]
    return pos_min, pos_nom, pos_max


def envelope_distance_curve(
    rel_pos0: np.ndarray,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_turn_rate_deg_sec: float,
    a_min_speed_mps: float,
    a_max_speed_mps: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_turn_rate_deg_sec: float,
    b_min_speed_mps: float,
    b_max_speed_mps: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Sample the exact fixed-time envelope distance `d(t)` over a time grid."""
    distances_m = np.empty(times_s.shape[0], dtype=np.float64)
    for idx in range(times_s.shape[0]):
        distances_m[idx] = _min_distance_to_relative_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=a_heading0_deg,
            a_target_heading_deg=a_target_heading_deg,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min_speed_mps,
            a_max_speed_mps=a_max_speed_mps,
            b_heading0_deg=b_heading0_deg,
            b_target_heading_deg=b_target_heading_deg,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min_speed_mps,
            b_max_speed_mps=b_max_speed_mps,
            t_s=float(times_s[idx]),
        )
    return distances_m


def corridor_polygon(traj_min: np.ndarray, traj_max: np.ndarray) -> np.ndarray:
    """Return a closed polygon spanning the min/max speed trajectory corridor."""
    return np.vstack((traj_min, traj_max[::-1]))


def focus_window_end_times(
    projection_time_s: float,
    closest_time_s: float,
    a_turn_end_s: float,
    b_turn_end_s: float,
) -> tuple[float, float]:
    """Choose spatial and distance-curve windows that focus on the relevant interaction."""
    spatial_focus_end_s = min(
        projection_time_s,
        max(150.0, closest_time_s + 60.0, a_turn_end_s + 30.0, b_turn_end_s + 30.0),
    )
    time_focus_end_s = min(
        projection_time_s,
        max(180.0, closest_time_s + 90.0, a_turn_end_s + 45.0, b_turn_end_s + 45.0),
    )
    return spatial_focus_end_s, time_focus_end_s


def square_bounds(points_xy: np.ndarray, pad_m: float) -> dict[str, float]:
    """Return square XY bounds with padding around a point cloud."""
    min_x = float(np.min(points_xy[:, 0]))
    max_x = float(np.max(points_xy[:, 0]))
    min_y = float(np.min(points_xy[:, 1]))
    max_y = float(np.max(points_xy[:, 1]))
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    half_span_m = 0.5 * max(max_x - min_x, max_y - min_y) + pad_m
    return {
        "x_min": centre_x - half_span_m,
        "x_max": centre_x + half_span_m,
        "y_min": centre_y - half_span_m,
        "y_max": centre_y + half_span_m,
        "half_span_m": half_span_m,
    }


def build_turn_debug_payload(
    *,
    a_lat: float,
    a_lon: float,
    a_heading0_deg: float,
    a_target_heading_deg: float,
    a_speed_kt: float,
    a_turn_rate_deg_sec: float,
    b_lat: float,
    b_lon: float,
    b_heading0_deg: float,
    b_target_heading_deg: float,
    b_speed_kt: float,
    b_turn_rate_deg_sec: float,
    separation_threshold_m: float,
    speed_diff_kt: float,
    projection_time_s: float,
    turn_speed_schedule_uncertainty_kt: float = 0.0,
) -> dict[str, Any]:
    """Build the full turn-aware debug payload consumed by the web app."""
    is_sep, min_distance_m, closest_time_s = catch_up_projection_interval_with_turns(
        a_lat=a_lat,
        a_lon=a_lon,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_speed_kt=a_speed_kt,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        b_lat=b_lat,
        b_lon=b_lon,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_speed_kt=b_speed_kt,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        separation_threshold_m=separation_threshold_m,
        speed_diff_kt=speed_diff_kt,
        projection_time_s=projection_time_s,
        turn_speed_schedule_uncertainty_kt=turn_speed_schedule_uncertainty_kt,
    )

    a_nom_speed_mps = a_speed_kt * KT_TO_MPS
    b_nom_speed_mps = b_speed_kt * KT_TO_MPS
    a_min_speed_mps, a_max_speed_mps = speed_bounds_mps(a_speed_kt, speed_diff_kt)
    b_min_speed_mps, b_max_speed_mps = speed_bounds_mps(b_speed_kt, speed_diff_kt)
    a_turn_end_s = turn_duration_s(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec)
    b_turn_end_s = turn_duration_s(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec)

    a_xy0 = latlon_to_local_xy(a_lat, a_lon, a_lat, a_lon)
    b_xy0 = latlon_to_local_xy(b_lat, b_lon, a_lat, a_lon)
    rel_pos0 = a_xy0 - b_xy0  # pyright: ignore[reportOperatorIssue]

    spatial_focus_end_s, time_focus_end_s = focus_window_end_times(
        projection_time_s,
        closest_time_s,
        a_turn_end_s,
        b_turn_end_s,
    )
    traj_sample_count = min(361, max(161, int(spatial_focus_end_s / 1.5) + 1))
    traj_times_s = np.linspace(0.0, spatial_focus_end_s, traj_sample_count)
    if 0.0 < closest_time_s < spatial_focus_end_s:
        traj_times_s = np.unique(np.concatenate((traj_times_s, np.asarray([closest_time_s], dtype=np.float64))))

    curve_sample_count = min(601, max(241, int(projection_time_s / 2.0) + 1))
    curve_times_s = np.linspace(0.0, projection_time_s, curve_sample_count)
    if 0.0 < closest_time_s < projection_time_s:
        curve_times_s = np.unique(np.concatenate((curve_times_s, np.asarray([closest_time_s], dtype=np.float64))))

    a_traj_nom = sample_trajectory_xy(
        a_xy0,
        a_heading0_deg,
        a_target_heading_deg,
        a_nom_speed_mps,
        a_turn_rate_deg_sec,
        traj_times_s,
    )
    b_traj_nom = sample_trajectory_xy(
        b_xy0,
        b_heading0_deg,
        b_target_heading_deg,
        b_nom_speed_mps,
        b_turn_rate_deg_sec,
        traj_times_s,
    )
    a_traj_min = sample_trajectory_xy(
        a_xy0,
        a_heading0_deg,
        a_target_heading_deg,
        a_min_speed_mps,
        a_turn_rate_deg_sec,
        traj_times_s,
    )
    a_traj_max = sample_trajectory_xy(
        a_xy0,
        a_heading0_deg,
        a_target_heading_deg,
        a_max_speed_mps,
        a_turn_rate_deg_sec,
        traj_times_s,
    )
    b_traj_min = sample_trajectory_xy(
        b_xy0,
        b_heading0_deg,
        b_target_heading_deg,
        b_min_speed_mps,
        b_turn_rate_deg_sec,
        traj_times_s,
    )
    b_traj_max = sample_trajectory_xy(
        b_xy0,
        b_heading0_deg,
        b_target_heading_deg,
        b_max_speed_mps,
        b_turn_rate_deg_sec,
        traj_times_s,
    )

    envelope_distances_m = envelope_distance_curve(
        rel_pos0=rel_pos0,
        a_heading0_deg=a_heading0_deg,
        a_target_heading_deg=a_target_heading_deg,
        a_turn_rate_deg_sec=a_turn_rate_deg_sec,
        a_min_speed_mps=a_min_speed_mps,
        a_max_speed_mps=a_max_speed_mps,
        b_heading0_deg=b_heading0_deg,
        b_target_heading_deg=b_target_heading_deg,
        b_turn_rate_deg_sec=b_turn_rate_deg_sec,
        b_min_speed_mps=b_min_speed_mps,
        b_max_speed_mps=b_max_speed_mps,
        times_s=curve_times_s,
    )
    a_curve_nom = sample_trajectory_xy(
        a_xy0,
        a_heading0_deg,
        a_target_heading_deg,
        a_nom_speed_mps,
        a_turn_rate_deg_sec,
        curve_times_s,
    )
    b_curve_nom = sample_trajectory_xy(
        b_xy0,
        b_heading0_deg,
        b_target_heading_deg,
        b_nom_speed_mps,
        b_turn_rate_deg_sec,
        curve_times_s,
    )
    nominal_distances_m = np.linalg.norm(a_curve_nom - b_curve_nom, axis=1)
    time_focus_mask = curve_times_s <= time_focus_end_s + 1e-9
    curve_times_focus_s = curve_times_s[time_focus_mask]
    envelope_distances_focus_nm = envelope_distances_m[time_focus_mask] * M_TO_NMI
    nominal_distances_focus_nm = nominal_distances_m[time_focus_mask] * M_TO_NMI

    frames: list[dict[str, Any]] = []
    relative_points: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    closest_frame_index = 0
    closest_delta_s = float("inf")
    for idx, t_s in enumerate(traj_times_s):
        a_pos_min, a_pos_nom, a_pos_max = speed_segment_at_time(
            a_xy0,
            a_heading0_deg,
            a_target_heading_deg,
            a_turn_rate_deg_sec,
            a_nom_speed_mps,
            a_min_speed_mps,
            a_max_speed_mps,
            float(t_s),
        )
        b_pos_min, b_pos_nom, b_pos_max = speed_segment_at_time(
            b_xy0,
            b_heading0_deg,
            b_target_heading_deg,
            b_turn_rate_deg_sec,
            b_nom_speed_mps,
            b_min_speed_mps,
            b_max_speed_mps,
            float(t_s),
        )
        rel_hull = _relative_position_hull_at_time(
            rel_pos0=rel_pos0,
            a_heading0_deg=a_heading0_deg,
            a_target_heading_deg=a_target_heading_deg,
            a_turn_rate_deg_sec=a_turn_rate_deg_sec,
            a_min_speed_mps=a_min_speed_mps,
            a_max_speed_mps=a_max_speed_mps,
            b_heading0_deg=b_heading0_deg,
            b_target_heading_deg=b_target_heading_deg,
            b_turn_rate_deg_sec=b_turn_rate_deg_sec,
            b_min_speed_mps=b_min_speed_mps,
            b_max_speed_mps=b_max_speed_mps,
            t_s=float(t_s),
        )
        rel_nom = a_pos_nom - b_pos_nom  # pyright: ignore[reportOperatorIssue]
        rel_closest_point, _ = closest_point_on_convex_polygon(np.zeros(2, dtype=np.float64), rel_hull)
        relative_points.extend((rel_hull, rel_nom[None, :], rel_closest_point[None, :]))
        nominal_distance_nm = float(np.linalg.norm(rel_nom) * M_TO_NMI)
        envelope_distance_nm = float(np.linalg.norm(rel_closest_point) * M_TO_NMI)
        frames.append(
            {
                "time_s": float(t_s),
                "time_min": float(t_s / 60.0),
                "a": {"min": a_pos_min.tolist(), "nominal": a_pos_nom.tolist(), "max": a_pos_max.tolist()},
                "b": {"min": b_pos_min.tolist(), "nominal": b_pos_nom.tolist(), "max": b_pos_max.tolist()},
                "relative_hull": rel_hull.tolist(),
                "relative_nominal": rel_nom.tolist(),
                "relative_closest_point": rel_closest_point.tolist(),
                "nominal_distance_nm": nominal_distance_nm,
                "envelope_distance_nm": envelope_distance_nm,
            }
        )
        delta_s = abs(float(t_s) - closest_time_s)
        if delta_s < closest_delta_s:
            closest_delta_s = delta_s
            closest_frame_index = idx

    a_nom_turn_end_xy = None
    if 0.0 < a_turn_end_s <= projection_time_s:
        a_nom_turn_end_xy = sample_trajectory_xy(
            a_xy0,
            a_heading0_deg,
            a_target_heading_deg,
            a_nom_speed_mps,
            a_turn_rate_deg_sec,
            np.asarray([a_turn_end_s], dtype=np.float64),
        )[0]
    b_nom_turn_end_xy = None
    if 0.0 < b_turn_end_s <= projection_time_s:
        b_nom_turn_end_xy = sample_trajectory_xy(
            b_xy0,
            b_heading0_deg,
            b_target_heading_deg,
            b_nom_speed_mps,
            b_turn_rate_deg_sec,
            np.asarray([b_turn_end_s], dtype=np.float64),
        )[0]

    spatial_points = [a_traj_nom, b_traj_nom, a_traj_min, a_traj_max, b_traj_min, b_traj_max]
    if a_nom_turn_end_xy is not None:
        spatial_points.append(a_nom_turn_end_xy[None, :])
    if b_nom_turn_end_xy is not None:
        spatial_points.append(b_nom_turn_end_xy[None, :])
    spatial_bounds = square_bounds(np.vstack(spatial_points), pad_m=max(0.35 * separation_threshold_m, 1500.0))
    relative_bounds = square_bounds(np.vstack(relative_points), pad_m=max(0.4 * separation_threshold_m, 750.0))

    return {
        "model_mode": "turn_aware",
        "origins": {"a": a_xy0.tolist(), "b": b_xy0.tolist()},
        "summary": {
            "is_separated": bool(is_sep),
            "min_distance_nm": float(min_distance_m * M_TO_NMI),
            "closest_time_min": float(closest_time_s / 60.0),
            "closest_time_s": float(closest_time_s),
        },
        "frame_interval_s": float(traj_times_s[1] - traj_times_s[0]) if traj_times_s.shape[0] > 1 else 0.0,
        "closest_frame_index": closest_frame_index,
        "frames": frames,
        "trajectories": {
            "times_s": traj_times_s.tolist(),
            "a": {
                "nominal": a_traj_nom.tolist(),
                "min": a_traj_min.tolist(),
                "max": a_traj_max.tolist(),
                "corridor": corridor_polygon(a_traj_min, a_traj_max).tolist(),
            },
            "b": {
                "nominal": b_traj_nom.tolist(),
                "min": b_traj_min.tolist(),
                "max": b_traj_max.tolist(),
                "corridor": corridor_polygon(b_traj_min, b_traj_max).tolist(),
            },
        },
        "distance_curve": {
            "times_s": curve_times_focus_s.tolist(),
            "envelope_nm": envelope_distances_focus_nm.tolist(),
            "nominal_nm": nominal_distances_focus_nm.tolist(),
            "threshold_nm": float(separation_threshold_m * M_TO_NMI),
        },
        "turns": {
            "a": {
                "start_heading_deg": float(a_heading0_deg),
                "target_heading_deg": float(a_target_heading_deg),
                "turn_rate_deg_sec": float(a_turn_rate_deg_sec),
                "end_time_s": float(a_turn_end_s),
                "end_position": None if a_nom_turn_end_xy is None else a_nom_turn_end_xy.tolist(),
            },
            "b": {
                "start_heading_deg": float(b_heading0_deg),
                "target_heading_deg": float(b_target_heading_deg),
                "turn_rate_deg_sec": float(b_turn_rate_deg_sec),
                "end_time_s": float(b_turn_end_s),
                "end_position": None if b_nom_turn_end_xy is None else b_nom_turn_end_xy.tolist(),
            },
        },
        "bounds_xy": spatial_bounds,
        "relative_bounds_xy": relative_bounds,
        "separation_threshold_m": float(separation_threshold_m),
        "info": {
            "a_heading_summary": heading_summary(a_heading0_deg, a_target_heading_deg, a_turn_rate_deg_sec, a_speed_kt),
            "b_heading_summary": heading_summary(b_heading0_deg, b_target_heading_deg, b_turn_rate_deg_sec, b_speed_kt),
            "speed_diff_kt": float(speed_diff_kt),
            "turn_speed_schedule_uncertainty_kt": float(turn_speed_schedule_uncertainty_kt),
            "spatial_focus_end_s": float(spatial_focus_end_s),
            "time_focus_end_s": float(time_focus_end_s),
            "projection_time_s": float(projection_time_s),
        },
    }
