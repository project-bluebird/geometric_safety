"""Trajectory sampling and scenarios shared by the web and standalone visualisers.

Everything in this module is auxiliary support for the interactive FastAPI
visualiser and the standalone ``scripts/visualise.py`` demos.
The core solver in ``geometric_safety.relevant_aircraft`` does not depend on
any of these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from geometric_safety.relevant_aircraft import (
    _min_distance_to_relative_hull_at_time,
    _turn_displacement_basis,
)
from geometric_safety.util import (
    DEG_TO_RAD,
    EARTH_RADIUS_IN_METERS,
    KT_TO_MPS,
    RAD_TO_DEG,
    shortest_turn_parameters,
)

DEMO_REF_LAT = 51.0
DEMO_REF_LON = -1.0


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
    """Return the deterministic duration of a commanded constant-rate turn."""
    return float(shortest_turn_parameters(heading0_deg, target_heading_deg, turn_rate_deg_sec)[2])


def heading_summary(
    heading0_deg: float,
    target_heading_deg: float,
    turn_rate_deg_sec: float,
    speed_kt: float,
) -> str:
    """Format one aircraft's lateral plan for debug text overlays and tooltips."""
    angle, _, _ = shortest_turn_parameters(heading0_deg, target_heading_deg, turn_rate_deg_sec)
    if angle == 0.0:
        return f"{heading0_deg:.0f} deg straight @ {speed_kt:.0f} kt"
    return f"{heading0_deg:.0f}->{target_heading_deg:.0f} deg @ {abs(turn_rate_deg_sec):.1f} deg/s, {speed_kt:.0f} kt"


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
