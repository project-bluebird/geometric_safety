"""Public input contracts across lateral, vertical and compiled utility paths."""

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal, Protocol, cast, overload

import numpy as np
import pytest

from geometric_safety import (
    catch_up_projection_interval,
    catch_up_projection_interval_with_turns,
    catch_up_projection_interval_with_turns_and_vertical,
    catch_up_projection_interval_with_vertical,
    relevant_aircraft as ra,
)
from geometric_safety._instrumentation import python_function
from geometric_safety.util import EARTH_RADIUS_IN_METERS, KT_TO_MPS, convex_hull, latlon_to_local_xy
from geometric_safety.vertical import (
    level_at_time_fl,
    time_to_vertical_overlap_resolution,
    vertical_band_gap_fl,
    vertical_bands_are_resolved,
)

type RealScalar = float | np.integer | np.floating
type LegacyKind = Literal["straight", "turning", "straight_vertical", "turning_vertical"]
type LegacyResult = tuple[bool, float, float] | tuple[bool, float, float, float]


class ScalarDispatcher(Protocol):
    """Numba's scalar-call boundary, including NumPy values and invalid declarations."""

    def __call__(self, **arguments: RealScalar) -> LegacyResult: ...


@overload
def _legacy(kind: Literal["straight", "turning"], **changes: RealScalar) -> tuple[bool, float, float]: ...


@overload
def _legacy(
    kind: Literal["straight_vertical", "turning_vertical"], **changes: RealScalar
) -> tuple[bool, float, float, float]: ...


def _legacy(kind: LegacyKind, **changes: RealScalar) -> LegacyResult:
    arguments: dict[str, RealScalar] = {
        "a_lat": 51.0,
        "a_lon": -1.0,
        "a_speed_kt": 250.0,
        "b_lat": 51.1,
        "b_lon": -1.0,
        "b_speed_kt": 250.0,
        "separation_threshold_m": 5556.0,
        "speed_diff_kt": 10.0,
        "projection_time_s": 120.0,
    }
    if kind.startswith("straight"):
        arguments.update(a_heading=0.0, b_heading=90.0)
    else:
        arguments.update(
            a_heading0_deg=0.0,
            a_target_heading_deg=90.0,
            a_turn_rate_deg_sec=3.0,
            b_heading0_deg=90.0,
            b_target_heading_deg=90.0,
            b_turn_rate_deg_sec=0.0,
        )
    if kind.endswith("vertical"):
        arguments.update(a_current_fl=300.0, a_selected_fl=300.0, b_current_fl=300.0, b_selected_fl=300.0)
    arguments.update(changes)
    dispatchers = {
        "straight": catch_up_projection_interval,
        "turning": catch_up_projection_interval_with_turns,
        "straight_vertical": catch_up_projection_interval_with_vertical,
        "turning_vertical": catch_up_projection_interval_with_turns_and_vertical,
    }
    # Exercise the compiled dispatcher without converting scalar types to the
    # Python annotations; invalid declarations must also reach runtime validation.
    return cast("ScalarDispatcher", dispatchers[kind])(**arguments)


@pytest.mark.parametrize("kind", ["straight_vertical", "turning_vertical"])
def test_nonfinite_selected_level_cannot_certify_a_colocated_pair(kind: LegacyKind) -> None:
    with pytest.raises(ValueError, match="finite"):
        _legacy(kind, b_lat=51.0, a_selected_fl=math.nan)


def test_negative_width_cannot_certify_a_reachable_conflict() -> None:
    latitude = 51.0
    a_lon = -1.0
    b_lon = a_lon + math.degrees(57692.0 / (EARTH_RADIUS_IN_METERS * math.cos(math.radians(latitude))))
    b_lat = latitude + math.degrees(18378.0 / EARTH_RADIUS_IN_METERS)
    arguments = {
        "a_heading0_deg": 65.19,
        "a_target_heading_deg": 70.95,
        "a_turn_rate_deg_sec": 1.887,
        "a_speed_kt": 312.78,
        "b_heading0_deg": 282.90,
        "b_target_heading_deg": 282.90,
        "b_speed_kt": 268.53,
        "b_lat": b_lat,
        "b_lon": b_lon,
        "separation_threshold_m": 9260.0,
        "projection_time_s": 300.0,
    }
    assert not _legacy("turning", **arguments, speed_diff_kt=126.65)[0]
    result = _legacy("turning", **arguments, speed_diff_kt=-126.65)
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


def test_integer_envelope_matches_float_on_exported_turn_api() -> None:
    expected = _legacy("turning", turn_speed_schedule_uncertainty_kt=10.0)
    assert _legacy("turning", turn_speed_schedule_uncertainty_kt=10) == expected


def test_vertical_shortcut_still_validates_turn_declarations() -> None:
    with pytest.raises(ValueError, match="finite"):
        _legacy("turning_vertical", b_current_fl=350.0, b_selected_fl=350.0, a_heading0_deg=math.nan)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_convex_hull_accepts_real_array_dtypes(dtype: type[np.signedinteger] | type[np.floating]) -> None:
    hull = convex_hull(np.array([[0, 0], [4, 0], [0, 3]], dtype=dtype))
    assert hull.dtype == np.float64
    assert {tuple(point) for point in hull} == {(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)}


def test_nonfinite_hull_points_do_not_disappear() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [math.nan, 5.0], [1.0, 1.0], [0.0, 1.0]])
    assert np.isnan(convex_hull(points)).all()


def test_longitude_normalisation_handles_multiple_revolutions() -> None:
    position = latlon_to_local_xy(80.0, 719.99, 80.0, -0.01)
    assert np.linalg.norm(position) < 1e-6
    result = catch_up_projection_interval(80.0, -0.01, 90.0, 400.0, 80.0, 719.99, 270.0, 400.0, 9260.0, 0.0, 300.0)
    assert not result[0]
    assert result[1] < 1e-6


@pytest.mark.parametrize("kind", ["straight_vertical", "turning_vertical"])
@pytest.mark.parametrize(
    "field",
    [
        "a_current_fl",
        "a_selected_fl",
        "b_current_fl",
        "b_selected_fl",
        "vertical_rate_fpm",
        "required_vertical_gap_fl",
        "vertical_rounding_fl",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("separated", [False, True])
def test_vertical_declarations_are_validated_before_every_shortcut(
    kind: LegacyKind, field: str, value: float, separated: bool
) -> None:
    changes = {"b_current_fl": 350.0, "b_selected_fl": 350.0} if separated else {}
    changes[field] = value
    with pytest.raises(ValueError, match="finite"):
        _legacy(kind, **changes)


@pytest.mark.parametrize("kind", ["straight", "turning", "straight_vertical", "turning_vertical"])
@pytest.mark.parametrize(
    "field", ["a_speed_kt", "b_speed_kt", "speed_diff_kt", "separation_threshold_m", "projection_time_s"]
)
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_lateral_scalars_cannot_be_hidden_by_vertical_separation(
    kind: LegacyKind, field: str, value: float
) -> None:
    changes = {"b_current_fl": 350.0, "b_selected_fl": 350.0} if kind.endswith("vertical") else {}
    changes[field] = value
    result = _legacy(kind, **changes)
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


@pytest.mark.parametrize("kind", ["straight", "turning", "straight_vertical", "turning_vertical"])
@pytest.mark.parametrize("field", ["a_lat", "a_lon", "b_lat", "b_lon"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_never_certify(kind: LegacyKind, field: str, value: float) -> None:
    changes = {"b_current_fl": 350.0, "b_selected_fl": 350.0} if kind.endswith("vertical") else {}
    changes[field] = value
    result = _legacy(kind, **changes)
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


@pytest.mark.parametrize(
    "field",
    [
        "a_heading0_deg",
        "a_target_heading_deg",
        "a_turn_rate_deg_sec",
        "b_heading0_deg",
        "b_target_heading_deg",
        "b_turn_rate_deg_sec",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_turn_shortcut_rejects_nonfinite_motion_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _legacy("turning_vertical", b_current_fl=350.0, b_selected_fl=350.0, **{field: value})


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_turn_shortcut_rejects_invalid_envelopes(value: float) -> None:
    with pytest.raises(ValueError, match="envelope"):
        _legacy("turning_vertical", b_current_fl=350.0, b_selected_fl=350.0, turn_speed_schedule_uncertainty_kt=value)


def test_vertical_shortcut_allows_unused_zero_rate_turns() -> None:
    result = _legacy("turning_vertical", b_current_fl=350.0, b_selected_fl=350.0, a_turn_rate_deg_sec=0.0)
    assert result[0]
    assert result[2:] == (0.0, 0.0)
    with pytest.raises(ValueError, match="nonzero"):
        _legacy("turning_vertical", a_turn_rate_deg_sec=0.0)


@pytest.mark.parametrize(
    "field",
    ["eta_const_m", "eta_rate_mps", "turn_speed_schedule_uncertainty_kt", "turn_delay_max_s", "turn_rate_error_max"],
)
@pytest.mark.parametrize("value", [0, np.int64(0), np.float32(0.125), False])
def test_envelope_scalar_types_match_their_float_values(field: str, value: RealScalar) -> None:
    assert _legacy("turning", **{field: value}) == _legacy("turning", **{field: float(value)})


@pytest.mark.parametrize("value", [0, np.int64(0), np.float32(0.125), False])
def test_combined_turn_envelope_scalar_types(value: RealScalar) -> None:
    assert _legacy("turning_vertical", turn_speed_schedule_uncertainty_kt=value) == _legacy(
        "turning_vertical", turn_speed_schedule_uncertainty_kt=float(value)
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_public_vertical_helpers_reject_nonfinite_motion(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        level_at_time_fl(300.0, value, 1000.0, 0.0)
    for helper in (vertical_band_gap_fl, vertical_bands_are_resolved, time_to_vertical_overlap_resolution):
        with pytest.raises(ValueError, match="finite"):
            helper(300.0, value, 350.0, 350.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("row", [0, 1, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_nonfinite_hull_rows_are_rejected_in_every_position(value: float, row: int, axis: int) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]])
    points[row, axis] = value
    hull = convex_hull(points)
    assert hull.shape == (1, 2)
    assert np.isnan(hull).all()
    result = catch_up_projection_interval(
        51.0, -1.0, 0.0, 250.0, 51.1, -1.0, 90.0, 250.0, 5556.0, 10.0, 120.0, velocity_hull=hull
    )
    assert not result[0]


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_invalid_subdivision_width_has_unavailable_diagnostics(value: float) -> None:
    query = {
        "rel_pos0": np.array([20_000.0, 0.0]),
        "a_heading0_deg": 0.0,
        "a_target_heading_deg": 90.0,
        "a_speed_kt": 250.0,
        "a_turn_rate_deg_sec": 3.0,
        "b_heading0_deg": 90.0,
        "b_target_heading_deg": 90.0,
        "b_speed_kt": 250.0,
        "b_turn_rate_deg_sec": 0.0,
        "separation_threshold_m": 9260.0,
        "speed_diff_kt": 10.0,
        "projection_time_s": 120.0,
        "min_cert_interval_s": value,
    }
    for solver in (
        ra._catch_up_projection_interval_with_turns_local,
        python_function(ra._catch_up_projection_interval_with_turns_local),
    ):
        result = solver(**query)
        assert not result[0]
        assert math.isnan(result[1])
        assert math.isnan(result[2])


@pytest.mark.parametrize("revolutions", [-4, -2, -1, 0, 1, 2, 4])
def test_wrapped_longitudes_preserve_a_local_offset(revolutions: int) -> None:
    expected = latlon_to_local_xy(51.001, -0.999, 51.0, -1.0)
    actual = latlon_to_local_xy(51.001, -0.999 + 360.0 * revolutions, 51.0, -1.0)
    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("kind", ["straight_vertical", "turning_vertical"])
def test_legacy_vertical_conventions_remain_available(kind: Literal["straight_vertical", "turning_vertical"]) -> None:
    encounter = {"a_current_fl": 300.0, "a_selected_fl": 300.0, "b_current_fl": 300.0, "b_selected_fl": 330.0}
    positive = _legacy(kind, **encounter, vertical_rate_fpm=1000.0, vertical_rounding_fl=0)
    assert positive == _legacy(kind, **encounter, vertical_rate_fpm=-1000.0, vertical_rounding_fl=-10)
    assert len(positive) == 4
    assert positive[3] == pytest.approx(60.0)
    zero_gap = _legacy(kind, **encounter, required_vertical_gap_fl=0.0)
    assert zero_gap[0]
    assert len(zero_gap) == 4
    assert zero_gap[3] == 0.0
    assert zero_gap == _legacy(kind, **encounter, required_vertical_gap_fl=-1.0)
    fractional_grid = _legacy(kind, **encounter, vertical_rounding_fl=2.5)
    assert len(fractional_grid) == 4
    assert fractional_grid[3] == pytest.approx(60.0, abs=1e-6)


@pytest.mark.parametrize("kind", ["straight", "straight_vertical"])
@pytest.mark.parametrize("field", ["a_heading", "b_heading"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_straight_shortcuts_retain_unavailable_diagnostics(kind: LegacyKind, field: str, value: float) -> None:
    changes = {"b_current_fl": 350.0, "b_selected_fl": 350.0} if kind.endswith("vertical") else {}
    changes[field] = value
    result = _legacy(kind, **changes)
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


@pytest.mark.parametrize("value", [0, np.float32(0.0)])
def test_integer_and_single_precision_envelopes_preserve_straight_delegation(value: RealScalar) -> None:
    result = _legacy("turning", a_target_heading_deg=0.0, eta_const_m=value, turn_delay_max_s=value)
    assert result == _legacy("straight")


@pytest.mark.parametrize("kind", ["straight", "turning", "straight_vertical", "turning_vertical"])
def test_zero_horizon_does_not_hide_invalid_speed_bounds(kind: LegacyKind) -> None:
    changes = {"b_current_fl": 350.0, "b_selected_fl": 350.0} if kind.endswith("vertical") else {}
    result = _legacy(kind, **changes, speed_diff_kt=-1.0, projection_time_s=0.0)
    assert not result[0]
    assert math.isnan(result[1])
    assert math.isnan(result[2])


@pytest.mark.parametrize("axis", [0, 1])
def test_single_nonfinite_vertex_is_not_a_hull(axis: int) -> None:
    point = np.zeros((1, 2))
    point[0, axis] = math.nan
    assert np.isnan(convex_hull(point)).all()


def test_validation_survives_first_compilation_through_fast_math(tmp_path: Path) -> None:
    # The ordinary suite may compile strict callees first. An empty cache and a
    # separate interpreter exercise compilation through the outer fast-math caller.
    source = """import sys
sys.path.insert(0, sys.argv[1])

import math
import numba
import numpy as np

from geometric_safety import catch_up_projection_interval_with_turns_and_vertical
from geometric_safety.util import convex_hull


@numba.njit(fastmath=True)
def combined(speed: float, width: float, level: float) -> tuple[bool, float, float, float]:
    return catch_up_projection_interval_with_turns_and_vertical(
        51.0, -1.0, 0.0, 90.0, speed, 3.0, 300.0, level,
        51.0, -1.0, 90.0, 90.0, 250.0, 0.0, 350.0, 350.0,
        5556.0, width, 120.0,
    )


@numba.njit(fastmath=True)
def hull(points: np.ndarray) -> np.ndarray:
    return convex_hull(points)


try:
    combined(250.0, 10.0, math.nan)
except ValueError:
    print("vertical nan rejected")
else:
    raise AssertionError("nonfinite vertical declaration accepted")
for speed, width in ((math.nan, 10.0), (250.0, math.nan), (250.0, -1.0)):
    result = combined(speed, width, 300.0)
    assert not result[0], result
    assert math.isnan(result[1]), result
    print("invalid lateral bounds rejected", speed, width)
for axis in (0, 1):
    points = np.zeros((5, 2))
    points[1, axis] = math.nan
    actual = hull(points)
    assert np.isnan(actual).all(), actual
print("hull nonfinite rejected")"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", source, str(Path(__file__).resolve().parents[1])],
        env=dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hull nonfinite rejected" in result.stdout


@pytest.mark.parametrize("heading", [1e20, -1e20, sys.float_info.max, -sys.float_info.max])
@pytest.mark.parametrize("profile", ["straight", "turn_from", "turn_to"])
@pytest.mark.parametrize("vertical", [False, True])
def test_large_finite_headings_preserve_collision_geometry(heading: float, profile: str, vertical: bool) -> None:
    reduced = math.fmod(heading, 360.0)
    start = reduced - 90.0 if profile == "turn_to" else reduced
    target = start if profile == "straight" else start + 90.0
    initial, final = math.radians(start), math.radians(target)
    if profile == "straight":
        east, north = 10_000.0 * math.sin(initial), 10_000.0 * math.cos(initial)
    else:
        # Integrate the circular arc and its 70-second straight continuation.
        omega = math.radians(3.0)
        east = 100.0 * ((math.cos(initial) - math.cos(final)) / omega + 70.0 * math.sin(final))
        north = 100.0 * ((math.sin(final) - math.sin(initial)) / omega + 70.0 * math.cos(final))
    kind: LegacyKind
    if profile == "straight":
        kind = "straight_vertical" if vertical else "straight"
        normal = {"a_heading": start, "b_heading": 0.0}
        declared = dict(normal, a_heading=heading)
    else:
        kind = "turning_vertical" if vertical else "turning"
        normal = {
            "a_heading0_deg": start,
            "a_target_heading_deg": target,
            "a_turn_rate_deg_sec": 3.0,
            "b_heading0_deg": 0.0,
            "b_target_heading_deg": 0.0,
            "b_turn_rate_deg_sec": 0.0,
        }
        declared = dict(normal)
        declared["a_target_heading_deg" if profile == "turn_to" else "a_heading0_deg"] = heading
    common = {
        "a_lat": 0.0,
        "a_lon": 0.0,
        "a_speed_kt": 100.0 / KT_TO_MPS,
        "b_speed_kt": 0.0,
        "b_lat": math.degrees(north / EARTH_RADIUS_IN_METERS),
        "b_lon": math.degrees(east / EARTH_RADIUS_IN_METERS),
        "speed_diff_kt": 0.0,
        "projection_time_s": 100.0,
        "separation_threshold_m": 1.0,
    }
    expected = _legacy(kind, **common, **normal)
    actual = _legacy(kind, **common, **declared)
    assert not expected[0]
    assert expected[1] < 1e-6
    assert not actual[0]
    assert actual[1:] == pytest.approx(expected[1:], abs=1e-6)


@pytest.mark.parametrize("heading", [1e20, -1e20, sys.float_info.max, -sys.float_info.max])
def test_large_heading_turn_profile_keeps_its_continuous_lift(heading: float) -> None:
    start = math.fmod(heading, 360.0)
    target = start + 90.0
    for time in (0.0, 10.0, 20.0, 30.0, 100.0):
        actual = ra._heading_at_time_from_turn_parameters(heading, target, 3.0, 30.0, time)
        assert actual == pytest.approx(start + 3.0 * min(time, 30.0), abs=1e-12)
    assert ra._heading_at_time_from_turn_parameters(heading, heading, 0.0, 0.0, 100.0) == start
