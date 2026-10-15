"""Geographic entry points preserve the planar contract and coordinate orientation."""

from dataclasses import replace
import math
from typing import Literal, TypedDict

import pytest

from geometric_safety import (
    HeadingMotion,
    certify_heading_separation,
    certify_heading_separation_from_latlon,
    certify_heading_separation_with_vertical,
    certify_heading_separation_with_vertical_from_latlon,
)
from geometric_safety.util import EARTH_RADIUS_IN_METERS, latlon_to_local_xy


class _Coordinates(TypedDict):
    a_lat: float
    a_lon: float
    b_lat: float
    b_lon: float


class _LateralParameters(TypedDict):
    separation_threshold_m: float
    horizon_s: float
    relative_position_error_m: float
    relative_velocity_error_mps: float


class _VerticalParameters(TypedDict):
    a_current_fl: float
    a_selected_fl: float
    b_current_fl: float
    b_selected_fl: float
    minimum_vertical_rate_fpm: float
    required_vertical_gap_fl: float
    vertical_rounding_fl: int


def _straight(heading: float, speed: float = 100.0) -> HeadingMotion:
    return HeadingMotion(
        heading0_deg=heading,
        target_heading_deg=heading,
        turn_rate_deg_s=0.0,
        speed_min_mps=speed,
        speed_max_mps=speed,
    )


def _coordinates() -> _Coordinates:
    return {"a_lat": 51.0, "a_lon": 0.0, "b_lat": 51.1, "b_lon": -0.5}


def _lateral() -> _LateralParameters:
    return {
        "separation_threshold_m": 9260.0,
        "horizon_s": 120.0,
        "relative_position_error_m": 71.0,
        "relative_velocity_error_mps": 2.3,
    }


def _vertical() -> _VerticalParameters:
    return {
        "a_current_fl": 103.0,
        "a_selected_fl": 103.0,
        "b_current_fl": 103.0,
        "b_selected_fl": 123.0,
        "minimum_vertical_rate_fpm": 600.0,
        "required_vertical_gap_fl": 7.0,
        "vertical_rounding_fl": 5,
    }


def _legacy_relative_position(coordinates: _Coordinates) -> tuple[float, float]:
    offset = latlon_to_local_xy(coordinates["b_lat"], coordinates["b_lon"], coordinates["a_lat"], coordinates["a_lon"])
    return -float(offset[0]), -float(offset[1])


@pytest.mark.parametrize("latitude", [-51.0, 0.0, 51.0])
@pytest.mark.parametrize("turning", [False, True])
def test_geographic_results_match_explicit_existing_projection(latitude: float, turning: bool) -> None:
    coordinates = _coordinates()
    coordinates.update(a_lat=latitude, b_lat=latitude + 0.1)
    a = replace(_straight(270.0), speed_min_mps=95.0, speed_max_mps=110.0, heading_deviation_max_deg=0.5)
    if turning:
        a = replace(a, target_heading_deg=310.0, turn_rate_deg_s=1.5)
    b = replace(_straight(90.0), speed_min_mps=80.0, speed_max_mps=120.0)
    expected = certify_heading_separation(_legacy_relative_position(coordinates), a, b, **_lateral())
    actual = certify_heading_separation_from_latlon(a, b, **coordinates, **_lateral())
    assert actual == expected


@pytest.mark.parametrize("axis", ["east", "north"])
@pytest.mark.parametrize("latitude", [-35.0, 0.0, 51.0])
def test_approaching_and_diverging_motion_use_a_minus_b(axis: str, latitude: float) -> None:
    coordinates: _Coordinates = {"a_lat": latitude, "a_lon": 0.0, "b_lat": latitude, "b_lon": 0.0}
    if axis == "east":
        coordinates["b_lon"] = 0.1
        heading = 90.0
        initial_distance = EARTH_RADIUS_IN_METERS * math.radians(0.1) * math.cos(math.radians(latitude))
    else:
        coordinates["b_lat"] += 0.1
        heading = 0.0
        initial_distance = EARTH_RADIUS_IN_METERS * math.radians(0.1)
    approaching = certify_heading_separation_from_latlon(
        _straight(heading),
        _straight(heading + 180.0),
        **coordinates,
        separation_threshold_m=1000.0,
        horizon_s=100.0,
    )
    diverging = certify_heading_separation_from_latlon(
        _straight(heading + 180.0),
        _straight(heading),
        **coordinates,
        separation_threshold_m=1000.0,
        horizon_s=100.0,
    )
    assert approaching.status == "unresolved"
    assert approaching.base_min_distance_m == pytest.approx(0.0, abs=1e-8)
    assert approaching.base_closest_time_s == pytest.approx(initial_distance / 200.0)
    assert diverging.status == "certified_safe"
    assert diverging.base_min_distance_m == pytest.approx(initial_distance)
    assert diverging.base_closest_time_s == 0.0


@pytest.mark.parametrize(("a_lon", "b_lon", "a_heading"), [(179.9, -179.9, 90.0), (-179.9, 179.9, 270.0)])
def test_date_line_encounters_follow_the_short_longitude_difference(
    a_lon: float, b_lon: float, a_heading: float
) -> None:
    result = certify_heading_separation_from_latlon(
        _straight(a_heading),
        _straight(a_heading + 180.0),
        a_lat=0.0,
        a_lon=a_lon,
        b_lat=0.0,
        b_lon=b_lon,
        separation_threshold_m=1000.0,
        horizon_s=120.0,
    )
    distance = EARTH_RADIUS_IN_METERS * math.radians(0.2)
    assert result.status == "unresolved"
    assert result.base_min_distance_m == pytest.approx(0.0, abs=1e-8)
    assert result.base_closest_time_s == pytest.approx(distance / 200.0)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_date_line_conversion_preserves_representable_small_offsets(direction: float) -> None:
    a_lon = math.nextafter(direction * 180.0, 0.0)
    result = certify_heading_separation_from_latlon(
        _straight(0.0, 0.0),
        _straight(0.0, 0.0),
        a_lat=0.0,
        a_lon=a_lon,
        b_lat=0.0,
        b_lon=-direction * 180.0,
        separation_threshold_m=0.0,
        horizon_s=0.0,
    )
    expected_distance = EARTH_RADIUS_IN_METERS * math.radians(180.0 - abs(a_lon))
    assert result.base_min_distance_m == pytest.approx(expected_distance, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("a_lon", "b_lon"), [(0.0, 0.0), (180.0, -180.0), (-180.0, 180.0)])
def test_coincident_positions_are_not_laterally_certified(a_lon: float, b_lon: float) -> None:
    result = certify_heading_separation_from_latlon(
        _straight(0.0),
        _straight(90.0),
        a_lat=51.0,
        a_lon=a_lon,
        b_lat=51.0,
        b_lon=b_lon,
        separation_threshold_m=1.0,
        horizon_s=60.0,
    )
    assert result.status == "unresolved"
    assert result.base_min_distance_m == 0.0


@pytest.mark.parametrize("case", ["shortened", "beyond_horizon", "no_progress", "never", "separated", "unrounded"])
def test_combined_geographic_results_preserve_vertical_options_and_diagnostics(case: str) -> None:
    coordinates, lateral, vertical = _coordinates(), _lateral(), _vertical()
    if case == "beyond_horizon":
        lateral["horizon_s"] = 30.0
    elif case == "no_progress":
        vertical["minimum_vertical_rate_fpm"] = 0.0
    elif case == "never":
        vertical["b_selected_fl"] = vertical["b_current_fl"]
    elif case == "separated":
        vertical["b_current_fl"] = vertical["b_selected_fl"]
    elif case == "unrounded":
        vertical["vertical_rounding_fl"] = 0
    a = replace(_straight(270.0), target_heading_deg=300.0, turn_rate_deg_s=1.0, speed_min_mps=90.0)
    b = _straight(90.0)
    expected = certify_heading_separation_with_vertical(
        _legacy_relative_position(coordinates), a, b, **lateral, **vertical
    )
    actual = certify_heading_separation_with_vertical_from_latlon(a, b, **coordinates, **lateral, **vertical)
    assert actual == expected


def test_combined_date_line_check_uses_the_same_short_lateral_geometry() -> None:
    a, b = _straight(90.0), _straight(270.0)
    coordinates: _Coordinates = {"a_lat": 0.0, "a_lon": 179.9, "b_lat": 0.0, "b_lon": -179.9}
    lateral = _lateral()
    lateral.update(separation_threshold_m=1000.0, relative_position_error_m=0.0, relative_velocity_error_mps=0.0)
    vertical = _vertical()
    vertical.update(
        a_current_fl=100.0,
        a_selected_fl=100.0,
        b_current_fl=100.0,
        b_selected_fl=120.0,
        minimum_vertical_rate_fpm=1000.0,
        required_vertical_gap_fl=10.0,
        vertical_rounding_fl=10,
    )
    full = certify_heading_separation_from_latlon(a, b, **coordinates, **lateral)
    result = certify_heading_separation_with_vertical_from_latlon(a, b, **coordinates, **lateral, **vertical)
    assert full.status == "unresolved"
    assert result.status == "certified_safe"
    assert result.vertical_resolution_time_s == result.lateral_horizon_s == 60.0
    lateral["horizon_s"] = 60.0
    assert result.lateral_result == certify_heading_separation_from_latlon(a, b, **coordinates, **lateral)


def test_unsupported_motion_is_preserved_except_when_vertical_separation_suffices() -> None:
    a, b = replace(_straight(270.0), heading_law="route_turn"), _straight(90.0)
    coordinates, lateral, vertical = _coordinates(), _lateral(), _vertical()
    assert certify_heading_separation_from_latlon(a, b, **coordinates, **lateral).status == "unsupported"
    assert (
        certify_heading_separation_with_vertical_from_latlon(a, b, **coordinates, **lateral, **vertical).status
        == "unsupported"
    )
    vertical["b_current_fl"] = vertical["b_selected_fl"]
    result = certify_heading_separation_with_vertical_from_latlon(a, b, **coordinates, **lateral, **vertical)
    assert result.status == "certified_safe"
    assert result.lateral_result is None


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("a_lat", math.nan),
        ("a_lat", math.inf),
        ("a_lat", -90.0),
        ("a_lat", 90.0),
        ("a_lat", 90.1),
        ("b_lat", math.nan),
        ("b_lat", -math.inf),
        ("b_lat", -90.0),
        ("b_lat", 90.0),
        ("b_lat", -90.1),
        ("a_lon", math.nan),
        ("a_lon", math.inf),
        ("a_lon", -180.1),
        ("a_lon", 180.1),
        ("b_lon", math.nan),
        ("b_lon", -math.inf),
        ("b_lon", -180.1),
        ("b_lon", 180.1),
    ],
)
def test_invalid_coordinates_raise_even_with_separated_altitude_bands(
    combined: bool, field: Literal["a_lat", "a_lon", "b_lat", "b_lon"], value: float
) -> None:
    coordinates = _coordinates()
    coordinates[field] = value
    a, b = _straight(270.0), _straight(90.0)
    if combined:
        vertical = _vertical()
        vertical["b_current_fl"] = vertical["b_selected_fl"]
        with pytest.raises(ValueError, match=field):
            certify_heading_separation_with_vertical_from_latlon(a, b, **coordinates, **_lateral(), **vertical)
    else:
        with pytest.raises(ValueError, match=field):
            certify_heading_separation_from_latlon(a, b, **coordinates, **_lateral())


@pytest.mark.parametrize("b_lon", [-180.0, 180.0])
@pytest.mark.parametrize("combined", [False, True])
def test_half_world_longitude_differences_are_rejected(b_lon: float, combined: bool) -> None:
    coordinates = _coordinates()
    coordinates["b_lon"] = b_lon
    if combined:
        with pytest.raises(ValueError, match="180 degrees apart"):
            certify_heading_separation_with_vertical_from_latlon(
                _straight(270.0), _straight(90.0), **coordinates, **_lateral(), **_vertical()
            )
    else:
        with pytest.raises(ValueError, match="180 degrees apart"):
            certify_heading_separation_from_latlon(_straight(270.0), _straight(90.0), **coordinates, **_lateral())


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "field", ["horizon_s", "separation_threshold_m", "relative_position_error_m", "relative_velocity_error_mps"]
)
def test_lateral_numeric_validation_is_preserved(
    combined: bool,
    field: Literal["horizon_s", "separation_threshold_m", "relative_position_error_m", "relative_velocity_error_mps"],
) -> None:
    lateral = _lateral()
    lateral[field] = -1.0
    if combined:
        with pytest.raises(ValueError, match=field):
            certify_heading_separation_with_vertical_from_latlon(
                _straight(270.0), _straight(90.0), **_coordinates(), **lateral, **_vertical()
            )
    else:
        with pytest.raises(ValueError, match=field):
            certify_heading_separation_from_latlon(_straight(270.0), _straight(90.0), **_coordinates(), **lateral)
