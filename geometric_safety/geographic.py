"""Heading checks from geographic positions in a shared local planar model."""

import math

from geometric_safety.combined import CombinedHeadingAssuranceResult, certify_heading_separation_with_vertical
from geometric_safety.heading import HeadingAssuranceResult, HeadingMotion, certify_heading_separation
from geometric_safety.util import latlon_to_local_xy
from geometric_safety.vertical import DEFAULT_VERTICAL_ROUNDING_FL, DEFAULT_VERTICAL_SEPARATION_FL


def _relative_position_from_latlon(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> tuple[float, float]:
    """Return A minus B, in metres east/north with A as the local origin."""
    for name, latitude in (("a_lat", a_lat), ("b_lat", b_lat)):
        if not math.isfinite(latitude) or not -90.0 < latitude < 90.0:
            raise ValueError(f"{name} must be finite and strictly between -90 and 90 degrees")
    for name, longitude in (("a_lon", a_lon), ("b_lon", b_lon)):
        if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
            raise ValueError(f"{name} must be finite and between -180 and 180 degrees")

    longitude_difference = b_lon - a_lon
    if longitude_difference > 180.0:
        longitude_difference = (b_lon - 180.0) - (a_lon + 180.0)
    elif longitude_difference < -180.0:
        longitude_difference = (b_lon + 180.0) - (a_lon - 180.0)
    if abs(longitude_difference) == 180.0:
        raise ValueError("positions 180 degrees apart in longitude have no unique local longitude difference")

    # Longitude has no absolute scale in this projection. Passing the wrapped
    # difference directly avoids adding and subtracting the reference again.
    b_from_a = latlon_to_local_xy(lat=b_lat, lon=longitude_difference, ref_lat=a_lat, ref_lon=0.0)
    return -float(b_from_a[0]), -float(b_from_a[1])


def certify_heading_separation_from_latlon(
    a: HeadingMotion,
    b: HeadingMotion,
    *,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    separation_threshold_m: float,
    horizon_s: float,
    relative_position_error_m: float = 0.0,
    relative_velocity_error_mps: float = 0.0,
) -> HeadingAssuranceResult:
    """Check heading separation from simultaneous latitude/longitude positions.

    Coordinates are degrees, with east-positive longitude. A is the origin
    of the existing local east/north projection; the shorter signed longitude
    difference handles the date line. Both heading descriptions are used as
    directions in that common plane, clockwise from its north axis.

    This is a local planar approximation, not a geodesic separation check.
    Projection and heading-frame errors are not estimated automatically. Any
    needed bounds must hold over the checked horizon and be included in the
    declared deviation bounds. Speeds, thresholds and errors use the same SI
    units and contract as certify_heading_separation. The result has the same
    status and diagnostics as the corresponding Cartesian query.

    Non-finite coordinates, latitudes at or beyond the poles, longitudes
    outside [-180, 180], and an exactly 180-degree longitude difference raise
    ValueError. Valid coordinates alone do not establish projection accuracy.
    """
    relative_position_m = _relative_position_from_latlon(a_lat, a_lon, b_lat, b_lon)
    return certify_heading_separation(
        relative_position_m,
        a,
        b,
        separation_threshold_m=separation_threshold_m,
        horizon_s=horizon_s,
        relative_position_error_m=relative_position_error_m,
        relative_velocity_error_mps=relative_velocity_error_mps,
    )


def certify_heading_separation_with_vertical_from_latlon(
    a: HeadingMotion,
    b: HeadingMotion,
    *,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    a_current_fl: float,
    a_selected_fl: float,
    b_current_fl: float,
    b_selected_fl: float,
    minimum_vertical_rate_fpm: float,
    separation_threshold_m: float,
    horizon_s: float,
    required_vertical_gap_fl: float = DEFAULT_VERTICAL_SEPARATION_FL,
    vertical_rounding_fl: int = DEFAULT_VERTICAL_ROUNDING_FL,
    relative_position_error_m: float = 0.0,
    relative_velocity_error_mps: float = 0.0,
) -> CombinedHeadingAssuranceResult:
    """Check lateral or vertical separation from latitude/longitude positions.

    The coordinate model and validation are those of
    certify_heading_separation_from_latlon, including the local projection
    limitations. Coordinates are validated even when altitude bands already
    suffice. Position and heading descriptions refer to the same start time.

    All other inputs and results follow certify_heading_separation_with_vertical.
    Each aircraft must progress towards its selected level immediately at no
    less than minimum_vertical_rate_fpm, without overshoot, then remain there.
    The minimum is a nonnegative magnitude in ft/min. The shared conversion
    supplies the relative position; the combined solver determines the lateral
    horizon and applies all declared deviations over that interval.
    """
    relative_position_m = _relative_position_from_latlon(a_lat, a_lon, b_lat, b_lon)
    return certify_heading_separation_with_vertical(
        relative_position_m,
        a,
        b,
        a_current_fl=a_current_fl,
        a_selected_fl=a_selected_fl,
        b_current_fl=b_current_fl,
        b_selected_fl=b_selected_fl,
        minimum_vertical_rate_fpm=minimum_vertical_rate_fpm,
        separation_threshold_m=separation_threshold_m,
        horizon_s=horizon_s,
        required_vertical_gap_fl=required_vertical_gap_fl,
        vertical_rounding_fl=vertical_rounding_fl,
        relative_position_error_m=relative_position_error_m,
        relative_velocity_error_mps=relative_velocity_error_mps,
    )
