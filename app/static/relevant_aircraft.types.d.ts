export type Status = "certified_safe" | "unresolved" | "unsupported";
export type Point = [number, number];
export type Bounds = [number, number, number, number];
export type AircraftKey = "a" | "b";

export interface AircraftInput {
  lat: number;
  lon: number;
  heading_deg: number;
  target_heading_deg: number;
  turn_rate_deg_s: number;
  speed_min_kt: number;
  speed_max_kt: number;
  heading_deviation_deg: number;
  current_fl: number;
  selected_fl: number;
}

export interface VisualParams {
  a: AircraftInput;
  b: AircraftInput;
  separation_threshold_nm: number;
  horizon_s: number;
  relative_position_error_m: number;
  relative_velocity_error_mps: number;
  vertical_enabled: boolean;
  minimum_vertical_rate_fpm: number;
  required_vertical_gap_fl: number;
  vertical_rounding_fl: number;
  grid_density: number;
  view_half_extent_nm: number;
}

export type LateralAircraftInput = Omit<AircraftInput, "current_fl" | "selected_fl">;

// Inactive vertical fields are omitted; the endpoint supplies their defaults.
export type VisualRequest =
  | (VisualParams & { vertical_enabled: true })
  | (Omit<
      VisualParams,
      | "a"
      | "b"
      | "vertical_enabled"
      | "minimum_vertical_rate_fpm"
      | "required_vertical_gap_fl"
      | "vertical_rounding_fl"
    > & {
      a: LateralAircraftInput;
      b: LateralAircraftInput;
      vertical_enabled: false;
    });

export type NumericParameter = Exclude<
  keyof VisualParams,
  "a" | "b" | "vertical_enabled"
>;

interface LateralDecision {
  status: Status;
  reason: string;
  inflation_m: number;
}

// The solver's diagnostic kind determines which distance and time fields exist.
export type LateralResult = LateralDecision &
  (
    | {
        diagnostic_kind: "straight_minimum" | "sampled_minimum";
        base_min_distance_m: number;
        base_closest_time_s: number;
        base_bound_interval_s: null;
      }
    | {
        diagnostic_kind: "interval_bound";
        base_min_distance_m: number;
        base_closest_time_s: null;
        base_bound_interval_s: Point;
      }
    | {
        diagnostic_kind: null;
        base_min_distance_m: null;
        base_closest_time_s: null;
        base_bound_interval_s: null;
      }
  );

export interface Certificate {
  status: Status;
  reason: string;
  horizon_s: number;
  lateral_horizon_s: number;
  vertical_resolution_time_s: number | null;
  lateral: LateralResult | null;
}

export interface Paths {
  minimum: Point[];
  midpoint: Point[];
  maximum: Point[];
}

export interface GeometryFrame {
  relative_hull: Point[];
  closest_point: Point;
  base_distance_m: number;
  midpoint_distance_m: number;
}

export interface Geometry {
  a: Paths;
  b: Paths;
  a_hull_speed_bounds_mps: Point;
  b_hull_speed_bounds_mps: Point;
  frames: GeometryFrame[];
  effective_threshold_m: number | null;
}

export interface VerticalFrame {
  a_band_fl: Point;
  b_band_fl: Point;
  gap_fl: number;
}

export interface DetailPayload {
  certificate: Certificate;
  times_s: number[];
  geometry: Geometry | null;
  geometry_note: string | null;
  vertical: VerticalFrame[] | null;
}

export interface GridPayload {
  east_m: number[];
  north_m: number[];
  lats: number[];
  lons: number[];
  statuses: Status[][];
  counts: Record<Status, number>;
}

export interface Preset {
  id: string;
  title: string;
  description: string;
  params: VisualParams;
}

export interface Plot {
  ctx: CanvasRenderingContext2D;
  point: (value: Point) => Point;
  inverse: (value: Point) => Point;
  box: { left: number; right: number; top: number; bottom: number };
  bounds: Bounds;
}
