"use strict";

/**
 * @typedef {import("./relevant_aircraft.types").AircraftInput} AircraftInput
 * @typedef {import("./relevant_aircraft.types").AircraftKey} AircraftKey
 * @typedef {import("./relevant_aircraft.types").Bounds} Bounds
 * @typedef {import("./relevant_aircraft.types").Certificate} Certificate
 * @typedef {import("./relevant_aircraft.types").DetailPayload} DetailPayload
 * @typedef {import("./relevant_aircraft.types").Geometry} Geometry
 * @typedef {import("./relevant_aircraft.types").GeometryFrame} GeometryFrame
 * @typedef {import("./relevant_aircraft.types").GridPayload} GridPayload
 * @typedef {import("./relevant_aircraft.types").LateralResult} LateralResult
 * @typedef {import("./relevant_aircraft.types").NumericParameter} NumericParameter
 * @typedef {import("./relevant_aircraft.types").Paths} Paths
 * @typedef {import("./relevant_aircraft.types").Plot} Plot
 * @typedef {import("./relevant_aircraft.types").Point} Point
 * @typedef {import("./relevant_aircraft.types").Preset} Preset
 * @typedef {import("./relevant_aircraft.types").Status} Status
 * @typedef {import("./relevant_aircraft.types").VerticalFrame} VerticalFrame
 * @typedef {import("./relevant_aircraft.types").VisualParams} VisualParams
 */

(() => {
  /** @param {string} id @returns {HTMLElement} */
  function $(id) {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLElement))
      throw new Error(`Missing visualiser element: ${id}.`);
    return element;
  }

  /** @param {string} id @returns {HTMLInputElement} */
  function inputElement(id) {
    const element = $(id);
    if (!(element instanceof HTMLInputElement))
      throw new Error(`Expected an input: ${id}.`);
    return element;
  }

  /** @param {string} id @returns {HTMLSelectElement} */
  function selectElement(id) {
    const element = $(id);
    if (!(element instanceof HTMLSelectElement))
      throw new Error(`Expected a select: ${id}.`);
    return element;
  }

  /** @param {string} id @returns {HTMLDetailsElement} */
  function detailsElement(id) {
    const element = $(id);
    if (!(element instanceof HTMLDetailsElement))
      throw new Error(`Expected a details element: ${id}.`);
    return element;
  }

  /** @param {string} id @returns {HTMLCanvasElement} */
  function canvasElement(id) {
    const element = $(id);
    if (!(element instanceof HTMLCanvasElement))
      throw new Error(`Expected a canvas: ${id}.`);
    return element;
  }

  const inputs = [...document.querySelectorAll("input")].filter(
    (input) => input.dataset.path !== undefined,
  );
  const NM = 1852;
  const theme = getComputedStyle(document.documentElement);
  const colour = {
    a: "#285fba",
    b: theme.getPropertyValue("--aircraft-b").trim(),
    base: "#2e5c81",
    threshold: "#c67422",
    safe: "#339985",
    unresolved: theme.getPropertyValue("--unresolved").trim(),
    unsupported: "#94a3b8",
  };
  /** @type {DetailPayload | null} */
  let detail = null;
  /** @type {VisualParams | null} */
  let params = null;
  /** @type {GridPayload | null} */
  let grid = null;
  /** @type {Plot | null} */
  let gridPlot = null;
  /** @type {Bounds | null} */
  let relativeBounds = null;
  let selected = 0,
    relativeBoundsCount = 0,
    revision = 0;
  /** @type {AbortController | null} */
  let detailRequest = null;
  /** @type {AbortController | null} */
  let gridRequest = null;
  /** @type {number | null} */
  let pending = null;
  /** @param {string} id @param {string} value @returns {void} */
  const text = (id, value) => {
    $(id).textContent = value;
  };
  /** @param {number} value @param {number} [places] @returns {string} */
  const fmt = (value, places = 1) => {
    const rounded = Number(value).toFixed(places);
    return Number(rounded) === 0 ? "0" : rounded.replace(/\.0$/, "");
  };
  /** @param {number} value @returns {string} */
  const seconds = (value) => `${fmt(value)} s`;
  /** @param {number} value @returns {string} */
  const nm = (value) => `${(value / NM).toFixed(2)} NM`;

  /** @type {readonly AircraftKey[]} */
  const aircraftKeys = ["a", "b"];
  /** @type {readonly (keyof AircraftInput)[]} */
  const aircraftFields = [
    "lat",
    "lon",
    "heading_deg",
    "target_heading_deg",
    "turn_rate_deg_s",
    "speed_min_kt",
    "speed_max_kt",
    "heading_deviation_deg",
    "current_fl",
    "selected_fl",
  ];
  /** @type {readonly NumericParameter[]} */
  const parameterFields = [
    "separation_threshold_nm",
    "horizon_s",
    "relative_position_error_m",
    "relative_velocity_error_mps",
    "minimum_vertical_rate_fpm",
    "required_vertical_gap_fl",
    "vertical_rounding_fl",
    "grid_density",
    "view_half_extent_nm",
  ];
  const examples = checked(
    JSON.parse($("presetData").textContent ?? ""),
    isPresets,
    "encounter examples",
  );

  /** @param {unknown} value @returns {value is Record<string, unknown>} */
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /** @param {unknown} value @returns {value is number} */
  function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  /**
   * @template T
   * @param {unknown} value
   * @param {(item: unknown) => item is T} accepts
   * @returns {value is T[]}
   */
  function isArrayOf(value, accepts) {
    return Array.isArray(value) && value.every(accepts);
  }

  /** @param {unknown} value @returns {value is Point} */
  function isPoint(value) {
    return isArrayOf(value, isNumber) && value.length === 2;
  }

  /** @param {unknown} value @returns {value is Status} */
  function isStatus(value) {
    return (
      value === "certified_safe" ||
      value === "unresolved" ||
      value === "unsupported"
    );
  }

  /** @param {Record<string, unknown>} value @param {readonly string[]} fields @returns {boolean} */
  function hasNumbers(value, fields) {
    return fields.every((field) => isNumber(value[field]));
  }

  /** @param {unknown} value @returns {value is AircraftInput} */
  function isAircraft(value) {
    return isRecord(value) && hasNumbers(value, aircraftFields);
  }

  /** @param {unknown} value @returns {value is VisualParams} */
  function isParams(value) {
    return (
      isRecord(value) &&
      isAircraft(value.a) &&
      isAircraft(value.b) &&
      hasNumbers(value, parameterFields) &&
      typeof value.vertical_enabled === "boolean"
    );
  }

  /** @param {unknown} value @returns {value is Preset} */
  function isPreset(value) {
    return (
      isRecord(value) &&
      typeof value.id === "string" &&
      typeof value.title === "string" &&
      typeof value.description === "string" &&
      isParams(value.params)
    );
  }

  /** @param {unknown} value @returns {value is Preset[]} */
  function isPresets(value) {
    return isArrayOf(value, isPreset) && value.length > 0;
  }

  /** @param {unknown} value @returns {value is LateralResult} */
  function isLateral(value) {
    if (
      !isRecord(value) ||
      !isStatus(value.status) ||
      typeof value.reason !== "string" ||
      !isNumber(value.inflation_m)
    )
      return false;
    if (value.diagnostic_kind === null)
      return (
        value.base_min_distance_m === null &&
        value.base_closest_time_s === null &&
        value.base_bound_interval_s === null
      );
    if (!isNumber(value.base_min_distance_m)) return false;
    if (value.diagnostic_kind === "interval_bound")
      return (
        value.base_closest_time_s === null &&
        isPoint(value.base_bound_interval_s)
      );
    return (
      (value.diagnostic_kind === "straight_minimum" ||
        value.diagnostic_kind === "sampled_minimum") &&
      isNumber(value.base_closest_time_s) &&
      value.base_bound_interval_s === null
    );
  }

  /** @param {unknown} value @returns {value is Certificate} */
  function isCertificate(value) {
    return (
      isRecord(value) &&
      isStatus(value.status) &&
      typeof value.reason === "string" &&
      hasNumbers(value, ["horizon_s", "lateral_horizon_s"]) &&
      (value.vertical_resolution_time_s === null ||
        isNumber(value.vertical_resolution_time_s)) &&
      (value.lateral === null || isLateral(value.lateral))
    );
  }

  /** @param {unknown} value @returns {value is Paths} */
  function isPaths(value) {
    return (
      isRecord(value) &&
      isArrayOf(value.minimum, isPoint) &&
      isArrayOf(value.midpoint, isPoint) &&
      isArrayOf(value.maximum, isPoint)
    );
  }

  /** @param {unknown} value @returns {value is GeometryFrame} */
  function isGeometryFrame(value) {
    return (
      isRecord(value) &&
      isArrayOf(value.relative_hull, isPoint) &&
      value.relative_hull.length > 0 &&
      isPoint(value.closest_point) &&
      hasNumbers(value, ["base_distance_m", "midpoint_distance_m"])
    );
  }

  /** @param {unknown} value @returns {value is Geometry} */
  function isGeometry(value) {
    return (
      isRecord(value) &&
      isPaths(value.a) &&
      isPaths(value.b) &&
      isPoint(value.a_hull_speed_bounds_mps) &&
      isPoint(value.b_hull_speed_bounds_mps) &&
      isArrayOf(value.frames, isGeometryFrame) &&
      (value.effective_threshold_m === null ||
        isNumber(value.effective_threshold_m))
    );
  }

  /** @param {unknown} value @returns {value is VerticalFrame} */
  function isVerticalFrame(value) {
    return (
      isRecord(value) &&
      isPoint(value.a_band_fl) &&
      isPoint(value.b_band_fl) &&
      isNumber(value.gap_fl)
    );
  }

  /** @param {unknown} value @returns {value is DetailPayload} */
  function isDetail(value) {
    if (
      !isRecord(value) ||
      !isCertificate(value.certificate) ||
      !isArrayOf(value.times_s, isNumber) ||
      value.times_s.length === 0 ||
      !(value.geometry === null || isGeometry(value.geometry)) ||
      !(
        value.geometry_note === null || typeof value.geometry_note === "string"
      ) ||
      !(value.vertical === null || isArrayOf(value.vertical, isVerticalFrame))
    )
      return false;
    const count = value.times_s.length;
    return (
      (value.geometry === null ||
        (value.geometry.frames.length === count &&
          [value.geometry.a, value.geometry.b].every((paths) =>
            [paths.minimum, paths.midpoint, paths.maximum].every(
              (path) => path.length === count,
            ),
          ))) &&
      (value.vertical === null || value.vertical.length === count)
    );
  }

  /** @param {unknown} value @returns {value is GridPayload} */
  function isGrid(value) {
    if (
      !isRecord(value) ||
      !isArrayOf(value.east_m, isNumber) ||
      value.east_m.length < 2
    )
      return false;
    const eastCount = value.east_m.length;
    return (
      isArrayOf(value.north_m, isNumber) &&
      value.north_m.length >= 2 &&
      isArrayOf(value.lats, isNumber) &&
      value.lats.length === value.north_m.length &&
      isArrayOf(value.lons, isNumber) &&
      value.lons.length === eastCount &&
      isArrayOf(value.statuses, (row) => isArrayOf(row, isStatus)) &&
      value.statuses.length === value.north_m.length &&
      value.statuses.every((row) => row.length === eastCount) &&
      isRecord(value.counts) &&
      hasNumbers(value.counts, [
        "certified_safe",
        "unresolved",
        "unsupported",
      ]) &&
      isPoint(value.b_position_m)
    );
  }

  /**
   * @template T
   * @param {unknown} value
   * @param {(item: unknown) => item is T} accepts
   * @param {string} description
   * @returns {T}
   */
  function checked(value, accepts, description) {
    if (!accepts(value)) throw new Error(`Invalid ${description} payload.`);
    return value;
  }

  /** @param {unknown} error @returns {string} */
  function errorMessage(error) {
    return error instanceof Error ? error.message : String(error);
  }

  /** @param {Point} point @returns {Point} */
  function toNM(point) {
    return [point[0] / NM, point[1] / NM];
  }

  /** @param {string} id @returns {number} */
  function readNumber(id) {
    const input = inputElement(id);
    return Number(input.value) * Number(input.dataset.scale);
  }

  /** @param {AircraftKey} key @returns {AircraftInput} */
  function readAircraft(key) {
    return {
      lat: readNumber(`${key}_lat`),
      lon: readNumber(`${key}_lon`),
      heading_deg: readNumber(`${key}_heading_deg`),
      target_heading_deg: readNumber(`${key}_target_heading_deg`),
      turn_rate_deg_s: readNumber(`${key}_turn_rate_deg_s`),
      speed_min_kt: readNumber(`${key}_speed_min_kt`),
      speed_max_kt: readNumber(`${key}_speed_max_kt`),
      heading_deviation_deg: readNumber(`${key}_heading_deviation_deg`),
      current_fl: readNumber(`${key}_current_fl`),
      selected_fl: readNumber(`${key}_selected_fl`),
    };
  }

  /** @returns {VisualParams} */
  function readInputs() {
    return {
      a: readAircraft("a"),
      b: readAircraft("b"),
      separation_threshold_nm: readNumber("separation_threshold_nm"),
      horizon_s: readNumber("horizon_s"),
      relative_position_error_m: readNumber("relative_position_error_m"),
      relative_velocity_error_mps: readNumber("relative_velocity_error_mps"),
      vertical_enabled: inputElement("vertical_enabled").checked,
      minimum_vertical_rate_fpm: readNumber("minimum_vertical_rate_fpm"),
      required_vertical_gap_fl: readNumber("required_vertical_gap_fl"),
      vertical_rounding_fl: readNumber("vertical_rounding_fl"),
      grid_density: readNumber("grid_density"),
      view_half_extent_nm: readNumber("view_half_extent_nm"),
    };
  }

  /** @param {VisualParams} value @returns {void} */
  function fillInputs(value) {
    for (const key of aircraftKeys) {
      for (const field of aircraftFields) {
        const input = inputElement(`${key}_${field}`);
        input.value = String(value[key][field] / Number(input.dataset.scale));
      }
    }
    for (const field of parameterFields) {
      const input = inputElement(field);
      input.value = String(value[field] / Number(input.dataset.scale));
    }
    inputElement("vertical_enabled").checked = value.vertical_enabled;
    updateVerticalControls();
  }

  /** @returns {void} */
  function updateVerticalControls() {
    document.querySelectorAll(".vertical-input").forEach((el) => {
      if (el instanceof HTMLElement)
        el.hidden = !inputElement("vertical_enabled").checked;
    });
  }

  /** @param {boolean} [keepGrid] @returns {void} */
  function invalidate(keepGrid = false) {
    revision += 1;
    detailRequest?.abort();
    gridRequest?.abort();
    if (pending !== null) clearTimeout(pending);
    detail = null;
    ["plots", "timeControls", "resultFacts", "geometryNotice"].forEach((id) => {
      $(id).hidden = true;
    });
    $("resultCard").dataset.status = "pending";
    text("resultTitle", "Inputs changed");
    text("statusBadge", "Not evaluated");
    text("resultReason", "Evaluate the encounter to check these inputs.");
    if (!keepGrid) {
      grid = null;
      gridPlot = null;
      $("gridCanvas").hidden = true;
      text("gridStatus", "Map needs evaluation for these inputs.");
    }
  }

  /** @returns {boolean} */
  function validateInputs() {
    const invalid = inputs.find((input) => !input.checkValidity());
    if (!invalid) return true;
    for (
      let ancestor = invalid.parentElement;
      ancestor;
      ancestor = ancestor.parentElement
    )
      if (ancestor instanceof HTMLDetailsElement) ancestor.open = true;
    invalid.reportValidity();
    showError(
      `Check ${invalid.parentElement?.textContent?.trim() ?? invalid.id}.`,
    );
    return false;
  }

  /** @param {string} message @returns {void} */
  function showError(message) {
    $("resultCard").dataset.status = "error";
    text("statusBadge", "Invalid input");
    text("resultTitle", "The encounter could not be evaluated");
    text("resultReason", message);
  }

  /** @param {string} url @param {VisualParams} value @param {AbortSignal} signal @returns {Promise<unknown>} */
  async function post(url, value, signal) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(value),
      signal,
    });
    if (!response.ok) {
      /** @type {unknown} */
      const body = await response.json().catch(() => null);
      const error = isRecord(body) ? body.detail : null;
      if (Array.isArray(error))
        throw new Error(
          error
            .map((/** @type {unknown} */ item) => {
              if (
                !isRecord(item) ||
                !Array.isArray(item.loc) ||
                typeof item.msg !== "string"
              )
                return "Invalid request.";
              return `${item.loc.slice(1).join(" ")}: ${item.msg}`;
            })
            .join("; "),
        );
      throw new Error(
        typeof error === "string"
          ? error
          : `Request failed (${response.status}).`,
      );
    }
    return response.json();
  }

  /** @returns {Promise<void>} */
  async function evaluate() {
    if (!validateInputs()) return;
    detailRequest?.abort();
    const request = new AbortController();
    detailRequest = request;
    const currentRevision = revision;
    const query = readInputs();
    text("resultTitle", "Evaluating encounter…");
    text("statusBadge", "Working");
    try {
      const data = checked(
        await post("/api/time_sweep", query, request.signal),
        isDetail,
        "encounter",
      );
      if (currentRevision !== revision || request.signal.aborted) return;
      params = query;
      detail = data;
      relativeBounds = null;
      const preferred =
        data.certificate.vertical_resolution_time_s ??
        data.certificate.lateral?.base_closest_time_s ??
        0;
      selected = nearestIndex(Math.min(preferred, windowEnd()));
      renderSummary();
      render();
    } catch (error) {
      if (!request.signal.aborted && currentRevision === revision)
        showError(errorMessage(error));
    }
  }

  /** @returns {void} */
  function renderSummary() {
    if (!detail || !params) return;
    const c = detail.certificate,
      lateral = c.lateral;
    $("resultCard").dataset.status = c.status;
    const labels = {
      certified_safe: "Certified",
      unresolved: "Unresolved",
      unsupported: "Unsupported",
    };
    text("statusBadge", labels[c.status]);
    text(
      "resultTitle",
      c.status === "certified_safe"
        ? `Separation certified for ${fmt(c.horizon_s / 60)} min`
        : c.status === "unresolved"
          ? "Separation has not been certified"
          : "The required motion is unsupported",
    );
    const explanation =
      c.status === "unresolved"
        ? " This result does not establish a conflict."
        : "";
    text(
      "resultReason",
      c.reason.charAt(0).toUpperCase() + c.reason.slice(1) + "." + explanation,
    );
    $("resultFacts").hidden = false;
    text(
      "lateralMetric",
      lateral === null ? "Not needed" : `0 to ${seconds(c.lateral_horizon_s)}`,
    );
    const v = c.vertical_resolution_time_s;
    text(
      "verticalMetric",
      !params.vertical_enabled
        ? "Not used"
        : v === null
          ? "Not established"
          : v === 0
            ? "From the start"
            : `From ${seconds(v)}`,
    );
    text(
      "allowanceMetric",
      lateral === null
        ? "Not needed"
        : lateral.diagnostic_kind === null
          ? "Unavailable"
          : nm(lateral.inflation_m),
    );
    const fraction =
      c.horizon_s > 0
        ? c.lateral_horizon_s / c.horizon_s
        : lateral === null
          ? 0
          : 1;
    $("lateralSegment").style.width = `${100 * fraction}%`;
    $("verticalSegment").style.width = `${100 * (1 - fraction)}%`;
    text(
      "timelineCaption",
      lateral === null
        ? "The altitude bands remain separated throughout the requested horizon."
        : v !== null && v < c.horizon_s
          ? `Lateral separation is required through ${seconds(c.lateral_horizon_s)}. Vertical separation covers the remaining horizon.`
          : "Lateral separation is required throughout the requested horizon.",
    );
    let diagnostic = "No lateral diagnostic was needed.";
    if (lateral) {
      if (lateral.diagnostic_kind === "straight_minimum")
        diagnostic = `Base-hull minimum: ${nm(lateral.base_min_distance_m)} at ${seconds(lateral.base_closest_time_s)}.`;
      else if (lateral.diagnostic_kind === "sampled_minimum")
        diagnostic = `Smallest solver sample: ${nm(lateral.base_min_distance_m)} at ${seconds(lateral.base_closest_time_s)}. This is not a continuous-time minimum.`;
      else if (lateral.diagnostic_kind === "interval_bound")
        diagnostic = `Base-hull lower bound: ${nm(lateral.base_min_distance_m)} over ${seconds(lateral.base_bound_interval_s[0])} to ${seconds(lateral.base_bound_interval_s[1])}.`;
      else diagnostic = "No lateral distance diagnostic is available.";
    }
    text("diagnostic", diagnostic);
    $("geometryNotice").hidden = !detail.geometry_note;
    text("geometryNotice", detail.geometry_note || "");
    for (const id of ["spatialCard", "distanceCard", "hullCard"])
      $(id).hidden = !detail.geometry;
    $("verticalCard").hidden = !detail.vertical;
    $("plots").hidden = !detail.geometry && !detail.vertical;
    $("timeControls").hidden = $("plots").hidden;
  }

  /** @returns {number} */
  function windowEnd() {
    if (!detail) return 0;
    return Math.min(
      detail.certificate.horizon_s,
      selectElement("plotWindow").value === "full"
        ? Infinity
        : Number(selectElement("plotWindow").value),
    );
  }

  /** @param {number} time @returns {number} */
  function nearestIndex(time) {
    if (!detail) return 0;
    const times = detail.times_s;
    return times.reduce(
      (best, t, i) =>
        Math.abs(t - time) < Math.abs(times[best] - time) ? i : best,
      0,
    );
  }

  /** @returns {number} */
  function visibleCount() {
    if (!detail) return 0;
    const index = detail.times_s.findIndex((time) => time > windowEnd());
    return index < 0 ? detail.times_s.length : index;
  }

  /** @param {Point[]} points @param {number} [minimumSpan] @returns {Bounds} */
  function extent(points, minimumSpan = 2) {
    const xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
      cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const dx = Math.max(minimumSpan, Math.max(...xs) - Math.min(...xs)) * 1.22;
    const dy = Math.max(minimumSpan, Math.max(...ys) - Math.min(...ys)) * 1.22;
    return [cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2];
  }

  /**
   * @param {string} id
   * @param {Bounds} bounds
   * @param {string} xLabel
   * @param {string} yLabel
   * @param {boolean} [equal]
   * @param {number[] | null} [yTicks]
   * @returns {Plot}
   */
  function axes(id, bounds, xLabel, yLabel, equal = false, yTicks = null) {
    const canvas = canvasElement(id),
      rect = canvas.getBoundingClientRect(),
      dpr = window.devicePixelRatio || 1;
    const width = rect.width,
      height = rect.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas drawing is unavailable.");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const box = { left: 49, right: width - 14, top: 20, bottom: height - 39 };
    let [x0, x1, y0, y1] = bounds;
    if (equal) {
      const scale = Math.min(
        (box.right - box.left) / (x1 - x0),
        (box.bottom - box.top) / (y1 - y0),
      );
      const cx = (x0 + x1) / 2,
        cy = (y0 + y1) / 2;
      const dx = (box.right - box.left) / scale,
        dy = (box.bottom - box.top) / scale;
      [x0, x1, y0, y1] = [cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2];
    }
    /** @param {Point} value @returns {Point} */
    const point = ([x, y]) => [
      box.left + ((x - x0) / (x1 - x0)) * (box.right - box.left),
      box.bottom - ((y - y0) / (y1 - y0)) * (box.bottom - box.top),
    ];
    /** @param {Point} value @returns {Point} */
    const inverse = ([x, y]) => [
      x0 + ((x - box.left) / (box.right - box.left)) * (x1 - x0),
      y0 + ((box.bottom - y) / (box.bottom - box.top)) * (y1 - y0),
    ];
    ctx.fillStyle = "#f6f9fb";
    ctx.fillRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    ctx.font = "10px Segoe UI, sans-serif";
    for (let tick = 0; tick <= 4; tick++) {
      const f = tick / 4,
        x = box.left + f * (box.right - box.left);
      ctx.strokeStyle = "#dce5ec";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, box.top);
      ctx.lineTo(x, box.bottom);
      ctx.stroke();
      ctx.fillStyle = "#526779";
      ctx.textAlign = "center";
      ctx.fillText(fmt(x0 + f * (x1 - x0)), x, box.bottom + 15);
    }
    const verticalTicks =
      yTicks ??
      Array.from({ length: 5 }, (_, tick) => y0 + (tick / 4) * (y1 - y0));
    for (const value of verticalTicks) {
      const y = point([x0, value])[1];
      ctx.beginPath();
      ctx.moveTo(box.left, y);
      ctx.lineTo(box.right, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(fmt(value), box.left - 7, y + 3);
    }
    ctx.textAlign = "center";
    ctx.fillText(xLabel, (box.left + box.right) / 2, height - 5);
    ctx.save();
    ctx.translate(11, (box.top + box.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    ctx.clip();
    return { ctx, point, inverse, box, bounds: [x0, x1, y0, y1] };
  }

  /**
   * @param {Plot} plot
   * @param {Point[]} points
   * @param {string} stroke
   * @param {number[]} [dash]
   * @param {number} [width]
   * @param {string | null} [fill]
   * @returns {void}
   */
  function line(plot, points, stroke, dash = [], width = 1.8, fill = null) {
    if (!points.length) return;
    const { ctx, point } = plot;
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = point(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (fill) {
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * @param {Plot} plot
   * @param {Point} value
   * @param {string} fill
   * @param {string} [label]
   * @param {number} [radius]
   * @param {boolean} [labelBelow]
   * @returns {void}
   */
  function dot(plot, value, fill, label = "", radius = 4, labelBelow = false) {
    const [x, y] = plot.point(value),
      ctx = plot.ctx;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.3;
    ctx.stroke();
    if (label) {
      ctx.font = "bold 11px Segoe UI, sans-serif";
      const alignRight = x + 7 + ctx.measureText(label).width > plot.box.right;
      ctx.textAlign = alignRight ? "right" : "left";
      const labelX = x + (alignRight ? -7 : 7),
        labelY = y + (labelBelow ? 14 : -7);
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.strokeText(label, labelX, labelY);
      ctx.fillText(label, labelX, labelY);
    }
  }

  /**
   * @param {Plot} plot
   * @param {number} radius
   * @param {string} stroke
   * @param {string | null} [fill]
   * @param {number[]} [dash]
   * @returns {void}
   */
  function circle(plot, radius, stroke, fill = null, dash = []) {
    const centre = plot.point([0, 0]),
      edge = plot.point([radius, 0]),
      ctx = plot.ctx;
    ctx.beginPath();
    ctx.arc(...centre, Math.abs(edge[0] - centre[0]), 0, 2 * Math.PI);
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /** @param {Plot} plot @param {number} time @param {string} stroke @param {string} [label] @returns {void} */
  function markTime(plot, time, stroke, label = "") {
    if (time < plot.bounds[0] || time > plot.bounds[1]) return;
    const x = plot.point([time, 0])[0],
      { ctx, box } = plot;
    ctx.strokeStyle = stroke;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, box.top);
    ctx.lineTo(x, box.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.font = "10px Segoe UI, sans-serif";
      ctx.fillStyle = stroke;
      ctx.textAlign = x > (box.left + box.right) / 2 ? "right" : "left";
      ctx.fillText(
        label,
        x + (ctx.textAlign === "right" ? -5 : 5),
        box.top + 12,
      );
    }
  }

  /** @param {Plot} plot @returns {void} */
  function shadeAfterVertical(plot) {
    if (!detail) return;
    const c = detail.certificate;
    if (
      c.vertical_resolution_time_s === null ||
      c.vertical_resolution_time_s >= windowEnd()
    )
      return;
    const x = plot.point([c.lateral_horizon_s, 0])[0];
    plot.ctx.fillStyle = "#dfe7ec99";
    plot.ctx.fillRect(
      x,
      plot.box.top,
      plot.box.right - x,
      plot.box.bottom - plot.box.top,
    );
  }

  /** @param {number} count @returns {void} */
  function drawSpatial(count) {
    const g = detail?.geometry;
    if (!g) return;
    const all = [g.a, g.b].flatMap((paths) =>
      [paths.minimum, paths.maximum].flatMap((path) =>
        path.slice(0, count).map(toNM),
      ),
    );
    const plot = axes(
      "spatialCanvas",
      extent(all),
      "East from A’s start (NM)",
      "North (NM)",
      true,
    );
    /** @type {Array<[Paths, AircraftKey]>} */
    const aircraftPaths = [
      [g.a, "a"],
      [g.b, "b"],
    ];
    for (const [paths, key] of aircraftPaths) {
      line(
        plot,
        paths.minimum.slice(0, count).map(toNM),
        colour[key],
        [3, 4],
        1,
      );
      line(
        plot,
        paths.maximum.slice(0, count).map(toNM),
        colour[key],
        [3, 4],
        1,
      );
      line(plot, paths.midpoint.slice(0, count).map(toNM), colour[key]);
      line(
        plot,
        [paths.minimum[selected], paths.maximum[selected]].map(toNM),
        colour[key],
        [],
        4,
      );
      dot(
        plot,
        toNM(paths.midpoint[0]),
        colour[key],
        `${key.toUpperCase()} start`,
        2.7,
        true,
      );
      dot(plot, toNM(paths.midpoint[selected]), colour[key], key.toUpperCase());
    }
  }

  /** @param {number} count @returns {void} */
  function drawDistance(count) {
    if (!detail || !params || !detail.geometry) return;
    const g = detail.geometry,
      c = detail.certificate,
      times = detail.times_s.slice(0, count);
    const distances = g.frames
      .slice(0, count)
      .map((f) => f.base_distance_m / NM);
    const threshold =
      g.effective_threshold_m === null ? null : g.effective_threshold_m / NM;
    const top =
      Math.max(
        ...distances,
        threshold ?? 0,
        params.separation_threshold_nm,
        1,
      ) * 1.15;
    const plot = axes(
      "distanceCanvas",
      [0, Math.max(windowEnd(), 1), 0, top],
      "Time (s)",
      "Distance (NM)",
    );
    shadeAfterVertical(plot);
    line(
      plot,
      [
        [0, params.separation_threshold_nm],
        [windowEnd(), params.separation_threshold_nm],
      ],
      "#7f8f9c",
      [2, 4],
      1,
    );
    if (threshold !== null)
      line(
        plot,
        [
          [0, threshold],
          [Math.min(windowEnd(), c.lateral_horizon_s), threshold],
        ],
        colour.threshold,
        [],
        2,
      );
    line(
      plot,
      times.map((time, i) => [time, distances[i]]),
      colour.base,
      [],
      2,
    );
    if (c.vertical_resolution_time_s !== null)
      markTime(
        plot,
        c.vertical_resolution_time_s,
        "#237968",
        "Vertical separation",
      );
    const lateral = c.lateral;
    if (lateral?.diagnostic_kind === "interval_bound") {
      const [left, right] = lateral.base_bound_interval_s;
      line(
        plot,
        [
          [left, lateral.base_min_distance_m / NM],
          [right, lateral.base_min_distance_m / NM],
        ],
        colour.threshold,
        [3, 2],
        3,
      );
    }
    markTime(plot, detail.times_s[selected], "#64748b");
    dot(
      plot,
      [detail.times_s[selected], g.frames[selected].base_distance_m / NM],
      colour.base,
    );
  }

  /** @param {number} count @returns {void} */
  function drawHull(count) {
    if (!detail || !params || !detail.geometry) return;
    const g = detail.geometry,
      c = detail.certificate,
      frame = g.frames[selected];
    const effectiveThreshold = g.effective_threshold_m;
    const required =
      c.lateral !== null &&
      effectiveThreshold !== null &&
      detail.times_s[selected] <= c.lateral_horizon_s;
    const radius = required
      ? effectiveThreshold / NM
      : params.separation_threshold_nm;
    const hull = frame.relative_hull.map(toNM);
    if (relativeBounds === null || relativeBoundsCount !== count) {
      const maxRadius = Math.max(
        params.separation_threshold_nm,
        (g.effective_threshold_m ?? 0) / NM,
      );
      const points = g.frames
        .slice(0, count)
        .flatMap((f) => f.relative_hull.map(toNM));
      relativeBounds = extent(
        [...points, [-maxRadius, -maxRadius], [maxRadius, maxRadius]],
        1,
      );
      relativeBoundsCount = count;
    }
    const plot = axes(
      "hullCanvas",
      relativeBounds,
      "Relative east, A minus B (NM)",
      "Relative north (NM)",
      true,
    );
    circle(
      plot,
      radius,
      required ? colour.threshold : "#94a3b8",
      required ? "#f5dfbe70" : "#dfe7ec70",
    );
    circle(plot, params.separation_threshold_nm, "#8a969f", null, [3, 4]);
    line(plot, hull, colour.base, [], 2.5, "#32668d38");
    const closest = toNM(frame.closest_point);
    line(plot, [[0, 0], closest], colour.base, [4, 4], 1);
    dot(plot, [0, 0], "#354958", "0", 2.5);
    dot(plot, closest, colour.base);
    text(
      "hullCaption",
      required
        ? `Base-hull distance ${nm(frame.base_distance_m)}. The outer circle is ${nm(effectiveThreshold)}: spacing plus the allowance used for this lateral interval. The dotted circle shows the requested spacing.`
        : `Base-hull distance ${nm(frame.base_distance_m)}. Lateral separation is not required at this time; the grey circle shows the requested spacing for reference.`,
    );
  }

  /** @param {number} count @returns {void} */
  function drawVertical(count) {
    if (!detail || !params || !detail.vertical) return;
    const frames = detail.vertical.slice(0, count),
      times = detail.times_s.slice(0, count);
    const levels = frames.flatMap((f) => [...f.a_band_fl, ...f.b_band_fl]);
    const low = Math.min(...levels),
      high = Math.max(...levels),
      pad = Math.max(3, (high - low) * 0.2);
    const tickStep = Math.max(10, 10 * Math.ceil((high - low + 2 * pad) / 50));
    const bottom = Math.floor((low - pad) / tickStep) * tickStep;
    const top = Math.ceil((high + pad) / tickStep) * tickStep;
    const ticks = Array.from(
      { length: Math.round((top - bottom) / tickStep) + 1 },
      (_, i) => bottom + i * tickStep,
    );
    const plot = axes(
      "verticalCanvas",
      [0, Math.max(1, windowEnd()), bottom, top],
      "Time (s)",
      "Flight level",
      false,
      ticks,
    );
    for (const key of aircraftKeys) {
      /** @type {Point[]} */
      const lower = times.map((time, i) => [
        time,
        frames[i][`${key}_band_fl`][0],
      ]);
      /** @type {Point[]} */
      const upper = times.map((time, i) => [
        time,
        frames[i][`${key}_band_fl`][1],
      ]);
      line(
        plot,
        [...lower, ...upper.toReversed()],
        colour[key],
        [],
        1.3,
        `${colour[key]}28`,
      );
    }
    const v = detail.certificate.vertical_resolution_time_s;
    if (v !== null) markTime(plot, v, "#237968", "Vertical separation");
    markTime(plot, detail.times_s[selected], "#64748b");
    text(
      "verticalCaption",
      `Unrounded band gap at this time: ${fmt(detail.vertical[selected].gap_fl, 2)} FL; required: ${fmt(params.required_vertical_gap_fl)} FL. Bands include all faster progress. The marked resolution time also respects the rounding grid.`,
    );
  }

  /** @returns {void} */
  function render() {
    if (!detail) return;
    const count = visibleCount();
    selected = Math.min(selected, count - 1);
    inputElement("timeSlider").max = String(count - 1);
    inputElement("timeSlider").value = String(selected);
    text("selectedTime", seconds(detail.times_s[selected]));
    text(
      "windowCaption",
      `Showing ${seconds(windowEnd())} of the ${seconds(detail.certificate.horizon_s)} horizon. The decision covers the full requested horizon.`,
    );
    if (detail.geometry) {
      drawSpatial(count);
      drawDistance(count);
      drawHull(count);
    }
    if (detail.vertical) drawVertical(count);
  }

  /** @returns {Promise<void>} */
  async function evaluateGrid() {
    if (!validateInputs()) return;
    gridRequest?.abort();
    const request = new AbortController();
    gridRequest = request;
    const currentRevision = revision;
    text("gridStatus", "Evaluating map…");
    try {
      const data = checked(
        await post("/api/grid", readInputs(), request.signal),
        isGrid,
        "position map",
      );
      if (currentRevision !== revision || request.signal.aborted) return;
      grid = data;
      $("gridCanvas").hidden = false;
      drawGrid();
      text(
        "gridStatus",
        `${data.counts.certified_safe} certified · ${data.counts.unresolved} unresolved · ${data.counts.unsupported} unsupported cells`,
      );
    } catch (error) {
      if (!request.signal.aborted && currentRevision === revision)
        text("gridStatus", errorMessage(error));
    }
  }

  /** @returns {void} */
  function drawGrid() {
    if (!grid || !detailsElement("mapSection").open) return;
    const east = grid.east_m.map((v) => v / NM),
      north = grid.north_m.map((v) => v / NM);
    const step = east[1] - east[0];
    const plot = axes(
      "gridCanvas",
      [
        east[0] - step / 2,
        east[east.length - 1] + step / 2,
        north[0] - step / 2,
        north[north.length - 1] + step / 2,
      ],
      "B east from A (NM)",
      "B north from A (NM)",
      true,
    );
    gridPlot = plot;
    for (let i = 0; i < north.length; i++)
      for (let j = 0; j < east.length; j++) {
        const [x, y] = plot.point([east[j] - step / 2, north[i] + step / 2]);
        const [right, bottom] = plot.point([
          east[j] + step / 2,
          north[i] - step / 2,
        ]);
        const status = grid.statuses[i][j];
        plot.ctx.fillStyle =
          status === "certified_safe" ? colour.safe : colour[status];
        plot.ctx.fillRect(x, y, right - x + 0.3, bottom - y + 0.3);
      }
    dot(plot, [0, 0], colour.a, "A");
    dot(plot, toNM(grid.b_position_m), colour.b, "B");
  }

  /** @param {string} id @returns {void} */
  function chooseExample(id) {
    const example = examples.find((item) => item.id === id);
    if (!example) return;
    invalidate();
    fillInputs(example.params);
    text("presetDescription", example.description);
    evaluate();
  }

  inputs.forEach((input) => {
    input.addEventListener("input", () => {
      invalidate();
      selectElement("preset").value = "custom";
      text("presetDescription", "Custom motion and separation bounds.");
      updateVerticalControls();
    });
    input.addEventListener("change", () => {
      pending = window.setTimeout(evaluate, 150);
    });
  });
  $("editInputs").addEventListener("click", () => {
    detailsElement("encounterInputs").open =
      !detailsElement("encounterInputs").open;
  });
  $("encounterInputs").addEventListener("toggle", () => {
    $("editInputs").setAttribute(
      "aria-expanded",
      String(detailsElement("encounterInputs").open),
    );
  });
  $("inputs").addEventListener("submit", (event) => {
    event.preventDefault();
    if (pending !== null) clearTimeout(pending);
    evaluate();
  });
  $("preset").addEventListener("change", () =>
    chooseExample(selectElement("preset").value),
  );
  $("timeSlider").addEventListener("input", () => {
    selected = Number(inputElement("timeSlider").value);
    render();
  });
  $("plotWindow").addEventListener("change", render);
  $("startBtn").addEventListener("click", () => {
    selected = 0;
    render();
  });
  $("limitBtn").addEventListener("click", () => {
    if (!detail) return;
    if (detail.certificate.lateral_horizon_s > windowEnd())
      selectElement("plotWindow").value = "full";
    selected = nearestIndex(detail.certificate.lateral_horizon_s);
    render();
  });
  $("refreshGridBtn").addEventListener("click", evaluateGrid);
  $("mapSection").addEventListener("toggle", () => {
    if (grid) drawGrid();
  });
  $("gridCanvas").addEventListener("click", (event) => {
    if (!grid || !gridPlot) return;
    const rect = $("gridCanvas").getBoundingClientRect(),
      px = event.clientX - rect.left,
      py = event.clientY - rect.top;
    const box = gridPlot.box;
    if (px < box.left || px > box.right || py < box.top || py > box.bottom)
      return;
    const [east, north] = gridPlot.inverse([px, py]).map((v) => v * NM);
    const n = grid.east_m.length,
      step = grid.east_m[1] - grid.east_m[0];
    const i = Math.max(
      0,
      Math.min(n - 1, Math.round((north - grid.north_m[0]) / step)),
    );
    const j = Math.max(
      0,
      Math.min(n - 1, Math.round((east - grid.east_m[0]) / step)),
    );
    invalidate(true);
    inputElement("b_lat").value = String(grid.lats[i]);
    inputElement("b_lon").value = String(grid.lons[j]);
    grid.b_position_m = [grid.east_m[j], grid.north_m[i]];
    selectElement("preset").value = "custom";
    text("presetDescription", "Aircraft B’s position selected from the map.");
    drawGrid();
    evaluate();
  });
  window.addEventListener("resize", () => {
    render();
    drawGrid();
  });
  chooseExample(examples[0].id);
})();
