"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const examples = JSON.parse($("presetData").textContent);
  const inputs = [...document.querySelectorAll("[data-path]")];
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
  let detail = null,
    params = null,
    grid = null,
    gridPlot = null,
    selected = 0,
    relativeBounds = null,
    relativeBoundsCount = 0;
  let revision = 0,
    detailRequest = null,
    gridRequest = null,
    pending = null;
  const text = (id, value) => {
    $(id).textContent = value;
  };
  const fmt = (value, places = 1) => {
    const rounded = Number(value).toFixed(places);
    return Number(rounded) === 0 ? "0" : rounded.replace(/\.0$/, "");
  };
  const seconds = (value) => `${fmt(value)} s`;
  const nm = (value) => `${(value / NM).toFixed(2)} NM`;

  function readInputs() {
    const value = { a: {}, b: {} };
    for (const input of inputs) {
      const path = input.dataset.path.split(".");
      const target = path.length === 2 ? value[path[0]] : value;
      target[path.at(-1)] =
        input.type === "checkbox"
          ? input.checked
          : Number(input.value) * Number(input.dataset.scale);
    }
    return value;
  }

  function fillInputs(value) {
    for (const input of inputs) {
      const path = input.dataset.path.split(".");
      const field =
        path.length === 2 ? value[path[0]][path[1]] : value[path[0]];
      if (input.type === "checkbox") input.checked = field;
      else input.value = field / Number(input.dataset.scale);
    }
    updateVerticalControls();
  }

  function updateVerticalControls() {
    document.querySelectorAll(".vertical-input").forEach((el) => {
      el.hidden = !$("vertical_enabled").checked;
    });
  }

  function invalidate(keepGrid = false) {
    revision += 1;
    detailRequest?.abort();
    gridRequest?.abort();
    clearTimeout(pending);
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
    showError(`Check ${invalid.parentElement.textContent.trim()}.`);
    return false;
  }

  function showError(message) {
    $("resultCard").dataset.status = "error";
    text("statusBadge", "Invalid input");
    text("resultTitle", "The encounter could not be evaluated");
    text("resultReason", message);
  }

  async function post(url, value, signal) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(value),
      signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const error = body?.detail;
      if (Array.isArray(error))
        throw new Error(
          error
            .map((item) => `${item.loc.slice(1).join(" ")}: ${item.msg}`)
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
      const data = await post("/api/time_sweep", query, request.signal);
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
        showError(error.message);
    }
  }

  function renderSummary() {
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

  function windowEnd() {
    return Math.min(
      detail.certificate.horizon_s,
      $("plotWindow").value === "full"
        ? Infinity
        : Number($("plotWindow").value),
    );
  }

  function nearestIndex(time) {
    return detail.times_s.reduce(
      (best, t, i) =>
        Math.abs(t - time) < Math.abs(detail.times_s[best] - time) ? i : best,
      0,
    );
  }

  function visibleCount() {
    const index = detail.times_s.findIndex((time) => time > windowEnd());
    return index < 0 ? detail.times_s.length : index;
  }

  function extent(points, minimumSpan = 2) {
    const xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
      cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const dx = Math.max(minimumSpan, Math.max(...xs) - Math.min(...xs)) * 1.22;
    const dy = Math.max(minimumSpan, Math.max(...ys) - Math.min(...ys)) * 1.22;
    return [cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2];
  }

  function axes(id, bounds, xLabel, yLabel, equal = false, yTicks = null) {
    const canvas = $(id),
      rect = canvas.getBoundingClientRect(),
      dpr = window.devicePixelRatio || 1;
    const width = rect.width,
      height = rect.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
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
    const point = ([x, y]) => [
      box.left + ((x - x0) / (x1 - x0)) * (box.right - box.left),
      box.bottom - ((y - y0) / (y1 - y0)) * (box.bottom - box.top),
    ];
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

  function shadeAfterVertical(plot) {
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

  function drawSpatial(count) {
    const g = detail.geometry;
    const toNM = (p) => p.map((v) => v / NM);
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
    for (const [paths, key] of [
      [g.a, "a"],
      [g.b, "b"],
    ]) {
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

  function drawDistance(count) {
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

  function drawHull(count) {
    const g = detail.geometry,
      c = detail.certificate,
      frame = g.frames[selected];
    const required =
      c.lateral !== null && detail.times_s[selected] <= c.lateral_horizon_s;
    const radius =
      required && g.effective_threshold_m !== null
        ? g.effective_threshold_m / NM
        : params.separation_threshold_nm;
    const hull = frame.relative_hull.map((p) => p.map((v) => v / NM));
    if (relativeBounds === null || relativeBoundsCount !== count) {
      const maxRadius = Math.max(
        params.separation_threshold_nm,
        (g.effective_threshold_m ?? 0) / NM,
      );
      const points = g.frames
        .slice(0, count)
        .flatMap((f) => f.relative_hull.map((p) => p.map((v) => v / NM)));
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
    const closest = frame.closest_point.map((v) => v / NM);
    line(plot, [[0, 0], closest], colour.base, [4, 4], 1);
    dot(plot, [0, 0], "#354958", "0", 2.5);
    dot(plot, closest, colour.base);
    text(
      "hullCaption",
      required
        ? `Base-hull distance ${nm(frame.base_distance_m)}. The outer circle is ${nm(g.effective_threshold_m)}: spacing plus the allowance used for this lateral interval. The dotted circle shows the requested spacing.`
        : `Base-hull distance ${nm(frame.base_distance_m)}. Lateral separation is not required at this time; the grey circle shows the requested spacing for reference.`,
    );
  }

  function drawVertical(count) {
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
    for (const key of ["a", "b"]) {
      const lower = times.map((time, i) => [
        time,
        frames[i][`${key}_band_fl`][0],
      ]);
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

  function render() {
    if (!detail) return;
    const count = visibleCount();
    selected = Math.min(selected, count - 1);
    $("timeSlider").max = count - 1;
    $("timeSlider").value = selected;
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

  async function evaluateGrid() {
    if (!validateInputs()) return;
    gridRequest?.abort();
    const request = new AbortController();
    gridRequest = request;
    const currentRevision = revision;
    text("gridStatus", "Evaluating map…");
    try {
      const data = await post("/api/grid", readInputs(), request.signal);
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
        text("gridStatus", error.message);
    }
  }

  function drawGrid() {
    if (!grid || !$("mapSection").open) return;
    const east = grid.east_m.map((v) => v / NM),
      north = grid.north_m.map((v) => v / NM);
    const step = east[1] - east[0];
    const plot = axes(
      "gridCanvas",
      [
        east[0] - step / 2,
        east.at(-1) + step / 2,
        north[0] - step / 2,
        north.at(-1) + step / 2,
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
    dot(
      plot,
      grid.b_position_m.map((v) => v / NM),
      colour.b,
      "B",
    );
  }

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
      $("preset").value = "custom";
      text("presetDescription", "Custom motion and separation bounds.");
      updateVerticalControls();
    });
    input.addEventListener("change", () => {
      pending = setTimeout(evaluate, 150);
    });
  });
  $("editInputs").addEventListener("click", () => {
    $("encounterInputs").open = !$("encounterInputs").open;
  });
  $("encounterInputs").addEventListener("toggle", () => {
    $("editInputs").setAttribute(
      "aria-expanded",
      String($("encounterInputs").open),
    );
  });
  $("inputs").addEventListener("submit", (event) => {
    event.preventDefault();
    clearTimeout(pending);
    evaluate();
  });
  $("preset").addEventListener("change", () =>
    chooseExample($("preset").value),
  );
  $("timeSlider").addEventListener("input", () => {
    selected = Number($("timeSlider").value);
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
      $("plotWindow").value = "full";
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
    $("b_lat").value = grid.lats[i];
    $("b_lon").value = grid.lons[j];
    grid.b_position_m = [grid.east_m[j], grid.north_m[i]];
    $("preset").value = "custom";
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
