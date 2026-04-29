console.log("[ceiling-rcp] app.js build 8 — auto-detect");

// ─── STATE ────────────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  plan: null,
  imageBitmap: null,
  view: { scale: 1, tx: 0, ty: 0 },
  // mode: "select" | "draw_room" | "draw_main" | "draw_region"
  mode: "select",
  // Tool within select mode: "select" | "insert-vertex" | "delete-vertex"
  tool: "select",
  // Draft polygon being placed: array of [x, z] in world metres
  draft: [],
  // Selected polygon: { kind: "room"|"main"|"region", regionId? }
  selection: null,
  drag: null,
  panning: null,
  hover: { world: null },
  // Map of { kind: "room"|"main"|"region:N" → ImageBitmap } for heatmaps
  heatmaps: new Map(),
};

// ─── DOM ──────────────────────────────────────────────────────────────────
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const readout = document.getElementById("readout");
const banner = document.getElementById("banner");

document.getElementById("btn-pick-folder").onclick = () => {
  document.getElementById("file-input").click();
};
document.getElementById("btn-pick-files").onclick = () => {
  document.getElementById("file-input-flat").click();
};
document.getElementById("file-input").addEventListener("change",
  e => uploadFiles(e.target.files));
document.getElementById("file-input-flat").addEventListener("change",
  e => uploadFiles(e.target.files));

document.getElementById("btn-room").onclick = () => startDraw("room");
document.getElementById("btn-main").onclick = () => startDraw("main");
document.getElementById("btn-region").onclick = () => startDraw("region");

document.querySelectorAll(".tool").forEach(b => {
  b.onclick = () => {
    const t = b.dataset.tool;
    if (t === "cancel-draw") { cancelDraw(); return; }
    cancelDraw();
    state.tool = t;
    state.mode = "select";
    setActiveTool(t);
    updateToolHint();
    draw();
  };
});

document.getElementById("btn-snap").onclick = snapPolygons;
document.getElementById("btn-pdf").onclick = downloadPdf;
document.getElementById("btn-export").onclick = exportPlan;
document.getElementById("btn-autodetect").onclick = autoDetect;

window.addEventListener("resize", resizeCanvas);
window.addEventListener("keydown", onKey);

canvas.addEventListener("mousedown", onMouseDown);
canvas.addEventListener("mousemove", onMouseMove);
canvas.addEventListener("mouseup", onMouseUp);
canvas.addEventListener("dblclick", onDblClick);
canvas.addEventListener("wheel", onWheel, { passive: false });
canvas.addEventListener("contextmenu", e => e.preventDefault());

// ─── COORDS ───────────────────────────────────────────────────────────────
function worldToImg(x, z) {
  const g = state.plan.grid;
  return {
    u: (g.max_x - x) * g.pixels_per_metre,
    v: (g.max_z - z) * g.pixels_per_metre,
  };
}
function imgToWorld(u, v) {
  const g = state.plan.grid;
  return {
    x: g.max_x - u / g.pixels_per_metre,
    z: g.max_z - v / g.pixels_per_metre,
  };
}
function imgToCanvas(u, v) {
  return { cx: u * state.view.scale + state.view.tx,
           cy: v * state.view.scale + state.view.ty };
}
function canvasToImg(cx, cy) {
  return { u: (cx - state.view.tx) / state.view.scale,
           v: (cy - state.view.ty) / state.view.scale };
}
function canvasToWorld(cx, cy) {
  const i = canvasToImg(cx, cy);
  return imgToWorld(i.u, i.v);
}

// ─── UPLOAD ───────────────────────────────────────────────────────────────
const KEEP_EXTENSIONS = new Set([".obj", ".mtl", ".jpg", ".jpeg", ".png", ".json"]);
const SKIP_DIR_SEGMENTS = new Set([
  "keyframes", "depth", "confidence", "cameras",
  "corrected_cameras", "images", "corrected_images",
]);
const SKIP_FILENAMES = new Set([
  ".DS_Store", "thumbnail.jpg", "polycam.mp4",
  "ceiling_geometry.json", "ceiling_geometry_preview.png",
  "roomplan.json",
]);

function shouldKeepUpload(rel) {
  const parts = rel.split("/");
  const name = parts[parts.length - 1];
  if (SKIP_FILENAMES.has(name)) return false;
  for (const seg of parts.slice(0, -1)) if (SKIP_DIR_SEGMENTS.has(seg)) return false;
  const dot = name.lastIndexOf(".");
  if (dot < 0) return false;
  return KEEP_EXTENSIONS.has(name.slice(dot).toLowerCase());
}

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) {
    setReport("Browser returned 0 files.", "err"); return;
  }
  const fd = new FormData();
  let kept = 0;
  for (const f of fileList) {
    const rel = f.webkitRelativePath && f.webkitRelativePath.length ? f.webkitRelativePath : f.name;
    if (!shouldKeepUpload(rel)) continue;
    fd.append("files", f, rel.split("/").pop());
    kept++;
  }
  if (kept === 0) { setReport("No mesh files in selection.", "err"); return; }
  for (const f of fileList) {
    const rel = f.webkitRelativePath && f.webkitRelativePath.length ? f.webkitRelativePath : f.name;
    if (shouldKeepUpload(rel)) fd.append("paths", rel);
  }
  setReport(`Uploading ${kept} files…`, "muted");
  try {
    const r = await fetch("/api/sessions", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    state.sessionId = data.session_id;
    document.getElementById("session-label").textContent = `session ${data.session_id}`;
    showReport(data.report);
    if (data.report.ok) {
      setReport("Processing…", "muted");
      await runProcess();
    }
  } catch (e) {
    setReport("Upload failed: " + e.message, "err");
  }
}

async function runProcess() {
  const fd = new FormData();
  fd.append("ppm", "150");
  const r = await fetch(`/api/sessions/${state.sessionId}/process`,
    { method: "POST", body: fd });
  if (!r.ok) { setReport("Processing failed.", "err"); return; }
  state.plan = await r.json();
  await loadCeilingImage();
  document.getElementById("workflow").hidden = false;
  document.getElementById("btn-export").disabled = false;
  fitView();
  refreshPolygonsList();
  draw();
}

async function loadCeilingImage() {
  const r = await fetch(`/api/sessions/${state.sessionId}/image/ceiling.jpg?_=${Date.now()}`);
  state.imageBitmap = await createImageBitmap(await r.blob());
}

// ─── REPORT ───────────────────────────────────────────────────────────────
function showReport(rep) {
  const el = document.getElementById("upload-report");
  const lines = [];
  lines.push(`<span class="${rep.ok ? "ok" : "err"}">${rep.ok ? "OK" : "Cannot process"}</span>`);
  if (rep.obj) lines.push(`obj: ${rep.obj}`);
  if (rep.mtl) lines.push(`mtl: ${rep.mtl}`);
  if (rep.mesh_info) lines.push(`alignment: ${rep.mesh_info}`);
  lines.push(`textures: ${rep.textures_found} found`);
  for (const w of rep.warnings) lines.push(`<span class="warn">⚠ ${w}</span>`);
  for (const e of rep.errors) lines.push(`<span class="err">✗ ${e}</span>`);
  el.innerHTML = lines.join("<br>");
  el.classList.add("show");
}
function setReport(text, cls) {
  const el = document.getElementById("upload-report");
  el.innerHTML = `<span class="${cls || ""}">${text}</span>`;
  el.classList.add("show");
}

// ─── DRAW STATE MACHINE ───────────────────────────────────────────────────
function startDraw(kind) {
  if (kind === "main" && !state.plan.room) return;
  if (kind === "region" && !state.plan.main) return;
  state.mode = "draw_" + kind;
  state.draft = [];
  state.selection = null;
  banner.textContent = {
    room: "Drawing ROOM outline — click vertices, click first or press Enter to close. Esc = cancel",
    main: "Drawing MAIN CEILING — defines height datum (relative = 0)",
    region: "Drawing CEILING REGION — relative to main ceiling",
  }[kind];
  banner.classList.add("show");
  document.querySelector(".tool[data-tool='cancel-draw']").disabled = false;
  setStepDrawing(kind, true);
  draw();
}

function cancelDraw() {
  if (!state.mode.startsWith("draw_")) return;
  const kind = state.mode.slice(5);
  setStepDrawing(kind, false);
  state.mode = "select";
  state.draft = [];
  banner.classList.remove("show");
  document.querySelector(".tool[data-tool='cancel-draw']").disabled = true;
  draw();
}

async function commitDraft() {
  if (!state.mode.startsWith("draw_")) return;
  if (state.draft.length < 3) return;
  const kind = state.mode.slice(5);
  const polygon = state.draft.slice();

  setStepDrawing(kind, false);
  state.draft = [];
  banner.classList.remove("show");
  document.querySelector(".tool[data-tool='cancel-draw']").disabled = true;
  state.mode = "select";

  if (kind === "room") {
    const r = await fetch(`/api/sessions/${state.sessionId}/room`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.room = d.room;
      state.plan.room_heatmap = d.room_heatmap;
      if (d.room_heatmap) await refreshHeatmap("room", d.room_heatmap);
      markStepDone("room");
      unlockStep("main");
      document.getElementById("btn-autodetect").disabled = false;
    }
  } else if (kind === "main") {
    const r = await fetch(`/api/sessions/${state.sessionId}/main`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.main = d.main;
      await refreshHeatmap("main", d.main);
      markStepDone("main");
      unlockStep("region");
      // Recompute relative_y on existing regions (server already does this)
      const planR = await fetch(`/api/sessions/${state.sessionId}/plan`);
      state.plan = await planR.json();
    }
  } else if (kind === "region") {
    const r = await fetch(`/api/sessions/${state.sessionId}/region`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.regions = state.plan.regions || [];
      state.plan.regions.push(d.region);
      await refreshHeatmap("region:" + d.region.id, d.region);
    }
  }
  refreshPolygonsList();
  draw();
}

async function refreshHeatmap(key, polyObj) {
  const b64 = polyObj?.heatmap_png_b64;
  if (!b64) { state.heatmaps.delete(key); return; }
  const blob = await (await fetch("data:image/png;base64," + b64)).blob();
  state.heatmaps.set(key, await createImageBitmap(blob));
}

async function deletePolygon(kind, regionId) {
  if (kind === "region") {
    await fetch(`/api/sessions/${state.sessionId}/region/${regionId}`,
      { method: "DELETE" });
    state.plan.regions = state.plan.regions.filter(r => r.id !== regionId);
    state.heatmaps.delete("region:" + regionId);
  } else if (kind === "room") {
    await fetch(`/api/sessions/${state.sessionId}/room`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null }),
    });
    state.plan.room = null;
    document.getElementById("step-room").classList.remove("done");
    document.getElementById("step-room").classList.add("active");
    document.getElementById("step-main").classList.add("locked");
    document.getElementById("btn-main").disabled = true;
  } else if (kind === "main") {
    await fetch(`/api/sessions/${state.sessionId}/main`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null }),
    });
    state.plan.main = null;
    state.heatmaps.delete("main");
    document.getElementById("step-main").classList.remove("done");
    document.getElementById("step-main").classList.add("active");
    document.getElementById("step-region").classList.add("locked");
    document.getElementById("btn-region").disabled = true;
    // Refresh regions to clear relative_y
    for (const r of state.plan.regions || []) r.relative_y = null;
  }
  state.selection = null;
  refreshPolygonsList();
  draw();
}

function setStepDrawing(kind, on) {
  const btn = document.getElementById("btn-" + kind);
  btn.classList.toggle("drawing", on);
}
function markStepDone(kind) {
  document.getElementById("step-" + kind).classList.remove("active");
  document.getElementById("step-" + kind).classList.add("done");
}
function unlockStep(kind) {
  const li = document.getElementById("step-" + kind);
  li.classList.remove("locked");
  li.classList.add("active");
  document.getElementById("btn-" + kind).disabled = false;
}

function setActiveTool(name) {
  document.querySelectorAll(".tool").forEach(b =>
    b.classList.toggle("active", b.dataset.tool === name));
}

function updateToolHint() {
  const hints = {
    "select": "Click a vertex to drag. Click a polygon to select it.",
    "insert-vertex": "Select a polygon, then click on one of its edges to insert a vertex there.",
    "delete-vertex": "Select a polygon, then click one of its vertices to remove it.",
  };
  const el = document.getElementById("tool-hint");
  if (el) el.textContent = hints[state.tool] || "";
}

async function autoDetect() {
  if (!state.sessionId) return;
  if (!state.plan?.room) { setBanner("Trace the room outline first."); return; }
  if (!confirm(
    "Auto-detect will replace any main ceiling and regions you've drawn. " +
    "Continue?"
  )) return;
  setBanner("Auto-detecting heights…");
  const r = await fetch(`/api/sessions/${state.sessionId}/auto_detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!r.ok) {
    setBanner("Auto-detect failed: " + (await r.text()).slice(0, 160), true);
    return;
  }
  state.plan = await r.json();
  state.heatmaps.clear();
  if (state.plan.room_heatmap) await refreshHeatmap("room", state.plan.room_heatmap);
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
  markStepDone("main");
  unlockStep("region");
  refreshPolygonsList();
  draw();
  const n = (state.plan.regions || []).length + (state.plan.main ? 1 : 0);
  setBanner(`Auto-detect found ${n} planes — refine, snap, or export.`);
  setTimeout(() => banner.classList.remove("show"), 3000);
}

async function snapPolygons() {
  if (!state.sessionId) return;
  if (!state.plan?.room) { setBanner("Trace the room outline first."); return; }
  setBanner("Snapping borders…");
  const r = await fetch(`/api/sessions/${state.sessionId}/snap`, { method: "POST" });
  if (!r.ok) {
    setBanner("Snap failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan = await r.json();
  state.heatmaps.clear();
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || []) await refreshHeatmap("region:" + reg.id, reg);
  refreshPolygonsList();
  draw();
  setBanner("Snapped — borders shared, no gaps or overlap.");
  setTimeout(() => banner.classList.remove("show"), 2500);
}

function setBanner(text, isErr = false) {
  banner.textContent = text;
  banner.classList.toggle("show", true);
  banner.style.background = isErr ? "rgba(239, 83, 80, 0.95)" : "";
}

async function downloadPdf() {
  if (!state.sessionId) return;
  if (!state.plan?.room) { setBanner("Trace the room outline first."); return; }
  setBanner("Generating PDF…");
  let blob;
  try {
    const r = await fetch(`/api/sessions/${state.sessionId}/pdf`);
    if (!r.ok) {
      setBanner("PDF failed: " + (await r.text()).slice(0, 120), true);
      return;
    }
    blob = await r.blob();
  } catch (e) {
    setBanner("PDF fetch failed: " + e.message, true);
    return;
  }
  // Safari ignores `download` on a programmatic click of an unattached anchor.
  // Append → click → remove → revoke.
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `rcp_${state.sessionId}.pdf`;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  setBanner("PDF downloaded.");
  setTimeout(() => banner.classList.remove("show"), 2000);
}

// ─── POLYGON LIST ─────────────────────────────────────────────────────────
function refreshPolygonsList() {
  const ul = document.getElementById("polygons-list");
  ul.innerHTML = "";

  const addRow = (key, label, color, meta, selKey, currentNotes, allowNotes) => {
    const li = document.createElement("li");
    li.className = "poly-row" + (state.selection?.key === selKey ? " active" : "");
    const head = document.createElement("div");
    head.className = "poly-head";
    head.innerHTML =
      `<div class="swatch" style="background:${color}"></div>` +
      `<div class="label">${label}</div>` +
      `<div class="meta">${meta}</div>` +
      `<button class="del-btn" title="Delete">×</button>`;
    li.appendChild(head);

    if (allowNotes) {
      const noteRow = document.createElement("div");
      noteRow.className = "poly-notes";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Notes (e.g. white plaster, oak battens)";
      input.value = currentNotes || "";
      input.onclick = (e) => e.stopPropagation();
      let timer = null;
      input.oninput = () => {
        clearTimeout(timer);
        timer = setTimeout(() => saveNotes(selKey, input.value), 350);
      };
      input.onblur = () => saveNotes(selKey, input.value);
      noteRow.appendChild(input);
      li.appendChild(noteRow);
    }

    li.onclick = (e) => {
      if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
      state.selection = { key: selKey };
      refreshPolygonsList();
      updateSelectionInfo();
      draw();
    };
    head.querySelector(".del-btn").onclick = (e) => {
      e.stopPropagation();
      const [k, idStr] = selKey.split(":");
      deletePolygon(k, idStr ? parseInt(idStr, 10) : null);
    };
    ul.appendChild(li);
  };

  if (state.plan.room) {
    addRow("room", "Room outline", "#ffe082",
      `${state.plan.room.length} verts`, "room", null, false);
  }
  if (state.plan.main) {
    const s = state.plan.main.stats;
    addRow("main", state.plan.main.label || "Main Ceiling (1)",
      state.plan.main.tint || "#80cbc4",
      `0 mm  σ${(s.std_y * 1000).toFixed(0)} mm`,
      "main", state.plan.main.notes, true);
  }
  for (const r of state.plan.regions || []) {
    const s = r.stats;
    const rel = r.relative_y;
    const relTxt = rel === null || rel === undefined
      ? "—"
      : (rel >= 0 ? "+" : "") + (rel * 1000).toFixed(0) + " mm";
    addRow("region:" + r.id, r.label || `Ceiling Region (${r.id + 2})`,
      r.tint || "#ff7043",
      `${relTxt}  σ${(s.std_y * 1000).toFixed(0)} mm`,
      "region:" + r.id, r.notes, true);
  }
  updateSelectionInfo();
}

async function saveNotes(selKey, value) {
  if (selKey === "main") {
    if (!state.plan.main) return;
    state.plan.main.notes = value;
    await fetch(`/api/sessions/${state.sessionId}/main/notes`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: value }),
    });
  } else if (selKey.startsWith("region:")) {
    const id = parseInt(selKey.slice(7), 10);
    const r = state.plan.regions.find(r => r.id === id);
    if (!r) return;
    r.notes = value;
    await fetch(`/api/sessions/${state.sessionId}/region/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: value }),
    });
  }
}

function updateSelectionInfo() {
  const el = document.getElementById("selection-info");
  if (!state.selection) { el.textContent = "Nothing selected"; return; }
  const key = state.selection.key;
  if (key === "room") {
    el.innerHTML = `<b>Room outline</b><br>${state.plan.room.length} vertices`;
  } else if (key === "main") {
    const m = state.plan.main; if (!m) return;
    el.innerHTML = renderStats("Main ceiling (datum)", m.stats, null);
  } else if (key.startsWith("region:")) {
    const id = parseInt(key.slice(7), 10);
    const r = state.plan.regions.find(r => r.id === id);
    if (!r) return;
    el.innerHTML = renderStats(r.label || `region ${id}`, r.stats, r.relative_y);
  }
}

function renderStats(title, s, relativeY) {
  const fmt_std = (s.std_y * 1000).toFixed(0);
  const fmt_rel = relativeY === null || relativeY === undefined
    ? "0 mm (datum)"
    : (relativeY >= 0 ? "+" : "") + (relativeY * 1000).toFixed(0) + " mm";
  const valid_pct = (s.valid_frac * 100).toFixed(0);
  const range_mm = ((s.max_y - s.min_y) * 1000).toFixed(0);
  const warn = s.std_y > 0.05
    ? `<div class="warn-line">⚠ high variance (${fmt_std} mm) — likely clipped a bulkhead</div>`
    : "";
  const validWarn = s.valid_frac < 0.6
    ? `<div class="warn-line">⚠ only ${valid_pct}% of polygon has LiDAR coverage</div>`
    : "";
  return `
    <b>${title}</b><br>
    relative height: ${fmt_rel}<br>
    σ inside polygon: ${fmt_std} mm<br>
    range inside polygon: ${range_mm} mm<br>
    LiDAR coverage: ${valid_pct}%
    ${warn}${validWarn}
  `;
}

// ─── DRAWING ──────────────────────────────────────────────────────────────
function resizeCanvas() {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  draw();
}

function fitView() {
  if (!state.plan?.grid) return;
  const g = state.plan.grid;
  const sx = canvas.width / g.width;
  const sy = canvas.height / g.height;
  const s = Math.min(sx, sy) * 0.95;
  state.view.scale = s;
  state.view.tx = (canvas.width - g.width * s) / 2;
  state.view.ty = (canvas.height - g.height * s) / 2;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.plan?.grid) return;

  ctx.save();
  ctx.translate(state.view.tx, state.view.ty);
  ctx.scale(state.view.scale, state.view.scale);

  // Base render
  if (state.imageBitmap) {
    ctx.drawImage(state.imageBitmap, 0, 0);
  } else {
    ctx.fillStyle = "#1a1d20";
    ctx.fillRect(0, 0, state.plan.grid.width, state.plan.grid.height);
  }

  // Room shaded mask first (lowest layer), then per-polygon heatmaps.
  if (state.plan.room_heatmap) drawHeatmap("room", state.plan.room_heatmap);
  drawHeatmap("main", state.plan.main);
  for (const r of state.plan.regions || []) {
    drawHeatmap("region:" + r.id, r);
  }

  // Existing polygons
  if (state.plan.room) drawPolygon(state.plan.room, {
    stroke: "#ffe082", lineWidth: 2, dashed: true,
    selected: state.selection?.key === "room",
    selectedKey: "room",
  });
  if (state.plan.main) drawPolygon(state.plan.main.polygon, {
    stroke: state.plan.main.tint || "#80cbc4", lineWidth: 1.8,
    selected: state.selection?.key === "main",
    selectedKey: "main",
  });
  for (const r of state.plan.regions || []) {
    drawPolygon(r.polygon, {
      stroke: r.tint || "#ff7043", lineWidth: 1.5,
      selected: state.selection?.key === "region:" + r.id,
      selectedKey: "region:" + r.id,
    });
  }
  // Hover preview for insert-vertex
  if (state.tool === "insert-vertex" && state.selection && state.hover.world) {
    drawInsertPreview();
  }

  // Draft polygon being drawn
  if (state.mode.startsWith("draw_") && state.draft.length > 0) {
    drawDraft();
  }

  ctx.restore();

  if (state.hover.world) {
    readout.textContent =
      `x ${state.hover.world.x.toFixed(2)} m   z ${state.hover.world.z.toFixed(2)} m`;
  }
}

function drawHeatmap(key, polyObj) {
  if (!polyObj) return;
  const bm = state.heatmaps.get(key);
  if (!bm) return;
  const bbox = polyObj.heatmap_bbox_px;
  if (!bbox) return;
  ctx.drawImage(bm, bbox[0], bbox[1]);
}

function drawPolygon(poly, opts) {
  if (!poly || poly.length < 2) return;
  ctx.beginPath();
  for (let i = 0; i < poly.length; i++) {
    const p = worldToImg(poly[i][0], poly[i][1]);
    if (i === 0) ctx.moveTo(p.u, p.v); else ctx.lineTo(p.u, p.v);
  }
  ctx.closePath();
  ctx.strokeStyle = opts.stroke;
  ctx.lineWidth = (opts.lineWidth || 1.5) / state.view.scale;
  ctx.setLineDash(opts.dashed ? [8 / state.view.scale, 5 / state.view.scale] : []);
  ctx.stroke();
  ctx.setLineDash([]);

  if (opts.selected) {
    const r = 5 / state.view.scale;
    for (const p of poly) {
      const v = worldToImg(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(v.u, v.v, r, 0, Math.PI * 2);
      ctx.fillStyle = opts.stroke; ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5 / state.view.scale;
      ctx.stroke();
    }
  }
}

function drawInsertPreview() {
  const poly = polygonForKey(state.selection.key);
  if (!poly || poly.length < 2) return;
  const hover = worldToImg(state.hover.world.x, state.hover.world.z);
  const edge = nearestEdgeImg(poly, hover.u, hover.v);
  if (!edge.proj || edge.dist > 12 / state.view.scale) return;
  const a = worldToImg(poly[edge.index][0], poly[edge.index][1]);
  const b = worldToImg(poly[(edge.index + 1) % poly.length][0],
                        poly[(edge.index + 1) % poly.length][1]);
  ctx.beginPath();
  ctx.moveTo(a.u, a.v); ctx.lineTo(b.u, b.v);
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 3 / state.view.scale;
  ctx.stroke();
  const p = worldToImg(edge.proj[0], edge.proj[1]);
  ctx.beginPath();
  ctx.arc(p.u, p.v, 5 / state.view.scale, 0, Math.PI * 2);
  ctx.fillStyle = "#00e5ff";
  ctx.fill();
}

function drawDraft() {
  const poly = state.draft;
  ctx.beginPath();
  for (let i = 0; i < poly.length; i++) {
    const p = worldToImg(poly[i][0], poly[i][1]);
    if (i === 0) ctx.moveTo(p.u, p.v); else ctx.lineTo(p.u, p.v);
  }
  // Hover preview line back to first vertex
  if (state.hover.world) {
    const h = worldToImg(state.hover.world.x, state.hover.world.z);
    ctx.lineTo(h.u, h.v);
  }
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 2 / state.view.scale;
  ctx.setLineDash([6 / state.view.scale, 4 / state.view.scale]);
  ctx.stroke();
  ctx.setLineDash([]);

  const r = 5 / state.view.scale;
  for (let i = 0; i < poly.length; i++) {
    const v = worldToImg(poly[i][0], poly[i][1]);
    ctx.beginPath();
    ctx.arc(v.u, v.v, r, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#fff" : "#00e5ff";
    ctx.fill();
    ctx.strokeStyle = "#003a4a";
    ctx.lineWidth = 1.5 / state.view.scale;
    ctx.stroke();
  }
}

// ─── INPUT ────────────────────────────────────────────────────────────────
function getMouse(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function onMouseDown(e) {
  if (!state.plan) return;
  const m = getMouse(e);

  // Pan with middle / right / shift+left
  if (e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey)) {
    state.panning = { x: m.x, y: m.y, tx: state.view.tx, ty: state.view.ty };
    return;
  }

  if (state.mode.startsWith("draw_")) {
    const w = canvasToWorld(m.x, m.y);
    // Click on first vertex to close
    if (state.draft.length >= 3) {
      const first = worldToImg(state.draft[0][0], state.draft[0][1]);
      const cur = worldToImg(w.x, w.z);
      const dpx = Math.hypot(first.u - cur.u, first.v - cur.v) * state.view.scale;
      if (dpx < 12) { commitDraft(); return; }
    }
    state.draft.push([w.x, w.z]);
    draw();
    return;
  }

  // Insert-vertex mode: click an edge of the selected polygon to inject a vertex.
  if (state.tool === "insert-vertex" && state.selection) {
    const poly = polygonForKey(state.selection.key);
    if (!poly) return;
    const img = canvasToImg(m.x, m.y);
    const edge = nearestEdgeImg(poly, img.u, img.v);
    if (edge.proj && edge.dist < 12 / state.view.scale) {
      poly.splice(edge.index + 1, 0, [edge.proj[0], edge.proj[1]]);
      pushPolygonForKey(state.selection.key);
      draw();
      return;
    }
  }

  // Delete-vertex mode: click an existing vertex to remove it (min 3 verts).
  if (state.tool === "delete-vertex" && state.selection) {
    const poly = polygonForKey(state.selection.key);
    if (!poly || poly.length <= 3) return;
    const vi = nearestVertexIndex(poly, m.x, m.y, 12);
    if (vi >= 0) {
      poly.splice(vi, 1);
      pushPolygonForKey(state.selection.key);
      draw();
      return;
    }
  }

  // Select mode: click on a polygon vertex to drag, or click polygon edge to select
  const hit = hitTest(m.x, m.y);
  if (hit) {
    state.selection = { key: hit.key };
    if (state.tool === "select" && hit.vertexIndex != null) state.drag = hit;
    refreshPolygonsList();
    draw();
  } else {
    state.selection = null;
    refreshPolygonsList();
    draw();
  }
}

function nearestVertexIndex(poly, cx, cy, thresholdPx) {
  let best = -1, bestD = thresholdPx;
  for (let i = 0; i < poly.length; i++) {
    const ic = imgToCanvas(...Object.values(worldToImg(poly[i][0], poly[i][1])));
    const d = Math.hypot(ic.cx - cx, ic.cy - cy);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function nearestEdgeImg(poly, u, v) {
  let best = -1, bestD = Infinity, bestProj = null;
  for (let i = 0; i < poly.length; i++) {
    const a = worldToImg(poly[i][0], poly[i][1]);
    const b = worldToImg(poly[(i + 1) % poly.length][0], poly[(i + 1) % poly.length][1]);
    const dx = b.u - a.u, dy = b.v - a.v;
    const L2 = dx * dx + dy * dy || 1;
    let t = ((u - a.u) * dx + (v - a.v) * dy) / L2;
    t = Math.max(0, Math.min(1, t));
    const cu = a.u + t * dx, cv = a.v + t * dy;
    const d = Math.hypot(cu - u, cv - v);
    if (d < bestD) { bestD = d; best = i; bestProj = imgToWorld(cu, cv); }
  }
  return { index: best, dist: bestD, proj: bestProj ? [bestProj.x, bestProj.z] : null };
}

function onMouseMove(e) {
  if (!state.plan) return;
  const m = getMouse(e);
  state.hover.world = canvasToWorld(m.x, m.y);

  if (state.panning) {
    state.view.tx = state.panning.tx + (m.x - state.panning.x);
    state.view.ty = state.panning.ty + (m.y - state.panning.y);
    draw(); return;
  }

  if (state.drag) {
    const w = canvasToWorld(m.x, m.y);
    const poly = polygonForKey(state.drag.key);
    if (poly) poly[state.drag.vertexIndex] = [w.x, w.z];
    draw(); return;
  }

  if (state.mode.startsWith("draw_")) draw();
  else readout.textContent =
    `x ${state.hover.world.x.toFixed(2)} m   z ${state.hover.world.z.toFixed(2)} m`;
}

async function onMouseUp(e) {
  if (state.panning) { state.panning = null; return; }
  if (state.drag) {
    const key = state.drag.key;
    state.drag = null;
    // Push the polygon to the server so analyse refreshes.
    await pushPolygonForKey(key);
  }
}

function onDblClick(e) {
  if (state.mode.startsWith("draw_") && state.draft.length >= 3) commitDraft();
}

function onWheel(e) {
  e.preventDefault();
  if (!state.plan) return;
  const m = getMouse(e);
  const before = canvasToImg(m.x, m.y);
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  state.view.scale *= factor;
  const after = canvasToImg(m.x, m.y);
  state.view.tx += (after.u - before.u) * state.view.scale;
  state.view.ty += (after.v - before.v) * state.view.scale;
  draw();
}

function onKey(e) {
  if (e.key === "Enter" && state.mode.startsWith("draw_")) {
    if (state.draft.length >= 3) commitDraft();
  } else if (e.key === "Escape") {
    cancelDraw();
  }
}

// ─── HIT TEST + POLYGON HELPERS ───────────────────────────────────────────
function polygonForKey(key) {
  if (key === "room") return state.plan.room;
  if (key === "main") return state.plan.main?.polygon;
  if (key.startsWith("region:")) {
    const id = parseInt(key.slice(7), 10);
    return state.plan.regions.find(r => r.id === id)?.polygon;
  }
  return null;
}

function hitTest(cx, cy) {
  // Order candidates so the currently-selected polygon's vertices win
  // ties — otherwise dragging a region near a room corner would grab the
  // room's vertex instead.
  const all = [];
  if (state.plan.room) all.push({ key: "room", poly: state.plan.room });
  if (state.plan.main) all.push({ key: "main", poly: state.plan.main.polygon });
  for (const r of state.plan.regions || [])
    all.push({ key: "region:" + r.id, poly: r.polygon });

  const selKey = state.selection?.key;
  const ordered = selKey
    ? [...all.filter(c => c.key === selKey), ...all.filter(c => c.key !== selKey)]
    : all;

  // Take the closest vertex within threshold across ALL candidates, but
  // bias the selected polygon by counting its distance as 80% so a tie
  // resolves in its favour.
  let best = null, bestD = 11;
  for (const c of ordered) {
    const bias = (c.key === selKey) ? 0.8 : 1.0;
    for (let i = 0; i < c.poly.length; i++) {
      const p = imgToCanvas(...Object.values(worldToImg(c.poly[i][0], c.poly[i][1])));
      const d = Math.hypot(p.cx - cx, p.cy - cy) * bias;
      if (d < bestD) { bestD = d; best = { key: c.key, vertexIndex: i }; }
    }
  }
  if (best) return best;

  // Edge hit just selects the polygon. Innermost (smallest) first.
  const inside = ordered.filter(c => pointInPolygonCanvas(c.poly, cx, cy));
  if (inside.length === 0) return null;
  inside.sort((a, b) => polygonAreaCanvas(a.poly) - polygonAreaCanvas(b.poly));
  return { key: inside[0].key };
}

function polygonAreaCanvas(poly) {
  let a = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = imgToCanvas(...Object.values(worldToImg(poly[i][0], poly[i][1])));
    const q = imgToCanvas(...Object.values(worldToImg(poly[(i + 1) % n][0], poly[(i + 1) % n][1])));
    a += p.cx * q.cy - q.cx * p.cy;
  }
  return Math.abs(a) / 2;
}

function pointInPolygonCanvas(poly, cx, cy) {
  // Convert poly to canvas pixels and run ray-cast
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = imgToCanvas(...Object.values(worldToImg(poly[i][0], poly[i][1])));
    const b = imgToCanvas(...Object.values(worldToImg(poly[j][0], poly[j][1])));
    if (((a.cy > cy) !== (b.cy > cy)) &&
        (cx < (b.cx - a.cx) * (cy - a.cy) / (b.cy - a.cy) + a.cx)) {
      inside = !inside;
    }
  }
  return inside;
}

async function pushPolygonForKey(key) {
  if (key === "room") {
    const r = await fetch(`/api/sessions/${state.sessionId}/room`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: state.plan.room }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.room_heatmap = d.room_heatmap;
      if (d.room_heatmap) await refreshHeatmap("room", d.room_heatmap);
      else state.heatmaps.delete("room");
    }
  } else if (key === "main") {
    const r = await fetch(`/api/sessions/${state.sessionId}/main`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: state.plan.main.polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.main = d.main;
      await refreshHeatmap("main", d.main);
      // relative_y on regions changed → reload plan
      const planR = await fetch(`/api/sessions/${state.sessionId}/plan`);
      state.plan = await planR.json();
    }
  } else if (key.startsWith("region:")) {
    const id = parseInt(key.slice(7), 10);
    const reg = state.plan.regions.find(r => r.id === id);
    const r = await fetch(`/api/sessions/${state.sessionId}/region/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: reg.polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      Object.assign(reg, d.region);
      await refreshHeatmap("region:" + id, d.region);
    }
  }
  refreshPolygonsList();
  draw();
}

// ─── EXPORT ───────────────────────────────────────────────────────────────
async function exportPlan() {
  if (!state.sessionId) return;
  const r = await fetch(`/api/sessions/${state.sessionId}/export`);
  const data = await r.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ceiling_rcp_${state.sessionId}.json`;
  a.click();
}

// ─── BOOT ─────────────────────────────────────────────────────────────────
async function loadFromUrlParam() {
  const sid = new URLSearchParams(window.location.search).get("session");
  if (!sid) return;
  document.getElementById("session-label").textContent = `session ${sid} (loading…)`;
  try {
    const r = await fetch(`/api/sessions/${sid}/plan`);
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    state.sessionId = sid;
    state.plan = await r.json();
    document.getElementById("session-label").textContent = `session ${sid}`;
    document.getElementById("upload-section").hidden = true;
    document.getElementById("workflow").hidden = false;
    document.getElementById("btn-export").disabled = false;
    if (state.plan.room) {
      markStepDone("room"); unlockStep("main");
      document.getElementById("btn-autodetect").disabled = false;
    }
    if (state.plan.room_heatmap)
      await refreshHeatmap("room", state.plan.room_heatmap);
    if (state.plan.main) {
      markStepDone("main"); unlockStep("region");
      await refreshHeatmap("main", state.plan.main);
    }
    for (const r of state.plan.regions || [])
      await refreshHeatmap("region:" + r.id, r);
    await loadCeilingImage();
    fitView();
    refreshPolygonsList();
    draw();
  } catch (e) {
    document.getElementById("session-label").textContent =
      `session ${sid} — failed: ${e.message}`;
  }
}

resizeCanvas();
loadFromUrlParam();
