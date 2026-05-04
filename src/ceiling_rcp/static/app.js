console.log("[ceiling-rcp] app.js build 12 — units, scan settings, vertex UX");

// ─── UNITS ────────────────────────────────────────────────────────────────
// World coords stay in metres throughout. These helpers turn metres into
// the user's chosen display string. Mirror of src/ceiling_rcp/units.py —
// keep the two in sync.

const INCHES_PER_METRE = 39.3700787401574803;

function _gcd(a, b) { return b === 0 ? a : _gcd(b, a % b); }

function _roundToSixteenth(inches) {
  const totalSixteenths = Math.round(inches * 16);
  const feet = Math.floor(totalSixteenths / (12 * 16));
  const rem = totalSixteenths - feet * 12 * 16;
  const wholeInches = Math.floor(rem / 16);
  const sixteenths = rem - wholeInches * 16;
  return [feet, wholeInches, sixteenths];
}

function _fraction(sixteenths) {
  if (sixteenths === 0) return "";
  const g = _gcd(sixteenths, 16);
  return `${sixteenths / g}/${16 / g}`;
}

function _imperialLength(metres) {
  const sign = metres < 0 ? "-" : "";
  const inches = Math.abs(metres) * INCHES_PER_METRE;
  const [feet, whole, sixteenths] = _roundToSixteenth(inches);
  const frac = _fraction(sixteenths);
  if (feet === 0) {
    if (whole === 0 && frac === "") return `${sign}0"`;
    if (whole === 0) return `${sign}${frac}"`;
    if (frac) return `${sign}${whole} ${frac}"`;
    return `${sign}${whole}"`;
  }
  let inchesPart = `${whole}`;
  if (frac) inchesPart += ` ${frac}`;
  return `${sign}${feet}'-${inchesPart}"`;
}

function _imperialHeightDelta(metres) {
  if (metres === 0) return '0"';
  const sign = metres < 0 ? "−" : "+";
  const inches = Math.abs(metres) * INCHES_PER_METRE;
  const [feet, whole, sixteenths] = _roundToSixteenth(inches);
  const frac = _fraction(sixteenths);
  if (feet === 0) {
    if (whole === 0 && frac === "") return '0"';
    if (whole === 0) return `${sign}${frac}"`;
    if (frac) return `${sign}${whole} ${frac}"`;
    return `${sign}${whole}"`;
  }
  let inchesPart = `${whole}`;
  if (frac) inchesPart += ` ${frac}`;
  return `${sign}${feet}'-${inchesPart}"`;
}

function unitsSystem() {
  return state.plan?.units || "metric";
}

function formatLength(metres) {
  if (unitsSystem() === "imperial") return _imperialLength(metres);
  if (Math.abs(metres) < 1.0) return `${Math.round(metres * 1000)} mm`;
  return `${metres.toFixed(2)} m`;
}

function formatHeightDelta(metres) {
  if (unitsSystem() === "imperial") return _imperialHeightDelta(metres);
  if (metres === 0) return "0 mm";
  const sign = metres < 0 ? "−" : "+";
  return `${sign}${Math.round(Math.abs(metres) * 1000)} mm`;
}

// ─── TOPOLOGY HELPERS ─────────────────────────────────────────────────────
// Post-snap, plan.topology is the source of truth: a planar graph where
// shared edges between two faces are stored ONCE. Vertex drags here update
// the topology, which then re-derives every affected face polygon — so
// dragging a wall between two ceiling regions moves both at the same time.

const TOPOLOGY_VERTEX_TOL_M = 0.005;  // 5 mm — tighter than RDP simplify

function topologyVertexAtWorld(x, z, tolM = TOPOLOGY_VERTEX_TOL_M) {
  // Returns the topology vertex id within `tolM` of (x, z), or -1 if none.
  // Default tolerance (5mm) is for vertex-pickup on drag — exact-match
  // resolution. Pass a wider tolerance for click-to-delete (~5cm).
  const topo = state.plan?.topology;
  if (!topo) return -1;
  let best = -1, bestD = tolM;
  for (let i = 0; i < topo.vertices.length; i++) {
    const v = topo.vertices[i];
    const d = Math.hypot(v[0] - x, v[1] - z);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function nearestTopologyEdge(x, z) {
  // Return {edgeId, proj: [x, z], distPx} for the topology edge closest
  // to (x, z). Distance is computed in canvas pixels for tool-radius
  // checks; projection is in world coords for the insert payload.
  const topo = state.plan?.topology;
  if (!topo) return null;
  let best = null;
  let bestD = Infinity;
  for (const e of topo.edges) {
    for (let k = 0; k < e.vertices.length - 1; k++) {
      const a = topo.vertices[e.vertices[k]];
      const b = topo.vertices[e.vertices[k + 1]];
      const dx = b[0] - a[0], dz = b[1] - a[1];
      const L2 = dx * dx + dz * dz;
      if (L2 < 1e-12) continue;
      let t = ((x - a[0]) * dx + (z - a[1]) * dz) / L2;
      t = Math.max(0, Math.min(1, t));
      const px = a[0] + t * dx, pz = a[1] + t * dz;
      const dWorld = Math.hypot(px - x, pz - z);
      if (dWorld < bestD) {
        bestD = dWorld;
        best = { edgeId: e.id, proj: [px, pz], distWorld: dWorld };
      }
    }
  }
  if (best) {
    // Convert world distance to canvas pixels using the grid + view scale
    // so the tool-radius threshold (12 px) is consistent with other tools.
    const ppm = state.plan.grid.pixels_per_metre;
    best.distPx = best.distWorld * ppm * state.view.scale;
  }
  return best;
}

async function insertVertexOnTopologyEdge(edgeId, projWorld) {
  const r = await fetch(
    `/api/sessions/${state.sessionId}/topology/edge/${edgeId}/insert_vertex`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ position: projWorld }),
    },
  );
  if (!r.ok) {
    setBanner("Insert vertex failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan = await r.json();
  await refreshAllHeatmaps();
  refreshPolygonsList();
  draw();
}

async function deleteTopologyVertex(vid) {
  const r = await fetch(
    `/api/sessions/${state.sessionId}/topology/vertex/${vid}`,
    { method: "DELETE" },
  );
  if (!r.ok) {
    setBanner("Delete vertex failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan = await r.json();
  await refreshAllHeatmaps();
  refreshPolygonsList();
  draw();
}

async function unsnapTopology() {
  if (!confirm("Un-snap clears the shared-edge topology. The current ceiling regions stay; you can edit them and re-snap. Continue?")) return;
  const r = await fetch(`/api/sessions/${state.sessionId}/topology`, { method: "DELETE" });
  if (!r.ok) {
    setBanner("Un-snap failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan = await r.json();
  await refreshAllHeatmaps();
  refreshPolygonsList();
  draw();
  setBanner("Un-snapped — edit polygons then re-snap.");
  setTimeout(() => banner.classList.remove("show"), 2500);
}

function snapAlongOrPerp(x, z, anchor, refDir) {
  // Snap the cursor position so it lies on either the line through
  // ``anchor`` along ``refDir`` (collinear with the reference edge) or on
  // the perpendicular through ``anchor`` — whichever is closer to the
  // cursor. Returns ``[x, z]`` or ``null`` if the reference is degenerate.
  const L = Math.hypot(refDir[0], refDir[1]);
  if (L < 1e-6) return null;
  const ux = refDir[0] / L, uz = refDir[1] / L;
  const px = -uz, pz = ux;
  const rx = x - anchor[0], rz = z - anchor[1];
  const tParallel = rx * ux + rz * uz;
  const tPerp = rx * px + rz * pz;
  if (Math.abs(tPerp) > Math.abs(tParallel)) {
    return [anchor[0] + tPerp * px, anchor[1] + tPerp * pz];
  }
  return [anchor[0] + tParallel * ux, anchor[1] + tParallel * uz];
}

function constrainShiftSnap(x, z) {
  // Drawing case. The anchor is the LAST drafted vertex; the direction
  // reference is the edge from the second-to-last to the last vertex.
  const n = state.draft.length;
  if (n === 0) return null;
  if (n === 1) {
    // Only one vertex placed: snap horizontal/vertical from it.
    const a = state.draft[0];
    const dx = x - a[0], dz = z - a[1];
    return Math.abs(dx) > Math.abs(dz) ? [a[0] + dx, a[1]] : [a[0], a[1] + dz];
  }
  const a = state.draft[n - 2];
  const b = state.draft[n - 1];
  return snapAlongOrPerp(x, z, b, [b[0] - a[0], b[1] - a[1]]);
}

function constrainShiftSnapForDrag(x, z, dragInfo) {
  // Drag case. Snap the moving vertex relative to its ring neighbours in
  // the *selected* face (post-snap) or the polygon being dragged
  // (pre-snap). Reference edge is prev_prev → prev — same construction as
  // the drawing case.
  const ring = dragRingFor(dragInfo);
  if (!ring) return null;
  const { poly, dragIdx } = ring;
  if (poly.length < 3 || dragIdx < 0) return null;
  const n = poly.length;
  const prev = poly[(dragIdx - 1 + n) % n];
  const prevPrev = poly[(dragIdx - 2 + n) % n];
  return snapAlongOrPerp(x, z, prev, [prev[0] - prevPrev[0], prev[1] - prevPrev[1]]);
}

function dragRingFor(dragInfo) {
  // Returns the ring `{poly, dragIdx}` to snap against. For pre-snap edits
  // it's the polygon being dragged; for topology edits it's the selected
  // face's ring (with the dragged vertex located by world-coord match).
  if (dragInfo.topologyVid != null) {
    const topo = state.plan?.topology;
    if (!topo?.vertices) return null;
    const target = topo.vertices[dragInfo.topologyVid];
    if (!target) return null;
    const sel = state.selection?.key;
    let face = null;
    if (sel === "main") face = topo.faces?.find(f => f.id === 0);
    else if (sel?.startsWith("region:")) {
      const rid = parseInt(sel.slice(7), 10);
      face = topo.faces?.find(f => f.region_id === rid || f.id === rid + 1);
    }
    if (!face) {
      // Fallback: any face whose ring contains this world point.
      face = (topo.faces || []).find(f =>
        (f.polygon || []).some(([px, pz]) =>
          Math.hypot(px - target[0], pz - target[1]) < 1e-6));
    }
    if (!face?.polygon) return null;
    const dragIdx = face.polygon.findIndex(([px, pz]) =>
      Math.hypot(px - target[0], pz - target[1]) < 1e-6);
    return { poly: face.polygon, dragIdx };
  }
  const poly = polygonForKey(dragInfo.key);
  if (!poly) return null;
  return { poly, dragIdx: dragInfo.vertexIndex };
}

async function refreshAllHeatmaps() {
  state.heatmaps.clear();
  if (state.plan.room_heatmap) await refreshHeatmap("room", state.plan.room_heatmap);
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
}

function rederiveFacePolygons() {
  // Walk every face's edge ring against the current vertex pool; mutate
  // both `face.polygon` and the legacy main/regions polygon arrays the
  // renderer reads, so a single vertex drag updates every shape that
  // shares it.
  const topo = state.plan?.topology;
  if (!topo) return;
  const edgesById = new Map(topo.edges.map(e => [e.id, e]));
  for (const face of topo.faces) {
    const pts = [];
    for (const h of face.ring) {
      const e = edgesById.get(h.edge);
      if (!e) continue;
      const verts = h.rev ? e.vertices.slice().reverse() : e.vertices;
      for (let k = 0; k < verts.length - 1; k++) {
        const v = topo.vertices[verts[k]];
        pts.push([v[0], v[1]]);
      }
    }
    face.polygon = pts;
    if (face.id === 0 && state.plan.main) {
      state.plan.main.polygon = pts;
    } else if (face.id !== 0) {
      const rid = face.region_id;
      const r = (state.plan.regions || []).find(r => r.id === rid);
      if (r) r.polygon = pts;
    }
  }
}

async function pushTopologyVertices() {
  const topo = state.plan?.topology;
  if (!topo) return;
  const r = await fetch(
    `/api/sessions/${state.sessionId}/topology/vertices`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vertices: topo.vertices }),
    },
  );
  if (!r.ok) {
    setBanner("Topology update failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan = await r.json();
  state.heatmaps.clear();
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
  refreshPolygonsList();
  draw();
}

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
  hover: { world: null, vertex: null },
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
document.getElementById("btn-column").onclick = () => startDraw("column");

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
document.getElementById("btn-unsnap").onclick = unsnapTopology;
document.getElementById("btn-pdf").onclick = downloadPdf;
document.getElementById("btn-export").onclick = exportPlan;
document.getElementById("btn-apply-variance").onclick = applyMaxVariance;
document.querySelectorAll(".seg-btn[data-units]").forEach(b => {
  b.onclick = () => setUnits(b.dataset.units);
});
document.querySelectorAll("[data-project]").forEach(inp => {
  inp.addEventListener("input", () => scheduleProjectPush(
    inp.dataset.project, inp.value));
});
document.getElementById("input-north-deg").addEventListener("input", (e) => {
  const v = ((parseFloat(e.target.value) || 0) % 360 + 360) % 360;
  setNorthAngleUI(v);
  scheduleProjectPush("north_deg", v);
});
document.getElementById("input-print-north").addEventListener("change", (e) => {
  scheduleProjectPush("print_north", e.target.checked);
});
{
  const svg = document.getElementById("north-picker");
  svg.addEventListener("click", (e) => {
    const rect = svg.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    // Page-up = 0°, clockwise positive (compass convention).
    let angle = Math.atan2(dx, -dy) * 180 / Math.PI;
    if (angle < 0) angle += 360;
    angle = Math.round(angle);
    document.getElementById("input-north-deg").value = angle;
    setNorthAngleUI(angle);
    scheduleProjectPush("north_deg", angle);
  });
}
document.getElementById("btn-add-register-row").onclick = () => {
  appendRegisterRow({ rev: "", date: "", by: "", note: "" });
  pushDrawingRegister();
};

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
  if (kind === "column" && !state.plan.room) return;
  state.mode = "draw_" + kind;
  state.draft = [];
  state.selection = null;
  banner.textContent = {
    room: "Drawing ROOM outline — click vertices, click first or press Enter to close. Esc = cancel",
    main: "Drawing MAIN CEILING — defines height datum (relative = 0)",
    region: "Drawing CEILING REGION — relative to main ceiling",
    column: "Drawing COLUMN — ceiling regions stop at its boundary. Hold Shift to lock 90°.",
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
      unlockStep("column");
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
      if (d.snapped && d.plan) {
        // Server auto re-snapped — replace the whole plan and refresh
        // every heatmap because face shapes (and therefore stats) all
        // changed.
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else {
        state.plan.regions = state.plan.regions || [];
        state.plan.regions.push(d.region);
        await refreshHeatmap("region:" + d.region.id, d.region);
      }
    } else {
      setBanner("Add region failed: " + (await r.text()).slice(0, 120), true);
    }
  } else if (kind === "column") {
    const r = await fetch(`/api/sessions/${state.sessionId}/obstruction`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon, kind: "column" }),
    });
    if (r.ok) {
      const d = await r.json();
      if (d.snapped && d.plan) {
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else {
        state.plan.obstructions = state.plan.obstructions || [];
        state.plan.obstructions.push(d.obstruction);
      }
    } else {
      setBanner("Add column failed: " + (await r.text()).slice(0, 120), true);
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
  if (kind === "column") {
    const r = await fetch(`/api/sessions/${state.sessionId}/obstruction/${regionId}`,
      { method: "DELETE" });
    if (r.ok) {
      const d = await r.json();
      if (d.snapped && d.plan) {
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else {
        state.plan.obstructions = (state.plan.obstructions || []).filter(o => o.id !== regionId);
      }
    }
    state.selection = null;
    refreshPolygonsList();
    draw();
    return;
  }
  if (kind === "region") {
    const r = await fetch(`/api/sessions/${state.sessionId}/region/${regionId}`,
      { method: "DELETE" });
    if (r.ok) {
      const d = await r.json();
      if (d.snapped && d.plan) {
        // Auto re-snap after delete: the deleted region's area is
        // re-absorbed by neighbours, so every face needs a fresh heatmap.
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else {
        state.plan.regions = state.plan.regions.filter(r => r.id !== regionId);
        state.heatmaps.delete("region:" + regionId);
      }
    }
  } else if (kind === "room") {
    await fetch(`/api/sessions/${state.sessionId}/room`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null }),
    });
    state.plan.room = null;
    state.heatmaps.delete("room");
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
    "select": "Click a vertex to drag. Press Delete over a vertex to remove it.",
    "insert-vertex": "Select a polygon, then click on one of its edges to insert a vertex there.",
  };
  const el = document.getElementById("tool-hint");
  if (el) el.innerHTML = hints[state.tool] || "";
}

// ─── PROJECT METADATA ────────────────────────────────────────────────────
const _projectPushTimers = {};
function scheduleProjectPush(key, value) {
  if (!state.sessionId) return;
  if (!state.plan.project) state.plan.project = {};
  state.plan.project[key] = value;
  clearTimeout(_projectPushTimers[key]);
  _projectPushTimers[key] = setTimeout(() => pushProject({ [key]: value }), 350);
}

async function pushProject(patch) {
  if (!state.sessionId) return;
  const r = await fetch(`/api/sessions/${state.sessionId}/project`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) {
    setBanner("Project save failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  const d = await r.json();
  state.plan.project = d.project;
}

function setNorthAngleUI(deg) {
  const needle = document.getElementById("north-needle");
  if (needle) needle.setAttribute("transform", `rotate(${deg})`);
}

function appendRegisterRow(row) {
  const tbody = document.querySelector("#register-table tbody");
  const tr = document.createElement("tr");
  for (const key of ["rev", "date", "by", "note"]) {
    const td = document.createElement("td");
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = row[key] || "";
    inp.dataset.regField = key;
    inp.addEventListener("input", () => scheduleRegisterPush());
    td.appendChild(inp);
    tr.appendChild(td);
  }
  const tdDel = document.createElement("td");
  const del = document.createElement("button");
  del.className = "del-row"; del.textContent = "×"; del.title = "Remove";
  del.onclick = () => { tr.remove(); pushDrawingRegister(); };
  tdDel.appendChild(del);
  tr.appendChild(tdDel);
  tbody.appendChild(tr);
}

let _regTimer = null;
function scheduleRegisterPush() {
  clearTimeout(_regTimer);
  _regTimer = setTimeout(pushDrawingRegister, 400);
}

function pushDrawingRegister() {
  const rows = [];
  document.querySelectorAll("#register-table tbody tr").forEach(tr => {
    const row = {};
    tr.querySelectorAll("input[data-reg-field]").forEach(inp => {
      row[inp.dataset.regField] = inp.value;
    });
    if (row.rev || row.date || row.by || row.note) rows.push(row);
  });
  pushProject({ drawing_register: rows });
}

function syncProjectPanel(project) {
  if (!project) return;
  document.querySelectorAll("[data-project]").forEach(inp => {
    const k = inp.dataset.project;
    if (k in project) inp.value = project[k] ?? "";
  });
  const deg = Number(project.north_deg) || 0;
  document.getElementById("input-north-deg").value = Math.round(deg);
  setNorthAngleUI(deg);
  document.getElementById("input-print-north").checked =
    project.print_north !== false;
  // Drawing register table.
  const tbody = document.querySelector("#register-table tbody");
  tbody.innerHTML = "";
  for (const row of (project.drawing_register || [])) {
    appendRegisterRow(row);
  }
}

async function setUnits(value) {
  if (!state.sessionId) return;
  if (state.plan?.units === value) return;
  const r = await fetch(`/api/sessions/${state.sessionId}/units`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  if (!r.ok) {
    setBanner("Units update failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  state.plan.units = value;
  applyUnitsToggleUI();
  refreshPolygonsList();
  // Variance preview, region label info, etc. reflect the new units.
  const sel = state.selection;
  if (sel) updateSelectionInfo();
}

function applyUnitsToggleUI() {
  const sys = state.plan?.units || "metric";
  document.querySelectorAll(".seg-btn[data-units]").forEach(b => {
    b.classList.toggle("active", b.dataset.units === sys);
  });
}

async function applyMaxVariance() {
  if (!state.sessionId) return;
  const inp = document.getElementById("input-max-variance");
  const v = Number.parseFloat(inp.value);
  if (!Number.isFinite(v) || v < 0.5 || v > 6.0) {
    setBanner("Enter a value between 0.5 and 6.0 m.", true);
    return;
  }
  if (state.plan?.topology) {
    const ok = confirm(
      "Re-rendering will drop the current snap (you'll need to Snap again). " +
      "Continue?"
    );
    if (!ok) return;
  }
  setBanner("Re-rendering with new ceiling-variance limit…");
  const r = await fetch(`/api/sessions/${state.sessionId}/scan_settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_ceiling_variance_m: v }),
  });
  if (!r.ok) {
    setBanner("Re-render failed: " + (await r.text()).slice(0, 160), true);
    return;
  }
  state.plan = await r.json();
  state.heatmaps.clear();
  if (state.plan.room_heatmap)
    await refreshHeatmap("room", state.plan.room_heatmap);
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
  await loadCeilingImage();
  fitView();
  refreshPolygonsList();
  draw();
  setBanner("Re-rendered. Re-snap if you had snapped.");
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
  // Show Un-snap only when a topology exists; show Snap always pre-snap.
  const hasTopology = !!state.plan?.topology;
  document.getElementById("btn-unsnap").hidden = !hasTopology;
  document.getElementById("btn-snap").textContent =
    hasTopology ? "Re-snap (rebuild from current polygons)" : "Snap polygons";

  const ul = document.getElementById("polygons-list");
  ul.innerHTML = "";

  const addRow = (key, label, color, meta, selKey, currentNotes, allowNotes, allowTintEdit) => {
    const li = document.createElement("li");
    li.className = "poly-row" + (state.selection?.key === selKey ? " active" : "");
    const head = document.createElement("div");
    head.className = "poly-head";
    if (allowTintEdit) {
      head.innerHTML =
        `<input type="color" class="swatch swatch-input" value="${color}" title="Change tint">` +
        `<div class="label">${label}</div>` +
        `<div class="meta">${meta}</div>` +
        `<button class="del-btn" title="Delete">×</button>`;
    } else {
      head.innerHTML =
        `<div class="swatch" style="background:${color}"></div>` +
        `<div class="label">${label}</div>` +
        `<div class="meta">${meta}</div>` +
        `<button class="del-btn" title="Delete">×</button>`;
    }
    li.appendChild(head);

    if (allowTintEdit) {
      const sw = head.querySelector(".swatch-input");
      sw.onclick = (e) => e.stopPropagation();
      let tintTimer = null;
      sw.oninput = () => {
        clearTimeout(tintTimer);
        tintTimer = setTimeout(() => pushTintForKey(selKey, sw.value), 250);
      };
      sw.onchange = () => {
        clearTimeout(tintTimer);
        pushTintForKey(selKey, sw.value);
      };
    }

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
      `${state.plan.room.length} verts`, "room", null, false, false);
  }
  if (state.plan.main) {
    const s = state.plan.main.stats;
    addRow("main", state.plan.main.label || "Main Ceiling (1)",
      state.plan.main.tint || "#80cbc4",
      `${formatHeightDelta(0)}  variance ${formatLength(s.std_y)}`,
      "main", state.plan.main.notes, true, true);
  }
  for (const r of state.plan.regions || []) {
    const s = r.stats;
    const rel = r.relative_y;
    const relTxt = rel === null || rel === undefined ? "—" : formatHeightDelta(rel);
    addRow("region:" + r.id, r.label || `Ceiling Region (${r.id + 2})`,
      r.tint || "#ff7043",
      `${relTxt}  variance ${formatLength(s.std_y)}`,
      "region:" + r.id, r.notes, true, true);
  }
  for (const o of state.plan.obstructions || []) {
    addRow("column:" + o.id, o.label || `Column (${o.id + 1})`,
      "#ffffff", `${o.polygon.length} verts`,
      "column:" + o.id, null, false, false);
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
  const fmt_std = formatLength(s.std_y);
  const fmt_rel = relativeY === null || relativeY === undefined
    ? `${formatHeightDelta(0)} (datum)`
    : formatHeightDelta(relativeY);
  const valid_pct = (s.valid_frac * 100).toFixed(0);
  const fmt_range = formatLength(s.max_y - s.min_y);
  const warn = s.std_y > 0.05
    ? `<div class="warn-line">⚠ high variance (${fmt_std}) — likely clipped a bulkhead</div>`
    : "";
  const validWarn = s.valid_frac < 0.6
    ? `<div class="warn-line">⚠ only ${valid_pct}% of polygon has LiDAR coverage</div>`
    : "";
  return `
    <b>${title}</b><br>
    relative height: ${fmt_rel}<br>
    variance inside polygon: ${fmt_std}<br>
    range inside polygon: ${fmt_range}<br>
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
  // Obstructions (columns) — hatched fill on top so the negative space
  // reads above any region tint that briefly bleeds through after a snap.
  for (const o of state.plan.obstructions || []) {
    drawObstruction(o);
  }
  // Hover preview for insert-vertex
  if (state.tool === "insert-vertex" && state.selection && state.hover.world) {
    drawInsertPreview();
  }

  // Draft polygon being drawn
  if (state.mode.startsWith("draw_") && state.draft.length > 0) {
    drawDraft();
  }

  // Hovered-vertex highlight (drawn on top of everything so it's always
  // visible). Communicates "Delete will hit *this* one".
  if (state.hover.vertex && !state.drag && !state.mode.startsWith("draw_")) {
    let world = null;
    const h = state.hover.vertex;
    if (h.topologyVid != null) {
      world = state.plan?.topology?.vertices?.[h.topologyVid] || null;
    } else {
      const poly = polygonForKey(h.key);
      world = poly ? poly[h.vertexIndex] : null;
    }
    if (world) {
      const v = worldToImg(world[0], world[1]);
      const r = 7 / state.view.scale;
      ctx.beginPath();
      ctx.arc(v.u, v.v, r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fill();
      ctx.strokeStyle = "#00e5ff";
      ctx.lineWidth = 2 / state.view.scale;
      ctx.stroke();
    }
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

function drawObstruction(o) {
  if (!o?.polygon || o.polygon.length < 3) return;
  const poly = o.polygon;
  // Build the polygon path in image coords and fill with white so the
  // ceiling underneath reads as masked-off, then hatch on top.
  ctx.save();
  ctx.beginPath();
  for (let i = 0; i < poly.length; i++) {
    const p = worldToImg(poly[i][0], poly[i][1]);
    if (i === 0) ctx.moveTo(p.u, p.v); else ctx.lineTo(p.u, p.v);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fill();
  // Hatched stroke pattern: clip to the polygon, then stroke a grid of
  // diagonal lines. Cheap and crisp at any zoom.
  ctx.clip();
  const minU = Math.min(...poly.map(p => worldToImg(p[0], p[1]).u));
  const maxU = Math.max(...poly.map(p => worldToImg(p[0], p[1]).u));
  const minV = Math.min(...poly.map(p => worldToImg(p[0], p[1]).v));
  const maxV = Math.max(...poly.map(p => worldToImg(p[0], p[1]).v));
  const step = Math.max(6, 10 / state.view.scale);
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1.0 / state.view.scale;
  ctx.beginPath();
  for (let s = minU - (maxV - minV); s < maxU + (maxV - minV); s += step) {
    ctx.moveTo(s, minV);
    ctx.lineTo(s + (maxV - minV), maxV);
  }
  ctx.stroke();
  ctx.restore();
  // Outline last so it sits above the hatch.
  ctx.beginPath();
  for (let i = 0; i < poly.length; i++) {
    const p = worldToImg(poly[i][0], poly[i][1]);
    if (i === 0) ctx.moveTo(p.u, p.v); else ctx.lineTo(p.u, p.v);
  }
  ctx.closePath();
  const isSelected = state.selection?.key === "column:" + o.id;
  ctx.strokeStyle = isSelected ? "#00e5ff" : "#222";
  ctx.lineWidth = (isSelected ? 2.0 : 1.4) / state.view.scale;
  ctx.stroke();
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

  // Pan with middle / right / shift+left (but NOT in draw mode — there
  // shift means "snap the next vertex to 0° / 90° relative to the
  // previous edge").
  const drawing = state.mode.startsWith("draw_");
  if (e.button === 1 || e.button === 2 ||
      (e.button === 0 && e.shiftKey && !drawing)) {
    state.panning = { x: m.x, y: m.y, tx: state.view.tx, ty: state.view.ty };
    return;
  }

  if (drawing) {
    let w = canvasToWorld(m.x, m.y);
    if (e.shiftKey) {
      const c = constrainShiftSnap(w.x, w.z);
      if (c) w = { x: c[0], z: c[1] };
    }
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

  // Insert-vertex mode: click an edge to inject a vertex.
  // Post-snap: finds the nearest topology edge so the new vertex is added
  // to BOTH faces sharing it; pre-snap: works on the selected polygon.
  if (state.tool === "insert-vertex") {
    if (state.plan?.topology) {
      const w = canvasToWorld(m.x, m.y);
      const hit = nearestTopologyEdge(w.x, w.z);
      if (hit && hit.distPx < 12) {
        insertVertexOnTopologyEdge(hit.edgeId, hit.proj);
        return;
      }
      return;
    }
    if (!state.selection) return;
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

  // (Vertex deletion is handled by the Delete key in onKey, not by a tool.)

  // Select mode: click on a polygon vertex to drag, or click polygon edge to select
  const hit = hitTest(m.x, m.y);
  if (hit) {
    state.selection = { key: hit.key };
    if (state.tool === "select" && hit.vertexIndex != null) {
      state.drag = hit;
      // Post-snap, this vertex is shared with every other face that owns
      // the same junction. Resolve to a topology vertex so the drag moves
      // them all in lockstep.
      const poly = polygonForKey(hit.key);
      if (poly) {
        const v = poly[hit.vertexIndex];
        const vid = topologyVertexAtWorld(v[0], v[1]);
        if (vid >= 0) state.drag.topologyVid = vid;
      }
    }
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

  // While drawing with shift held, snap the hover preview so the next
  // edge will continue collinearly OR turn at exactly 90° — whichever
  // direction the cursor is closer to. Drawn-vertex commit reads from
  // hover.world too, so the click lands exactly on the preview line.
  if (state.mode.startsWith("draw_") && e.shiftKey) {
    const c = constrainShiftSnap(state.hover.world.x, state.hover.world.z);
    if (c) state.hover.world = { x: c[0], z: c[1] };
  }

  if (state.panning) {
    state.view.tx = state.panning.tx + (m.x - state.panning.x);
    state.view.ty = state.panning.ty + (m.y - state.panning.y);
    draw(); return;
  }

  if (state.drag) {
    let w = canvasToWorld(m.x, m.y);
    if (e.shiftKey) {
      // Constrain the dragged-to point to be 0/90° relative to the prev
      // edge of the selected face's ring — same intent as draw-mode shift
      // but anchored to the existing polygon instead of an in-progress
      // draft.
      const c = constrainShiftSnapForDrag(w.x, w.z, state.drag);
      if (c) w = { x: c[0], z: c[1] };
    }
    if (state.drag.topologyVid != null) {
      // Move the shared topology vertex; all incident face polygons follow.
      state.plan.topology.vertices[state.drag.topologyVid] = [w.x, w.z];
      rederiveFacePolygons();
    } else {
      const poly = polygonForKey(state.drag.key);
      if (poly) poly[state.drag.vertexIndex] = [w.x, w.z];
    }
    draw(); return;
  }

  // Track which vertex (if any) the cursor is hovering over — used by the
  // Delete-key handler and as a visual affordance in the canvas redraw.
  state.hover.vertex = findHoveredVertex(m.x, m.y, 12);

  if (state.mode.startsWith("draw_")) draw();
  else {
    readout.textContent =
      `x ${state.hover.world.x.toFixed(2)} m   z ${state.hover.world.z.toFixed(2)} m`;
    draw();
  }
}

function findHoveredVertex(cx, cy, thresholdPx) {
  // Post-snap: prefer topology vertices so a junction shared by N faces
  // resolves to the single shared id (consistent with the drag path).
  if (state.plan?.topology) {
    const verts = state.plan.topology.vertices || [];
    let best = -1, bestD = thresholdPx;
    for (let vid = 0; vid < verts.length; vid++) {
      const ic = imgToCanvas(...Object.values(worldToImg(verts[vid][0], verts[vid][1])));
      const d = Math.hypot(ic.cx - cx, ic.cy - cy);
      if (d < bestD) { bestD = d; best = vid; }
    }
    if (best >= 0) return { topologyVid: best };
  }
  // Pre-snap (or for column polygons that aren't part of topology): hit-
  // test every visible polygon's vertices.
  const candidates = [];
  if (state.plan?.room) candidates.push({ key: "room", poly: state.plan.room });
  if (state.plan?.main) candidates.push({ key: "main", poly: state.plan.main.polygon });
  for (const r of state.plan?.regions || [])
    candidates.push({ key: "region:" + r.id, poly: r.polygon });
  for (const o of state.plan?.obstructions || [])
    candidates.push({ key: "column:" + o.id, poly: o.polygon });
  let best = null, bestD = thresholdPx;
  for (const c of candidates) {
    if (!c.poly) continue;
    for (let i = 0; i < c.poly.length; i++) {
      const ic = imgToCanvas(...Object.values(worldToImg(c.poly[i][0], c.poly[i][1])));
      const d = Math.hypot(ic.cx - cx, ic.cy - cy);
      if (d < bestD) { bestD = d; best = { key: c.key, vertexIndex: i }; }
    }
  }
  return best;
}

async function onMouseUp(e) {
  if (state.panning) { state.panning = null; return; }
  if (state.drag) {
    const wasTopologyDrag = state.drag.topologyVid != null;
    const key = state.drag.key;
    state.drag = null;
    if (wasTopologyDrag) {
      // Server replays the move against every face the vertex belongs to,
      // re-runs the height analysis, and returns a fresh plan.
      await pushTopologyVertices();
    } else {
      await pushPolygonForKey(key);
    }
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
  } else if ((e.key === "Delete" || e.key === "Backspace") && e.type === "keydown") {
    // Delete the vertex under the cursor. Drag is the default click
    // behaviour; this is the only path to vertex deletion.
    if (state.mode.startsWith("draw_")) return;  // don't intercept while drawing
    if (!state.hover.vertex) return;
    e.preventDefault();
    deleteHoveredVertex();
  } else if (e.key === "Shift" && state.mode.startsWith("draw_") && state.hover.world) {
    // Re-snap the preview the instant shift is pressed/released, even
    // if the mouse hasn't moved since.
    const c = e.type === "keydown"
      ? constrainShiftSnap(state.hover.world.x, state.hover.world.z)
      : null;
    if (c) state.hover.world = { x: c[0], z: c[1] };
    draw();
  }
}

async function deleteHoveredVertex() {
  const h = state.hover.vertex;
  if (!h) return;
  if (h.topologyVid != null) {
    await deleteTopologyVertex(h.topologyVid);
    state.hover.vertex = null;
    return;
  }
  // Pre-snap (or column) — splice the vertex out of the polygon and push.
  const poly = polygonForKey(h.key);
  if (!poly || poly.length <= 3) {
    setBanner("Polygon needs at least 3 vertices.", true);
    return;
  }
  poly.splice(h.vertexIndex, 1);
  state.hover.vertex = null;
  await pushPolygonForKey(h.key);
  draw();
}
window.addEventListener("keyup", onKey);

// ─── HIT TEST + POLYGON HELPERS ───────────────────────────────────────────
function polygonForKey(key) {
  if (key === "room") return state.plan.room;
  if (key === "main") return state.plan.main?.polygon;
  if (key.startsWith("region:")) {
    const id = parseInt(key.slice(7), 10);
    return state.plan.regions.find(r => r.id === id)?.polygon;
  }
  if (key.startsWith("column:")) {
    const id = parseInt(key.slice(7), 10);
    return (state.plan.obstructions || []).find(o => o.id === id)?.polygon;
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
  // Columns are hit-testable too — they're not part of the topology graph
  // but the user must still be able to drag/select their vertices.
  for (const o of state.plan.obstructions || [])
    all.push({ key: "column:" + o.id, poly: o.polygon });

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

async function pushTintForKey(key, tint) {
  if (!state.sessionId || !key || !tint) return;
  const topo = state.plan?.topology;
  let url, body;
  if (topo) {
    // Find the face id corresponding to the legacy key.
    let faceId = null;
    if (key === "main") {
      faceId = 0;
    } else if (key.startsWith("region:")) {
      const rid = parseInt(key.slice(7), 10);
      const f = (topo.faces || []).find(
        f => f.region_id === rid || (f.kind !== "main" && f.id === rid + 1));
      if (f) faceId = f.id;
    }
    if (faceId == null) return;
    url = `/api/sessions/${state.sessionId}/topology/face/${faceId}/tint`;
    body = { tint };
  } else if (key === "main") {
    url = `/api/sessions/${state.sessionId}/main/tint`;
    body = { tint };
  } else if (key.startsWith("region:")) {
    const rid = parseInt(key.slice(7), 10);
    url = `/api/sessions/${state.sessionId}/region/${rid}`;
    body = { tint };
  } else {
    return;  // columns / room don't support tint editing
  }
  const r = await fetch(url, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    setBanner("Tint update failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  const d = await r.json();
  // Topology endpoint returns the full plan; main/region endpoints return
  // just the updated entity. Apply both shapes.
  if (d.plan) {
    state.plan = d.plan;
  } else if (d.main) {
    state.plan.main = d.main;
  } else if (d.region) {
    const reg = state.plan.regions.find(r => r.id === d.region.id);
    if (reg) Object.assign(reg, d.region);
  }
  // Refresh heatmap for the affected face.
  if (key === "main" && state.plan.main) {
    await refreshHeatmap("main", state.plan.main);
  } else if (key.startsWith("region:")) {
    const rid = parseInt(key.slice(7), 10);
    const reg = state.plan.regions.find(r => r.id === rid);
    if (reg) await refreshHeatmap("region:" + rid, reg);
  }
  refreshPolygonsList();
  draw();
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
  } else if (key.startsWith("column:")) {
    const id = parseInt(key.slice(7), 10);
    const obs = (state.plan.obstructions || []).find(o => o.id === id);
    if (!obs) return;
    const r = await fetch(`/api/sessions/${state.sessionId}/obstruction/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: obs.polygon }),
    });
    if (r.ok) {
      const d = await r.json();
      // Server auto re-snaps when topology exists; replace the whole plan
      // so the topology + heatmaps stay coherent.
      if (d.snapped && d.plan) {
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else if (d.obstruction) {
        Object.assign(obs, d.obstruction);
      }
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
    const sv = state.plan?.scan_settings?.max_ceiling_variance_m;
    if (typeof sv === "number" && Number.isFinite(sv)) {
      document.getElementById("input-max-variance").value = sv.toFixed(1);
    }
    if (!state.plan.units) state.plan.units = "metric";
    applyUnitsToggleUI();
    syncProjectPanel(state.plan.project);
    if (state.plan.room) {
      markStepDone("room"); unlockStep("main");
      unlockStep("column");
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
