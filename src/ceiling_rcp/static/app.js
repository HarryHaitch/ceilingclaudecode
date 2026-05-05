console.log("[ceiling-rcp] app.js build 13 — services overlay (symbols)");

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

// Absolute world-Y height — used as the histogram axis label since
// each bin lives in absolute metres, not relative to the datum.
function formatHeightAbs(metres) {
  if (unitsSystem() === "imperial") {
    const inches = metres * 39.3700787;
    const feet = Math.floor(inches / 12);
    const remIn = Math.round(inches - feet * 12);
    return `${feet}'-${remIn}"`;
  }
  return `${metres.toFixed(2)} m`;
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

// Snap radius for "snap to nearest existing room/interface" geometry,
// in canvas pixels. The close-loop check uses 12 px; 14 here is the
// equivalent radius for snapping to a vertex / midpoint / segment.
const SNAP_TO_EXISTING_PX = 14;

function snapToNearestExisting(x, z) {
  // Returns [sx, sz] of the nearest snap target (vertex, midpoint, or
  // any point along a room or interface segment) within
  // SNAP_TO_EXISTING_PX of canvas distance, else null. Caller skips
  // this when Shift is held — Shift is the override that re-routes to
  // the constrainShiftSnap ortho-lock path.
  //
  // Vertices and midpoints get a small bias factor (multiplied
  // distance) so they win ties against generic segment-projection
  // snaps — the user's cursor near a corner should grab the corner,
  // not a slightly-closer point on the wall passing through it.
  if (!state.plan) return null;

  const cur = worldToImg(x, z);
  let bestD = SNAP_TO_EXISTING_PX;
  let bestPt = null;
  const VERTEX_BIAS = 0.85;  // sub-1 = corners win ties

  function consider(tx, tz, bias) {
    const t = worldToImg(tx, tz);
    const d = Math.hypot(t.u - cur.u, t.v - cur.v) * state.view.scale * bias;
    if (d < bestD) { bestD = d; bestPt = [tx, tz]; }
  }
  function considerSegment(a, b) {
    // Project the cursor onto the segment (a → b) in image space,
    // clamp to [0, 1], and use the foot as a snap candidate. Lets
    // the user click ANYWHERE along a chord and have the click
    // land exactly on the chord — without this, a chord that ends
    // partway across the room is invisible to snap except at its
    // two vertices and one midpoint.
    const ai = worldToImg(a[0], a[1]);
    const bi = worldToImg(b[0], b[1]);
    const dx = bi.u - ai.u;
    const dz = bi.v - ai.v;
    const len2 = dx * dx + dz * dz;
    if (len2 < 1e-9) return;
    let t = ((cur.u - ai.u) * dx + (cur.v - ai.v) * dz) / len2;
    t = Math.max(0, Math.min(1, t));
    const fu = ai.u + t * dx;
    const fv = ai.v + t * dz;
    const d = Math.hypot(fu - cur.u, fv - cur.v) * state.view.scale;
    if (d < bestD) {
      bestD = d;
      // Linear interp in world space — image and world both linear,
      // so the projection's `t` is the same in either basis.
      bestPt = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    }
  }

  const room = state.plan.room || [];
  const n = room.length;
  for (let i = 0; i < n; i++) {
    const cur_v = room[i];
    const next_v = room[(i + 1) % n];
    consider(cur_v[0], cur_v[1], VERTEX_BIAS);
    consider((cur_v[0] + next_v[0]) / 2, (cur_v[1] + next_v[1]) / 2, VERTEX_BIAS);
    considerSegment(cur_v, next_v);
  }
  for (const iface of state.plan.interfaces || []) {
    const line = iface.polyline || [];
    const m = line.length;
    if (m < 2) continue;
    for (let i = 0; i < m; i++) {
      consider(line[i][0], line[i][1], VERTEX_BIAS);
    }
    const stop = iface.closed ? m : m - 1;
    for (let i = 0; i < stop; i++) {
      const a = line[i];
      const b = line[(i + 1) % m];
      consider((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, VERTEX_BIAS);
      considerSegment(a, b);
    }
  }
  return bestPt;
}

function applyTraceSnap(world, shift) {
  // Single entry point for the trace tool's cursor snap. Shift held
  // → ortho-lock relative to the previous draft edge (existing
  // behaviour). Otherwise → snap to nearest existing room / interface
  // vertex / midpoint. Returns the snapped {x, z} (and updates
  // state.hover.snapped so drawDraft can highlight the snap target).
  state.hover.snapped = false;
  if (shift) {
    const c = constrainShiftSnap(world.x, world.z);
    if (c) return { x: c[0], z: c[1] };
    return world;
  }
  const c = snapToNearestExisting(world.x, world.z);
  if (c) {
    state.hover.snapped = true;
    return { x: c[0], z: c[1] };
  }
  return world;
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
    const mainId = state.plan?.main_face_id ?? 0;
    let face = null;
    if (sel === "main") face = topo.faces?.find(f => f.id === mainId);
    else if (sel?.startsWith("region:")) {
      const rid = parseInt(sel.slice(7), 10);
      face = topo.faces?.find(f => f.region_id === rid
        || (f.id !== mainId && f.id === rid + 1));
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
  const mainId = state.plan?.main_face_id ?? 0;
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
    if (face.id === mainId && state.plan.main) {
      state.plan.main.polygon = pts;
    } else if (face.id !== mainId) {
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
  hover: { world: null, vertex: null, snapped: false },
  // Map of { kind: "room"|"main"|"region:N" → ImageBitmap } for heatmaps
  heatmaps: new Map(),
  // While the histogram slider is being dragged, holds the
  // pink/black checker overlay returned by /face_below as
  // {key, threshold, bm, bbox}. Cleared on mouseup.
  belowOverlay: null,
  // Ceiling-services symbols overlay. ``doc`` is the loaded symbols.json
  // (canonical schema; see sam3_symbols.py). ``visible`` toggles the
  // whole overlay on/off via the right-sidebar Print button. ``hidden``
  // is a Set of class keys the user has individually unticked in the
  // legend. ``selectedId`` is the symbol whose info card is showing.
  symbols: {
    doc: null,
    visible: false,
    hidden: new Set(),
    selectedId: null,
    status: "absent",
  },
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
document.getElementById("btn-interface").onclick = () => startDraw("interface");
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

document.getElementById("btn-define").onclick = defineCeilings;
document.getElementById("btn-unsnap").onclick = unsnapTopology;
document.getElementById("btn-pdf").onclick = downloadPdf;
document.getElementById("btn-export").onclick = exportPlan;
document.getElementById("btn-apply-min-height").onclick = applyMinCeilingHeight;
document.getElementById("btn-apply-min-height-imp").onclick = applyMinCeilingHeight;
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
  let dragging = false;
  function applyAngleFromEvent(e) {
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
  }
  svg.addEventListener("mousedown", (e) => {
    dragging = true;
    svg.style.cursor = "grabbing";
    applyAngleFromEvent(e);
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    applyAngleFromEvent(e);
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    svg.style.cursor = "grab";
  });
}
document.getElementById("btn-add-register-row").onclick = () => {
  appendRegisterRow({ rev: "", date: "", by: "", note: "" });
  pushDrawingRegister();
};

// Right-sidebar tab toggle (Regions / Services). The Services pane is
// blank until a session loads — symbols are fetched lazily by
// loadSymbolsForSession() once a plan is in.
document.querySelectorAll(".right-tab[data-rtab]").forEach(b => {
  b.onclick = () => setRightTab(b.dataset.rtab);
});
document.getElementById("btn-print-symbols").onclick = () => {
  setSymbolsVisible(!state.symbols.visible);
};
document.getElementById("btn-generate-symbols").onclick = generateSymbols;

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
  document.getElementById("right-panel").hidden = false;
  document.getElementById("btn-export").disabled = false;
  fitView();
  refreshPolygonsList();
  await loadSymbolsForSession();
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
  if (kind === "interface" && !state.plan.room) return;
  if (kind === "column" && !state.plan.room) return;
  state.mode = "draw_" + kind;
  state.draft = [];
  state.selection = null;
  state.draftClosed = false;
  banner.textContent = {
    room: "Drawing ROOM outline — click vertices, click first or press Enter to close. Esc = cancel",
    interface: "Tracing INTERFACE — cursor auto-snaps to nearby vertices. Click first vertex to close (ring) or end on room outline (chord). Shift = ortho lock (overrides snap).",
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
  const kind = state.mode.slice(5);
  // Interfaces commit at 2+ vertices (an open chord); polygons need 3+.
  if (kind === "interface") {
    if (state.draft.length < 2) return;
  } else if (state.draft.length < 3) return;
  const polygon = state.draft.slice();
  const closed = !!state.draftClosed;

  setStepDrawing(kind, false);
  state.draft = [];
  state.draftClosed = false;
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
      unlockStep("interface");
      unlockStep("column");
      document.getElementById("btn-define").disabled = false;
    }
  } else if (kind === "interface") {
    const r = await fetch(`/api/sessions/${state.sessionId}/interface`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polyline: polygon, closed }),
    });
    if (r.ok) {
      const d = await r.json();
      state.plan.interfaces = state.plan.interfaces || [];
      state.plan.interfaces.push(d.interface);
    } else {
      setBanner("Add interface failed: " + (await r.text()).slice(0, 120), true);
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
  } else if (kind === "interface") {
    await fetch(`/api/sessions/${state.sessionId}/interface/${regionId}`,
      { method: "DELETE" });
    state.plan.interfaces = (state.plan.interfaces || [])
      .filter(i => i.id !== regionId);
  } else if (kind === "room") {
    await fetch(`/api/sessions/${state.sessionId}/room`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null }),
    });
    state.plan.room = null;
    state.heatmaps.delete("room");
    document.getElementById("step-room").classList.remove("done");
    document.getElementById("step-room").classList.add("active");
    document.getElementById("step-interface").classList.add("locked");
    document.getElementById("btn-interface").disabled = true;
    document.getElementById("btn-define").disabled = true;
  } else if (kind === "main") {
    // Legacy: delete main face. After cluster D the user re-derives by
    // clicking Define ceilings again.
    await fetch(`/api/sessions/${state.sessionId}/main`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null }),
    });
    state.plan.main = null;
    state.heatmaps.delete("main");
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
  syncMinHeightUnitsUI();
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

function readMinCeilingHeightInputM() {
  // Returns the user-entered floor height in metres, regardless of
  // which units toggle is active. Imperial → ft+in is added back into
  // metres before we hit the server (the server's stored value is
  // always metric).
  if (unitsSystem() === "imperial") {
    const ft = Number.parseFloat(
      document.getElementById("input-min-height-ft").value);
    const inch = Number.parseFloat(
      document.getElementById("input-min-height-in").value);
    if (!Number.isFinite(ft) || !Number.isFinite(inch)) return NaN;
    return (ft * 12 + inch) * 0.0254;
  }
  return Number.parseFloat(
    document.getElementById("input-min-height").value);
}

function writeMinCeilingHeightInputM(metres) {
  if (!Number.isFinite(metres)) return;
  document.getElementById("input-min-height").value = metres.toFixed(1);
  const totalIn = metres * 39.3700787;
  const ft = Math.floor(totalIn / 12);
  const inch = Math.round(totalIn - ft * 12);
  document.getElementById("input-min-height-ft").value = String(ft);
  document.getElementById("input-min-height-in").value = String(inch);
}

function syncMinHeightUnitsUI() {
  // Show the metric or imperial input row based on the current units.
  const imperial = unitsSystem() === "imperial";
  document.getElementById("row-min-height-metric").hidden = imperial;
  document.getElementById("row-min-height-imperial").hidden = !imperial;
}

async function applyMinCeilingHeight() {
  if (!state.sessionId) return;
  const v = readMinCeilingHeightInputM();
  if (!Number.isFinite(v) || v < 0.5 || v > 6.0) {
    setBanner("Enter a value between 0.5 and 6.0 m (1'-7\" to 19'-8\").", true);
    return;
  }
  if (state.plan?.topology) {
    const ok = confirm(
      "Re-rendering will drop the current snap (you'll need to Snap again). " +
      "Continue?"
    );
    if (!ok) return;
  }
  setBanner("Re-rendering with new minimum ceiling height…");
  const r = await fetch(`/api/sessions/${state.sessionId}/scan_settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ min_ceiling_height_m: v }),
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

async function defineCeilings() {
  if (!state.sessionId) return;
  if (!state.plan?.room) { setBanner("Trace the room outline first."); return; }
  setBanner("Polygonising room + interfaces…");
  const r = await fetch(`/api/sessions/${state.sessionId}/define_ceilings`,
    { method: "POST" });
  if (!r.ok) {
    setBanner("Define ceilings failed: " + (await r.text()).slice(0, 200), true);
    return;
  }
  state.plan = await r.json();
  state.heatmaps.clear();
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
  refreshPolygonsList();
  draw();

  // Diagnose the result. With N open chords (each cutting one face into
  // two) and M closed-ring islands, we expect 1 + N + M faces in the
  // common case. A shortfall means a chord didn't node — surface it
  // via banner and a console log so the user can debug specific cases.
  const diag = state.plan.last_define_diagnostic;
  const facesNow = 1 + (state.plan.regions || []).length;
  const expected = diag
    ? 1 + diag.input_chord_count + diag.input_closed_count
    : facesNow;
  if (diag) console.log("define_ceilings diagnostic:", diag);
  if (diag && (facesNow < expected || diag.dangling_segments > 0)) {
    const missing = expected - facesNow;
    setBanner(
      `Ceilings defined — ${facesNow} face(s), but ${missing} chord(s) didn't `
      + `cut (${diag.dangling_segments} dangling segment(s)). See console.`,
      true,
    );
  } else {
    setBanner(`Ceilings defined — ${facesNow} face(s).`);
    setTimeout(() => banner.classList.remove("show"), 2500);
  }
}

async function pushMainFace(selKey) {
  if (!state.sessionId) return;
  const r = await fetch(`/api/sessions/${state.sessionId}/main_face`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: selKey }),
  });
  if (!r.ok) {
    setBanner("Main-face swap failed: " + (await r.text()).slice(0, 200), true);
    return;
  }
  const d = await r.json();
  if (d.noop) return;
  state.plan = d.plan || d;
  state.heatmaps.clear();
  if (state.plan.main) await refreshHeatmap("main", state.plan.main);
  for (const reg of state.plan.regions || [])
    await refreshHeatmap("region:" + reg.id, reg);
  refreshPolygonsList();
  draw();
  setBanner("Datum swapped — heights now relative to the selected face.");
  setTimeout(() => banner.classList.remove("show"), 2200);
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
function ceilingMetaHtml(face, isMain) {
  // Two-line meta for ceiling rows. addRow wraps this in
  // <div class="meta meta-grid">; the grid (right-aligned values)
  // makes "Height: +12 mm" line up under "Spread:  23 mm" without
  // hand-tuned spacing.
  const stats = face?.stats || {};
  let heightVal;
  if (isMain) {
    heightVal = formatHeightDelta(0);
  } else {
    const rel = face?.relative_y;
    heightVal = (rel === null || rel === undefined)
      ? "—" : formatHeightDelta(rel);
  }
  const spreadVal = (stats.std_y !== undefined && stats.std_y !== null)
    ? formatLength(stats.std_y) : "—";
  return `<span class="meta-k">Height:</span><span class="meta-v">${heightVal}</span>`
    + `<span class="meta-k">Spread:</span><span class="meta-v">${spreadVal}</span>`;
}

function refreshPolygonsList() {
  const hasTopology = !!state.plan?.topology;
  document.getElementById("btn-unsnap").hidden = !hasTopology;
  document.getElementById("btn-define").textContent =
    hasTopology ? "Re-define ceilings" : "Define ceilings";

  const ul = document.getElementById("polygons-list");
  ul.innerHTML = "";

  const addRow = (key, label, color, meta, selKey, currentNotes, allowNotes, allowTintEdit, face, mainFlag) => {
    // mainFlag is one of {undefined, "main", "region", "off"}:
    //  "main"   → render a checked Main radio (this row IS the datum)
    //  "region" → render an unchecked Main radio (clicking promotes it)
    //  "off"    → no radio (room outline, columns, interfaces)
    const li = document.createElement("li");
    li.className = "poly-row" + (state.selection?.key === selKey ? " active" : "");
    const head = document.createElement("div");
    head.className = "poly-head";
    const radioHtml = mainFlag === "main"
      ? `<input type="radio" class="main-radio" name="main-face" checked title="Datum face (relative_y = 0)">`
      : mainFlag === "region"
        ? `<input type="radio" class="main-radio" name="main-face" title="Make this the datum face">`
        : "";
    const isCeiling = (mainFlag === "main" || mainFlag === "region");
    const metaCls = "meta" + (isCeiling ? " meta-grid" : "");
    if (allowTintEdit) {
      head.innerHTML =
        radioHtml +
        `<input type="color" class="swatch swatch-input" value="${color}" title="Change tint">` +
        `<div class="label">${label}</div>` +
        `<div class="${metaCls}">${meta}</div>` +
        `<button class="del-btn" title="Delete">×</button>`;
    } else {
      head.innerHTML =
        radioHtml +
        `<div class="swatch" style="background:${color}"></div>` +
        `<div class="label">${label}</div>` +
        `<div class="${metaCls}">${meta}</div>` +
        `<button class="del-btn" title="Delete">×</button>`;
    }
    li.appendChild(head);

    if (mainFlag === "region") {
      const radio = head.querySelector(".main-radio");
      if (radio) {
        radio.onclick = (e) => e.stopPropagation();
        radio.onchange = () => pushMainFace(selKey);
      }
    } else if (mainFlag === "main") {
      const radio = head.querySelector(".main-radio");
      if (radio) radio.onclick = (e) => e.stopPropagation();
    }

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

    // Notes input first — the user types text more often than they
    // tweak the slider, so it sits closer to the row's identifying head.
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

    if (face && face.histogram && face.histogram.counts) {
      const sparkWrap = document.createElement("div");
      sparkWrap.className = "poly-spark";
      sparkWrap.onclick = (e) => e.stopPropagation();
      sparkWrap.appendChild(renderHistogramSparkline(face, selKey));
      li.appendChild(sparkWrap);
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
      `${state.plan.room.length} verts`, "room", null, false, false, null, "off");
  }
  for (const iface of state.plan.interfaces || []) {
    const closedTxt = iface.closed ? "ring" : "chord";
    addRow("interface:" + iface.id, `Interface ${iface.id + 1}`,
      "#00e5ff",
      `${iface.polyline.length} verts • ${closedTxt}`,
      "interface:" + iface.id, null, false, false, null, "off");
  }
  if (state.plan.main) {
    addRow("main", state.plan.main.label || "Main Ceiling (1)",
      state.plan.main.tint || "#80cbc4",
      ceilingMetaHtml(state.plan.main, /*isMain*/ true),
      "main", state.plan.main.notes, true, true, state.plan.main, "main");
  }
  for (const r of state.plan.regions || []) {
    addRow("region:" + r.id, r.label || `Ceiling Region (${r.id + 2})`,
      r.tint || "#ff7043",
      ceilingMetaHtml(r, /*isMain*/ false),
      "region:" + r.id, r.notes, true, true, r, "region");
  }
  for (const o of state.plan.obstructions || []) {
    addRow("column:" + o.id, o.label || `Column (${o.id + 1})`,
      "#ffffff", `${o.polygon.length} verts`,
      "column:" + o.id, null, false, false, null, "off");
  }
  updateSelectionInfo();
}

// ─── HISTOGRAM SPARKLINE ──────────────────────────────────────────────────
// Each region row carries a per-pixel height histogram (bins in absolute
// metres). The user drags a vertical marker to pick the height that
// becomes that face's reported "ceiling height" — useful when the
// polygon spans a non-flat ceiling (vault, services bump) and the mean
// alone is ambiguous. Main's marker defines zero for the whole drawing.

function renderHistogramSparkline(face, selKey) {
  const hist = face.histogram;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(SVG_NS, "svg");
  // 2× height (was 40). Three bands stacked vertically: peak% on top
  // of the bars, the bars themselves, then the axis row (mm offsets
  // from main + below%/slider/above% all on one baseline).
  const W = 264;
  const H = 80;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.classList.add("hist-spark");

  if (!hist || !hist.counts || hist.counts.length === 0) {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", W / 2);
    t.setAttribute("y", H / 2);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("dominant-baseline", "middle");
    t.setAttribute("font-size", "10");
    t.setAttribute("fill", "#8b939c");
    t.textContent = "no histogram — re-snap to compute";
    svg.appendChild(t);
    return svg;
  }

  const counts = hist.counts;
  const edges = hist.bin_edges_m;
  const minY = hist.min_y;
  const maxY = hist.max_y;
  const range = Math.max(1e-6, maxY - minY);
  const sy = (face.selected_y ?? face.stats?.mean_y);
  const meanY = face.stats?.mean_y;
  const tint = face.tint || "#80cbc4";
  const isMain = (selKey === "main");
  // Datum = main's selected_y. For the main row we ARE the datum, so
  // labels just read 0 / +/- relative to the main itself's selected_y.
  const datum = state.plan?.main?.selected_y
             ?? state.plan?.main?.stats?.mean_y
             ?? meanY ?? sy;

  const padX = 6;
  const padTop = 14;          // peak% header row
  const padBot = 30;          // axis labels (top sub-row) + readouts (bot)
  const usableW = W - 2 * padX;
  const usableH = H - padTop - padBot;
  const totalCount = counts.reduce((a, b) => a + b, 0) || 1;
  const maxCount = Math.max(1, ...counts);
  const peakPct = (maxCount / totalCount) * 100;

  const xForY = (y) => padX + ((y - minY) / range) * usableW;

  // Bars
  const barW = usableW / counts.length;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i];
    if (!c) continue;
    const h = (c / maxCount) * usableH;
    const x = padX + i * barW;
    const y = padTop + (usableH - h);
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", Math.max(1, barW));
    rect.setAttribute("height", h);
    rect.setAttribute("fill", tint);
    rect.setAttribute("opacity", "0.8");
    svg.appendChild(rect);
  }

  function makeText(x, y, anchor, fill, weight, size, content) {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", x);
    t.setAttribute("y", y);
    t.setAttribute("text-anchor", anchor);
    t.setAttribute("font-size", String(size));
    t.setAttribute("fill", fill);
    if (weight) t.setAttribute("font-weight", weight);
    t.textContent = content;
    return t;
  }

  // Header: peak frequency as a number — answers "how flat is this
  // ceiling?" at a glance.
  svg.appendChild(makeText(W - padX, padTop - 4, "end", "#8b939c", "600", 9,
    `peak ${peakPct.toFixed(1)}%`));
  svg.appendChild(makeText(padX, padTop - 4, "start", "#8b939c", null, 9,
    `${counts.length} bins • ${(hist.bin_w_m * 1000).toFixed(0)} mm`));

  // Datum (main's selected_y) line — 0 mm on the relative axis. Only
  // draw it when it falls inside this face's histogram range.
  if (Number.isFinite(datum) && datum >= minY && datum <= maxY) {
    const dx = xForY(datum);
    const datumLine = document.createElementNS(SVG_NS, "line");
    datumLine.setAttribute("x1", dx);
    datumLine.setAttribute("x2", dx);
    datumLine.setAttribute("y1", padTop);
    datumLine.setAttribute("y2", padTop + usableH);
    datumLine.setAttribute("stroke", "#888");
    datumLine.setAttribute("stroke-width", "1");
    datumLine.setAttribute("stroke-dasharray", "3,2");
    svg.appendChild(datumLine);
  }

  // Mean reference (kept faint so it doesn't compete with the datum
  // line when they're far apart, e.g. a vault region).
  if (Number.isFinite(meanY) && Math.abs(meanY - datum) > 1e-4) {
    const mx = xForY(meanY);
    const meanLine = document.createElementNS(SVG_NS, "line");
    meanLine.setAttribute("x1", mx);
    meanLine.setAttribute("x2", mx);
    meanLine.setAttribute("y1", padTop);
    meanLine.setAttribute("y2", padTop + usableH);
    meanLine.setAttribute("stroke", "#bbb");
    meanLine.setAttribute("stroke-width", "0.6");
    meanLine.setAttribute("stroke-dasharray", "1,2");
    svg.appendChild(meanLine);
  }

  // Selected marker (red, draggable)
  const handleSx = xForY(sy);
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", handleSx);
  line.setAttribute("x2", handleSx);
  line.setAttribute("y1", padTop - 2);
  line.setAttribute("y2", padTop + usableH + 2);
  line.setAttribute("stroke", "#ef5350");
  line.setAttribute("stroke-width", "1.8");
  svg.appendChild(line);
  const blob = document.createElementNS(SVG_NS, "circle");
  blob.setAttribute("cx", handleSx);
  blob.setAttribute("cy", padTop + 2);
  blob.setAttribute("r", "4");
  blob.setAttribute("fill", "#ef5350");
  svg.appendChild(blob);

  // Axis row: min / 0 / max in mm relative to the datum.
  const yAxis = padTop + usableH + 11;
  const minRel = minY - datum;
  const maxRel = maxY - datum;
  svg.appendChild(makeText(padX, yAxis, "start", "#8b939c", null, 9,
    formatHeightDelta(minRel)));
  svg.appendChild(makeText(W - padX, yAxis, "end", "#8b939c", null, 9,
    formatHeightDelta(maxRel)));
  if (datum >= minY && datum <= maxY) {
    const zeroX = xForY(datum);
    svg.appendChild(makeText(zeroX, yAxis, "middle", "#666", "600", 9, "0"));
  }

  // Below% / slider readout (relative mm) / above% — all on the same
  // bottom baseline. Recomputed on every drag so the user sees the
  // split shift live as they move the marker.
  function partitionByY(threshold) {
    let below = 0;
    for (let i = 0; i < counts.length; i++) {
      const binCenter = (edges[i] + edges[i + 1]) / 2;
      if (binCenter < threshold) below += counts[i];
    }
    return { below, above: totalCount - below };
  }
  const yReadout = H - 4;
  const initial = partitionByY(sy);
  const belowLabel = makeText(padX, yReadout, "start", "#8b939c", null, 10,
    `↓ ${(initial.below / totalCount * 100).toFixed(0)}%`);
  svg.appendChild(belowLabel);
  const aboveLabel = makeText(W - padX, yReadout, "end", "#8b939c", null, 10,
    `${(initial.above / totalCount * 100).toFixed(0)}% ↑`);
  svg.appendChild(aboveLabel);
  const selLabel = makeText(handleSx, yReadout, "middle", "#ef5350", "700", 11,
    isMain ? "0" : formatHeightDelta(sy - datum));
  svg.appendChild(selLabel);

  // Drag interaction — clamp to [minY, maxY] and snap to bin width so
  // the on-screen marker only ever falls on a bin centre.
  svg.style.cursor = "ew-resize";
  svg.style.userSelect = "none";
  let dragging = false;
  let lastY = sy;
  // Throttle the below-overlay fetches: at most one in flight at a
  // time, queue the latest threshold and fire it when the previous
  // returns. Avoids a request storm on fast drags.
  let overlayInflight = false;
  let overlayQueued = null;
  function requestOverlay(threshold) {
    if (overlayInflight) { overlayQueued = threshold; return; }
    overlayInflight = true;
    const url = `/api/sessions/${state.sessionId}/face_below`
      + `?key=${encodeURIComponent(selKey)}&y=${threshold}`;
    fetch(url).then(async (r) => {
      if (!r.ok) return;
      const bboxHdr = r.headers.get("X-Bbox") || "0,0,0,0";
      const bbox = bboxHdr.split(",").map(Number);
      const blob = await r.blob();
      const bm = await createImageBitmap(blob);
      // The user may have stopped dragging by the time this returns —
      // only commit the overlay if the slider is still active for the
      // same face.
      if (dragging) {
        state.belowOverlay = { key: selKey, threshold, bm, bbox };
        draw();
      }
    }).catch(() => {}).finally(() => {
      overlayInflight = false;
      if (overlayQueued != null && dragging) {
        const next = overlayQueued;
        overlayQueued = null;
        requestOverlay(next);
      }
    });
  }
  function applyFromEvent(e) {
    const rect = svg.getBoundingClientRect();
    const xPx = e.clientX - rect.left;
    const xUnits = (xPx * W) / rect.width;  // SVG viewBox-aware
    const norm = Math.max(0, Math.min(1, (xUnits - padX) / usableW));
    const newY = minY + norm * range;
    const px = xForY(newY);
    line.setAttribute("x1", px);
    line.setAttribute("x2", px);
    blob.setAttribute("cx", px);
    selLabel.setAttribute("x", px);
    selLabel.textContent = isMain ? "0" : formatHeightDelta(newY - datum);
    const part = partitionByY(newY);
    belowLabel.textContent = `↓ ${(part.below / totalCount * 100).toFixed(0)}%`;
    aboveLabel.textContent = `${(part.above / totalCount * 100).toFixed(0)}% ↑`;
    lastY = newY;
    requestOverlay(newY);
  }
  svg.addEventListener("mousedown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    dragging = true;
    applyFromEvent(e);
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    applyFromEvent(e);
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    overlayQueued = null;
    state.belowOverlay = null;
    draw();
    pushSelectedY(selKey, lastY);
  });

  return svg;
}

async function pushSelectedY(selKey, newY) {
  if (!state.sessionId) return;
  let url;
  if (selKey === "main") {
    url = `/api/sessions/${state.sessionId}/main/selected_y`;
  } else if (selKey.startsWith("region:")) {
    const id = parseInt(selKey.slice(7), 10);
    url = `/api/sessions/${state.sessionId}/region/${id}/selected_y`;
  } else {
    return;
  }
  const r = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value: newY }),
  });
  if (!r.ok) {
    setBanner("Height update failed: " + (await r.text()).slice(0, 120), true);
    return;
  }
  // Server's recompute touched relative_y on every face — re-read the
  // whole plan so the UI's labels stay in sync (especially after a
  // main.selected_y drag, which changes every region's delta).
  const planResp = await fetch(`/api/sessions/${state.sessionId}/plan`);
  if (!planResp.ok) return;
  state.plan = await planResp.json();
  refreshPolygonsList();
  if (state.selection) updateSelectionInfo();
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
    ? `<div class="warn-line">⚠ high spread (${fmt_std}) — likely clipped a bulkhead</div>`
    : "";
  const validWarn = s.valid_frac < 0.6
    ? `<div class="warn-line">⚠ only ${valid_pct}% of polygon has LiDAR coverage</div>`
    : "";
  return `
    <b>${title}</b><br>
    relative height: ${fmt_rel}<br>
    spread inside polygon: ${fmt_std}<br>
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

  // Below-threshold overlay — drawn while the user drags the histogram
  // slider. Pixels in the active face whose Y < the slider value render
  // as a pink/black checker so the user can see what would be cropped
  // out if they commit the marker's current position. Cleared on
  // slider mouseup.
  if (state.belowOverlay && state.belowOverlay.bm) {
    const ov = state.belowOverlay;
    ctx.drawImage(ov.bm, ov.bbox[0], ov.bbox[1]);
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
  // Interface polylines / rings (cyan, drawn above region fills so the
  // user can see what they've traced even after Define ceilings).
  for (const iface of state.plan.interfaces || []) {
    drawInterface(iface);
  }
  // Ceiling-services symbols overlay (Diffusers, Downlights, …). Drawn
  // above interfaces so the symbols sit on top of the linework, but
  // below the live drafting feedback (draft polygon / snap indicator)
  // so the user's in-progress edit always wins visually.
  if (state.symbols.visible && state.symbols.doc) {
    drawSymbolsOverlay();
  }
  // Hover preview for insert-vertex
  if (state.tool === "insert-vertex" && state.selection && state.hover.world) {
    drawInsertPreview();
  }

  // Draft polygon being drawn
  if (state.mode.startsWith("draw_") && state.draft.length > 0) {
    drawDraft();
  }

  // Snap indicator — green ring on the locked-onto target. drawDraft
  // does this for the second-and-later vertices, but the user also
  // needs feedback while placing the FIRST vertex (when state.draft is
  // still empty and drawDraft is skipped). Otherwise the first click
  // looks freehand even when the cursor is locked onto a wall.
  if (state.mode.startsWith("draw_") && state.draft.length === 0
      && state.hover.snapped && state.hover.world) {
    const h = worldToImg(state.hover.world.x, state.hover.world.z);
    ctx.beginPath();
    ctx.arc(h.u, h.v, 8 / state.view.scale, 0, Math.PI * 2);
    ctx.strokeStyle = "#7cff79";
    ctx.lineWidth = 2 / state.view.scale;
    ctx.stroke();
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

function drawInterface(iface) {
  const pts = iface.polyline || [];
  if (pts.length < 2) return;
  const selKey = "interface:" + iface.id;
  const selected = state.selection?.key === selKey;
  // Once a topology is in place, the chord is a face boundary that
  // already gets stroked by the topology renderer. Drawing it again on
  // top would just be noise. Fade unselected interfaces to a hint so
  // the user knows the chord is still an editable object (delete from
  // the Regions panel, drag a vertex to nudge then auto-redefine), but
  // it's not visually competing with the actual face outlines.
  const faded = !!state.plan?.topology && !selected;
  ctx.save();
  if (faded) ctx.globalAlpha = 0.7;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const p = worldToImg(pts[i][0], pts[i][1]);
    if (i === 0) ctx.moveTo(p.u, p.v); else ctx.lineTo(p.u, p.v);
  }
  if (iface.closed) {
    const p0 = worldToImg(pts[0][0], pts[0][1]);
    ctx.lineTo(p0.u, p0.v);
  }
  ctx.strokeStyle = selected ? "#ffffff" : "#00e5ff";
  ctx.lineWidth = (selected ? 2.4 : 1.8) / state.view.scale;
  ctx.setLineDash(faded ? [6 / state.view.scale, 4 / state.view.scale] : []);
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertex dots so the user can grab them post-trace. Hide them when
  // faded — the row in the Regions panel still surfaces the chord for
  // delete, and clicking it re-selects to bring vertices back.
  if (!faded) {
    const r = 4 / state.view.scale;
    for (let i = 0; i < pts.length; i++) {
      const v = worldToImg(pts[i][0], pts[i][1]);
      ctx.beginPath();
      ctx.arc(v.u, v.v, r, 0, Math.PI * 2);
      ctx.fillStyle = "#00e5ff";
      ctx.fill();
      ctx.strokeStyle = "#003a4a";
      ctx.lineWidth = 1 / state.view.scale;
      ctx.stroke();
    }
  }
  ctx.restore();
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

  // Snap indicator: green ring around the cursor when it's locked onto
  // an existing room / interface vertex or midpoint. Tells the user
  // "the click will land exactly here, not where the cursor is."
  if (state.hover.snapped && state.hover.world) {
    const h = worldToImg(state.hover.world.x, state.hover.world.z);
    ctx.beginPath();
    ctx.arc(h.u, h.v, 8 / state.view.scale, 0, Math.PI * 2);
    ctx.strokeStyle = "#7cff79";
    ctx.lineWidth = 2 / state.view.scale;
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
    w = applyTraceSnap(w, e.shiftKey);
    const isIface = state.mode === "draw_interface";
    // Click on first vertex to close (a ring, for interface; a polygon
    // for room/column). Min vertices: 3 for both.
    if (state.draft.length >= 3) {
      const first = worldToImg(state.draft[0][0], state.draft[0][1]);
      const cur = worldToImg(w.x, w.z);
      const dpx = Math.hypot(first.u - cur.u, first.v - cur.v) * state.view.scale;
      if (dpx < 12) {
        if (isIface) state.draftClosed = true;
        commitDraft();
        return;
      }
    }
    state.draft.push([w.x, w.z]);
    draw();
    return;
  }

  // Symbol overlay click — when the symbols layer is visible and the
  // click landed on a symbol, show its info card and don't fall through
  // to the polygon hit-test. Click on empty canvas (anywhere not on a
  // symbol) clears the symbol selection.
  if (state.symbols.visible && state.tool === "select") {
    const sid = hitTestSymbol(m.x, m.y);
    if (sid != null) {
      selectSymbol(sid);
      return;
    }
    // Fall through — let the polygon hit-test run so the user can still
    // pick regions / vertices through gaps in the symbols.
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
      // Post-snap, a face vertex is shared with every other face at the
      // same junction. Resolve to a topology vertex so the drag moves
      // them all in lockstep. Interfaces are NOT part of the topology
      // — they live on top of it as the linework that define_ceilings
      // consumes — so dragging an interface vertex must stay scoped to
      // that interface even if it coincides with a topology vertex.
      const isInterface = hit.key.startsWith("interface:");
      const poly = polygonForKey(hit.key);
      if (poly && !isInterface) {
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

  // While drawing, snap the hover preview either to a nearby existing
  // room / interface vertex or midpoint (default), or — when Shift is
  // held — to a 0/90° lock relative to the previous draft edge. Commit
  // (onMouseDown) reads from hover.world via the same applyTraceSnap
  // path, so the click lands exactly on the preview marker.
  if (state.mode.startsWith("draw_")) {
    state.hover.world = applyTraceSnap(state.hover.world, e.shiftKey);
  } else {
    state.hover.snapped = false;
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
  // Interfaces are NOT part of the topology graph — they're the
  // tracing-time linework that `define_ceilings` consumes — so check
  // them before the topology branch so a chord vertex on top of a
  // topology junction can still be edited as the chord's vertex.
  for (const iface of state.plan?.interfaces || []) {
    const line = iface.polyline || [];
    for (let i = 0; i < line.length; i++) {
      const ic = imgToCanvas(...Object.values(worldToImg(line[i][0], line[i][1])));
      const d = Math.hypot(ic.cx - cx, ic.cy - cy);
      if (d < thresholdPx) {
        return { key: "interface:" + iface.id, vertexIndex: i };
      }
    }
  }
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
    // Interfaces commit at 2+ vertices (open chord); polygons need 3+.
    const isIface = state.mode === "draw_interface";
    const min = isIface ? 2 : 3;
    if (state.draft.length >= min) commitDraft();
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
    // if the mouse hasn't moved since. Shift held → ortho lock; not
    // held → fall back to snap-to-existing.
    state.hover.world = applyTraceSnap(state.hover.world, e.type === "keydown");
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
  // Pre-snap (or column / interface) — splice the vertex out and push.
  const poly = polygonForKey(h.key);
  if (!poly) return;
  // Min vertex count: 2 for an open interface chord, 3 for everything
  // else (closed polygons + closed interface rings).
  const isOpenChord = h.key.startsWith("interface:")
    && !((state.plan.interfaces || [])
      .find(i => "interface:" + i.id === h.key)?.closed);
  const minLen = isOpenChord ? 2 : 3;
  if (poly.length <= minLen) {
    setBanner(`Need at least ${minLen} vertices.`, true);
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
  if (key.startsWith("interface:")) {
    const id = parseInt(key.slice(10), 10);
    return (state.plan.interfaces || []).find(i => i.id === id)?.polyline;
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
  // Interfaces sit on top of the topology faces and stay editable after
  // define_ceilings — moving a chord vertex re-runs polygonisation
  // server-side. Push them last so a coincident region/main vertex still
  // wins when the user is editing the underlying ceiling outlines.
  for (const i of state.plan.interfaces || [])
    all.push({ key: "interface:" + i.id, poly: i.polyline });

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
    const mainId = state.plan?.main_face_id ?? 0;
    if (key === "main") {
      faceId = mainId;
    } else if (key.startsWith("region:")) {
      const rid = parseInt(key.slice(7), 10);
      const f = (topo.faces || []).find(
        f => f.region_id === rid || (f.id !== mainId && f.id === rid + 1));
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
  } else if (key.startsWith("interface:")) {
    const id = parseInt(key.slice(10), 10);
    const iface = (state.plan.interfaces || []).find(i => i.id === id);
    if (!iface) return;
    const r = await fetch(`/api/sessions/${state.sessionId}/interface/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polyline: iface.polyline }),
    });
    if (r.ok) {
      const d = await r.json();
      // If a topology already existed, the server re-ran define_ceilings
      // and returned a fresh plan — adopt it wholesale so the topology +
      // heatmaps reflect the moved interface.
      if (d.redefined && d.plan) {
        state.plan = d.plan;
        await refreshAllHeatmaps();
      } else if (d.interface) {
        Object.assign(iface, d.interface);
      }
    } else {
      setBanner("Interface update failed: " + (await r.text()).slice(0, 120), true);
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

// ─── CEILING-SERVICES SYMBOLS ─────────────────────────────────────────────
// The Services tab loads an optional symbols.json from the server (or a
// preloaded example for the canned demo session) and renders the symbols
// on top of the ortho. Click toggles "print" mode; per-class checkboxes
// in the legend hide individual classes; clicking a symbol shows its
// label index, length, width, and centroid in the info card.

function setRightTab(name) {
  document.querySelectorAll(".right-tab[data-rtab]").forEach(b => {
    b.classList.toggle("active", b.dataset.rtab === name);
  });
  document.querySelectorAll(".right-tab-pane").forEach(p => {
    p.hidden = p.dataset.rpane !== name;
  });
}

function setServicesStatus(text, cls = "") {
  const el = document.getElementById("services-status");
  if (!el) return;
  el.textContent = text;
  el.className = "muted";
  if (cls) el.classList.add(cls);
}

function symbolClasses() {
  return state.symbols.doc?.symbol_classes || {};
}
function symbolList() {
  return state.symbols.doc?.symbols || [];
}
function pxPerCmFromSymbols() {
  // Fall back to the plan grid if the symbols doc is missing one
  // (shouldn't happen — empty_doc still emits a grid).
  const ppm = state.symbols.doc?.grid?.pixels_per_metre
    ?? state.plan?.grid?.pixels_per_metre
    ?? 100;
  return ppm / 100.0;
}

async function loadSymbolsForSession() {
  if (!state.sessionId) return;
  try {
    const r = await fetch(`/api/sessions/${state.sessionId}/symbols`);
    if (!r.ok) {
      console.warn("symbols fetch failed:", r.status);
      return;
    }
    const doc = await r.json();
    state.symbols.doc = doc;
    state.symbols.status = doc.status || (doc.symbols?.length ? "ok" : "absent");
    // Reset transient state — preserve nothing from a prior session.
    state.symbols.hidden.clear();
    state.symbols.selectedId = null;
  } catch (e) {
    console.warn("symbols fetch threw:", e);
    state.symbols.doc = null;
    state.symbols.status = "absent";
  }
  refreshSymbolsUI();
}

function refreshSymbolsUI() {
  const doc = state.symbols.doc;
  const printBtn = document.getElementById("btn-print-symbols");
  const symbols = symbolList();
  const haveAny = symbols.length > 0;
  printBtn.disabled = !haveAny;
  printBtn.setAttribute("aria-pressed", state.symbols.visible ? "true" : "false");
  printBtn.textContent = state.symbols.visible
    ? "Hide symbols on plan"
    : "Print symbols on plan";

  if (state.symbols.status === "absent" || !haveAny) {
    setServicesStatus(
      "No symbols generated for this session yet. Use Detect symbols (SAM 3) "
      + "to run the segmentation pipeline.", ""
    );
  } else if (state.symbols.status === "stale_grid") {
    setServicesStatus(
      "Symbols exist but were generated against a different ortho grid — "
      + "regenerate after the latest scan re-render.", "warn"
    );
  } else {
    setServicesStatus(`${symbols.length} symbols loaded.`, "ok");
  }
  refreshSymbolsLegend();
  refreshSymbolInfo();
}

function refreshSymbolsLegend() {
  const ul = document.getElementById("symbols-legend");
  if (!ul) return;
  ul.innerHTML = "";
  ul.classList.remove("muted");
  const classes = symbolClasses();
  const symbols = symbolList();
  if (Object.keys(classes).length === 0 || symbols.length === 0) {
    ul.classList.add("muted");
    ul.innerHTML = '<li>No symbols loaded yet.</li>';
    return;
  }
  // Counts include only classes actually present, but legend rows always
  // include every defined class so the user can re-enable an empty class
  // after generating a fresh batch.
  const counts = {};
  for (const s of symbols) counts[s.class] = (counts[s.class] || 0) + 1;
  for (const [key, def] of Object.entries(classes)) {
    const n = counts[key] || 0;
    const li = document.createElement("li");
    li.className = "legend-row" + (state.symbols.hidden.has(key) ? " hidden-cls" : "");
    li.innerHTML =
      `<input type="checkbox" ${state.symbols.hidden.has(key) ? "" : "checked"}>` +
      `<span class="legend-swatch" style="background:${def.color_hex}"></span>` +
      `<span class="legend-label">${def.label}</span>` +
      `<span class="legend-count">${n}</span>`;
    const cb = li.querySelector("input");
    cb.onclick = (e) => e.stopPropagation();
    cb.onchange = () => {
      if (cb.checked) state.symbols.hidden.delete(key);
      else state.symbols.hidden.add(key);
      // Hide the info card if the user just hid the class containing
      // the selected symbol.
      const sel = findSymbolById(state.symbols.selectedId);
      if (sel && state.symbols.hidden.has(sel.class)) {
        state.symbols.selectedId = null;
        refreshSymbolInfo();
      }
      li.classList.toggle("hidden-cls", state.symbols.hidden.has(key));
      draw();
    };
    ul.appendChild(li);
  }
}

function findSymbolById(id) {
  if (id == null) return null;
  return symbolList().find(s => s.id === id) || null;
}

function setSymbolsVisible(on) {
  state.symbols.visible = !!on;
  if (!on) state.symbols.selectedId = null;
  // If the user toggles symbols on, switch the right-tab to Services so
  // the legend / info card is in view.
  if (on) setRightTab("services");
  refreshSymbolsUI();
  draw();
}

function selectSymbol(id) {
  state.symbols.selectedId = id;
  setRightTab("services");
  refreshSymbolInfo();
  draw();
}

function refreshSymbolInfo() {
  const el = document.getElementById("symbol-info");
  if (!el) return;
  const s = findSymbolById(state.symbols.selectedId);
  if (!s) {
    el.classList.remove("has-selection");
    el.classList.add("muted");
    el.textContent = "Click a symbol on the plan to see its details.";
    return;
  }
  const def = (symbolClasses()[s.class]) || {};
  const cm = pxPerCmFromSymbols();
  // For circle-shaped symbols (Downlight, Sprinkler) the model carries
  // a fixed diameter rather than length × width; show the diameter row
  // so the info card stays accurate. Everything else falls back to the
  // measured length/width in cm.
  const isCircle = def.shape === "circle" || def.shape === "dot_in_circle";
  const fixedD = def.fixed_diameter_cm;
  const lenCm = s.length_px / cm;
  const widCm = s.width_px / cm;
  const cx = s.centroid_px[0];
  const cy = s.centroid_px[1];
  // Centroid in metres too — handier for users tagging fixtures
  // against the room outline (which they think of in metres).
  const grid = state.symbols.doc?.grid || state.plan?.grid;
  let mx = null, mz = null;
  if (grid) {
    mx = grid.max_x - cx / grid.pixels_per_metre;
    mz = grid.max_z - cy / grid.pixels_per_metre;
  }
  const sourceConcept = s.source?.concept;
  const nViews = s.source?.n_views;
  const score = s.source?.median_score;
  const sizeRow = isCircle && fixedD
    ? `<div class="sym-row"><span class="k">Diameter</span><span class="v">${fixedD.toFixed(1)} cm</span></div>`
    : `<div class="sym-row"><span class="k">Length</span><span class="v">${lenCm.toFixed(1)} cm</span></div>`
      + `<div class="sym-row"><span class="k">Width</span><span class="v">${widCm.toFixed(1)} cm</span></div>`;
  const centroidWorld = (mx != null)
    ? `<div class="sym-row"><span class="k">Centroid (m)</span>`
      + `<span class="v">${mx.toFixed(2)}, ${mz.toFixed(2)}</span></div>`
    : "";
  const provenance = (sourceConcept || nViews != null)
    ? `<div class="sym-row"><span class="k">Source</span>`
      + `<span class="v">${sourceConcept || "—"}`
      + (nViews != null ? `, ${nViews} views` : "")
      + (typeof score === "number" ? ` · score ${score.toFixed(2)}` : "")
      + `</span></div>`
    : "";
  el.classList.remove("muted");
  el.classList.add("has-selection");
  el.innerHTML =
    `<div class="sym-head">`
      + `<span class="legend-swatch" style="background:${def.color_hex || '#888'}"></span>`
      + `<span>${def.label || s.class} ${s.index}</span>`
    + `</div>`
    + `<div class="sym-row"><span class="k">ID</span><span class="v">${s.id}</span></div>`
    + `<div class="sym-row"><span class="k">Index</span><span class="v">${s.index}</span></div>`
    + sizeRow
    + `<div class="sym-row"><span class="k">Angle</span><span class="v">${(s.angle_deg ?? 0).toFixed(1)}°</span></div>`
    + `<div class="sym-row"><span class="k">Centroid (px)</span>`
      + `<span class="v">${cx.toFixed(0)}, ${cy.toFixed(0)}</span></div>`
    + centroidWorld
    + provenance;
}

// ─── Symbol rendering ────────────────────────────────────────────────────
//
// Drafting convention: every symbol body is **black-line + white-fill**.
// The white fill cuts a hole through the ceiling masks underneath so the
// fixture reads even on a busy plan, and a uniform black stroke means
// the plan looks like a published RCP rather than a colour-coded debug
// view. Class identity is carried by the *label* colour and the legend
// swatch — the geometric markers themselves don't need to compete.

const SYMBOL_STROKE = "#111111";
const SYMBOL_FILL   = "#ffffff";
const SYMBOL_LINE_PX = 1.4;   // canvas-px stroke width (un-scaled)

function _drawCircle(cu, cv, rPx, { fill = SYMBOL_FILL, stroke = SYMBOL_STROKE,
                                     lineWidth = SYMBOL_LINE_PX } = {}) {
  ctx.beginPath();
  ctx.arc(cu, cv, rPx, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth / state.view.scale;
    ctx.stroke();
  }
}

function _drawFilledDot(cu, cv, rPx, color = SYMBOL_STROKE) {
  ctx.beginPath();
  ctx.arc(cu, cv, rPx, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
}

function _drawRotatedRect(cu, cv, lengthPx, widthPx, angleDeg,
                          { fill = SYMBOL_FILL, stroke = SYMBOL_STROKE,
                            lineWidth = SYMBOL_LINE_PX } = {}) {
  const rad = angleDeg * Math.PI / 180;
  ctx.save();
  ctx.translate(cu, cv);
  ctx.rotate(rad);
  ctx.beginPath();
  ctx.rect(-lengthPx / 2, -widthPx / 2, lengthPx, widthPx);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth / state.view.scale;
    ctx.stroke();
  }
  ctx.restore();
}

function _drawCrossInRect(cu, cv, lengthPx, widthPx, angleDeg,
                          { stroke = SYMBOL_STROKE,
                            lineWidth = SYMBOL_LINE_PX } = {}) {
  const rad = angleDeg * Math.PI / 180;
  ctx.save();
  ctx.translate(cu, cv);
  ctx.rotate(rad);
  ctx.beginPath();
  ctx.moveTo(-lengthPx / 2, -widthPx / 2);
  ctx.lineTo(lengthPx / 2, widthPx / 2);
  ctx.moveTo(lengthPx / 2, -widthPx / 2);
  ctx.lineTo(-lengthPx / 2, widthPx / 2);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth / state.view.scale;
  ctx.stroke();
  ctx.restore();
}

function _symbolPxSize(sym, def, pxPerCm) {
  // Returns the [length_px, width_px] to draw this symbol at, falling
  // back to the class's default_size_cm when the per-symbol fields are
  // missing/zero. Mirrors sam3_symbols.render_symbols Python helper.
  const minPx = 8;
  const lengthPx = Math.max(minPx,
    sym.length_px || (def.default_size_cm?.[0] || 30) * pxPerCm);
  const widthPx  = Math.max(minPx,
    sym.width_px  || (def.default_size_cm?.[1] || 30) * pxPerCm);
  return [lengthPx, widthPx];
}

function drawSymbolsOverlay() {
  const doc = state.symbols.doc;
  if (!doc || !doc.symbols) return;
  const classes = doc.symbol_classes || {};
  const pxPerCm = pxPerCmFromSymbols();
  const labelPx = 12;       // CSS px for label text
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const s of doc.symbols) {
    if (state.symbols.hidden.has(s.class)) continue;
    const def = classes[s.class];
    if (!def) continue;
    // Class colour is reserved for the *label* and the legend swatch;
    // the symbol body itself follows the black-line / white-fill
    // drafting convention so it reads through the ceiling masks.
    const labelColor = def.color_hex || "#fff";
    const cu = s.centroid_px[0], cv = s.centroid_px[1];

    if (def.shape === "circle") {
      const dCm = def.fixed_diameter_cm || 20;
      _drawCircle(cu, cv, dCm / 2 * pxPerCm);
    } else if (def.shape === "dot_in_circle") {
      const dOuter = def.fixed_diameter_cm || 12;
      const dInner = def.fixed_inner_dot_diameter_cm || 5;
      _drawCircle(cu, cv, dOuter / 2 * pxPerCm);
      _drawFilledDot(cu, cv, dInner / 2 * pxPerCm);
    } else {
      // rect / rect_with_cross / rect_with_M
      const [lengthPx, widthPx] = _symbolPxSize(s, def, pxPerCm);
      _drawRotatedRect(cu, cv, lengthPx, widthPx, s.angle_deg || 0);
      if (def.shape === "rect_with_cross") {
        _drawCrossInRect(cu, cv, lengthPx, widthPx, s.angle_deg || 0);
      } else if (def.shape === "rect_with_M") {
        // Font size scales with the rect (image-pixel space), capped at
        // 50 % of the shorter side so it sits inside the box. Earlier
        // versions divided by state.view.scale, which decoupled the
        // font size from the rect and made the M dwarf the body when
        // zoomed out.
        const fontPx = Math.min(lengthPx, widthPx) * 0.5;
        ctx.save();
        ctx.fillStyle = SYMBOL_STROKE;
        ctx.font = `bold ${fontPx}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("M", cu, cv);
        ctx.restore();
      }
    }

    // Selection ring on top so it's visible no matter the shape.
    if (state.symbols.selectedId === s.id) {
      ctx.save();
      const r = 18 / state.view.scale + Math.max(20, ((s.length_px || 30) + (s.width_px || 30)) * 0.25);
      ctx.beginPath();
      ctx.arc(cu, cv, r, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffaa00";
      ctx.lineWidth = 2.5 / state.view.scale;
      ctx.setLineDash([6 / state.view.scale, 4 / state.view.scale]);
      ctx.stroke();
      ctx.restore();
    }

    // Label — class-coloured text with a white halo so it reads on the
    // dark ortho and the coloured ceiling tints. Skip when zoomed far
    // out so the labels don't cake into a wall of text.
    if (state.view.scale > 0.25) {
      const lab = `${def.label || s.class} ${s.index}`;
      ctx.save();
      ctx.font = `${labelPx / state.view.scale}px -apple-system, system-ui, sans-serif`;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.lineWidth = 3 / state.view.scale;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.fillStyle = labelColor;
      const labU = cu + 8;
      const labV = cv - 8;
      ctx.strokeText(lab, labU, labV);
      ctx.fillText(lab, labU, labV);
      ctx.restore();
    }
  }
  ctx.restore();
}

function hitTestSymbol(cx, cy) {
  // Hit test in canvas-pixel space — radius scales inversely with view
  // zoom so symbols stay grabbable when zoomed out. Larger fixtures
  // (LED panels, diffusers) take their bounding-circle radius; small
  // ones use a min-grab radius so a single-pixel cursor still lands.
  if (!state.symbols.doc) return null;
  const classes = symbolClasses();
  const pxPerCm = pxPerCmFromSymbols();
  let best = null;
  let bestD = Infinity;
  const minGrabPx = 12;  // canvas-pixel minimum
  for (const s of symbolList()) {
    if (state.symbols.hidden.has(s.class)) continue;
    const def = classes[s.class];
    if (!def) continue;
    const cu = s.centroid_px[0], cv = s.centroid_px[1];
    // Pick a radius in image-pixel space, then convert to canvas px.
    let rImg;
    if (def.shape === "circle" || def.shape === "dot_in_circle") {
      const d = def.fixed_diameter_cm || 20;
      rImg = d / 2 * pxPerCm;
    } else {
      const [lp, wp] = _symbolPxSize(s, def, pxPerCm);
      rImg = Math.max(lp, wp) / 2;
    }
    const ic = imgToCanvas(cu, cv);
    const rCanvas = Math.max(minGrabPx, rImg * state.view.scale);
    const d = Math.hypot(ic.cx - cx, ic.cy - cy);
    if (d <= rCanvas && d < bestD) {
      bestD = d;
      best = s.id;
    }
  }
  return best;
}

async function generateSymbols() {
  if (!state.sessionId) return;
  setServicesStatus("Running SAM 3 segmentation pipeline…", "");
  try {
    const r = await fetch(
      `/api/sessions/${state.sessionId}/symbols/generate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend: "local" }),
      },
    );
    if (!r.ok) {
      let msg;
      try {
        const body = await r.json();
        msg = body.detail || body.error || `HTTP ${r.status}`;
      } catch (_) {
        msg = `HTTP ${r.status}`;
      }
      setServicesStatus(msg, r.status === 501 ? "warn" : "err");
      return;
    }
    const data = await r.json();
    setServicesStatus(`Generated ${data.n_symbols ?? "?"} symbols.`, "ok");
    await loadSymbolsForSession();
    setSymbolsVisible(true);
  } catch (e) {
    setServicesStatus("Generation failed: " + e.message, "err");
  }
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
    document.getElementById("right-panel").hidden = false;
    document.getElementById("btn-export").disabled = false;
    const sv = state.plan?.scan_settings?.min_ceiling_height_m
            ?? state.plan?.scan_settings?.max_ceiling_variance_m;
    if (typeof sv === "number" && Number.isFinite(sv)) {
      writeMinCeilingHeightInputM(sv);
    }
    syncMinHeightUnitsUI();
    if (!state.plan.units) state.plan.units = "metric";
    applyUnitsToggleUI();
    syncProjectPanel(state.plan.project);
    if (state.plan.room) {
      markStepDone("room");
      unlockStep("interface");
      unlockStep("column");
      document.getElementById("btn-define").disabled = false;
    }
    if (state.plan.room_heatmap)
      await refreshHeatmap("room", state.plan.room_heatmap);
    if (state.plan.main) {
      await refreshHeatmap("main", state.plan.main);
    }
    for (const r of state.plan.regions || [])
      await refreshHeatmap("region:" + r.id, r);
    await loadCeilingImage();
    fitView();
    refreshPolygonsList();
    await loadSymbolsForSession();
    draw();
  } catch (e) {
    document.getElementById("session-label").textContent =
      `session ${sid} — failed: ${e.message}`;
  }
}

resizeCanvas();
loadFromUrlParam();
