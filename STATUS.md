# Status

**Tagged build: `good-wip-4-05052026` (2026-05-05).** Branch
`claude/admiring-fermi-4b18e7-impl`, based on `good-wip-3-05052026`.
Read this first when picking the work back up.

**Read-only filesystem snapshot:** `/Users/harishusic/Documents/Claude
Code/Good WIP 4 05052026/` (chmod a-w, with `repo.bundle` for full git
restore). Don't edit it; it's the rollback point.

## What's shipped (v0.2)

- **Web app**: upload Polycam folder → render textured top-down + height
  map → trace polygons in canvas → snap → download PDF.
- **Manual flow**: room outline → main ceiling → ceiling regions, with
  heatmap shading inside each polygon showing local height variance.
- **Auto-detect** (`POST /api/sessions/<id>/auto_detect`): histogram
  peaks + median filter + per-cluster CC + coverage filter. Production
  default.
- **Snap** (`POST /api/sessions/<id>/snap`): Voronoi-style nearest-
  polygon assignment. Polygons grow into gaps and meet at midlines.
- **PDF**: matplotlib-rendered architectural plan with coloured masks,
  height tags, dimensioned room outline, free-text notes per polygon.
- **Lab** (`debug/`): four-stage harness for iterating on auto-detect.
  - `python -m debug.snapshot_truth <id>` — capture ground truth from
    the web UI's plan.json into `debug/ground_truth/<id>.json`.
  - `python -m debug.segment_lab <id> --algo X` — single PNG.
  - `python -m debug.experiments <id>` — sweep + IoU leaderboard.

## Auto-detect performance (2026-04-29 sweep, session cc367350d007)

29 configurations across 11 algorithms scored by Hungarian-matched
mean IoU against a hand-traced 6-region ground truth (1 main + 5
regions). Full leaderboard in
[`debug/results/cc367350d007.csv`](debug/results/cc367350d007.csv).

**Top of the leaderboard:**

| rank | algo | IoU | weighted | min IoU | pred | s |
|---|---|---|---|---|---|---|
| 1 | felzenszwalb_s250 | 0.830 | 0.815 | 0.684 | 37 | 45 |
| 1 | felzenszwalb_s200 | 0.830 | 0.816 | 0.679 | 35 | 46 |
| 1 | felzenszwalb_s225 | 0.829 | 0.816 | 0.683 | 38 | 46 |
| 4 | felz_image_s200_3cm | 0.827 | 0.824 | 0.625 | 35 | 118 |
| 5 | **felz_merged_s200_3cm** | 0.825 | **0.825** | **0.707** | **22** | 45 |
| 6 | felz_merged_s200_2cm | 0.822 | 0.820 | 0.700 | 25 | 46 |
| 7 | felz_image_s200_3cm (RGB) | 0.622-0.827 | varies | varies | varies | 70-150 |
| 8 | felzenszwalb_s400-s1500 | 0.66 | 0.62 | ≤ 0.03 | 22-27 | 45 |
| 9 | watershed_* (4 variants) | 0.58 | 0.80 | 0.00 | 4 | 1 |
| 10 | region_growing_* | 0.34-0.42 | 0.50 | 0.00 | 4 | 1-300 |
| 11 | slic_merge_* | 0.41-0.60 | 0.40-0.61 | 0.01-0.28 | 51-85 | 13-19 |
| 12 | felz_combined / felz_image_only | 0.55-0.77 | varies | varies | varies | 70-150 |
| 13 | wall_constrained_* | 0.07 / FAIL | 0.41 | 0.00 | 1 | 0.1 |

**Production winner: `felz_merged_s200_3cm`.** Same IoU (0.825 vs 0.830)
as the raw plateau but with 22 polygons instead of 35, **best weighted
IoU**, and **best min IoU** — every truth region matched within 0.7
IoU. Editable shape, identical runtime.

**Findings**

1. **0.830 is the boundary-jitter ceiling** for height-only methods on
   this scan. Three different scales (200/225/250) hit it identically
   and the image-domain version matches it. The remaining gap is
   per-pixel boundary noise, not missing regions.
2. **Visual edges and height edges co-locate** on a clean
   architectural ceiling. Image-domain felzenszwalb (`felz_image_*`)
   ties height-domain at scale 200 (0.827 vs 0.830) and falls behind
   at every other scale. Mixing the channels strictly hurts.
3. **Histogram-peak seeding is the seed-count bottleneck.** Watershed,
   region_growing and region_growing_texture all cap at IoU 0.58
   because the prominence filter keeps only 4 of 6 height clusters.
   These algorithms run in 1-2 s — if seeded better they could
   leapfrog felz at 30× the speed.
4. **wall_constrained is structurally broken** — wall raster too
   thick, fragments the room into sub-min cells. Plus an `IndexError`
   at `min_cell_px_m2=0.6`. Fixable, low priority.

## MVP pivot

Auto-detect at IoU 0.83 is good enough for an MVP, especially for
small rooms with few zones. **Manual marking is an acceptable
fallback.** Algorithm work is parked here; production focus shifts to
making the manual-and-auto outputs read as proper architectural
drawings.

## What shipped this session (Good WIP 1 040526)

### Shared-edge topology

`api_snap` produces a planar topology — `{vertices, edges, faces}` —
built at corner resolution from the assignment image. Each shared
boundary is **one** edge that knows both face IDs; the PDF strokes
each edge once; vertex drag, vertex insert and vertex delete all
flow through `topology.vertices`, so a junction shared by two faces
moves both at the same time.

`plan["topology"]` lives next to legacy `main` / `regions` (the
latter kept as a derived view so the renderer didn't have to change).
`schema_version: 2` carries the migration; pre-v2 `plan.json` loads
with `topology = null` and the user re-snaps.

### Topology edit API (post-snap)

| Endpoint | Behaviour |
|---|---|
| `POST /snap` | Build / rebuild the topology |
| `DELETE /topology` | Un-snap; keep current main + regions as plain polygons |
| `PUT /topology/vertices` | Replace the vertex pool — drag, edge-drag |
| `POST /topology/edge/{id}/insert_vertex` | Split an edge; both faces gain the vertex |
| `DELETE /topology/vertex/{id}` | Interior vertex → spliced; degree-2 endpoint with same face pair → edges merge; junction → 400 |
| `PUT /topology/face/{id}/notes` | Notes per face; mirrored from `PUT /main/notes` and `PUT /region/{id}` (notes) |

**Auto-resnap** when topology is present:

- `POST /region` → adds region, snap rebuilds so it has shared edges.
- `DELETE /region/{id}` → removes region, snap reabsorbs its area.
- `POST /obstruction` and `DELETE /obstruction/{id}` — same.
- `PUT /main` with a polygon → clears topology (user is editing
  underneath), then user re-snaps when ready.

**Polygon-edit guards.** `PUT /region/{id}` with a `polygon` field
returns 409 once a topology exists — direct polygon edits would
desync from the topology, so callers must use the topology endpoints
or un-snap first.

### Shift-snap (drawing)

While drawing any polygon (room, main, region, column), holding
**Shift** locks the next vertex to either continue the previous
edge collinearly or turn at exactly 90° — whichever the cursor is
closer to. With one vertex placed, Shift snaps to horizontal /
vertical from that point. Implemented in `app.js`:`constrainShiftSnap`,
read by both the hover-preview render and the click-commit. Also
handled on Shift keydown / keyup so the preview snaps the instant
the modifier is held without needing mouse motion.

### Negative space (columns / structural obstructions)

A new `plan["obstructions"]` list of `{id, polygon, label, kind}`
entries lives alongside `regions`. Drawn after the room outline; the
new "Add column / structural" step in the workflow. They render with
a hatched fill on the canvas and in the PDF.

**Snap honours them**: `api_snap` subtracts every obstruction polygon
from `room_mask` before doing Voronoi assignment, so ceiling regions
*stop* at the column boundary — the topology gains real boundary
edges around the column hole. Region stats no longer include
column-wall LiDAR noise. CRUD endpoints
(`POST/PUT/DELETE /api/sessions/{id}/obstruction[/{id}]`) auto re-snap
when a topology exists.

This is the "Option C" middle ground from the design discussion:
obstructions stored separately (not first-class topology faces with
multi-ring support), but snap-aware so geometry and stats are clean.

### Source-of-the-work pointers

- `src/ceiling_rcp/topology.py` — build-from-assignment, RDP simplify,
  DCEL face-ring traversal, `insert_vertex_on_edge`, `delete_vertex`.
- `src/ceiling_rcp/server.py` — schema migration, `api_snap` (now
  obstruction-aware), the topology / obstruction CRUD endpoints,
  legacy-edit 409 guards, PDF that strokes each edge once and hatches
  obstructions.
- `src/ceiling_rcp/static/app.js` build 10 — topology vertex drag,
  shift-snap, column draw flow + hatched canvas render, Un-snap
  button, auto-resnap response handling.
- `src/ceiling_rcp/static/index.html` — Un-snap button, "Add column"
  step.

## What shipped this session (Good WIP 2 04052026)

Iteration on top of `good-wip-1-040526` driven by the user's 15-item
feedback list. Plan: `~/.claude/plans/this-is-the-ceiling-rcp-expressive-bumblebee.md`.

### Ortho render becomes "ceiling-aware"

`mesh.ceiling_face_mask` replaces the old strict 30° downward-cone
filter with two combined gates:

1. Wider 60° tilt cone — tilted bulkhead edges and vault flanks now
   render instead of leaving black holes in the ortho.
2. Height-band filter — area-weighted 95th-percentile Y of the
   downward-facing faces is treated as the "ceiling top". Anything
   more than `max_ceiling_variance_m` below it gets dropped before
   the raster runs.

`max_ceiling_variance_m` defaults to 1.5 m (couches & floor culled
in a 2.4–3 m room) and is **user-editable** via a number input at
the top of the workflow ("Scan settings"). `PUT /scan_settings`
re-renders ceiling.jpg + height.npy and refreshes per-polygon stats
and heatmaps; the topology is dropped (re-snap required). Stored
as `plan["scan_settings"]["max_ceiling_variance_m"]`.

`make_grid` pad widened from 0.3 m → 0.9 m so polygons drawn just
outside the ortho image still land inside the snap raster.

### Editor UX overhaul

- **Vertex deletion is keyboard-only.** "Delete vertex" tool button
  removed; default tool is drag. Hover any vertex (cyan ring) and
  press `Delete` / `Backspace` to remove it. Insert-vertex stays as
  a separate tool (less common, less mode-error risk).
- **Shift-snap during drag.** Holding Shift while dragging a vertex
  constrains it to be 0° or 90° relative to the prev edge of the
  selected face's ring. Mirrors the existing draw-time constraint;
  works for both pre-snap polygon edits and topology-vertex drags.
- **Columns are first-class.** Drag/insert/delete works on column
  vertices via `hitTest` + `findHoveredVertex` + a column branch in
  `pushPolygonForKey` that hits `PUT /obstruction/{oid}` (server
  auto-re-snaps).
- **Editable tints.** Side-panel swatches are now `<input type="color">`.
  New endpoints: `PUT /main/tint`, `PUT /topology/face/{fid}/tint`,
  and a `tint` field on `PUT /region/{rid}`. Heatmap PNGs re-render
  with the new tint.
- **Auto-detect button removed** from the UI (endpoint kept server-
  side as dead code; debug/ harness still uses the same algorithms).
- **Room heatmap clears** on room delete (was a persistent ghost mask).

### Imperial / metric

New `src/ceiling_rcp/units.py` (server) and a JS mirror in `app.js`.
Helpers: `format_length`, `format_height_delta`. Imperial format is
feet & inches with 1/16″ fractions, e.g. `4'-3 5/8"`, `+5 7/8"`.
Toggle is the *first* control in the side panel; persisted as
`plan["units"]`. PDF reads it for length labels, height deltas, and
the scale bar.

### Multi-ring topology — column holes

`topology.py` now keeps every ring per face (largest CCW = outer,
smaller CW = holes), exposed as `face["holes"]` (edge half-edges) and
`face["holes_polygons"]` (resolved [[x,z],…]). `_analyse_and_pack`
takes an optional `holes` parameter and subtracts those rings from
the mask, so per-face stats (mean Y, σ, range) no longer include
column-wall LiDAR. PDF renders faces via `matplotlib.path.Path` with
each hole as a separate sub-path so the column area is properly
*cut out* of the fill.

This closes the "Face holes / multi-component faces" gap from
WIP 1 — a column inside a face now reads correctly in stats, in the
PDF fill, and in label placement.

### Region label placement (pole of inaccessibility)

New `src/ceiling_rcp/polylabel.py` — Mapbox polylabel adapted to
honour holes. Replaces centroid-based label placement in the PDF.
Labels now sit inside L-shaped regions and avoid column holes.
Rotation matches the polygon's longest edge (wrapped to ±90° so
text is right-side-up).

### Architectural PDF (A1 landscape)

Complete redesign. Page is **A1 landscape** (841 × 594 mm,
33.11 × 23.39 in). Layout via `gridspec`: plan area (left), title
block (right strip, full height), legend strip (bottom).

**Title block** carries:
- Ortho thumbnail (top of strip; `imshow` of the rendered ceiling.jpg).
- Project metadata fields: PROJECT, ADDRESS, CLIENT, COMPANY,
  DRAWING NO. — all user-edited via the new "Project info" panel.
- Drawing register table: REV / DATE / BY / NOTE rows, add/remove
  via the panel.

**Plan area overlays:**
- North arrow at top-right — circle + filled triangle, clickable
  SVG picker on the frontend; checkbox to omit print.
- Scale bar at bottom-left — segmented 0—1—5—10 m or 0—1—5—10 ft.

**Legend strip** has three columns: ceiling zones (tint swatch +
label + relative height per face, two columns of up to 7 rows),
structural ("Columns" with cross-hatch swatch), services
(placeholder for future light symbols).

Other PDF tweaks from the same iteration:
- Length labels sit *on* the room outline (white bbox cuts the
  line) — the architectural ——[label]—— look — instead of being
  offset perpendicular outside.
- Skip-if-short heuristic: edges shorter than `1.5 × estimated label
  width` get no label.
- Column hatch upgraded to cross-hatch (`hatch="xx"`) and the
  per-column centroid label dropped — single "Columns" row in the
  legend instead.

### Schema bump (v3)

`PLAN_SCHEMA_VERSION = 3` adds `plan["units"]`, `plan["scan_settings"]`,
and `plan["project"]`. v2 plans backfill defaults on load
(`metric`, 1.5 m variance, blank project). v1 plans get the
topology field too; user re-snaps.

### New / updated endpoints

| Method | Path | Purpose |
|---|---|---|
| GET / PUT | `/api/sessions/{id}/project` | Title-block fields, drawing register, north heading |
| PUT | `/api/sessions/{id}/units` | "metric" / "imperial" |
| PUT | `/api/sessions/{id}/scan_settings` | `max_ceiling_variance_m` — re-renders ceiling.jpg/height.npy |
| PUT | `/api/sessions/{id}/main/tint` | Recolour main + refresh heatmap |
| PUT | `/api/sessions/{id}/topology/face/{fid}/tint` | Post-snap face tint, mirrored to legacy view |
| PUT | `/api/sessions/{id}/region/{rid}` | Now also accepts `tint` |

### Known v2 limits

- **Edge-drag tool** still not built (same as WIP 1 — vertex drag
  covers ~95% of edits).
- **Junction vertex deletion** still rejects with 400 (degree-3+
  vertices need a face-level edit; un-snap, edit, re-snap).
- **PDF column legend is generic** — single "Columns" row regardless
  of column count. Numbered C1/C2/… per the design discussion is
  deferred until a real project demands it.
- **Services legend is a placeholder** — empty box; light detection
  will populate it in the next iteration.
- **Drawing register row count** capped at ~14 (more spill off the
  title block). Fine for typical revisions; long histories truncate
  visually.

## What shipped this session (Good WIP 3 05052026)

Iteration on top of `good-wip-2-04052026` driven by the user's
nine-item feedback list. Work was clustered into four commits on
`claude/admiring-fermi-4b18e7-impl`:

| Commit | Cluster | Headline |
|---|---|---|
| `bc554d6` | A | Editor UX polish + PDF title-block tweaks |
| `83ba8c0` | B | Render PDF plan at a true architectural scale |
| `7b0b601` | C | Per-region height histograms + draggable height pick |
| `0e25a57` | D | Interface-tracing pipeline (replaces per-region tracing) |

### Cluster A — UX polish

- Project-info panel **defaults open**.
- North-arrow picker: brighter needle (`--text` instead of `#222`),
  red blob at the tip, `cursor: grab`, **drag** rather than click to
  set heading, "Drag to set North" hint.
- "Polygons" list renamed **"Regions"** and moved to a new 320 px
  **right sidebar** alongside "Selected"; columns ride along.
- PDF title-block strip narrowed 30 % (gridspec width ratio
  `3.2:1.0` → `3.2:0.7`).
- PDF field values bumped 10 pt → 14 pt (labels stay at 7 pt).
- PDF ceiling labels render **horizontally** — dropped the
  longest-edge rotation that read like a CAD section marker.
- PDF ortho thumbnail now letterboxes inside a fixed box (was
  stretched on tall rooms via `aspect="auto"`).

### Cluster B — true-scale plan rendering

- New `_choose_standard_scale(room_w, room_h, plan_w_in, plan_h_in,
  units)` picks the largest of `(20, 50, 100, 200)` metric or
  `(24, 48, 96, 192)` imperial that fits the room within the plan
  area at 10 mm of paper margin per side.
- `api_pdf` centres the room in a fixed paper-sized window:
  `window_w_m = plan_w_in × 0.0254 × scale_ratio`. 1 m on the page
  is exactly `1/scale_ratio` m on paper.
- Scale display moves from a plan-axes inset to the title block as a
  "SCALE  1:N" badge plus a graphic bar whose paper length equals the
  marks' world distance ÷ scale. Marks tuned per ratio via
  `SCALE_BAR_MARKS_METRIC` / `SCALE_BAR_MARKS_IMPERIAL` so the bar
  reads cleanly across the ladder (e.g. 1:50 shows `0 1 2 5 m`,
  1:200 shows `0 5 10 20 m`).
- Padding tightened from 25 mm → 10 mm of paper per side; a 21 m
  tall room on A1 now fits at 1:50 instead of being bumped to 1:100
  for no visual gain.

### Cluster C — height histograms + draggable height pick

- New `_height_histogram(height_map, mask, bin_w_m=0.005)` returns
  `{bin_edges_m, counts, min_y, max_y, bin_w_m}` for the valid
  (non-NaN) ceiling heights inside the mask. Sourced from the existing
  cone-band-filtered `height.npy` — no new triangle pass.
- `_analyse_and_pack` now emits `histogram` alongside the existing
  stats / heatmap fields.
- New `selected_y` field on main / regions / topology faces
  (absolute world Y, m). Defaults to `stats.mean_y`.
  `relative_y = face.selected_y − main.selected_y` is derived
  centrally by `_recompute_relatives(plan)` — called on every plan
  load (via `_migrate_plan`) and before every save in handlers that
  touch stats.
- New endpoints
  `PUT /api/sessions/{id}/main/selected_y` and
  `PUT /api/sessions/{id}/region/{rid}/selected_y` (body
  `{value: <metres>}`) plus `_mirror_selected_y_to_topology`.
- Sidebar: each row gets a 264 × 40 px **SVG sparkline** — bars in
  the face's tint, dashed grey line at `mean_y`, draggable red
  marker + blob at `selected_y`, with absolute-height labels at
  min / selected / max. Drag → live visual feedback → mouseup commits
  via the new endpoint and refreshes every row's `relative_y` label.
- "variance" → "spread" everywhere in the sidebar.
- Schema bump 3 → 4. Old plans back-fill `selected_y = mean_y` on
  first load; histograms repopulate on the next analyse pass
  (re-snap or any region edit).

### Cluster D — interface tracing replaces per-region tracing

- New plan field `interfaces = [{id, polyline, closed}]`. Open
  polylines are **chords** (subdivide whichever face contains them);
  closed polylines are **island rings** (a face wholly inside
  another).
- New plan field `main_face_id` (default 0) — lets the user pick
  the height datum without retracing.
- New endpoints:
  `POST /api/sessions/{id}/interface`,
  `PUT /api/sessions/{id}/interface/{iid}`,
  `DELETE /api/sessions/{id}/interface/{iid}`,
  `POST /api/sessions/{id}/define_ceilings`,
  `PUT /api/sessions/{id}/main_face`.
- `define_ceilings` uses `shapely.ops.polygonize` on the union of
  the room ring and every interface line. Filters out tiny
  artefacts (< 0.05 m²) and outside-room slivers, sorts by area
  descending, seeds the legacy `main` / `regions` views, and hands
  off to the existing `api_snap` so the topology / heatmaps /
  histograms build the same way they always did.
- `main_face` swap rewrites `plan.main` ↔ a chosen region and
  re-snaps so the topology stays canonical (face id 0 = main).
- Workflow steps simplified: **Room → Trace interface → Add column
  → Define ceilings → Export**. The old "Trace main ceiling" and
  "Add ceiling region" steps and buttons are removed (endpoints
  preserved server-side as a fallback).
- Trace-interface tool: click vertices, click the first vertex (with
  3+ points placed) to close as a ring, press Enter (with 2+ points)
  to commit as a chord.
- Interfaces render in cyan (`#00e5ff`) on the canvas above region
  fills, with vertex dots. They appear as info rows in the Regions
  panel for delete.
- Each region row in the panel gains a **Main radio** — clicking it
  on a non-main row swaps the datum. Currently-main row's radio is
  disabled-checked.
- Schema bump 4 → 5.

### Schema progression this session

- v3 → v4: face-level `selected_y` + `histogram` (cluster C).
- v4 → v5: `interfaces` + `main_face_id` (cluster D).
- Migrations are additive — old plans load with safe defaults and
  keep working. New tracing flow only activates when the user starts
  using `Trace interface`.

### Files most touched

| File | Cluster(s) |
|---|---|
| `src/ceiling_rcp/server.py` | A, B, C, D |
| `src/ceiling_rcp/static/app.js` | A, C, D |
| `src/ceiling_rcp/static/index.html` | A, C, D |
| `src/ceiling_rcp/static/style.css` | A, C, D |

`mesh.py`, `topology.py`, `polylabel.py`, `units.py`, `raster.py`,
`analyse.py` were *not* touched — their existing surfaces were
sufficient. The same is true of the segmentation lab in `debug/`.

### Verification

- API round-trips for selected_y (cluster C) and define_ceilings →
  main_face swap (cluster D) checked numerically against the
  topology-snapped session `0beced53c9df`.
- PDFs rendered for both metric (1:50) and imperial (1:48) at every
  cluster boundary; rasterised to PNG via `sips` and inspected
  visually.
- All static assets served at the bumped cache versions
  (`app.js?v=18`, `style.css?v=17`).

### Known v3 limits

- **No live snap-to-existing while tracing interfaces.** Chord
  endpoints rely on the user clicking close enough to the room
  outline; Shapely's `unary_union` tolerates small drift but
  obvious misses won't cut. Phase-2 improvement: snap the cursor
  to nearby room/interface vertices and edges with a Shift override.
- **Interface vertices aren't draggable post-trace.** Delete +
  re-trace is the only edit path. Easy follow-up.
- **`api_set_main` and `api_add_region` endpoints still alive**
  but unbuttoned — kept as a fallback / migration path.
- **Wide histogram ranges** observed on the existing topology-snapped
  session (e.g. region 0 spans 0.74–2.05 m). The histogram surfaces
  a pre-existing data-quality issue with the cone-band filter
  (`mesh.ceiling_face_mask`) letting through a few non-ceiling
  pixels. Worth tightening or revisiting `max_ceiling_variance_m`
  on a per-session basis.
- **Drawing register row count** still capped at 14 (carried from v2).
- **Services legend** still a placeholder (carried from v2 — light
  segmentation never started).

## What shipped this session (Good WIP 4 05052026)

Iteration on top of `good-wip-3-05052026` driven by the user's
20-item punch list (the bug + UX + scan-settings + page-size set
they sent in this session). Work landed as five cluster commits
plus follow-ups on `claude/admiring-fermi-4b18e7-impl`:

| Commit | Cluster | Headline |
|---|---|---|
| `b8eb603` | E | Bugs: main-face swap + PDF legend swatches |
| `919cb1b` | F | Interface tracing finish-up (snap, drag, fade) |
| `5671fdf` | G | Histogram + region-row UX overhaul |
| `21428ca` | H | Scan-settings rename + 2 % outlier trim |
| `980128c` | I | Page-size + scale selectors + sheet size in PDF |
| `ac6fd1b` | F+ | Harden trace snap + define_ceilings noding |
| `ca8f2fe` | F+ | Show snap indicator while placing the first vertex |
| `1093b32` | F+ | Make faded interface chords more visible |
| `96cf971` | F+ | define_ceilings: pairwise crossing + diagnostics |
| `111d344` | F+ | define_ceilings: extend snapped endpoints past target |
| `674978e` | F+ | define_ceilings: 2 cm visible-X overshoot, persisted |

### Cluster E — bugs

- **E1: Main-face swap.** Was reverting silently. Root cause: the
  topology branch in `api_swap_main_face` dropped the topology and
  re-snapped, but `api_snap`'s stage-1 region Voronoi let the
  previously-large main (now a region drawn polygon) overwrite
  every pixel of the small new main, so face id 0 ended with zero
  coverage and api_snap raised 409. Fix: post-snap, swap is now a
  metadata-only relabel — the vertices, edges and rings stay; only
  `kind`/`region_id`/`label`/`tint` flip on the two affected faces,
  and `plan.main_face_id` moves to the new main.
  `_refresh_topology_polygons` honours `main_face_id` to render the
  new roles. `api_snap` stamps `kind=main/region` on every face and
  resets `main_face_id=0` for fresh snaps. Frontend
  (`dragRingFor`, `rederiveFacePolygons`, `pushTintForKey`) uses
  `state.plan.main_face_id` everywhere it was assuming face id 0
  was main. `pushMainFace` shows a "Datum swapped" banner.
- **E2: PDF legend swatches were too vibrant.** New module-level
  `PLAN_FILL_ALPHA = 0.45` and `_muted_fill(hex)` helper pre-blend
  each tint with white at the same alpha. `_draw_legends` uses the
  muted RGB triple for the swatch facecolor so the eye reads
  "this swatch = this fill".

### Cluster F — interface tracing finish-up

- **F3: Live cursor snap to nearby room/interface geometry.** New
  `snapToNearestExisting` folds vertices, midpoints AND
  segment-foot projections (perpendicular onto each segment) into
  a candidate list — so the cursor locks onto a chord or wall
  *anywhere along it*, not just at vertices. Vertex/midpoint
  candidates get a 0.85× distance bias so corners win ties. Snap
  radius 14 px. New `applyTraceSnap` is the single entry point the
  trace tool reads from: Shift held → `constrainShiftSnap` (ortho
  lock); not held → `snapToNearestExisting`. Wired into
  `onMouseDown`'s draw branch, `onMouseMove`'s draw-mode preview,
  and the Shift keydown/keyup re-snap. Green ring renders around
  the cursor when locked onto a target — including for the very
  first vertex (drawDraft was previously skipped on empty drafts,
  so vertex 1 had no feedback).
- **F4: Server-side noding hardened.** Three layers:
  1. Per-vertex snap: every chord vertex (endpoints AND interior)
     projects onto the nearest point on any other line within
     `NODE_TOL_M = 0.20 m`. Snap point is also spliced into the
     target line as a new vertex via `_insert_vertex`, so
     `unary_union` sees an exact intersection rather than a
     near-miss.
  2. Pairwise crossing-point insertion: for every line pair, the
     geometric `intersection` is computed and each crossing point
     is spliced into both lines. Catches X-junction near-misses
     that float-drift would otherwise let `unary_union` silently
     ignore.
  3. **Endpoint overshoot**: when a chord endpoint snaps to a
     target line, the actual endpoint extends 2 cm past the target
     in the chord direction. This forces the geometry into an
     X-junction (chord crosses target) rather than a T-junction
     (chord ends exactly on target). polygonize handles
     X-junctions deterministically; T-junctions can fail to close
     rings on certain float configurations, which produced the
     "sometimes works, sometimes doesn't" intermittency the user
     reported. The 2 cm overshoot is persisted back to
     `plan.interfaces[i].polyline` so the canvas renders a small
     visible X — visual confirmation that the chord truly noded.
     Steady-state: the overshoot doesn't grow on re-define
     because the snap pass projects the existing endpoint back
     onto the target line first.
  4. Final `shapely.ops.snap(merged, merged, NODE_TOL_M / 2)`
     pass collapses any sub-tolerance float drift before
     polygonize.
- **F5: define_ceilings dangle diagnostics.** Switched
  `polygonize` → `polygonize_full` so dangling segments come back
  separately. The response carries `last_define_diagnostic` with
  `input_chord_count`, `input_closed_count`, `polygons_raw`,
  `polygons_after_filter`, `dangling_segments`, plus up to 8
  sample dangle endpoints. Frontend logs the diagnostic to the
  console on every Define and shows a warning banner when faces
  produced is fewer than expected.
- **F6: Interface vertex drag post-trace.** `polygonForKey`,
  `findHoveredVertex`, and `hitTest` learn about `"interface:N"`
  keys. Interfaces are checked BEFORE the topology branch in
  `findHoveredVertex` so a chord vertex coincident with a topology
  junction still drags as the chord. `onMouseDown` skips the
  topology-vertex resolve for interface drags so they stay scoped.
  `pushPolygonForKey` PUTs to `/interface/{id}` and adopts the
  response. Server-side: `api_update_interface` auto-runs
  `api_define_ceilings` when a topology already exists.
  `deleteHoveredVertex` uses min-2 for open chords, min-3 elsewhere.
- **F7: Hide consumed interface chords after Define.** Once a
  topology exists, `drawInterface` fades unselected chords to 70%
  alpha with a 6/4 dashed stroke and hides their vertex dots — the
  topology renderer already strokes the real face outline. Selecting
  the chord's row in the Regions panel un-fades it for vertex drag
  / hover-delete.

### Cluster G — histogram + region-row UX

- **G8** Notes input above the histogram.
- **G9** Histogram doubled in height (40 → 80 px).
- **G10** Slider readout in millimetres relative to the main
  ceiling — main is always 0; regions are +/- mm
  (`formatHeightDelta` goes imperial automatically).
- **G11** Axis labels reframed as offsets from `main.selected_y`:
  min on the left, max on the right, `0` in the middle when the
  datum falls inside the face's histogram range.
- **G12** Peak frequency shown as a number above the bars
  (`peak X.Y%`).
- **G13** `↓ N%` on the left and `M% ↑` on the right of the
  marker — recomputed live as the slider drags.
- **G14** Pink/black checker overlay. New endpoint
  `GET /face_below?key=<key>&y=<value>` returns an RGBA PNG sized
  to the face's bbox; pixels with Y < value render as a 4-px
  pink/black checker. Bbox carried in the X-Bbox header. Slider
  drag fetches it on every tick (one-in-flight throttling), the
  canvas draws it on top of the heatmap. Cleared on mouseup.
- **G15** Two-line `Height: / Spread:` row label with right-aligned
  values in a CSS grid so numbers line up regardless of sign.

### Cluster H — scan settings + outlier trim

- **H16** "Max ceiling height variance" → "Minimum ceiling height".
  `ceiling_face_mask` now filters by absolute world-Y threshold
  (was `top - variance`); raising the value reliably clips
  furniture (used to do the opposite). Default 2.0 m, range
  0.5–6.0 m. Imperial (ft + in) input mode appears when the units
  toggle is set to imperial; the value still round-trips as metres
  on the wire. Migration drops `max_ceiling_variance_m` and
  back-fills `min_ceiling_height_m: 2.0` (different semantic, no
  auto-conversion).
- **H17** 2 % tail trim. New `_trim_outliers(values, frac=0.02)`
  drops the lowest 2 % and highest 2 % of in-mask pixel heights
  before computing mean/std/min/max and before deriving histogram
  bin edges. Coverage metrics (`valid_frac`, `n_valid_px`,
  `n_total_px`) stay based on the full untrimmed sample.

### Cluster I — page-size + scale selectors + PDF sheet size

- **I18** Page-size dropdown in the Project info panel — eleven
  options grouped Metric (A4–A0) / Imperial (Letter, Tabloid,
  Arch B–E). Default A1. New `PAGE_SIZES` table replaces the
  hardcoded A1 figsize in `api_pdf`; default Arch D when units
  are imperial via `_resolve_page_size`.
- **I19** Scale-override dropdown. Default "Auto" runs the
  existing `_choose_standard_scale`; manual options cover both
  ladders. Stored as `plan.project.scale_override` (None = auto).
- **I20** Sheet-size sub-line in the title-block SCALE section
  (e.g. `A1 (841 × 594 mm)` or `Arch D (36" × 24")`).
- `api_set_project` accepts the new `page_size` and
  `scale_override` fields with validation.

### Schema progression this session

- v5 (carried) → v5 with additive defaults: `project.page_size`,
  `project.scale_override`, `scan_settings.min_ceiling_height_m`.
- `face.kind = "main" | "region"` now stamped on every topology
  face (was missing in WIP 3).
- Old plans migrate transparently — `_migrate_plan` pops the
  legacy `max_ceiling_variance_m` and seeds the new defaults.

### Files most touched

| File | Cluster(s) |
|---|---|
| `src/ceiling_rcp/server.py` | E, F, G, H, I |
| `src/ceiling_rcp/static/app.js` | E, F, G, H, I |
| `src/ceiling_rcp/static/index.html` | F, G, H, I |
| `src/ceiling_rcp/static/style.css` | G, I |
| `src/ceiling_rcp/mesh.py` | H (signature change) |

### Cache versions

- `app.js?v=18` → `app.js?v=26`
- `style.css?v=17` → `style.css?v=20`

### Verification

- `define_ceilings` exercised against simulated trace data via the
  dangle diagnostic (`polygons_after_filter` matches
  `1 + chord_count + closed_count` for every test pattern, with
  `dangling_segments == endpoint_overshoots`).
- Main-face swap round-tripped on snapped session
  `0beced53c9df`: face id changes, polygon shapes preserved,
  every region's `relative_y` recomputes against the new datum.
- PDF rendered for A1 metric (1:50), Arch D imperial (1:48), and
  A3 metric (1:100) at every cluster boundary; legend swatches
  visually match the plan fills.
- All static assets bumped (`app.js?v=26`, `style.css?v=20`).

### Known v4 limits

- **No imperial unit on the histogram axis tail labels** when the
  trim leaves the range under 1 mm — formatHeightDelta returns
  "0 mm" / "0\"" which is informative but not pretty.
- **Drawing register row count** still capped at 14 (carried from
  v2).
- **Services legend** still a placeholder (carried from v2 — light
  segmentation never started).
- **Numbered column references (C1 / C2 / …)** still on the
  backlog.

## Next session priorities

User-confirmed punch list, grouped into five proposed clusters.
Cluster E (bugs) is top-priority; the rest is feature work and the
order can be reshuffled. Full detail (motivation, file pointers,
acceptance criteria) lives in [`NEXT_SESSION.md`](NEXT_SESSION.md).

### Cluster E — bug fixes

1. **Main-ceiling swap doesn't stick.** Clicking the Main radio on
   a non-main row reverts after the API round-trip.
2. **PDF legend swatches are too vibrant** vs the 0.45-alpha plan
   fills they label — match the muted colour.

### Cluster F — interface-tracing finish-up (cluster D follow-ups)

3. Live snap-to-existing while tracing (cursor → nearest
   room-outline / interface vertex / midpoint, Shift override).
4. Server-side endpoint extension in `define_ceilings` so chords
   that miss by a few mm still cut.
5. Define ceilings missing chords — closes out once #3 / #4 ship.
6. Interface vertex drag post-trace (same shape as the topology
   vertex drag).
7. Interface chords disappear / fade after a successful Define
   (kept in the panel for delete).

### Cluster G — histogram + region-row UX

8. Notes input above the histogram.
9. Histogram 2× height.
10. Slider readout in relative mm (0 for main, ± for regions).
11. Histogram axis `−max(spread)` → `+max(spread)` with 0 marked,
    not absolute metres — eliminates the empty-tail problem.
12. Show peak % frequency as a number.
13. Show % of ceiling above / below the current slider position.
14. **Crop preview** — when the slider moves, render any pixels
    in the polygon below the slider in a bright pink/black
    checker overlay so the user sees what they're excluding.
15. Per-row label `Height: XX mm` / `Spread: XX mm`, vertically
    aligned.

### Cluster H — scan settings rename + outlier trim

16. Rename "Max ceiling height variance" → **"Minimum ceiling
    height"**, default 2.0 m, imperial-aware. Same plumbing,
    friendlier framing.
17. **New 2 %-trim outlier filter** on per-face stats + histogram —
    don't tighten the existing cone-band filter; trim the top
    2 % / bottom 2 % of valid pixel heights when computing
    `mean_y` / `std_y` / `min_y` / `max_y` and the histogram.
    Keeps `valid_frac` / `n_valid_px` / `n_total_px` honest. Drop
    into `_analyse_and_pack` as a single `_trim_outliers` helper.

### Cluster I — page size + scale selectors

18. **Page-size selector** in Project info — default A1; metric
    group A4–A0, imperial group US sheet sizes (default Arch D
    when imperial). Stored on `plan.project.page_size`.
19. **Scale selector** alongside, default to the current
    `_choose_standard_scale` auto-pick, manual override allowed,
    metric / imperial groups.
20. **Sheet size in the PDF's scale box** (e.g. `A1 (841 × 594 mm)`
    or `Arch D (24" × 36")`).

The page-size change cascades into `api_pdf` — A1 is hardcoded
today (`A1_W_IN, A1_H_IN = 33.11, 23.39`). Pull from a small
`PAGE_SIZES` table keyed on the selected size.

## Backlog (parked)

Items from previous "Next session priorities" the user hasn't
asked for this round — kept here so they don't get lost:

- Light segmentation + symbol placement (Services legend
  placeholder still empty).
- Edge-drag tool for shared topology boundaries.
- Numbered column references (C1 / C2 / …) in the legend.
- Cleaner Cmd+Z undo.

## Quick-resume CLI

```bash
# server (terminal 1)
ceiling-rcp-server --port 8765

# fresh session from a Polycam folder (terminal 2)
ceiling-rcp-init "Scan data/<your-folder>"

# lab against the captured truth
python -m debug.experiments cc367350d007
```

Project artefacts live under `sessions/`, lab outputs under
`debug_out/` and `debug/results*/`. All ignored by git except
`debug/ground_truth/` which carries the truth files.

## Files worth reading first

- [`README.md`](README.md) — overview, CLI table, API summary.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — scan-to-PDF data flow, plan
  JSON shape, coordinate convention.
- [`debug/README.md`](debug/README.md) — lab workflow, scoring
  details, how to add algorithms.
- This file — current state + next priorities.
