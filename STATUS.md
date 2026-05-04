# Status

**Tagged build: `Good WIP 1 040526` (2026-05-04).** Branch
`claude/nostalgic-poitras-f0c248`. Read this first when picking the
work back up.

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

## Known v1 limits

- **Edge-drag tool** (translate a shared boundary perpendicular to
  itself, sliding endpoints along their other incident edges) — not
  built. Vertex drag + insert + delete cover ~95% of editing needs;
  the architecturally-correct edge translate is a couple-hundred-line
  follow-up.
- **Face holes / multi-component faces** — the topology builder picks
  the largest ring per face and silently drops the rest. Real ceilings
  are simply connected so this hasn't bitten, but revisit if a snap
  ever produces a multi-component face.
- **Junction vertex deletion** — degree-3+ vertices reject with 400.
  To delete a junction the user has to delete the adjacent face (or
  un-snap) instead. Could add proper junction-collapse later.

## Next session priorities

The user has a list of feature changes to discuss in planning mode
before any code lands. The big topics already on the board:

### Light segmentation + symbol placement (priority 2 from the
prior plan)

Lights are visible as bright spots in `ceiling.jpg`. Both YOLOe26 and
SAM3.1 nail lights but their vocab doesn't cover the wider RCP family
(diffusers, smoke detectors, sprinklers, exit signs, fan-coils, …).
The straw-man stack from the design discussion was:

1. Bright-spot CC + shape-classify lights into strip / panel /
   downlight (works without ML).
2. SAM3.1-as-negative-space: invert SAM's ceiling mask, classify
   each hole by `{bbox, area, eccentricity, height-drop, brightness}`.
3. Review UI: every detection lands on the canvas, accept / re-classify
   / delete.
4. Fine-tune a small classifier on accumulated reviewed crops once
   ~50 scans worth of truth exists.

That conversation has not yet been picked back up — start the new
chat with the user's new feature list and re-scope from there.

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
