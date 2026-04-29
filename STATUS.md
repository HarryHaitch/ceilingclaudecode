# Status

Snapshot of the project at the end of the algorithm-sweep session. Read
this first when picking the work back up.

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

## Next session priorities (in order)

### 1. Shared polygon edges in the topology snap

Current `api_snap` (server.py:236) writes each polygon as its own
ring of vertices. Adjacent polygons end up with edges that *touch*
but are stored separately; if you drag one polygon's vertex the
neighbour's edge stays put. That isn't how architectural drawings
work.

**Goal:** edges shared between two polygons are stored *once* and
referenced by both. Editing one moves the other automatically. The
PDF should render each shared line exactly once.

**Approach to design:**

- Represent a snapped plan as a planar graph: vertices, edges
  (vertex-pair), faces (list of edge IDs).
- Convert the current per-polygon ring representation by:
  1. Rasterising the assignment image (we already build this in snap).
  2. Tracing each face's outer boundary.
  3. Merging duplicate vertices and shared edges across faces.
- Update the frontend to drag an *edge* (not a vertex chain that
  happens to overlap with a neighbour) — when an edge moves, both
  faces it bounds re-render.
- PDF rendering then iterates edges, not polygons, for the line
  layer; fills are still per-face.

This is the load-bearing change for "looks like an architectural
drawing".

### 2. Light segmentation + symbol placement

Lights are visible as bright spots in the textured render
(`ceiling.jpg`). Detect them, classify by shape, drop a CAD-style
symbol on the PDF aligned to the detection.

**Approach to design:**

- Threshold the textured render's brightness channel (or
  saturation-low + value-high in HSV) to mask candidate lights.
- Connected components → for each: bounding rect, oriented bounding
  rect, eccentricity.
- Classify: circle if eccentricity < 0.3 OR aspect ratio < 1.5;
  rectangle otherwise. Keep the orientation of the bounding rect for
  rectangles.
- Filter: discard CCs smaller than ~0.05 m² (noise) or larger than
  ~1 m² (likely not a light).
- Output schema: `{kind: "circle"|"rectangle", centre: [x, z],
  size: [w] | [w, h], rotation_deg: float}`.
- Render in PDF: circles as ⊙ symbol with diameter matching detection,
  rectangles as outlined frame matching detection.
- Stretch goal: detect linear diffusers (long thin rectangles),
  square panels (square rectangles), downlights (small circles).

The two priorities are independent — you can do them in either order
or in parallel.

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
