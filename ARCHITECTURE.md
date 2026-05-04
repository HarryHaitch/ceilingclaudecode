# Architecture

End-to-end data flow for `ceiling-rcp`. Read this if you want to
understand *why* a file exists; read the module docstrings for the
*how*.

## The journey of one scan

```
Polycam folder           (mesh + textures + alignment, on disk)
        │
        ▼   ceiling-rcp-init
sessions/<id>/upload/    (filtered copy: .obj/.mtl/textures/mesh_info.json)
        │
        ▼   ceiling-rcp-init  →  server.process_session
sessions/<id>/out/
  ceiling.jpg            top-down textured render (BGR, lowest-Y wins)
  height.npy             per-pixel world-Y, NaN where no LiDAR
  plan.json              authoritative plan object — mutated by every edit
        │
        ▼   ceiling-rcp-server (long-running)  +  static/index.html
browser canvas editor    user traces room → main → regions
        │
        ▼   POST /snap, POST /auto_detect, GET /pdf
sessions/<id>/out/plan.json  (refreshed)
debug_out/rcp_<id>.pdf       (one-page architectural PDF)
```

## Why each file exists

### `mesh.py`

Polycam exports are inconsistent (sometimes `mesh_info.json` lives in
a sibling folder, textures get flattened on upload, etc.).
`inspect_folder` is the tolerant validator that surfaces what's
present, what's missing, and what was moved. `load_mesh` then parses
the OBJ + MTL, applies `inv(alignmentTransform)` to bring vertices
into ARKit world space, and resolves textures even if the .mtl's
relative paths broke during upload. Returns a `Mesh` dataclass that
the rest of the pipeline consumes.

`ceiling_face_mask(mesh, max_tilt_deg=60.0, max_ceiling_variance_m=1.5)`
is the production filter for "what counts as ceiling". Two combined
gates: (1) downward-facing cone — wider than the legacy 30° so
tilted bulkheads and vault flanks render; (2) area-weighted
95th-percentile Y of those down-faces is treated as ceiling top, and
anything more than `max_ceiling_variance_m` below it is dropped.
Couches and floors fall out before the raster runs. The legacy
`downward_face_mask` is kept for the debug harness's wall-edge
rasteriser.

### `raster.py`

`render_textured_topdown` is the only function that touches mesh
textures. It walks every downward-facing triangle, projects it into
the XZ plan grid, picks up the warped texture patch via
`cv2.getAffineTransform`, and Z-buffers by lowest world-Y so the
output matches what an occupant would see looking up. **Returns both
the BGR canvas and the z-buffer** — the z-buffer is the height map
the rest of the pipeline analyses, so colour and height stay aligned
pixel-for-pixel.

### `planes.py`

Defines `PlanGrid` (the shared XZ↔pixel coordinate transform that
also implements the RCP X-mirror), helpers like `make_grid` and
`rasterise_faces`, plus a legacy `segment_ceiling` from the v0.1 auto
pipeline. `segment_ceiling` is no longer wired into the server but is
kept around for the `ceiling-rcp` debug CLI and as a reference
implementation.

### `analyse.py`

Per-polygon math: rasterise the polygon to a mask, compute mean Y / σ
/ valid fraction inside it, and emit a deviation-heatmap PNG (lightness
in the polygon's tint colour modulated by `(height − mean) / range`).
Also has `polygon_to_mask` and `height_map_to_storage` (`+inf → NaN`
for the saved height map).

### `polygons.py`

Mask-to-polygon conversion (with optional hole detection), and pure
polygon edit primitives (`insert_vertex_on_edge`, `delete_vertex`,
`delete_chain_between`). The frontend calls equivalents directly on
its in-memory polygon arrays; these server-side functions exist so
non-JS clients can still drive the same edits.

### `topology.py`

Builds the planar graph (`vertices`, `edges`, `faces`) traced from a
per-pixel assignment image. `build_from_assignment` traces edges at
corner resolution, simplifies via RDP, pools shared vertices, and
walks DCEL face rings. **Each face has both an outer ring and a list
of `holes` (CW rings)** — typically a column inside the face. The
edit primitives `insert_vertex_on_edge` and `delete_vertex` mutate
the dict in place; the server replays them and re-runs face analyses.

### `polylabel.py`

Mapbox pole-of-inaccessibility, hole-aware. Used by the PDF to place
each region's label inside the actual face — even L-shapes and faces
with column holes. ~70 LOC, no extra dependencies.

### `units.py`

Metric ↔ imperial display formatting. `format_length` (1234 mm /
1.23 m / 4'-3 5/8") and `format_height_delta` (signed). World
coordinates stay in metres internally; only display strings change.
Mirrored in `static/app.js` so the editor and the PDF format
identically.

### `server.py`

The FastAPI app. One file because the surface is small and every
endpoint shares the same load-plan / mutate / save-plan pattern. Key
pieces:

- `process_session` / `_do_render` — runs `inspect_folder` +
  `load_mesh` + `ceiling_face_mask` + `render_textured_topdown` +
  saves `ceiling.jpg`, `height.npy`, `plan.json`. `_do_render` is
  factored out so `PUT /scan_settings` can re-render without
  resetting user-drawn polygons.
- `_analyse_and_pack(polygon, holes=…, tint=…)` — wraps
  `analyse.analyse_polygon` + `_heatmap_from_mask`. Holes are
  rasterised and subtracted from the polygon mask before stats so
  per-face mean Y / σ exclude column interiors.
- `api_set_room` — also computes a *room* heatmap (white tint, ±15 cm
  range) so the user can see height variance across the whole room
  before drawing anything inside it.
- `api_set_scan_settings` — re-renders with a new
  `max_ceiling_variance_m`, refreshes per-polygon stats and
  heatmaps, drops the topology (re-snap required).
- `api_set_units`, `api_get_project` / `api_set_project` — display
  preferences and PDF title-block fields.
- `api_auto_detect` — the histogram-peaks → median-filter →
  per-cluster CC → coverage-filter → same-cluster absorption →
  band-restricted-stats pipeline, all in one function. ~150 lines, no
  intermediate state needed. Endpoint kept as dead code; the UI
  removed the trigger button in WIP 2.
- `api_snap` — Voronoi assignment: every room pixel goes to whichever
  drawn polygon owns it (region-drawn pixels win first, then main's
  drawn area, then nearest by distance transform). Obstructions are
  subtracted from the room mask before assignment so the topology
  builder traces around column holes. Then `build_from_assignment`
  emits `vertices/edges/faces` (with per-face `holes`).
- `api_topology_*` — vertex drag (`PUT /topology/vertices`),
  insert-on-edge, delete-vertex, set face notes / tint. All mutate
  the topology in place and call `_refresh_topology_polygons` which
  rebuilds each face's outer + holes polygons and re-runs analysis.
- `api_pdf` — A1-landscape matplotlib backend. Plan area + title
  block + legend strip via `gridspec`. Faces rendered as
  `matplotlib.path.Path` with hole sub-paths; labels via
  `polylabel`; length labels rotated and positioned *on* the room
  outline (white bbox cuts the line). Helpers
  `_draw_north_arrow`, `_draw_scale_bar`, `_draw_title_block`,
  `_draw_legends` are at module level so they can be tested
  in isolation.

### `static/index.html` + `app.js` + `style.css`

Single-page editor. No framework, no build step. State lives on a
global `state` object; every action mutates it and calls `draw()`.
Pencil draw, vertex select / drag / insert / delete, snap and PDF
buttons, per-polygon notes input. Heatmaps are PNGs returned by the
server, decoded into `ImageBitmap`s and drawn beneath the polygon
outlines.

### `init_session.py`

`ceiling-rcp-init` exists because Safari's file picker is hostile
(rejects multipart filenames containing `/`, can't read iCloud
placeholders, vague "Load failed" errors with no diagnostics). The
CLI bypasses the browser entirely: filters the Polycam folder to
just the mesh-relevant files (skip `.ply`, `.mp4`, keyframes, depth
maps), copies them into a fresh session under `sessions/<uuid>/upload/`,
calls `process_session`, and prints the URL.

### `cli.py`

The legacy `ceiling-rcp` debug CLI. Runs the full v0.1 auto-segmentation
pipeline against a scan folder and writes `out_<name>/` with debug
PNGs and a JSON. Useful when you want to see what the histogram +
periodic-feature detector produces without launching the server.

## Plan JSON shape

The single object every edit mutates and every read returns. Schema
version 3 (WIP 2). Older v1 / v2 plans get migrated transparently on
load.

```json
{
  "session_id": "abc123def456",
  "schema_version": 3,
  "report": { "ok": true, "warnings": [], "errors": [], ... },
  "units": "metric" | "imperial",
  "scan_settings": {
    "max_ceiling_variance_m": 1.5
  },
  "project": {
    "name": "...", "address": "...", "client": "...",
    "company": "...", "drawing_number": "...",
    "north_deg": 0.0, "print_north": true,
    "drawing_register": [
      {"rev": "A", "date": "2026-05-04", "by": "HH", "note": "..."}
    ]
  },
  "grid": { "min_x": ..., "max_x": ..., "min_z": ..., "max_z": ...,
            "pixels_per_metre": 150, "width": ..., "height": ... },
  "height_summary": { "min_y": ..., "max_y": ..., "median_y": ..., ... },
  "room": [[x, z], ...] | null,
  "room_heatmap": { "stats": {...}, "heatmap_png_b64": "...",
                    "heatmap_bbox_px": [x0, y0, x1, y1], ... } | null,
  "main": {
    "polygon": [[x, z], ...],
    "holes_polygons": [[[x, z], ...], ...],   // column rings inside main
    "label": "Main Ceiling (1)",
    "notes": "white plaster",
    "stats": { "mean_y": ..., "std_y": ..., "valid_frac": ..., ... },
    "heatmap_png_b64": "...",
    "heatmap_bbox_px": [...],
    "tint": "#80cbc4"
  } | null,
  "regions": [
    {
      "id": 0,
      "label": "Ceiling Region (2)",
      "notes": "oak battens",
      "polygon": [[x, z], ...],
      "holes_polygons": [...],
      "relative_y": 0.273,        // metres above main; null if no datum
      "stats": { ... },
      "heatmap_png_b64": "...",
      "heatmap_bbox_px": [...],
      "tint": "#ff7043"
    }, ...
  ],
  "obstructions": [
    {
      "id": 0,
      "kind": "column",
      "label": "Column (1)",
      "polygon": [[x, z], ...]
    }, ...
  ],
  "topology": {
    "vertices": [[x, z], ...],
    "edges": [
      { "id": 0, "vertices": [vid, ...], "faces": [fid_left, fid_right] }
    ],
    "faces": [
      {
        "id": 0,
        "kind": "main" | "region",
        "ring": [{"edge": eid, "rev": bool}, ...],
        "holes": [[{"edge": eid, "rev": bool}, ...], ...],
        "polygon": [[x, z], ...],
        "holes_polygons": [...],
        "label": "...", "notes": "...", "tint": "#...",
        "relative_y": 0.0, "stats": {...},
        "heatmap_png_b64": "...", "heatmap_bbox_px": [...]
      }, ...
    ]
  } | null,
  "snapped": false | true,
  "auto_detected": false | true
}
```

Pre-snap, `topology` is `null` and the user edits `main` /
`regions` / `obstructions` directly. After `POST /snap`, the topology
is the source of truth; `main` and `regions` become a derived view
that the server keeps in sync (so the existing frontend renderers
don't have to know about the topology). Direct polygon edits to
`main` / `regions[]` return 409 once a topology exists — callers
must use the topology endpoints or `DELETE /topology` first.

## Coordinate convention recap

- ARKit world space: X right, **Y up**, Z towards the back wall (depends
  on capture).
- The OBJ lives in *mesh space* — apply
  `inv(alignmentTransform).reshape(4, 4, order='F')` to vertices to
  bring them into ARKit world space.
- `PlanGrid.world_to_px` mirrors X (so the plan reads with floor-plan
  handedness) and flips Z (so +Z is up on the page). Every server
  rasteriser and the frontend's coord helpers go through this, so
  changing the convention is a single-file edit.

Full derivation, including failure modes ("camera in impossible
position", off-by-43°-rotation), in
[`docs/polycam_coordinate_system.md`](docs/polycam_coordinate_system.md).

## Where session data lives

```
sessions/
  <id>/
    upload/              the staged Polycam files
    out/
      ceiling.jpg
      height.npy         float32 H×W, NaN outside any down-face
      wall_edges.npy     bool H×W, lazily rebuilt by the lab
      plan.json
debug/
  ground_truth/<id>.json   committed; lab scoring target
  results/<id>.{json,csv}  regenerated each lab run; gitignored
debug_out/                 PNG outputs of segment_lab + experiments
```

Everything except `debug/ground_truth/` is regenerated from the
Polycam folder + a single `ceiling-rcp-init` invocation.
