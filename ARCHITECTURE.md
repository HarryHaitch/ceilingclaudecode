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

### `server.py`

The FastAPI app. One file because the surface is small and every
endpoint shares the same load-plan / mutate / save-plan pattern. Key
pieces:

- `process_session` — runs `inspect_folder` + `load_mesh` +
  `render_textured_topdown` + saves `ceiling.jpg`, `height.npy`,
  `plan.json`. No segmentation by default.
- `_analyse_and_pack` — wraps `analyse.analyse_polygon` +
  `_heatmap_from_mask` so every polygon endpoint returns the same
  `{stats, heatmap_bbox_px, heatmap_png_b64, ...}` shape.
- `api_set_room` — also computes a *room* heatmap (white tint, ±15 cm
  range) so the user can see height variance across the whole room
  before drawing anything inside it.
- `api_auto_detect` — the histogram-peaks → median-filter →
  per-cluster CC → coverage-filter → same-cluster absorption →
  band-restricted-stats pipeline, all in one function. ~150 lines, no
  intermediate state needed.
- `api_snap` — Voronoi assignment: every room pixel goes to whichever
  drawn polygon owns it (region-drawn pixels win first, then main's
  drawn area, then nearest by distance transform). Polygons are
  re-extracted from each label's mask via morph-close + DP simplify.
- `api_pdf` — matplotlib backend. Polygons filled in their tint
  colour, label + relative height + notes at centroid, room outline
  edges with mm dimensions.

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

The single object every edit mutates and every read returns:

```json
{
  "session_id": "abc123def456",
  "report": { "ok": true, "warnings": [], "errors": [], ... },
  "grid": { "min_x": ..., "max_x": ..., "min_z": ..., "max_z": ...,
            "pixels_per_metre": 150, "width": ..., "height": ... },
  "height_summary": { "min_y": ..., "max_y": ..., "median_y": ..., ... },
  "room": [[x, z], ...] | null,
  "room_heatmap": { "stats": {...}, "heatmap_png_b64": "...",
                    "heatmap_bbox_px": [x0, y0, x1, y1], ... } | null,
  "main": {
    "polygon": [[x, z], ...],
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
      "relative_y": 0.273,        // metres above main; null if no datum
      "stats": { ... },
      "heatmap_png_b64": "...",
      "heatmap_bbox_px": [...],
      "tint": "#ff7043"
    }, ...
  ],
  "snapped": false | true,
  "auto_detected": false | true
}
```

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
