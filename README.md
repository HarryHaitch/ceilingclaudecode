# ceiling-rcp

Reflected Ceiling Plan (RCP) generator from a Polycam textured mesh.

You scan a room with the Polycam app, drop the export into this tool,
trace (or auto-detect) the ceiling regions in a browser-based editor,
and get back a dimensioned PDF that reads like an architectural plan:
each ceiling polygon coloured by relative height, room outline tagged
with edge lengths, optional notes per region.

The repo also contains a **segmentation lab** (`debug/`) for iterating
on auto-detect algorithms against a hand-drawn ground truth.

## Quick start

```bash
# install (use the module form if `pip` isn't on PATH)
python3 -m pip install -e .                              # core deps
python3 -m pip install -e ".[dev]"                       # adds scikit-image for the lab

# verify the install points at this checkout (not a stray editable install
# from another path)
python3 -c "import ceiling_rcp; print(ceiling_rcp.__file__)"

# terminal 1 — long-running server
ceiling-rcp-server --port 8765

# terminal 2 — bootstrap a session from a Polycam folder
ceiling-rcp-init "Scan data/Lachys Polycam"
# → http://127.0.0.1:8765/?session=<id>

# open that URL in a browser
```

The browser walks you through:

1. **Pick units** — first control in the side panel: mm/m or ft-in.
   Affects PDF labels, scale bar, and side-panel display.
2. **Set scan settings** — `Max ceiling height variance` (default
   1.5 m) controls how far below the dominant ceiling the renderer
   will keep geometry. Lower = aggressive furniture culling; higher
   = handles vaulted / cathedral rooms. Re-render after changing.
3. **Trace polygons** — room outline → main ceiling → ceiling
   regions (+mm recesses / −mm bulkheads) → optional column
   obstructions. Hold **Shift** while clicking or dragging to snap
   to 0° / 90° relative to the previous edge. Click a vertex to
   drag; press **Delete** with a vertex hovered to remove it;
   "Insert vertex" tool button to add one mid-edge.
4. **Snap polygons** — pushes/pulls every polygon's borders so they
   share clean edges and tile the room without gaps or overlap.
   Adjacent edges become *one* shared topology edge: dragging it
   moves both faces.
5. **Fill in project info** — name, address, client, company,
   drawing number, north heading, drawing register. All flow into
   the title block of the exported PDF.
6. **Download PDF** — A1 landscape, title block on right, ortho
   thumbnail, scale bar, north arrow, ceiling-zone legend.

Each polygon shows a brightness shading inside it: pixels scanned
higher than that polygon's mean tint lighter, lower tint darker.
A clean flat ceiling is uniformly tinted; a clipped bulkhead jumps
out as a visibly different shade. Click any zone's swatch in the
side panel to recolour it.

## Where things live

| Path | What |
| --- | --- |
| `src/ceiling_rcp/mesh.py` | OBJ / MTL parser, `alignmentTransform` handling, `ceiling_face_mask` (60° cone + height-band filter) |
| `src/ceiling_rcp/raster.py` | Top-down textured render + per-pixel height map (z-buffer) |
| `src/ceiling_rcp/analyse.py` | Per-polygon mean Y, σ, deviation heatmap PNG |
| `src/ceiling_rcp/planes.py` | `PlanGrid` (XZ pixel ↔ world conversion); legacy auto-segmentation kept for the debug CLI |
| `src/ceiling_rcp/polygons.py` | Mask-to-polygon, polygon edit primitives |
| `src/ceiling_rcp/topology.py` | Planar graph builder (vertices/edges/faces with holes), post-snap edits |
| `src/ceiling_rcp/polylabel.py` | Pole-of-inaccessibility (Mapbox polylabel) for region label placement, hole-aware |
| `src/ceiling_rcp/units.py` | Metric ↔ imperial display formatting (length, height delta) |
| `src/ceiling_rcp/server.py` | FastAPI app: process, room/main/region/obstruction edit, scan-settings/units/project, snap, topology edits, A1 PDF |
| `src/ceiling_rcp/init_session.py` | `ceiling-rcp-init` — staging a scan into a server session without going through the browser |
| `src/ceiling_rcp/cli.py` | `ceiling-rcp` legacy debug CLI |
| `src/ceiling_rcp/static/` | Single-page canvas editor (no framework) — units toggle, project panel, swatch picker |
| `debug/` | Segmentation lab — algorithms, scoring, experiment harness |
| `docs/` | Coordinate-system reference and other design notes |

For the data-flow walkthrough see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For the experiment lab see [`debug/README.md`](debug/README.md).

## CLIs

| Command | Source | Purpose |
| --- | --- | --- |
| `ceiling-rcp-server [--port 8765]` | `server.py` | Run the FastAPI app + serve the canvas editor |
| `ceiling-rcp-init <scan_dir>` | `init_session.py` | Stage a Polycam folder into a server session and pre-process the height map |
| `ceiling-rcp <scan_dir>` | `cli.py` | Legacy: write debug renders + auto-segmented JSON without the server |
| `python -m debug.snapshot_truth <id>` | `debug/snapshot_truth.py` | Capture polygons drawn in the browser as ground truth |
| `python -m debug.segment_lab <id> --algo X` | `debug/segment_lab.py` | Run one algorithm and dump an overlay PNG |
| `python -m debug.experiments <id>` | `debug/experiments.py` | Sweep a parameter grid, score against truth, print a leaderboard |

## Web API

All endpoints live under `/api/sessions/`. See [`ARCHITECTURE.md`](ARCHITECTURE.md)
for the full list with bodies; the short version:

```
POST   /api/sessions                                  upload
POST   /api/sessions/{id}/process                     render + height map
GET    /api/sessions/{id}/plan                        full plan json
GET    /api/sessions/{id}/image/ceiling.jpg           textured render
GET    /api/sessions/{id}/export                      full plan json, download form

GET / PUT /api/sessions/{id}/project                  title-block + drawing register + north
PUT    /api/sessions/{id}/units                       "metric" | "imperial"
PUT    /api/sessions/{id}/scan_settings               max_ceiling_variance_m → re-renders

PUT    /api/sessions/{id}/room                        set / clear room polygon
PUT    /api/sessions/{id}/main                        set / clear main ceiling
PUT    /api/sessions/{id}/main/notes                  update main's notes
PUT    /api/sessions/{id}/main/tint                   recolour main
POST   /api/sessions/{id}/region                      add ceiling region
PUT    /api/sessions/{id}/region/{rid}                update polygon / label / notes / tint
DELETE /api/sessions/{id}/region/{rid}                delete region
POST   /api/sessions/{id}/obstruction                 add column
PUT    /api/sessions/{id}/obstruction/{oid}           update column polygon / label
DELETE /api/sessions/{id}/obstruction/{oid}           delete column

POST   /api/sessions/{id}/snap                        build planar topology, share edges
DELETE /api/sessions/{id}/topology                    un-snap
PUT    /api/sessions/{id}/topology/vertices           drag vertex(es)
POST   /api/sessions/{id}/topology/edge/{eid}/insert_vertex   split edge
DELETE /api/sessions/{id}/topology/vertex/{vid}       remove vertex
PUT    /api/sessions/{id}/topology/face/{fid}/notes   per-face notes
PUT    /api/sessions/{id}/topology/face/{fid}/tint    recolour face

POST   /api/sessions/{id}/auto_detect                 histogram-cluster auto-fill (legacy, no UI)
GET    /api/sessions/{id}/pdf                         A1 architectural PDF download
```

## Coordinate system

Polycam's OBJ lives in an axis-aligned mesh space, separate from the
ARKit world space the camera poses use. To bring the mesh into ARKit
world space (Y up, floor near 0), apply the **inverse** of
`alignmentTransform` from `mesh_info.json`. The full convention with
worked examples is in
[`docs/polycam_coordinate_system.md`](docs/polycam_coordinate_system.md).

The plan view uses the **Reflected Ceiling Plan** convention (X
mirrored relative to "looking up at the ceiling", Z up on page) — done
once in `PlanGrid.world_to_px`, so every downstream raster, polygon,
heatmap and PDF inherits it automatically.

## Project state

Per-session artefacts (`sessions/`, `out_*`, `debug_out/`,
`debug/results/`) and raw scans (`Scan data/`) are git-ignored — the
whole pipeline regenerates them from the Polycam input + a single
`ceiling-rcp-init` invocation. Captured ground truth lives under
`debug/ground_truth/` and **is** committed.

## Contact

H2 Engineering — haris@h2engineering.com.au
