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
# install
pip install -e .                                         # also installs scikit-image via dev deps if you want the lab
pip install -e ".[dev]"                                  # explicit form

# terminal 1 — long-running server
ceiling-rcp-server --port 8765

# terminal 2 — bootstrap a session from a Polycam folder
ceiling-rcp-init "Scan data/Lachys Polycam"
# → http://127.0.0.1:8765/?session=<id>

# open that URL in a browser
```

The browser walks you through:

1. **Trace room outline** — click vertices, click first to close.
2. **Trace main ceiling** OR click **Auto-detect from height map** —
   defines the height datum.
3. **Add ceiling regions** — recesses (+mm above main), bulkheads
   (−mm below main).
4. **Snap polygons** — push/pull every polygon's borders so they share
   clean edges and tile the room without gaps or overlap.
5. **Download PDF** — coloured masks, height tags, dimensioned room
   outline.

Each polygon shows a brightness shading inside it: pixels scanned
higher than that polygon's mean tint lighter, lower tint darker.
A clean flat ceiling is uniformly tinted; a clipped bulkhead jumps
out as a visibly different shade.

## Where things live

| Path | What |
| --- | --- |
| `src/ceiling_rcp/mesh.py` | OBJ / MTL parser, `alignmentTransform` handling, folder validator |
| `src/ceiling_rcp/raster.py` | Top-down textured render + per-pixel height map (z-buffer) |
| `src/ceiling_rcp/analyse.py` | Per-polygon mean Y, σ, deviation heatmap PNG |
| `src/ceiling_rcp/planes.py` | `PlanGrid` (XZ pixel ↔ world conversion); legacy auto-segmentation kept for the debug CLI |
| `src/ceiling_rcp/polygons.py` | Mask-to-polygon, polygon edit primitives |
| `src/ceiling_rcp/server.py` | FastAPI app: upload, process, room/main/region edit, snap, auto-detect, PDF |
| `src/ceiling_rcp/init_session.py` | `ceiling-rcp-init` — staging a scan into a server session without going through the browser |
| `src/ceiling_rcp/cli.py` | `ceiling-rcp` legacy debug CLI |
| `src/ceiling_rcp/static/` | Single-page canvas editor (no framework) |
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
POST   /api/sessions                        upload
POST   /api/sessions/{id}/process           render + height map
GET    /api/sessions/{id}/plan              full plan json
GET    /api/sessions/{id}/image/ceiling.jpg textured render

PUT    /api/sessions/{id}/room              set / clear room polygon
PUT    /api/sessions/{id}/main              set / clear main ceiling
POST   /api/sessions/{id}/region            add ceiling region
PUT    /api/sessions/{id}/region/{rid}      update polygon / label / notes
DELETE /api/sessions/{id}/region/{rid}      delete region
PUT    /api/sessions/{id}/main/notes        update main's notes

POST   /api/sessions/{id}/auto_detect       histogram-cluster auto-fill
POST   /api/sessions/{id}/snap              Voronoi-style border tidy

GET    /api/sessions/{id}/pdf               architectural PDF download
GET    /api/sessions/{id}/export            full plan json, download form
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
