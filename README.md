# ceiling-rcp

Reflected Ceiling Plan (RCP) generator from a Polycam textured mesh.

The user uploads a Polycam Space export (mesh + textures + alignment), the
tool:

1. Filters mesh triangles whose normals point downward (the ceiling and
   anything overhead).
2. Clusters those triangles by world-Y into ceiling-plane candidates,
   absorbing the millimetre-scale jitter of LiDAR triangles.
3. Detects regularly-repeating height features inside a candidate region
   (battens, tiles, coffers) via 2D periodicity analysis and merges them
   into a single composite plane carrying pattern metadata.
4. Derives an editable room-bounding polygon from the union footprint of
   downward-facing triangles.
5. Serves an interactive 2D editor: adjust the footprint, accept / redraw
   recommended ceiling-plane polygons, set heights, export.

## Layout

```
src/ceiling_rcp/
  mesh.py              OBJ/MTL parse, alignmentTransform, folder validator
  planes.py            down-face filter, height clustering, periodic merge
  polygons.py          footprint hull, polygon edit ops
  raster.py            top-down textured render and plane overlays
  server.py            FastAPI: upload, process, edit, export
  cli.py               offline batch / debug
  static/index.html    canvas-based 2D editor
Scan data/             test fixtures (Polycam exports)
```

## Install

```bash
pip install -e .
```

## CLI

```bash
ceiling-rcp "Scan data/Lachys Polycam/Lachys line"
# → writes plan + planes + footprint into ./out_<scan_name>/
```

## Server

```bash
ceiling-rcp-server                 # localhost:8000
# open http://localhost:8000/
```

## Coordinate system

Polycam stores its OBJ in axis-aligned mesh space. To get into ARKit
world space (Y up, floor near 0), apply the inverse of
`alignmentTransform` from `mesh_info.json`. The full convention is
documented in
[`docs/polycam_coordinate_system.md`](docs/polycam_coordinate_system.md).

## Contact

H2 Engineering — haris@h2engineering.com.au
