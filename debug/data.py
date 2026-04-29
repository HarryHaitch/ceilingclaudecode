"""Load and cache the inputs every algorithm needs.

For a given ``session_id`` (one of the dirs under ``./sessions/``), this
returns a :class:`SegInput` containing:

- ``height_map`` — float32 H×W per-pixel ceiling Y, NaN where there's no
  downward-facing geometry. Already cached by the server.
- ``room_mask`` — bool H×W, the user-drawn room outline (or the union of
  valid heightmap pixels if no room is set yet).
- ``wall_edges`` — bool H×W, the XZ projection of *near-horizontal*
  mesh triangles (those whose normal lies near the XZ plane). This is
  the strongest physical signal for ceiling-region boundaries: every
  bulkhead / recess / coffer transition has a vertical face whose XZ
  trace marks exactly where the regions divide. Computed once per
  session and cached as ``out/wall_edges.npy``.
- ``canvas`` — the textured top-down BGR render, used as the base layer
  for overlay PNGs.
- ``grid`` — the same :class:`PlanGrid` the server uses, so coordinate
  conversions match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ceiling_rcp.analyse import polygon_to_mask
from ceiling_rcp.mesh import inspect_folder, load_mesh
from ceiling_rcp.planes import PlanGrid


SESSIONS_DIR = Path("sessions")


@dataclass
class SegInput:
    height_map: np.ndarray
    room_mask: np.ndarray
    wall_edges: np.ndarray
    canvas: np.ndarray
    grid: PlanGrid
    session_id: str


def load_session(session_id: str, *, refresh_walls: bool = False) -> SegInput:
    sd = SESSIONS_DIR / session_id
    if not sd.is_dir():
        raise SystemExit(f"unknown session {session_id}; available: "
                         f"{sorted(p.name for p in SESSIONS_DIR.iterdir() if p.is_dir())}")

    plan_path = sd / "out" / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"session {session_id} not processed yet — run "
                         f"`ceiling-rcp-init <scan>` first.")
    plan = json.loads(plan_path.read_text())
    g = plan["grid"]
    grid = PlanGrid(
        min_x=g["min_x"], max_x=g["max_x"],
        min_z=g["min_z"], max_z=g["max_z"],
        pixels_per_metre=g["pixels_per_metre"],
    )

    height_map = np.load(sd / "out" / "height.npy")
    canvas_path = sd / "out" / "ceiling.jpg"
    canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"missing ceiling.jpg under {canvas_path}")

    if plan.get("room"):
        room_mask = polygon_to_mask([tuple(p) for p in plan["room"]], grid) > 0
    else:
        room_mask = ~np.isnan(height_map)

    walls_path = sd / "out" / "wall_edges.npy"
    if refresh_walls or not walls_path.exists():
        wall_edges = compute_wall_edges(sd, grid)
        np.save(walls_path, wall_edges.astype(np.uint8))
    else:
        wall_edges = np.load(walls_path).astype(bool)

    return SegInput(
        height_map=height_map,
        room_mask=room_mask,
        wall_edges=wall_edges,
        canvas=canvas,
        grid=grid,
        session_id=session_id,
    )


def compute_wall_edges(
    session_dir: Path,
    grid: PlanGrid,
    *,
    max_horizontal_tilt_deg: float = 15.0,
    max_vertical_span_m: float = 0.6,
    ceiling_band_m: float = 0.8,
    dilate_px: int = 1,
) -> np.ndarray:
    """Rasterise the XZ projection of *ceiling-relevant* horizontal triangles.

    A near-horizontal triangle has its normal close to the XZ plane.
    Without further filtering this catches every wall in the building,
    not just the bulkhead / coffer / recess transitions we care about.
    Two extra filters keep only the useful ones:

    - ``max_vertical_span_m``: the triangle's Y extent must be small.
      A floor-to-ceiling wall spans 2-3 m and is dropped; a bulkhead
      side typically spans 5-30 cm and is kept.
    - ``ceiling_band_m``: at least one vertex must sit within
      ±``ceiling_band_m`` of the median ceiling height (read from the
      cached height map). This drops walls whose tops or bottoms are
      far below the ceiling.

    The accepted triangles are filled into a binary mask and dilated
    so each thin sliver becomes a connected line.
    """
    upload = session_dir / "upload"
    rep = inspect_folder(upload)
    if not rep.ok:
        raise SystemExit(f"cannot load mesh from {upload}: {rep.errors}")
    mesh = load_mesh(rep)

    H, W = grid.height, grid.width
    edges = np.zeros((H, W), dtype=np.uint8)

    n = mesh.face_normals()
    threshold = np.sin(np.deg2rad(max_horizontal_tilt_deg))
    horiz = np.abs(n[:, 1]) < threshold

    tris_y = mesh.V[mesh.FV][:, :, 1]
    span = tris_y.max(axis=1) - tris_y.min(axis=1)
    short = span < max_vertical_span_m

    height_map = np.load(session_dir / "out" / "height.npy")
    ceiling_y = float(np.nanmedian(height_map))
    near_ceiling = (
        np.abs(tris_y - ceiling_y).min(axis=1) < ceiling_band_m
    )

    keep = horiz & short & near_ceiling
    keep_idx = np.where(keep)[0]
    print(f"  wall-edges: {len(keep_idx)} kept of {len(n)} "
          f"(horizontal {int(horiz.sum())}, "
          f"short-span {int((horiz & short).sum())}, "
          f"near-ceiling {int(keep.sum())})")

    if len(keep_idx) == 0:
        return edges.astype(bool)

    tris = mesh.V[mesh.FV[keep_idx]]
    u, v = grid.world_to_px(tris[..., 0], tris[..., 2])
    pts = np.stack([u, v], axis=-1).astype(np.int32)
    cv2.fillPoly(edges, pts, 1)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        edges = cv2.dilate(edges,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
                           iterations=1)
    return edges.astype(bool)


__all__ = ["SegInput", "load_session", "compute_wall_edges"]
