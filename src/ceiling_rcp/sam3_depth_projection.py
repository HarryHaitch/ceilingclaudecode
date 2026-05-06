"""Depth-map-based projection: image pixel → ARKit world point in one step.

Replaces the old ray-cast-through-ceiling-height pipeline. Polycam ships
per-keyframe LiDAR depth maps under ``keyframes/depth/<id>.png`` (uint16
millimetres at 192 × 256, no zero pixels) and confidence maps under
``keyframes/confidence/<id>.png`` (uint8 at 192 × 256, ARKit
``{0=low, 54=med, 255=high}`` levels). Both are stored in the *original*
1024 × 768 corrected-image orientation — the same orientation as
``corrected_images/<id>.jpg`` and the intrinsics in
``corrected_cameras/<id>.json``. The 192 × 256 → 1024 × 768 upscale is
exactly 4× in each axis.

Per the user's confirmed ceiling_server.py convention:

    x_c =  (u - cx) / fx · depth
    y_c = -(v - cy) / fy · depth        # ARKit Y up, image v down
    z_c = -depth                         # ARKit camera looks along -Z
    P_world = R @ P_cam + t              # R is cam-to-world

This module only handles **the original-image orientation**. Callers
must back-rotate any rotated-frame polygon (CW90 SAM 3 input coords)
into the source orientation *before* calling — see
``Detection.polygon_src`` in ``sam3_clusters.py`` which already does
this at parse time. The depth map and intrinsics never see the rotated
frame.

The geometry is camera-only — `height.npy` and the OBJ mesh play no
role in the projection. They keep their other roles (the ortho image,
the user-traced region heights) — but the SAM 3 polygon → world point
mapping is now driven purely by what the LiDAR sensor recorded for each
pixel SAM 3 cared about.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .planes import PlanGrid
from .sam3_projection import PolycamCamera


# ARKit confidence levels. Polycam stores them as 0/54/255 — we map to
# 0/1/2 internally so callers think in terms of ARKit's enum rather than
# Polycam's PNG encoding.
_ARKIT_CONF_LEVELS = {0: 0, 54: 1, 255: 2}

# Default minimum confidence: drop only "low" pixels (level 0). "Medium"
# (level 1) is plenty for ceiling fixtures, which are usually ~1–3 m
# from the camera and well within LiDAR's reliable range.
DEFAULT_MIN_CONFIDENCE = 1


# ─── DEPTH MAP I/O ────────────────────────────────────────────────────────

def load_depth(
    image_id: str, *,
    depth_dir: Path,
    confidence_dir: Path,
    target_size: tuple[int, int],     # (W, H) — full corrected-image size
    min_confidence: int = DEFAULT_MIN_CONFIDENCE,
) -> np.ndarray | None:
    """Load + upscale a Polycam depth map to ``target_size`` (W, H) and
    return depth in **metres** as a float32 array, with NaN where depth
    is zero or the LiDAR confidence is below ``min_confidence``.

    Returns ``None`` if either PNG is missing or the depth map has no
    valid pixels at the requested confidence floor.
    """
    dpath = depth_dir / f"{image_id}.png"
    cpath = confidence_dir / f"{image_id}.png"
    if not dpath.exists() or not cpath.exists():
        return None
    d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
    c = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
    if d is None or c is None:
        return None
    if d.dtype != np.uint16:
        d = d.astype(np.uint16)

    # Map Polycam's 0/54/255 → 0/1/2 (ARKit levels).
    level = np.zeros_like(c, dtype=np.uint8)
    for raw, lvl in _ARKIT_CONF_LEVELS.items():
        level[c == raw] = lvl
    # Anything not in the map (rare) is treated as low.

    valid = (d > 0) & (level >= min_confidence)

    # Nearest-neighbour upscale to the corrected-image resolution. NN is
    # what we want for depth at low LiDAR resolutions — bilinear fakes
    # values at depth discontinuities.
    W, H = target_size
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)

    out = d.astype(np.float32) / 1000.0   # mm → m
    out[~valid] = np.nan
    if not np.isfinite(out).any():
        return None
    return out


# ─── PIXEL → WORLD ────────────────────────────────────────────────────────

def project_pixels_to_world(
    uv_src: np.ndarray,                 # (N, 2) original-frame px (u, v)
    depth_m: np.ndarray,                # (H, W) float32 metres, NaN if unknown
    cam: PolycamCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of source-image pixels to ARKit-space world points
    using the per-pixel LiDAR depth.

    Returns ``(P_world, ok)`` where ``P_world`` is (N, 3) (NaN where the
    depth lookup failed) and ``ok`` is a boolean (N,) marking the
    successful samples. Out-of-bounds pixels and pixels with NaN depth
    are marked invalid.
    """
    N = uv_src.shape[0]
    H, W = depth_m.shape
    u = uv_src[:, 0]
    v = uv_src[:, 1]
    # Round to nearest pixel + clamp to image bounds.
    ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = depth_m[vi, ui]
    ok = in_bounds & np.isfinite(d)

    # Pinhole inverse-projection (matches ceiling_server.py briefing).
    x_c =  (u - cam.cx) / cam.fx * d
    y_c = -(v - cam.cy) / cam.fy * d
    z_c = -d
    P_cam = np.stack([x_c, y_c, z_c], axis=-1)            # (N, 3)
    # Cam-to-world: P_world = R @ P_cam + t  (R columns are cam axes in world)
    P_world = P_cam @ cam.R.T + cam.t
    P_world[~ok] = np.nan
    return P_world, ok


def project_polygon_to_world(
    poly_src: np.ndarray,               # (N, 2) source-frame px
    depth_m: np.ndarray,
    cam: PolycamCamera,
    *,
    min_valid_frac: float = 0.4,
) -> tuple[np.ndarray, bool]:
    """Project every polygon vertex through depth + intrinsics. Drop the
    polygon when fewer than ``min_valid_frac`` of vertices have a
    confident depth sample (Polycam's confidence map drops at depth
    discontinuities, so a real fixture's outline will have *some* NaN
    vertices — but a polygon with mostly-NaN samples is sitting on
    a sky/window/black region and isn't trustworthy).

    Returns ``(P_world, ok)``. When ``ok`` is False, ``P_world`` is the
    raw projection (still useful for diagnostics) but the caller should
    skip the detection.
    """
    P_world, valid = project_pixels_to_world(poly_src, depth_m, cam)
    frac = float(valid.mean()) if valid.size else 0.0
    if frac < min_valid_frac:
        return P_world, False
    # Backfill NaN vertices from the convex centroid of the valid ones,
    # so the polygon stays closed when we project to ortho. This is
    # purely cosmetic — clusters use the centroid, not the polygon
    # vertices, for spatial grouping.
    if not valid.all():
        good = P_world[valid]
        centroid = good.mean(axis=0)
        P_world[~valid] = centroid
    return P_world, True


# ─── WORLD → ORTHO ────────────────────────────────────────────────────────

def world_to_ortho_px(
    P_world: np.ndarray,                # (N, 3) ARKit-space points
    grid: PlanGrid,
) -> np.ndarray:
    """Rasterise ARKit-space world points to ortho pixel coords using
    the main app's ``PlanGrid``. The grid is *already* in ARKit space
    — :func:`ceiling_rcp.mesh.load_mesh` calls ``apply_alignment_inv``
    by default, so by the time ``make_grid`` builds the bbox the
    vertices are in ARKit. No M_align step is needed (and applying
    one rotates the symbols relative to the ortho, by ~the alignment
    matrix's rotation angle — small for Lachy's scan, ~30° for
    Scan example 2, where it was the visible bug)."""
    u, v = grid.world_to_px(P_world[:, 0].astype(np.float32),
                             P_world[:, 2].astype(np.float32))
    return np.stack([u, v], axis=-1)


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "load_depth",
    "project_pixels_to_world",
    "project_polygon_to_world",
    "world_to_ortho_px",
]
