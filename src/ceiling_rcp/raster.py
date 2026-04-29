"""Top-down textured raster of a Polycam mesh.

Lifted from the original ``polycam_ceiling_render.py`` prototype and
trimmed: same triangle-by-triangle UV → XZ affine warp, but operates on a
shared :class:`PlanGrid` so the resulting image lines up pixel-for-pixel
with the segmentation masks produced in :mod:`ceiling_rcp.planes`.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .mesh import Mesh
from .planes import PlanGrid


def render_textured_topdown(
    mesh: Mesh,
    face_indices: np.ndarray,
    grid: PlanGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise selected triangles into a top-down BGR image on ``grid``.

    Returns ``(canvas, height_map)`` where:

    - ``canvas`` is the textured BGR render (uint8, H×W×3).
    - ``height_map`` is the per-pixel world-Y of the lowest down-facing
      triangle covering that pixel (float32, H×W), with ``+inf`` where no
      face contributed. Convert to NaN before saving if downstream code
      expects that.

    The height map is the same z-buffer used to win triangle-vs-triangle
    contests during rendering, so the colour and the height are always
    consistent at every pixel.
    """
    W, H = grid.width, grid.height
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    textures: list[np.ndarray] = []
    for path in mesh.mat_textures:
        if path is None:
            textures.append(np.zeros((16, 16, 3), dtype=np.uint8))
        else:
            tex = cv2.imread(str(path), cv2.IMREAD_COLOR)
            textures.append(tex if tex is not None else np.zeros((16, 16, 3), dtype=np.uint8))

    n = len(face_indices)
    for k, fi in enumerate(face_indices):
        mat = int(mesh.FM[fi])
        if mat < 0 or mat >= len(textures):
            continue
        tex = textures[mat]
        th, tw = tex.shape[:2]

        vi = mesh.FV[fi]
        wx = mesh.V[vi, 0]; wy = mesh.V[vi, 1]; wz = mesh.V[vi, 2]
        u, v = grid.world_to_px(wx, wz)
        dst = np.stack([u, v], axis=1).astype(np.float32)

        ti = mesh.FT[fi]
        if ti.max() >= mesh.VT.shape[0]:
            continue
        uv = mesh.VT[ti]
        sx = uv[:, 0] * tw
        sy = (1.0 - uv[:, 1]) * th
        src = np.stack([sx, sy], axis=1).astype(np.float32)

        x0 = int(math.floor(dst[:, 0].min()))
        y0 = int(math.floor(dst[:, 1].min()))
        x1 = int(math.ceil(dst[:, 0].max())) + 1
        y1 = int(math.ceil(dst[:, 1].max())) + 1
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, W); y1 = min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        w = x1 - x0; h = y1 - y0

        dst_local = dst - np.array([x0, y0], dtype=np.float32)
        try:
            M = cv2.getAffineTransform(src, dst_local)
        except cv2.error:
            continue
        patch = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255)

        cy = float(wy.mean())
        zslice = zbuf[y0:y1, x0:x1]
        win = mask > 0
        winner = win & (cy < zslice)
        if not winner.any():
            continue
        cslice = canvas[y0:y1, x0:x1]
        cslice[winner] = patch[winner]
        zslice[winner] = cy
        canvas[y0:y1, x0:x1] = cslice
        zbuf[y0:y1, x0:x1] = zslice

    return canvas, zbuf


def overlay_regions(
    base: np.ndarray,
    regions,                # Sequence[PlaneRegion]
    grid: PlanGrid,
    *,
    alpha: float = 0.45,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Tint each region with a per-cluster colour and outline it."""
    if palette is None:
        palette = [
            (255, 102,  51),
            ( 51, 187, 255),
            (102, 255, 153),
            (255, 200,  51),
            (200,  51, 255),
            (255,  85, 200),
            (153, 255, 255),
            ( 85, 255,  85),
        ]

    out = base.copy()
    overlay = np.zeros_like(base)
    for r in regions:
        col = palette[r.cluster_idx % len(palette)]
        overlay[r.mask > 0] = col

    blended = cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)

    # Draw outlines with full saturation on top of the blend.
    for r in regions:
        col = palette[r.cluster_idx % len(palette)]
        contours, _ = cv2.findContours(r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(blended, contours, -1, col, max(1, int(grid.pixels_per_metre * 0.01)))
    return blended


def annotate_heights(
    img: np.ndarray, regions, *, font_scale: float = 0.6,
) -> np.ndarray:
    """Drop a ``Y=…m  A=…m²`` label at the centroid of each region."""
    out = img.copy()
    for r in regions:
        ys, xs = np.where(r.mask > 0)
        if xs.size == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        text = f"{r.y:.2f} m  {r.area_m2:.1f} m^2"
        cv2.putText(out, text, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


__all__ = [
    "render_textured_topdown",
    "overlay_regions",
    "annotate_heights",
]
