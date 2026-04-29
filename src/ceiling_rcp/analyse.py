"""Per-polygon height analysis.

A user draws a polygon over the textured ceiling render. We:

1. Rasterise the polygon to the same pixel grid as the height map.
2. Compute the mean world-Y of the height-map pixels inside the polygon,
   excluding pixels with no LiDAR coverage (NaN).
3. Build a deviation heatmap inside the polygon — each pixel's
   ``height - mean`` mapped to a diverging colour ramp, encoded as RGBA
   PNG bytes for direct overlay on the canvas.

The heightmap is whatever ``raster.render_textured_topdown`` produced as
its z-buffer (``+inf`` where no face contributed). We replace ``inf``
with NaN for the analysis so it doesn't pollute the average.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np

from .planes import PlanGrid


@dataclass
class PolygonAnalysis:
    mean_y: float           # in world metres (or NaN if polygon has no valid pixels)
    std_y: float            # standard deviation in metres
    valid_frac: float       # fraction of polygon-interior pixels that had height data
    n_valid_px: int
    n_total_px: int
    min_y: float
    max_y: float

    def as_dict(self) -> dict:
        return {
            "mean_y": _to_jsonable(self.mean_y),
            "std_y": _to_jsonable(self.std_y),
            "valid_frac": float(self.valid_frac),
            "n_valid_px": int(self.n_valid_px),
            "n_total_px": int(self.n_total_px),
            "min_y": _to_jsonable(self.min_y),
            "max_y": _to_jsonable(self.max_y),
        }


def _to_jsonable(x: float) -> float | None:
    return None if np.isnan(x) or np.isinf(x) else float(x)


def height_map_to_storage(zbuf: np.ndarray) -> np.ndarray:
    """Convert ``+inf`` placeholders to NaN so the array round-trips through
    standard tooling without surprises."""
    h = zbuf.astype(np.float32, copy=True)
    h[~np.isfinite(h)] = np.nan
    return h


def polygon_to_mask(polygon_world, grid: PlanGrid) -> np.ndarray:
    """Rasterise an XZ polygon (list of ``(x, z)`` tuples in metres) to a
    binary uint8 mask on ``grid``."""
    H, W = grid.height, grid.width
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polygon_world:
        return mask
    pts = np.array(polygon_world, dtype=np.float64)
    u, v = grid.world_to_px(pts[:, 0], pts[:, 1])
    poly_px = np.stack([u, v], axis=-1).astype(np.int32)
    cv2.fillPoly(mask, [poly_px], 255)
    return mask


def analyse_polygon(
    polygon_world, height_map: np.ndarray, grid: PlanGrid,
) -> PolygonAnalysis:
    mask = polygon_to_mask(polygon_world, grid) > 0
    n_total = int(mask.sum())
    if n_total == 0:
        return PolygonAnalysis(
            mean_y=float("nan"), std_y=float("nan"),
            valid_frac=0.0, n_valid_px=0, n_total_px=0,
            min_y=float("nan"), max_y=float("nan"),
        )
    sample = height_map[mask]
    valid = ~np.isnan(sample)
    if not valid.any():
        return PolygonAnalysis(
            mean_y=float("nan"), std_y=float("nan"),
            valid_frac=0.0, n_valid_px=0, n_total_px=n_total,
            min_y=float("nan"), max_y=float("nan"),
        )
    vals = sample[valid].astype(np.float64)
    return PolygonAnalysis(
        mean_y=float(vals.mean()),
        std_y=float(vals.std()),
        valid_frac=float(valid.sum()) / float(n_total),
        n_valid_px=int(valid.sum()),
        n_total_px=n_total,
        min_y=float(vals.min()),
        max_y=float(vals.max()),
    )


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    if len(s) == 3:
        s = "".join(c + c for c in s)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def deviation_heatmap_png(
    polygon_world,
    height_map: np.ndarray,
    grid: PlanGrid,
    *,
    mean_y: float | None = None,
    range_m: float = 0.05,
    alpha: float = 0.55,
    tint: str = "#80cbc4",
) -> tuple[bytes, tuple[int, int, int, int]]:
    """Render an RGBA PNG of ``(height - mean_y)`` clamped to ±``range_m``,
    masked to the polygon, in the polygon's own ``tint`` colour ramped
    by lightness. Pixels at the mean render at base lightness; pixels
    above the mean render lighter, below the mean darker.

    Pixels outside the polygon — or inside but missing height data —
    are fully transparent. Returns ``(png_bytes, (x0, y0, x1, y1))``
    in grid pixel coords so the frontend can position the overlay
    against the textured render.
    """
    mask = polygon_to_mask(polygon_world, grid) > 0
    if not mask.any():
        return b"", (0, 0, 0, 0)

    if mean_y is None:
        sample = height_map[mask]
        valid_sample = ~np.isnan(sample)
        mean_y = float(sample[valid_sample].mean()) if valid_sample.any() else 0.0

    dev = (height_map - float(mean_y)) / max(range_m, 1e-6)
    dev = np.clip(dev, -1.0, 1.0)

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub_dev = dev[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]
    sub_height = height_map[y0:y1, x0:x1]
    valid = sub_mask & ~np.isnan(sub_height)

    # Map dev ∈ [-1, 1] to lightness ∈ [0.35, 1.0] of the tint colour:
    #   dev = -1 → 0.35× tint (darkest)
    #   dev =  0 → 0.70× tint (base, the polygon's "average" appearance)
    #   dev = +1 → 1.00× tint with white blend (lightest)
    base_r, base_g, base_b = _hex_to_rgb(tint)

    # Lightness multiplier centred on 0.7
    lightness = 0.70 + 0.30 * sub_dev   # 0.40 … 1.00
    lightness = np.clip(lightness, 0.35, 1.0)

    # Above mean: blend toward white (255).
    boost = sub_dev.clip(0, 1)  # 0…1
    r = (base_r * lightness) + (255 - base_r * lightness) * (boost * 0.5)
    g = (base_g * lightness) + (255 - base_g * lightness) * (boost * 0.5)
    b = (base_b * lightness) + (255 - base_b * lightness) * (boost * 0.5)

    h, w = sub_dev.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = (valid * int(round(alpha * 255))).astype(np.uint8)

    # cv2 expects BGRA when encoding a 4-channel image.
    rgba_bgra = rgba[..., [2, 1, 0, 3]]
    ok, buf = cv2.imencode(".png", rgba_bgra)
    if not ok:
        return b"", (0, 0, 0, 0)
    return buf.tobytes(), (x0, y0, x1, y1)


__all__ = [
    "PolygonAnalysis",
    "polygon_to_mask",
    "analyse_polygon",
    "deviation_heatmap_png",
    "height_map_to_storage",
]
