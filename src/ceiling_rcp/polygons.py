"""Polygon construction and edit operations.

Two roles:

1. **Construction** — turn region masks into simplified polygons in metres
   (footprint and per-plane shapes), and approximate a tight outer hull
   around all downward-facing geometry for the editable room boundary.

2. **Editing** — primitives the frontend calls into via the server:
   insert vertex on edge, drag vertex, delete vertex, delete the chain
   between two selected vertices.

Polygons here are always lists of ``(x, z)`` metre tuples in world space.
The mask → polygon conversion uses the same :class:`PlanGrid` that the
segmentation built, so coordinates round-trip cleanly.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .planes import PlanGrid, PlaneRegion


Polygon = list[tuple[float, float]]   # [(x, z), …] metres in world XZ


# ─── MASK → POLYGON ───────────────────────────────────────────────────────────

def mask_to_polygon(
    mask: np.ndarray, grid: PlanGrid, *, simplify_m: float = 0.04,
) -> Polygon:
    """Take the largest external contour of ``mask`` and convert to world XZ.

    ``simplify_m`` is the Douglas-Peucker tolerance, expressed in metres
    so it scales correctly with ``grid.pixels_per_metre``.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps_px = max(1.0, simplify_m * grid.pixels_per_metre)
    approx = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        return []
    u = approx[:, 0].astype(np.float64)
    v = approx[:, 1].astype(np.float64)
    x, z = grid.px_to_world(u, v)
    return list(zip(x.tolist(), z.tolist()))


def mask_to_polygons_with_holes(
    mask: np.ndarray, grid: PlanGrid, *, simplify_m: float = 0.04,
) -> list[dict]:
    """Variant that also returns holes, as ``[{"shell": Polygon, "holes": [Polygon, …]}, …]``."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
    )
    if not contours or hierarchy is None:
        return []
    h = hierarchy[0]
    eps_px = max(1.0, simplify_m * grid.pixels_per_metre)

    out: list[dict] = []
    # In RETR_CCOMP, top-level contours have parent == -1; their children are holes.
    for i, hi in enumerate(h):
        if hi[3] != -1:
            continue
        shell = cv2.approxPolyDP(contours[i], eps_px, True).reshape(-1, 2)
        if shell.shape[0] < 3:
            continue
        sx, sz = grid.px_to_world(shell[:, 0].astype(np.float64),
                                  shell[:, 1].astype(np.float64))
        holes: list[Polygon] = []
        j = hi[2]
        while j != -1:
            hole = cv2.approxPolyDP(contours[j], eps_px, True).reshape(-1, 2)
            if hole.shape[0] >= 3:
                hx, hz = grid.px_to_world(hole[:, 0].astype(np.float64),
                                          hole[:, 1].astype(np.float64))
                holes.append(list(zip(hx.tolist(), hz.tolist())))
            j = h[j][0]  # next sibling
        out.append({
            "shell": list(zip(sx.tolist(), sz.tolist())),
            "holes": holes,
        })
    return out


def footprint_polygon(
    footprint_mask: np.ndarray, grid: PlanGrid, *, simplify_m: float = 0.15,
) -> Polygon:
    """Concave outline of all downward-facing geometry — initial room boundary.

    Default simplification of 15 cm keeps the footprint editable
    (~20–60 vertices for a typical room) while still tracking real wall
    setbacks. Users can refine with the in-canvas insert/delete tools.
    """
    return mask_to_polygon(footprint_mask, grid, simplify_m=simplify_m)


def region_polygon(region: PlaneRegion, grid: PlanGrid, *, simplify_m: float = 0.04) -> Polygon:
    return mask_to_polygon(region.mask, grid, simplify_m=simplify_m)


# ─── EDIT OPS (pure, on lists of (x, z) metres) ───────────────────────────────

def insert_vertex_on_edge(poly: Polygon, edge_index: int, point: tuple[float, float]) -> Polygon:
    """Insert ``point`` after ``poly[edge_index]`` (i.e. on edge i→i+1)."""
    if not (0 <= edge_index < len(poly)):
        raise IndexError(f"edge_index out of range: {edge_index}")
    return poly[: edge_index + 1] + [point] + poly[edge_index + 1 :]


def move_vertex(poly: Polygon, index: int, point: tuple[float, float]) -> Polygon:
    if not (0 <= index < len(poly)):
        raise IndexError(f"index out of range: {index}")
    out = poly[:]
    out[index] = point
    return out


def delete_vertex(poly: Polygon, index: int) -> Polygon:
    if len(poly) <= 3:
        raise ValueError("Polygon must keep at least 3 vertices.")
    if not (0 <= index < len(poly)):
        raise IndexError(f"index out of range: {index}")
    return poly[:index] + poly[index + 1 :]


def delete_chain_between(
    poly: Polygon, i: int, j: int, *, direction: str = "short",
) -> Polygon:
    """Remove every vertex *strictly* between ``poly[i]`` and ``poly[j]``.

    Both endpoints are kept; the polygon is bridged by the new edge i→j.

    Because a polygon is cyclic there are two arcs between any two
    vertices. ``direction="short"`` removes the arc with fewer interior
    points; ``"cw"`` (i → … → j going forward in list order) and
    ``"ccw"`` give the explicit choice.
    """
    n = len(poly)
    if not (0 <= i < n) or not (0 <= j < n):
        raise IndexError("vertex index out of range")
    if i == j:
        raise ValueError("i and j must differ")

    forward = (j - i) % n   # interior count + 1 when going forward
    backward = (i - j) % n
    forward_inner = forward - 1
    backward_inner = backward - 1

    if direction == "cw":
        remove_forward = True
    elif direction == "ccw":
        remove_forward = False
    else:
        remove_forward = forward_inner <= backward_inner

    if remove_forward:
        # Remove indices strictly between i and j going forward through the list.
        if i < j:
            kept = poly[: i + 1] + poly[j:]
        else:
            kept = poly[j : i + 1]
            # Rotate so it starts at the original j again (cosmetic).
    else:
        if i < j:
            kept = poly[i : j + 1]
        else:
            kept = poly[: j + 1] + poly[i:]

    if len(kept) < 3:
        raise ValueError("Resulting polygon would have fewer than 3 vertices.")
    return kept


# ─── HIT-TESTING (used server-side for snapping click → vertex / edge) ─────────

def closest_vertex(poly: Polygon, point: tuple[float, float]) -> tuple[int, float]:
    """Return ``(index, distance)`` of the polygon vertex nearest ``point``."""
    if not poly:
        return -1, float("inf")
    arr = np.asarray(poly)
    d = np.hypot(arr[:, 0] - point[0], arr[:, 1] - point[1])
    i = int(np.argmin(d))
    return i, float(d[i])


def closest_edge(
    poly: Polygon, point: tuple[float, float],
) -> tuple[int, float, tuple[float, float]]:
    """Return ``(edge_index, distance, projected_point)`` for the closest edge.

    ``edge_index`` is the index of the start vertex (so the edge is
    ``poly[i] → poly[(i+1) % len(poly)]``).
    """
    if len(poly) < 2:
        return -1, float("inf"), point
    pts = np.asarray(poly)
    n = len(poly)
    p = np.array(point)
    best_i = -1
    best_d = float("inf")
    best_pt = point
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = float((p - a) @ ab) / L2
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d = float(np.hypot(proj[0] - p[0], proj[1] - p[1]))
        if d < best_d:
            best_d = d
            best_i = i
            best_pt = (float(proj[0]), float(proj[1]))
    return best_i, best_d, best_pt


__all__ = [
    "Polygon",
    "mask_to_polygon",
    "mask_to_polygons_with_holes",
    "footprint_polygon",
    "region_polygon",
    "insert_vertex_on_edge",
    "move_vertex",
    "delete_vertex",
    "delete_chain_between",
    "closest_vertex",
    "closest_edge",
]
