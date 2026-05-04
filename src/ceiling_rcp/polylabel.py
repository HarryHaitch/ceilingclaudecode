"""Pole of inaccessibility — the centre of the largest inscribed circle in
a polygon.

Direct adaptation of Mapbox's polylabel algorithm
(https://github.com/mapbox/polylabel). Public-domain rewrite, ~70 LOC, no
extra dependencies. Honours optional inner rings ("holes") so labels in
ceiling regions with column cut-outs are placed inside the actual face,
not on top of a column.
"""
from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Sequence

Point = tuple[float, float]
Ring = Sequence[Point]


def _segdist_sq(px: float, py: float,
                ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    if t < 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    if t > 1:
        return (px - bx) ** 2 + (py - by) ** 2
    qx = ax + t * dx
    qy = ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _point_in_ring(px: float, py: float, ring: Ring) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _signed_distance(px: float, py: float, rings: Sequence[Ring]) -> float:
    """+ if inside the outer ring and outside all holes; − otherwise.
    Magnitude is the distance to the nearest edge across all rings."""
    if not rings:
        return -math.inf
    inside = _point_in_ring(px, py, rings[0])
    for hole in rings[1:]:
        if _point_in_ring(px, py, hole):
            inside = False
            break
    min_d2 = math.inf
    for ring in rings:
        n = len(ring)
        if n < 2:
            continue
        for i in range(n):
            ax, ay = ring[i]
            bx, by = ring[(i + 1) % n]
            d2 = _segdist_sq(px, py, ax, ay, bx, by)
            if d2 < min_d2:
                min_d2 = d2
    d = math.sqrt(min_d2) if min_d2 < math.inf else 0.0
    return d if inside else -d


def _bbox(ring: Ring) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def polylabel(
    outer: Ring,
    holes: Sequence[Ring] = (),
    *,
    precision: float = 0.01,
) -> tuple[float, float]:
    """Return the pole-of-inaccessibility (x, y) for ``outer`` minus
    ``holes``. ``precision`` is the maximum acceptable error in world
    units — 1 cm is fine for architectural plans (centimetre-level
    placement)."""
    if len(outer) < 3:
        return (0.0, 0.0)

    rings = (list(outer), *(list(h) for h in holes))
    minx, miny, maxx, maxy = _bbox(outer)
    cell_size = min(maxx - minx, maxy - miny)
    if cell_size == 0:
        return (minx, miny)
    h = cell_size / 2

    # Heap is ordered by max-possible distance for each cell (negative
    # because heapq is a min-heap).
    queue: list[tuple[float, float, float, float, float, float]] = []

    def push(cx: float, cy: float, half: float) -> tuple[float, float, float]:
        d = _signed_distance(cx, cy, rings)
        max_d = d + half * math.sqrt(2)
        heappush(queue, (-max_d, cx, cy, half, d, 0.0))
        return (cx, cy, d)

    # Seed grid.
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            push(x + h, y + h, h)
            y += cell_size
        x += cell_size

    # Best is the centroid initially.
    cx0 = sum(p[0] for p in outer) / len(outer)
    cy0 = sum(p[1] for p in outer) / len(outer)
    best_x, best_y, best_d = cx0, cy0, _signed_distance(cx0, cy0, rings)

    # Bbox-centre as a second initial guess (helps for thin polygons).
    bx_cx, bx_cy = (minx + maxx) / 2, (miny + maxy) / 2
    bd = _signed_distance(bx_cx, bx_cy, rings)
    if bd > best_d:
        best_x, best_y, best_d = bx_cx, bx_cy, bd

    while queue:
        neg_max_d, cx, cy, half, d, _ = heappop(queue)
        max_d = -neg_max_d
        if max_d - best_d <= precision:
            continue
        if d > best_d:
            best_x, best_y, best_d = cx, cy, d
        new_half = half / 2
        for dx, dy in ((-new_half, -new_half), (new_half, -new_half),
                       (-new_half, new_half), (new_half, new_half)):
            push(cx + dx, cy + dy, new_half)

    return (best_x, best_y)


def longest_edge_angle_deg(polygon: Ring) -> float:
    """Angle (degrees, in standard math convention with X right, Y up) of
    the polygon's longest edge, wrapped to [-90°, 90°] so labels read
    right-side-up."""
    if len(polygon) < 2:
        return 0.0
    longest_L = 0.0
    angle_deg = 0.0
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L > longest_L:
            longest_L = L
            angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg


__all__ = ["polylabel", "longest_edge_angle_deg"]
