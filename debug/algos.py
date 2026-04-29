"""Segmentation algorithms — each takes a :class:`SegInput`, returns an
``int32`` H×W label image (-1 = outside / unassigned, 0..N for clusters).

Add a new algorithm:

1. Write ``def my_algo(data: SegInput, **kwargs) -> np.ndarray``.
2. Register it in :data:`ALGOS` at the bottom of this file.
3. ``python -m debug.segment_lab <session_id> --algo my_algo``.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.signal import find_peaks

from .data import SegInput


# ─── helpers ─────────────────────────────────────────────────────────────────

def _peak_heights(
    valid_y: np.ndarray, *,
    bin_m: float = 0.005,
    sep_m: float = 0.10,
    prominence_frac: float = 0.05,
    max_clusters: int = 5,
) -> np.ndarray:
    if valid_y.size == 0:
        return np.array([])
    edges = np.arange(valid_y.min() - bin_m, valid_y.max() + 2 * bin_m, bin_m)
    hist, _ = np.histogram(valid_y, bins=edges)
    smooth = np.convolve(hist.astype(np.float64),
                         np.array([0.25, 0.5, 0.25]), mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    if smooth.max() == 0:
        return np.array([float(np.median(valid_y))])
    peaks, _ = find_peaks(smooth,
                          distance=max(1, int(sep_m / bin_m)),
                          prominence=prominence_frac * smooth.max())
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(smooth))])
    peaks = peaks[np.argsort(-smooth[peaks])][:max_clusters]
    peaks.sort()
    return centres[peaks]


def _heightmap_for_skimage(h: np.ndarray) -> np.ndarray:
    """Replace NaN with the global median so skimage routines don't choke."""
    out = h.copy()
    nan = np.isnan(out)
    if nan.any():
        out[nan] = float(np.nanmedian(out)) if (~nan).any() else 0.0
    return out


# ─── 1. histogram baseline (current production algorithm) ───────────────────

def histogram(data: SegInput) -> np.ndarray:
    """Current behaviour: histogram peaks → nearest-peak assignment →
    spatial-fill NaN → median denoise. Reference for comparison."""
    h, room = data.height_map, data.room_mask
    heights = h.copy()
    heights[~room] = np.nan
    valid = ~np.isnan(heights)
    if valid.sum() < 100:
        return np.full(h.shape, -1, dtype=np.int32)

    cluster_ys = _peak_heights(heights[valid])
    if cluster_ys.size == 0:
        return np.full(h.shape, -1, dtype=np.int32)

    flat = heights[valid]
    assign_flat = np.argmin(np.abs(flat[:, None] - cluster_ys[None, :]),
                            axis=1).astype(np.int32)
    labels = np.full(h.shape, -1, dtype=np.int32)
    labels[valid] = assign_flat

    nan_in_room = room & ~valid
    if nan_in_room.any():
        _, idx = distance_transform_edt(~valid, return_indices=True)
        ys, xs = np.where(nan_in_room)
        labels[ys, xs] = labels[idx[0, ys, xs], idx[1, ys, xs]]

    px_per_m = data.grid.pixels_per_metre
    k = max(3, int(round(0.10 * px_per_m)) | 1)
    sentinel = int(labels.max() + 100)
    a = np.where(labels >= 0, labels, sentinel).astype(np.int32)
    a = median_filter(a, size=k, mode="nearest")
    labels = np.where(room, a, -1)
    return labels


# ─── 2. felzenszwalb (graph segmentation, gradient-aware) ───────────────────

def felzenszwalb(data: SegInput, *, scale: float = 200.0,
                 sigma: float = 0.8) -> np.ndarray:
    """``skimage.segmentation.felzenszwalb`` on the heightmap.

    Treats the height image as an intensity map and merges adjacent
    pixels whose value gap is small. Larger ``scale`` → bigger regions.
    Boundaries fall naturally at sharp height changes — exactly the
    "gradient between adjacent triangles" signal you wanted.
    """
    from skimage.segmentation import felzenszwalb as fz

    px_per_m = data.grid.pixels_per_metre
    h_filled = _heightmap_for_skimage(data.height_map)
    seg = fz(h_filled.astype(np.float32),
             scale=scale, sigma=sigma,
             min_size=int(0.3 * px_per_m * px_per_m))
    seg = seg.astype(np.int32)
    seg[~data.room_mask] = -1
    seg = _compact_labels(seg)
    return seg


# ─── 3. region growing (gradient-gated growth from seeds) ───────────────────

def region_growing(data: SegInput, *, threshold_m: float = 0.02,
                   max_iters: int = 800) -> np.ndarray:
    """Seed each cluster with the largest CC of pixels close to its
    histogram peak, then iteratively dilate, accepting new pixels only
    where ``|height − cluster_mean| < threshold_m``.

    The growth stops naturally at bulkheads because the height jumps by
    more than ``threshold_m``. After every region stops growing, any
    remaining unassigned pixels go to the nearest-labelled neighbour.
    """
    h = data.height_map
    room = data.room_mask
    valid = ~np.isnan(h) & room
    if not valid.any():
        return np.full(h.shape, -1, dtype=np.int32)

    cluster_ys = _peak_heights(h[valid])
    if cluster_ys.size == 0:
        return np.full(h.shape, -1, dtype=np.int32)

    labels = np.full(h.shape, -1, dtype=np.int32)
    seeded = np.zeros(h.shape, dtype=bool)
    cluster_mean = []
    for ci, y in enumerate(cluster_ys):
        close = (np.abs(h - y) < 0.015) & valid & ~seeded
        if not close.any():
            cluster_mean.append(float(y))
            continue
        m = close.astype(np.uint8)
        n_cc, comp_labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n_cc < 2:
            cluster_mean.append(float(y))
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        seed_pixels = (comp_labels == biggest)
        labels[seed_pixels] = ci
        seeded |= seed_pixels
        cluster_mean.append(float(np.nanmean(h[seed_pixels])))

    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iters):
        any_growth = False
        for ci in range(len(cluster_ys)):
            current = labels == ci
            if not current.any():
                continue
            dilated = cv2.dilate(current.astype(np.uint8), cross, 1).astype(bool)
            cand = dilated & ~current & valid & (labels == -1) & room
            if not cand.any():
                continue
            mean_y = cluster_mean[ci]
            close = cand & (np.abs(h - mean_y) < threshold_m)
            if close.any():
                labels[close] = ci
                any_growth = True
                cluster_mean[ci] = float(np.nanmean(h[labels == ci]))
        if not any_growth:
            break

    # Fill remaining holes by nearest labelled neighbour.
    holes = room & (labels == -1)
    if holes.any() and (labels >= 0).any():
        _, idx = distance_transform_edt(labels < 0, return_indices=True)
        ys, xs = np.where(holes)
        labels[ys, xs] = labels[idx[0, ys, xs], idx[1, ys, xs]]
    labels[~room] = -1
    return _compact_labels(labels)


# ─── 4. wall-constrained partition (your idea) ──────────────────────────────

def wall_constrained(data: SegInput, *,
                     min_cell_px_m2: float = 0.3,
                     max_clusters: int = 5) -> np.ndarray:
    """Partition the room with the wall-edge raster as forced borders,
    then label each enclosed cell by its median height.

    1. ``cells = room_mask & ~wall_edges`` → connected components.
    2. For each cell, compute median ceiling height.
    3. Cluster the per-cell median heights into ≤``max_clusters`` groups
       (histogram peaks weighted by cell area).
    4. Assign each cell to its nearest cluster — every pixel in that cell
       gets the cluster label.

    Because the partition uses physical bulkhead/wall traces, region
    borders land exactly where the geometry says they should — no
    arbitrary nearest-peak boundary inside a flat plane.
    """
    h = data.height_map
    room = data.room_mask
    walls = data.wall_edges
    px_per_m = data.grid.pixels_per_metre
    min_px = max(50, int(min_cell_px_m2 * px_per_m * px_per_m))

    cells_mask = (room & ~walls).astype(np.uint8)
    n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(cells_mask, 8)

    cell_meds: list[float] = []
    cell_areas: list[int] = []
    valid_ids: list[int] = []
    for li in range(1, n_cc):
        if stats[li, cv2.CC_STAT_AREA] < min_px:
            continue
        m = cc_labels == li
        cell_h = h[m]
        cell_h = cell_h[~np.isnan(cell_h)]
        if cell_h.size < 50:
            continue
        cell_meds.append(float(np.median(cell_h)))
        cell_areas.append(int(stats[li, cv2.CC_STAT_AREA]))
        valid_ids.append(li)

    if not cell_meds:
        return np.full(h.shape, -1, dtype=np.int32)

    cell_meds_arr = np.array(cell_meds)
    cell_areas_arr = np.array(cell_areas)

    bin_m = 0.01
    edges = np.arange(cell_meds_arr.min() - bin_m,
                      cell_meds_arr.max() + 2 * bin_m, bin_m)
    if len(edges) < 3:
        cluster_ys = np.array([float(cell_meds_arr.mean())])
    else:
        hist, _ = np.histogram(cell_meds_arr, bins=edges, weights=cell_areas_arr)
        smooth = np.convolve(hist.astype(np.float64),
                             np.array([0.25, 0.5, 0.25]), mode="same")
        centres = 0.5 * (edges[:-1] + edges[1:])
        if smooth.max() == 0:
            cluster_ys = np.array([float(cell_meds_arr.mean())])
        else:
            peaks, _ = find_peaks(smooth, distance=max(1, int(0.10 / bin_m)),
                                  prominence=0.05 * smooth.max())
            if peaks.size == 0:
                peaks = np.array([int(np.argmax(smooth))])
            peaks = peaks[np.argsort(-smooth[peaks])][:max_clusters]
            peaks.sort()
            cluster_ys = centres[peaks]

    cell_clusters = np.argmin(
        np.abs(cell_meds_arr[:, None] - cluster_ys[None, :]), axis=1,
    )

    labels = np.full(h.shape, -1, dtype=np.int32)
    for cell_idx, cc_idx in enumerate(valid_ids):
        labels[cc_labels == cc_idx] = int(cell_clusters[cell_idx])

    # Wall pixels themselves stay -1 (they look like outlines on the overlay,
    # which is exactly the visualisation we want).
    return labels


# ─── 5. felzenszwalb + merge-adjacent-by-height post-pass ──────────────────

def felzenszwalb_merged(data: SegInput, *, scale: float = 200.0,
                        sigma: float = 0.8,
                        merge_threshold_m: float = 0.02) -> np.ndarray:
    """Felzenszwalb at ``scale``, then iteratively union-find adjacent
    regions whose mean heights agree within ±``merge_threshold_m``.

    Felzenszwalb naturally over-segments — boundaries are right but
    duplicated wherever the LiDAR mesh crosses its scale threshold
    inside a flat plane. Merging adjacent same-height regions collapses
    those duplicates back into one, while keeping real bulkhead steps.
    """
    from skimage.segmentation import felzenszwalb as fz
    px_per_m = data.grid.pixels_per_metre
    h_filled = _heightmap_for_skimage(data.height_map)
    seg = fz(h_filled.astype(np.float32),
             scale=scale, sigma=sigma,
             min_size=int(0.3 * px_per_m * px_per_m)).astype(np.int32)
    seg[~data.room_mask] = -1
    seg = _merge_adjacent_by_height(seg, data.height_map, merge_threshold_m)
    return _compact_labels(seg)


# ─── 6. multi-channel watershed ────────────────────────────────────────────

def watershed_multi(data: SegInput, *,
                    height_weight: float = 1.0,
                    texture_weight: float = 0.5,
                    wall_weight: float = 2.0,
                    seed_band_m: float = 0.015) -> np.ndarray:
    """Watershed seeded from histogram peaks, with elevation built from
    the gradient of the height map, the gradient of the textured
    render, and the wall-edge raster — all normalised then linearly
    combined. Boundaries fall on whichever of the three signals is
    strongest at each pixel.
    """
    h = data.height_map.copy()
    valid = ~np.isnan(h) & data.room_mask
    if valid.sum() < 100:
        return np.full(h.shape, -1, dtype=np.int32)
    h[np.isnan(h)] = float(np.nanmedian(h))

    def _grad(img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        g = np.abs(gx) + np.abs(gy)
        return g / max(float(g.max()), 1e-6)

    g_height = _grad(h)
    gray = cv2.cvtColor(data.canvas, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g_tex = _grad(gray)
    g_wall = data.wall_edges.astype(np.float32)

    elevation = (height_weight * g_height +
                 texture_weight * g_tex +
                 wall_weight * g_wall)
    elevation = elevation / max(float(elevation.max()), 1e-6)
    elev_u8 = (elevation * 255).astype(np.uint8)

    # Seed: largest CC of pixels within seed_band_m of each histogram peak.
    cluster_ys = _peak_heights(h[valid])
    if cluster_ys.size == 0:
        return np.full(h.shape, -1, dtype=np.int32)
    markers = np.zeros(h.shape, dtype=np.int32)
    next_id = 1
    for y in cluster_ys:
        close = (np.abs(h - y) < seed_band_m) & valid
        if not close.any():
            continue
        m = close.astype(np.uint8)
        n_cc, comp_labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n_cc < 2:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        markers[comp_labels == biggest] = next_id
        next_id += 1

    if next_id == 1:
        return np.full(h.shape, -1, dtype=np.int32)

    bgr = cv2.cvtColor(elev_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)  # mutates markers in-place

    # cv2.watershed marks boundaries with -1 and regions with positive ints.
    labels = markers.copy() - 1            # → -2 for boundaries, 0..N-1 for regions
    labels[labels == -2] = -1              # collapse boundaries to "unassigned"
    labels[~data.room_mask] = -1
    return _compact_labels(labels)


# ─── 7. region growing with texture-edge gating ────────────────────────────

def region_growing_texture(data: SegInput, *,
                           threshold_m: float = 0.02,
                           canny_low: int = 60,
                           canny_high: int = 160,
                           dilate_edges_px: int = 1,
                           max_iters: int = 800) -> np.ndarray:
    """Same growth logic as :func:`region_growing` but a Canny edge from
    the textured render also blocks the dilation step.

    Catches paint lines, batten gaps, tile grooves and other features
    that don't show up in the height map but mark a real region edge.
    """
    gray = cv2.cvtColor(data.canvas, cv2.COLOR_BGR2GRAY)
    edges_t = cv2.Canny(gray, canny_low, canny_high)
    if dilate_edges_px > 0:
        k = 2 * dilate_edges_px + 1
        edges_t = cv2.dilate(edges_t,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    edges_t = edges_t > 0

    h = data.height_map
    room = data.room_mask
    valid = ~np.isnan(h) & room
    if not valid.any():
        return np.full(h.shape, -1, dtype=np.int32)

    cluster_ys = _peak_heights(h[valid])
    if cluster_ys.size == 0:
        return np.full(h.shape, -1, dtype=np.int32)

    labels = np.full(h.shape, -1, dtype=np.int32)
    seeded = np.zeros(h.shape, dtype=bool)
    cluster_mean: list[float] = []
    for ci, y in enumerate(cluster_ys):
        close = (np.abs(h - y) < 0.015) & valid & ~seeded
        if not close.any():
            cluster_mean.append(float(y)); continue
        m = close.astype(np.uint8)
        n_cc, comp_labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n_cc < 2:
            cluster_mean.append(float(y)); continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        seed_pixels = (comp_labels == biggest)
        labels[seed_pixels] = ci
        seeded |= seed_pixels
        cluster_mean.append(float(np.nanmean(h[seed_pixels])))

    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iters):
        any_growth = False
        for ci in range(len(cluster_ys)):
            current = labels == ci
            if not current.any():
                continue
            dilated = cv2.dilate(current.astype(np.uint8), cross, 1).astype(bool)
            cand = (dilated & ~current & valid &
                    (labels == -1) & room & ~edges_t)
            if not cand.any():
                continue
            mean_y = cluster_mean[ci]
            close = cand & (np.abs(h - mean_y) < threshold_m)
            if close.any():
                labels[close] = ci
                any_growth = True
                cluster_mean[ci] = float(np.nanmean(h[labels == ci]))
        if not any_growth:
            break

    holes = room & (labels == -1)
    if holes.any() and (labels >= 0).any():
        _, idx = distance_transform_edt(labels < 0, return_indices=True)
        ys, xs = np.where(holes)
        labels[ys, xs] = labels[idx[0, ys, xs], idx[1, ys, xs]]
    labels[~room] = -1
    return _compact_labels(labels)


# ─── 8. SLIC superpixels merged by mean height ─────────────────────────────

def slic_merge(data: SegInput, *,
               n_segments: int = 200,
               compactness: float = 10.0,
               merge_threshold_m: float = 0.02) -> np.ndarray:
    """SLIC superpixels on the textured render, then merge adjacent
    superpixels whose mean heights agree within ±``merge_threshold_m``.

    Edges follow the rendered image (paint, battens, materials), heights
    decide which superpixels belong together. Strong on textured ceilings.
    """
    from skimage.segmentation import slic
    rgb = cv2.cvtColor(data.canvas, cv2.COLOR_BGR2RGB)
    seg = slic(rgb,
               n_segments=int(n_segments),
               compactness=float(compactness),
               start_label=0,
               mask=data.room_mask,
               channel_axis=-1).astype(np.int32)
    seg[~data.room_mask] = -1
    seg = _merge_adjacent_by_height(seg, data.height_map, merge_threshold_m)
    return _compact_labels(seg)


# ─── shared post-processing ─────────────────────────────────────────────────

def _merge_adjacent_by_height(
    labels: np.ndarray, height_map: np.ndarray, threshold_m: float,
) -> np.ndarray:
    """Union-find merge of adjacent regions whose valid-pixel means agree
    within ±``threshold_m``. One pass: union-find is transitive, so a
    chain ``A ~ B ~ C`` collapses to a single label even if ``A`` and
    ``C`` aren't directly adjacent."""
    if labels.max() < 0:
        return labels
    n = int(labels.max()) + 1
    means = np.full(n, np.nan, dtype=np.float64)
    for li in range(n):
        sample = height_map[labels == li]
        if sample.size == 0:
            continue
        valid = ~np.isnan(sample)
        if valid.any():
            means[li] = float(sample[valid].mean())

    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # 4-neighbour adjacency, vectorised
    pairs: set[tuple[int, int]] = set()
    for axis in (0, 1):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        m = (a != b) & (a >= 0) & (b >= 0)
        pa = a[m].astype(np.int64); pb = b[m].astype(np.int64)
        for x, y in zip(pa.tolist(), pb.tolist()):
            pairs.add((x, y) if x < y else (y, x))

    for i, j in pairs:
        if not (np.isnan(means[i]) or np.isnan(means[j])):
            if abs(means[i] - means[j]) < threshold_m:
                union(i, j)

    out = labels.copy()
    mask = labels >= 0
    out[mask] = np.array([find(int(x)) for x in labels[mask]], dtype=np.int32)
    return out


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber labels so they're a contiguous 0..N-1 (skipping -1)."""
    unique = np.unique(labels[labels >= 0])
    if unique.size == 0:
        return labels
    remap = -np.ones(int(unique.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(unique.size, dtype=np.int32)
    out = labels.copy()
    mask = labels >= 0
    out[mask] = remap[labels[mask]]
    return out


# ─── registry ───────────────────────────────────────────────────────────────

ALGOS: dict[str, Callable[..., np.ndarray]] = {
    "histogram": histogram,
    "felzenszwalb": felzenszwalb,
    "region_growing": region_growing,
    "wall_constrained": wall_constrained,
    "felzenszwalb_merged": felzenszwalb_merged,
    "watershed_multi": watershed_multi,
    "region_growing_texture": region_growing_texture,
    "slic_merge": slic_merge,
}
