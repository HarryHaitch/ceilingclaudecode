"""Ceiling-plane segmentation from a downward-facing-triangle mesh.

Pipeline:

1. ``cluster_heights`` — weighted 1D Y histogram → peak detection → list of
   candidate plane heights, each absorbing the millimetre-scale jitter of
   LiDAR triangles via a configurable Y tolerance.

2. ``assign_faces`` — assign each downward-facing face to its nearest
   accepted peak (within tolerance) or to "unclassified".

3. ``segment_regions`` — within each plane cluster, rasterise the union of
   faces' XZ projection, morphologically close to absorb sub-pixel gaps,
   and split into spatial regions via connected-components.

4. ``detect_periodic_composites`` — for pairs of plane regions that overlap
   in XZ, look for spatial periodicity in the "which-plane-wins" pattern
   (battens, tiles, coffers, …) and merge into a single composite plane
   tagged with pattern metadata.

5. ``polygons_from_regions`` — derive simplified polygons (in metres) from
   each region's mask. Lives in :mod:`ceiling_rcp.polygons`; this module
   only produces masks + region descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
from scipy.signal import find_peaks

from .mesh import Mesh, downward_face_mask


# ─── DATA ─────────────────────────────────────────────────────────────────────

@dataclass
class HeightCluster:
    """A height band around a peak in the weighted Y histogram."""

    y: float                  # peak height (m)
    y_lo: float               # band lower bound
    y_hi: float               # band upper bound
    area: float               # total face area in band (m^2)


@dataclass
class PlaneRegion:
    """One spatially-connected region of a height cluster."""

    cluster_idx: int          # which HeightCluster this came from
    y: float                  # cluster's nominal height
    mask: np.ndarray          # (H, W) uint8, 1 inside region
    area_m2: float            # region area in metres^2
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1 in mask pixels
    composite: dict | None = None  # pattern metadata if merged with another plane


@dataclass
class PlanGrid:
    """The XZ rasterisation grid all region masks share."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float
    pixels_per_metre: float

    @property
    def width(self) -> int:
        return int(np.ceil((self.max_x - self.min_x) * self.pixels_per_metre))

    @property
    def height(self) -> int:
        return int(np.ceil((self.max_z - self.min_z) * self.pixels_per_metre))

    def world_to_px(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Reflected Ceiling Plan convention: mirror X relative to the "looking
        # up at the ceiling" view so the plan reads with the same handedness
        # as a floor plan. +Z still maps to the top of the image.
        u = (self.max_x - x) * self.pixels_per_metre
        v = (self.max_z - z) * self.pixels_per_metre
        return u, v

    def px_to_world(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.max_x - u / self.pixels_per_metre
        z = self.max_z - v / self.pixels_per_metre
        return x, z


@dataclass
class SegmentationResult:
    grid: PlanGrid
    clusters: list[HeightCluster]
    regions: list[PlaneRegion]
    down_face_indices: np.ndarray   # face indices of all downward triangles
    down_face_y: np.ndarray         # centroid Y of those faces
    footprint_mask: np.ndarray      # (H, W) uint8, union of all downward faces
    debug: dict = field(default_factory=dict)


# ─── HEIGHT CLUSTERING ────────────────────────────────────────────────────────

def cluster_heights(
    y: np.ndarray,
    area: np.ndarray,
    *,
    bin_size_m: float = 0.005,
    min_band_area_m2: float = 0.5,
    band_tol_m: float = 0.05,
    min_peak_separation_m: float = 0.10,
    peak_prominence: float | None = None,
) -> list[HeightCluster]:
    """Find candidate ceiling heights by area-weighted 1D peak detection.

    A Y histogram is built at ``bin_size_m`` resolution and weighted by face
    area, so a single big ceiling outweighs many tiny noise triangles. Peaks
    are widened to a band of half-width ``band_tol_m``, which is the
    "close-enough to be the same plane" tolerance the user specified.

    ``peak_prominence`` defaults to 5% of the largest bin's area.
    """
    if y.size == 0:
        return []

    y_min = float(y.min()) - bin_size_m
    y_max = float(y.max()) + bin_size_m
    edges = np.arange(y_min, y_max + bin_size_m, bin_size_m)
    hist, _ = np.histogram(y, bins=edges, weights=area)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Light smoothing so neighbouring near-empty bins don't break a peak.
    kernel = np.array([0.25, 0.5, 0.25])
    hist_s = np.convolve(hist, kernel, mode="same")

    if peak_prominence is None:
        peak_prominence = 0.05 * float(hist_s.max())

    # Require peaks to be at least `min_peak_separation_m` apart so that two
    # bumps caused by mesh-triangle Y noise on a single physical ceiling
    # collapse to one peak (the larger).
    sep_bins = max(1, int(round(min_peak_separation_m / bin_size_m)))
    peaks, _ = find_peaks(hist_s, prominence=peak_prominence, distance=sep_bins)
    if peaks.size == 0:
        # Fallback: take the single highest bin.
        peaks = np.array([int(np.argmax(hist_s))])

    clusters: list[HeightCluster] = []
    for p in peaks:
        y_peak = float(centres[p])
        y_lo, y_hi = y_peak - band_tol_m, y_peak + band_tol_m
        in_band = (y >= y_lo) & (y <= y_hi)
        band_area = float(area[in_band].sum())
        if band_area < min_band_area_m2:
            continue
        clusters.append(HeightCluster(y=y_peak, y_lo=y_lo, y_hi=y_hi, area=band_area))

    # If two peaks ended up with overlapping bands, merge them (take the
    # higher-area one's height as the new centre).
    clusters.sort(key=lambda c: c.y)
    merged: list[HeightCluster] = []
    for c in clusters:
        if merged and c.y_lo <= merged[-1].y_hi:
            prev = merged[-1]
            keep_y = prev.y if prev.area >= c.area else c.y
            merged[-1] = HeightCluster(
                y=keep_y,
                y_lo=min(prev.y_lo, c.y_lo),
                y_hi=max(prev.y_hi, c.y_hi),
                area=prev.area + c.area,
            )
        else:
            merged.append(c)
    return merged


# ─── FACE ASSIGNMENT ──────────────────────────────────────────────────────────

def assign_faces(
    y: np.ndarray, clusters: list[HeightCluster]
) -> np.ndarray:
    """Assign each face to a cluster index, or -1 if it sits outside every band."""
    if not clusters:
        return np.full(y.shape, -1, dtype=np.int32)

    out = np.full(y.shape, -1, dtype=np.int32)
    # For overlapping bands (shouldn't happen post-merge but be safe), pick
    # the cluster whose peak is closest in Y.
    centres = np.array([c.y for c in clusters])
    los = np.array([c.y_lo for c in clusters])
    his = np.array([c.y_hi for c in clusters])

    for i in range(y.size):
        yi = y[i]
        in_band = (yi >= los) & (yi <= his)
        if not in_band.any():
            continue
        d = np.abs(centres - yi)
        d[~in_band] = np.inf
        out[i] = int(np.argmin(d))
    return out


# ─── XZ RASTERISATION ─────────────────────────────────────────────────────────

def make_grid(mesh: Mesh, down_face_indices: np.ndarray, *,
              pad_m: float = 0.3, pixels_per_metre: float = 100.0) -> PlanGrid:
    verts_used = np.unique(mesh.FV[down_face_indices].flatten())
    xs = mesh.V[verts_used, 0]
    zs = mesh.V[verts_used, 2]
    return PlanGrid(
        min_x=float(xs.min()) - pad_m,
        max_x=float(xs.max()) + pad_m,
        min_z=float(zs.min()) - pad_m,
        max_z=float(zs.max()) + pad_m,
        pixels_per_metre=float(pixels_per_metre),
    )


def rasterise_faces(
    mesh: Mesh, face_indices: np.ndarray, grid: PlanGrid
) -> np.ndarray:
    """Burn the XZ projection of selected triangles into a binary mask."""
    H, W = grid.height, grid.width
    mask = np.zeros((H, W), dtype=np.uint8)
    if face_indices.size == 0:
        return mask
    tris = mesh.V[mesh.FV[face_indices]]
    u, v = grid.world_to_px(tris[..., 0], tris[..., 2])
    pts = np.stack([u, v], axis=-1).astype(np.int32)  # (n, 3, 2)
    cv2.fillPoly(mask, pts, 255)
    return mask


# ─── REGION SEGMENTATION ──────────────────────────────────────────────────────

def segment_regions(
    mesh: Mesh,
    down_face_indices: np.ndarray,
    cluster_assignment: np.ndarray,
    clusters: list[HeightCluster],
    grid: PlanGrid,
    *,
    close_kernel_m: float = 0.05,
    min_region_area_m2: float = 0.25,
) -> list[PlaneRegion]:
    """Per cluster, raster → close → connected components → :class:`PlaneRegion`."""
    regions: list[PlaneRegion] = []
    px_per_m = grid.pixels_per_metre
    k = max(3, int(round(close_kernel_m * px_per_m)) | 1)  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    min_px = int(min_region_area_m2 * px_per_m * px_per_m)

    for ci, cluster in enumerate(clusters):
        sub = down_face_indices[cluster_assignment == ci]
        if sub.size == 0:
            continue
        m = rasterise_faces(mesh, sub, grid)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for lab in range(1, n):
            x, y, w, h, area_px = stats[lab]
            if area_px < min_px:
                continue
            comp = (labels == lab).astype(np.uint8) * 255
            regions.append(PlaneRegion(
                cluster_idx=ci,
                y=cluster.y,
                mask=comp,
                area_m2=area_px / (px_per_m * px_per_m),
                bbox_px=(int(x), int(y), int(x + w), int(y + h)),
            ))
    return regions


# ─── PERIODIC COMPOSITE DETECTION ─────────────────────────────────────────────

def _autocorr_2d(a: np.ndarray) -> np.ndarray:
    """Real-valued normalised 2D autocorrelation via FFT, zero-mean input."""
    a = a.astype(np.float32) - a.mean()
    F = np.fft.rfft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    P = (F * np.conj(F)).real
    R = np.fft.irfft2(P, s=(2 * a.shape[0], 2 * a.shape[1]))
    R = np.fft.fftshift(R)
    centre = R[R.shape[0] // 2, R.shape[1] // 2]
    if centre <= 1e-12:
        return np.zeros_like(R)
    return R / centre


def _shared_bbox(a: PlaneRegion, b: PlaneRegion) -> tuple[int, int, int, int] | None:
    ax0, ay0, ax1, ay1 = a.bbox_px
    bx0, by0, bx1, by1 = b.bbox_px
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax1, bx1); y1 = min(ay1, by1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def detect_periodic_composite(
    lower: PlaneRegion, upper: PlaneRegion, grid: PlanGrid,
    *,
    max_height_delta_m: float = 0.4,
    min_each_region_frac: float = 0.10,
    min_combined_fill_frac: float = 0.45,
    min_peak_strength: float = 0.35,
    min_pitch_m: float = 0.08,
    max_pitch_m: float = 0.6,
    min_alternations: int = 4,
) -> dict | None:
    """Detect a regularly-repeating two-height pattern between adjacent planes.

    Common targets: timber battens running over a recessed slot; suspended
    ceiling tiles whose grid lattice sits a few cm proud of the tile face;
    coffered grids where the soffit drops between cells.

    Conservative gating, in order:

    1. Cap on ``|Δy|`` (default 0.4 m) — far-apart heights are not the
       same physical assembly.
    2. The two regions must share a bounding box ≥ 8 px on each axis.
    3. Inside that shared bbox, both ``L``-only and ``U``-only pixels must
       individually exceed ``min_each_region_frac``, and their union must
       fill ``min_combined_fill_frac`` of the bbox. This rules out the
       failure mode where two unrelated regions just happen to have
       overlapping bboxes with all the L on one side and all the U on
       the other.
    4. 2D FFT autocorrelation of the {-1, 0, +1} "which plane wins" field
       must peak at a non-zero offset of at least ``min_pitch_m`` with
       strength ≥ ``min_peak_strength``.
    5. A 1D scan along the detected direction must actually alternate
       sign at least ``min_alternations`` times, confirming the pattern
       isn't a single shifted blob.

    Returns metadata on success::

        {"kind": "linear" | "grid",
         "pitch_m": float, "angle_deg": float,
         "strength": float, "perp_strength": float,
         "delta_y_m": float}

    ``lower`` is the lower-Y plane (closer to a viewer beneath the ceiling).
    """
    if abs(upper.y - lower.y) > max_height_delta_m:
        return None

    bb = _shared_bbox(lower, upper)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    bbox_px = (x1 - x0) * (y1 - y0)
    if bbox_px <= 0:
        return None

    L = lower.mask[y0:y1, x0:x1] > 0
    U = upper.mask[y0:y1, x0:x1] > 0
    L_only = L & ~U
    U_only = U & ~L
    L_frac = L_only.sum() / bbox_px
    U_frac = U_only.sum() / bbox_px
    fill_frac = (L | U).sum() / bbox_px
    if (L_frac < min_each_region_frac
            or U_frac < min_each_region_frac
            or fill_frac < min_combined_fill_frac):
        return None

    # +1 where upper wins, -1 where lower wins, 0 elsewhere.
    field = np.zeros(L.shape, dtype=np.float32)
    field[U_only] = 1.0
    field[L_only] = -1.0
    if (field != 0).sum() < 64:
        return None

    R = _autocorr_2d(field)
    H, W = R.shape
    cy, cx = H // 2, W // 2

    px_per_m = grid.pixels_per_metre
    rmin = max(3, int(min_pitch_m * px_per_m))
    rmax = int(max_pitch_m * px_per_m)
    if rmax <= rmin:
        return None
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (dist >= rmin) & (dist <= rmax)
    if not ring.any():
        return None

    R_ring = np.where(ring, R, -np.inf)
    idx = np.unravel_index(np.argmax(R_ring), R.shape)
    peak = float(R[idx])
    if peak < min_peak_strength:
        return None

    dy = idx[0] - cy
    dx = idx[1] - cx
    pitch_m = float(np.hypot(dx, dy)) / px_per_m
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Confirm alternation by scanning along the detected direction across
    # the centre of the field. A real periodic pattern flips sign many
    # times; a shifted-blob false positive flips at most twice.
    if not _confirms_alternation(field, angle_deg, min_alternations):
        return None

    # Probe perpendicular direction for grid vs linear classification.
    perp_angle = np.deg2rad(angle_deg + 90.0)
    perp_strength = 0.0
    for r in range(rmin, rmax + 1):
        py = int(round(cy + r * np.sin(perp_angle)))
        px = int(round(cx + r * np.cos(perp_angle)))
        if 0 <= py < H and 0 <= px < W:
            perp_strength = max(perp_strength, float(R[py, px]))

    kind = "grid" if perp_strength >= 0.6 * peak else "linear"

    return {
        "kind": kind,
        "pitch_m": pitch_m,
        "angle_deg": angle_deg,
        "strength": peak,
        "perp_strength": perp_strength,
        "delta_y_m": float(upper.y - lower.y),
        "L_frac": float(L_frac),
        "U_frac": float(U_frac),
    }


def _confirms_alternation(field: np.ndarray, angle_deg: float, min_alternations: int) -> bool:
    """Walk a line through the centre of ``field`` and count sign changes."""
    H, W = field.shape
    cy, cx = H // 2, W // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    diag = int(np.hypot(H, W))
    samples: list[float] = []
    for t in range(-diag, diag + 1):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < W and 0 <= y < H:
            v = float(field[y, x])
            if v != 0:
                samples.append(v)
    if len(samples) < 2:
        return False
    flips = sum(1 for a, b in zip(samples, samples[1:]) if a * b < 0)
    return flips >= min_alternations


# ─── TOP-LEVEL ENTRYPOINT ─────────────────────────────────────────────────────

def segment_ceiling(
    mesh: Mesh,
    *,
    max_tilt_deg: float = 30.0,
    pixels_per_metre: float = 100.0,
    band_tol_m: float = 0.05,
    bin_size_m: float = 0.005,
    min_band_area_m2: float = 0.5,
    min_region_area_m2: float = 0.25,
    close_kernel_m: float = 0.05,
) -> SegmentationResult:
    """End-to-end ceiling segmentation from a loaded :class:`Mesh`."""
    n = mesh.face_normals()
    c = mesh.face_centroids()
    a = mesh.face_areas()

    down = downward_face_mask(n, max_tilt_deg=max_tilt_deg)
    down_idx = np.where(down)[0]
    y_down = c[down_idx, 1]
    a_down = a[down_idx]

    grid = make_grid(mesh, down_idx, pixels_per_metre=pixels_per_metre)
    footprint_mask = rasterise_faces(mesh, down_idx, grid)
    footprint_mask = cv2.morphologyEx(
        footprint_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2,
    )

    clusters = cluster_heights(
        y_down, a_down,
        bin_size_m=bin_size_m,
        min_band_area_m2=min_band_area_m2,
        band_tol_m=band_tol_m,
    )

    assignment = assign_faces(y_down, clusters)
    regions = segment_regions(
        mesh, down_idx, assignment, clusters, grid,
        close_kernel_m=close_kernel_m,
        min_region_area_m2=min_region_area_m2,
    )

    return SegmentationResult(
        grid=grid,
        clusters=clusters,
        regions=regions,
        down_face_indices=down_idx,
        down_face_y=y_down,
        footprint_mask=footprint_mask,
    )


__all__ = [
    "HeightCluster",
    "PlaneRegion",
    "PlanGrid",
    "SegmentationResult",
    "cluster_heights",
    "assign_faces",
    "make_grid",
    "rasterise_faces",
    "segment_regions",
    "detect_periodic_composite",
    "segment_ceiling",
]
