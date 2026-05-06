"""Per-detection polygon backprojection + DBSCAN clustering across keyframes.

Why a separate path from :mod:`ceiling_rcp.sam3_projection`'s per-pixel
accumulator: small fixtures (sprinklers, downlights) lose to per-pixel
fusion because their masks are 5–15 px in source coords and sub-pixel
projection slop stops votes from stacking densely at the true location.
SAM 3 nonetheless detects these fixtures cleanly per keyframe — the
*instance* (a polygon + score) is well-defined even when the *mask
overlap* across keyframes is poor.

This module exploits that: for every detection, ray-cast the polygon
vertices onto the ceiling height field, push through ``M_align`` and
:class:`PlanGrid` to get one ortho-space polygon per detection, then
DBSCAN-cluster on the polygon centroids. Within each cluster the
member polygons are OR-fused so the final per-instance footprint
preserves shape (orientation, length, width) for big/long things while
small fixtures still aggregate to a clean blob.

The pipeline:

1. :func:`load_keyframe_detections` — reads ``detections.json``,
   back-rotates polygon vertices and bbox centres, and (for
   ``ceiling item``) culls detections whose source-image bbox centre
   falls inside a more-specific concept's mask in the same keyframe.
2. :func:`raycast_polygon` — converts a back-rotated polygon
   (original-frame px) to an ortho-space polygon by ray-casting each
   vertex through the ARKit camera onto the ceiling height field.
3. :func:`cluster_detections` — DBSCAN on ortho-centroids
   (eps in metres, configurable min-samples).
4. :func:`fuse_cluster` — rasterise the OR of member polygons into a
   tight crop, derive centroid + ``cv2.minAreaRect`` oriented bbox,
   and return both the binary footprint and the descriptor.

The driver in
``experiments/segmentation_projection/project_clusters.py`` wires
these together and renders the per-concept overlays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .planes import PlanGrid
from .sam3_projection import (
    PROMPTS, PROMPT_SLUG, PolycamCamera, back_rotate, load_alignment,
)


# ─── DETECTIONS ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One SAM 3 detection in a single keyframe, polygon already back-rotated
    into the *original* (1024 × 768) source-image coordinates.
    """
    image_id: str
    prompt: str
    instance_id: int
    score: float
    polygon_src: np.ndarray         # (N, 2) float in original image px (u, v)
    bbox_center_src: np.ndarray     # (2,) float — cheap centroid for culling
    area_px: int


def _back_rotate_point(uv_rot: np.ndarray, src_h: int = 768) -> np.ndarray:
    """rotated-frame (u_r, v_r) → original-frame (u_o, v_o).

    Each point in the CW90-rotated 768×1024 image at ``(u_r, v_r)`` lives
    at ``(v_r, src_h - 1 - u_r)`` in the original 1024×768 frame.
    Vectorised over an (N, 2) array.
    """
    if uv_rot.size == 0:
        return uv_rot.astype(np.float64)
    out = np.empty_like(uv_rot, dtype=np.float64)
    out[:, 0] = uv_rot[:, 1]
    out[:, 1] = (src_h - 1) - uv_rot[:, 0]
    return out


def load_keyframe_detections(
    image_id: str,
    *,
    results_dir: Path,
    masks_for_cull: dict[str, np.ndarray] | None = None,
    cull_overlap: float = 0.5,
    catchall_prompt: str = "ceiling item",
    specific_prompts: tuple[str, ...] = ("light", "vent"),
    src_h: int = 768,
    prompts: Iterable[str] = PROMPTS,
) -> dict[str, list[Detection]]:
    """Read ``detections.json`` and return back-rotated detections per prompt.

    If ``masks_for_cull`` (already back-rotated original-frame masks) are
    provided, drops every ``catchall_prompt`` detection whose bbox centre
    falls inside the union of the specific prompts' masks. This is the
    detection-level analogue of
    :func:`sam3_projection.cull_ceiling_items_against_specifics`.
    """
    det_path = results_dir / image_id / "detections.json"
    if not det_path.exists():
        return {}
    j = json.loads(det_path.read_text())
    out: dict[str, list[Detection]] = {p: [] for p in prompts}

    cull_mask = None
    if masks_for_cull:
        # All masks_for_cull entries share the same back-rotated source
        # shape; pick the first to size the cull buffer. Tolerate an
        # empty dict (a keyframe where every prompt produced no
        # detections) by skipping the cull pass entirely — there's
        # nothing to cull against.
        first_key = next(iter(masks_for_cull), None)
        if first_key is not None:
            cull_mask = np.zeros(masks_for_cull[first_key].shape, dtype=bool)
            for p in specific_prompts:
                if p in masks_for_cull:
                    cull_mask |= masks_for_cull[p] > 0

    for entry in j.get("prompts", []):
        prompt = entry.get("prompt")
        if prompt not in out:
            continue
        for d in entry.get("detections", []) or []:
            poly_rot = np.asarray(d.get("polygon", []), dtype=np.float64)
            if poly_rot.shape[0] < 3:
                continue
            poly_src = _back_rotate_point(poly_rot, src_h=src_h)
            box = d.get("box_xyxy")
            if box and len(box) == 4:
                cx_r = 0.5 * (box[0] + box[2])
                cy_r = 0.5 * (box[1] + box[3])
                bb_src = _back_rotate_point(np.array([[cx_r, cy_r]]), src_h=src_h)[0]
            else:
                bb_src = poly_src.mean(axis=0)

            # detection-level cull: drop catchall detections whose bbox
            # centre lands inside the specific-prompt union.
            if (
                cull_mask is not None
                and prompt == catchall_prompt
                and 0 <= int(bb_src[1]) < cull_mask.shape[0]
                and 0 <= int(bb_src[0]) < cull_mask.shape[1]
                and cull_mask[int(bb_src[1]), int(bb_src[0])]
            ):
                # Belt-and-braces: if the detection's polygon also has
                # significant overlap with the cull mask, definitely drop.
                continue

            out[prompt].append(Detection(
                image_id=image_id,
                prompt=prompt,
                instance_id=int(d.get("instance_id", 0)),
                score=float(d.get("score", 1.0)),
                polygon_src=poly_src,
                bbox_center_src=bb_src,
                area_px=int(d.get("area_px", 0)),
            ))
    return out


# ─── RAY-CAST: SOURCE PX → ORTHO PX ───────────────────────────────────────

def _src_px_to_camray(uv_src: np.ndarray, cam: PolycamCamera) -> np.ndarray:
    """Per ARKit briefing: camera-space ray for an image-plane pixel.

    ``x_c = (u - cx) / fx``, ``y_c = -(v - cy) / fy`` (v-flip back),
    ``z_c = -1`` (camera looks down ``-Z``). Returned rays are
    *un-normalised* — the caller can normalise or use the implicit
    ``z_c = -1`` to skip a square root.
    """
    x = (uv_src[:, 0] - cam.cx) / cam.fx
    y = -(uv_src[:, 1] - cam.cy) / cam.fy
    z = -np.ones_like(x)
    return np.stack([x, y, z], axis=1)


def _intersect_ceiling(
    ray_origins: np.ndarray, ray_dirs_world: np.ndarray,
    *, height: np.ndarray, grid: PlanGrid, M_align: np.ndarray,
    initial_h: float, max_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively intersect each ARKit ray with the per-pixel ceiling.

    For a horizontal ray bound that points up (ray_dir.y > 0):
      * 1st iter: assume ceiling at ``initial_h`` (median valid height,
        ARKit-y).
      * Convert hit point to mesh space, look up actual height.npy at
        that ortho pixel, recompute hit with the new height, repeat.

    Returns ``(hit_arkit, valid_mask)``. ``hit_arkit`` is the ARKit-space
    intersection point (NaN where the ray never hits). ``valid_mask`` is
    True for rays that hit a finite-height ortho pixel.
    """
    h_now = np.full(ray_origins.shape[0], initial_h, dtype=np.float64)
    valid = np.ones(ray_origins.shape[0], dtype=bool)
    hit = np.full_like(ray_origins, np.nan)

    for _ in range(max_iter):
        ry = ray_dirs_world[:, 1]
        denom_ok = ry > 1e-6  # ray must point up
        valid &= denom_ok
        if not valid.any():
            break
        d = np.where(denom_ok, (h_now - ray_origins[:, 1]) / np.where(denom_ok, ry, 1.0), -1.0)
        valid &= d > 0
        if not valid.any():
            break
        hit = ray_origins + d[:, None] * ray_dirs_world

        # Convert ARKit hits to mesh space via M_align (forward = ARKit→mesh).
        N = hit.shape[0]
        hh = np.hstack([hit, np.ones((N, 1))])
        hit_mesh = (M_align @ hh.T).T[:, :3]
        x_mesh = hit_mesh[:, 0]; z_mesh = hit_mesh[:, 2]
        u, v = grid.world_to_px(x_mesh.astype(np.float32), z_mesh.astype(np.float32))
        ui = np.clip(np.round(u).astype(np.int64), 0, grid.width - 1)
        vi = np.clip(np.round(v).astype(np.int64), 0, grid.height - 1)
        sampled_h = height[vi, ui]
        finite = np.isfinite(sampled_h)
        # Update plane height for next iteration where we have a valid
        # sample; freeze the rest.
        new_h = np.where(finite, sampled_h.astype(np.float64), h_now)
        # Convert mesh-space height back to ARKit-y. M_align's y row is
        # (0, 1, 0, t_y), so y_arkit = y_mesh + t_y? Actually it's the
        # reverse — M_align is mesh→ARKit's *inverse* per the load_alignment
        # docstring, meaning M_align here is ARKit→mesh. The y column of
        # the rotation block is (0, 1, 0): heights pass through unchanged
        # with only a translation. We compose: y_mesh = y_arkit + dy where
        # dy = M_align[1, 3] (column-major reshape ⇒ row 1 of column 3).
        # Going the other way: y_arkit = y_mesh - M_align[1, 3].
        h_now = new_h - M_align[1, 3]
        valid &= finite

    return hit, valid


# Soft sanity caps for the depth-based projection. These are NOT the
# old ray-cast tuning knobs — depth-map projection is geometry-driven,
# not heuristic. ``MAX_DIST_M`` rejects projections that landed
# absurdly far away (almost always a bad LiDAR sample at a window or
# mirror); ``MIN_DIST_M`` drops degenerate near-zero hits. Both are
# defensive — change only if you know exactly which artefact you're
# fighting.
MAX_DIST_M = 20.0
MIN_DIST_M = 0.05


def project_polygons_to_ortho(
    detections: list[Detection],
    cam: PolycamCamera,
    *,
    grid: PlanGrid,
    depth_m: np.ndarray | None = None,    # (H, W) float32 metres, NaN where unknown
    min_valid_frac: float = 0.4,
) -> list[dict]:
    """Project every detection's polygon to ortho coords using LiDAR
    depth — pixel + depth + intrinsics → ARKit world point in one step,
    no ceiling-height assumption.

    ``depth_m`` is the per-keyframe depth map, **already loaded and
    upscaled** to match the camera's image resolution (see
    :func:`ceiling_rcp.sam3_depth_projection.load_depth`). Pixels with
    NaN depth are unreliable LiDAR samples (usually low confidence at
    depth discontinuities) — when more than ``1 - min_valid_frac`` of a
    polygon's vertices have NaN depth, the whole detection is dropped.

    Returns one dict per surviving detection::

        {
          "image_id": str, "prompt": str, "score": float,
          "ortho_polygon": (N, 2) float ortho-px,
          "ortho_centroid_xz_m": (2,) float metres,
          "ortho_centroid_px": (2,) float ortho-px,
          "view_dist_m": float,
          "mask_diameter_cm": float,
        }
    """
    if not detections:
        return []
    if depth_m is None:
        # No depth → nothing we can do. Caller should have skipped this
        # keyframe, but tolerate the case rather than crash.
        return []
    from .sam3_depth_projection import (
        project_polygon_to_world, world_to_ortho_px,
    )

    t = cam.t              # camera position (ARKit)

    out: list[dict] = []
    for d in detections:
        P_world, ok = project_polygon_to_world(
            d.polygon_src, depth_m, cam, min_valid_frac=min_valid_frac,
        )
        if not ok:
            continue
        ortho_poly = world_to_ortho_px(P_world, grid)
        if ortho_poly.shape[0] < 3:
            continue

        # Centroid in ortho px.
        cu = float(ortho_poly[:, 0].mean())
        cv_ = float(ortho_poly[:, 1].mean())
        cx_m, cz_m = grid.px_to_world(np.array([cu]), np.array([cv_]))

        # Camera-to-centroid distance for the apparent-mask-diameter
        # calculation. Computed from ARKit-space directly — no need to
        # round-trip through mesh space.
        c_world = P_world.mean(axis=0)
        view = c_world - t
        dist = float(np.linalg.norm(view))
        if dist > MAX_DIST_M or dist < MIN_DIST_M:
            continue

        # Mean mask size in real-world units, using the source-image
        # polygon area + the centroid view distance: at distance d each
        # source pixel² covers (d/fx)·(d/fy) m². This reads the true
        # apparent fixture size SAM saw, not the cluster-OR footprint.
        src_area_px2 = float(abs(cv2.contourArea(d.polygon_src.astype(np.float32))))
        src_area_m2 = src_area_px2 * (dist / cam.fx) * (dist / cam.fy)
        mask_diameter_cm = float(2.0 * np.sqrt(max(src_area_m2, 0.0) / np.pi) * 100.0)

        out.append(dict(
            image_id=d.image_id, prompt=d.prompt, score=d.score,
            ortho_polygon=ortho_poly,
            ortho_centroid_px=np.array([cu, cv_]),
            ortho_centroid_xz_m=np.array([cx_m[0], cz_m[0]]),
            view_dist_m=dist,
            mask_diameter_cm=mask_diameter_cm,
            # Carry the source-image polygon through so downstream
            # cluster fusion can save it on the symbol for thumbnails.
            polygon_src=d.polygon_src,
        ))
    return out


# ─── DBSCAN ───────────────────────────────────────────────────────────────

def dbscan_cluster(
    points_xy: np.ndarray, *, eps: float, min_samples: int = 3,
) -> np.ndarray:
    """Tiny DBSCAN — labels in ``range(K)`` with -1 for noise.

    Avoids the sklearn dependency. ``points_xy`` is (N, 2), ``eps`` is in
    the same units as the points. Cost: O(N²) brute force; N is in the
    hundreds-to-low-thousands per concept here so this is fine.
    """
    n = points_xy.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    # Pairwise distances.
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    neighbours = [np.where(dist[i] <= eps)[0] for i in range(n)]
    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        nbi = neighbours[i]
        if nbi.size < min_samples:
            continue
        # start a new cluster
        labels[i] = cluster
        seeds = list(nbi)
        seen = set(seeds)
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster
            if labels[j] != cluster:
                continue
            nbj = neighbours[j]
            if nbj.size >= min_samples:
                for k in nbj:
                    if k not in seen:
                        seen.add(int(k)); seeds.append(int(k))
        cluster += 1
    return labels


# ─── PER-CLUSTER OR-FUSE ─────────────────────────────────────────────────

@dataclass
class ClusterInstance:
    id: int
    prompt: str
    n_views: int
    median_score: float
    centroid_px: tuple[float, float]
    centroid_xz_m: tuple[float, float]
    bbox_px: tuple[int, int, int, int]    # x0, y0, x1, y1
    rotated_rect: dict                    # cx_px, cy_px, w_px, h_px, angle_deg
    width_m: float
    length_m: float
    area_cm2: float
    mean_mask_diameter_cm: float = 0.0   # median of per-detection mask sizes
    image_ids: list[str] = field(default_factory=list)
    # Per-keyframe SAM 3 polygons in *original* (1024 × 768) image
    # coords, one entry per contributing detection. Carried through to
    # symbols.json so the viewer's thumbnail endpoint can outline the
    # detection on the saved per-frame photo without recomputing it.
    # Each entry: {"image_id": str, "polygon_src": [[u, v], ...]}.
    per_frame_polygons: list[dict] = field(default_factory=list)


def fuse_cluster(
    detections: list[dict],
    *,
    grid: PlanGrid,
    pad_px: int = 8,
) -> tuple[ClusterInstance, np.ndarray, tuple[int, int]] | None:
    """OR-fuse a cluster's ortho polygons into a tight crop.

    Returns ``(ClusterInstance, fused_mask, (x0, y0))`` where
    ``fused_mask`` is the binary footprint inside its bounding box
    (uint8 0/255) and ``(x0, y0)`` is the top-left corner of that
    crop in ortho-pixel coords.
    """
    if not detections:
        return None
    # Bbox of all polygons combined.
    all_pts = np.vstack([d["ortho_polygon"] for d in detections])
    x0 = max(0, int(np.floor(all_pts[:, 0].min())) - pad_px)
    y0 = max(0, int(np.floor(all_pts[:, 1].min())) - pad_px)
    x1 = min(grid.width,  int(np.ceil(all_pts[:, 0].max())) + pad_px)
    y1 = min(grid.height, int(np.ceil(all_pts[:, 1].max())) + pad_px)
    w = max(1, x1 - x0); h = max(1, y1 - y0)
    crop = np.zeros((h, w), dtype=np.uint8)
    for d in detections:
        poly = d["ortho_polygon"].copy()
        poly[:, 0] -= x0; poly[:, 1] -= y0
        cv2.fillPoly(crop, [poly.astype(np.int32)], 255)
    # Take the largest connected component to cancel scattered
    # polygon-vertex misses.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    fused = ((lab == keep).astype(np.uint8)) * 255

    # cv2.minAreaRect for oriented bbox.
    contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)        # (cx, cy), (w, h), angle
    (rcx, rcy), (rw, rh), angle = rect
    # in ortho-pixel coords, length = max(rw, rh), width = min
    length_px = float(max(rw, rh)); width_px = float(min(rw, rh))
    # cv2 angles are in (-90, 0]; normalise so length matches the long axis
    if rw < rh:
        angle = angle + 90.0
    px_per_m = grid.pixels_per_metre
    cm2_per_px = (100.0 / px_per_m) ** 2
    area_cm2 = float((fused > 0).sum() * cm2_per_px)

    diam_samples = [d.get("mask_diameter_cm", 0.0) for d in detections]
    diam_samples = [d for d in diam_samples if d > 0]
    mean_mask_diameter_cm = float(np.median(diam_samples)) if diam_samples else 0.0

    centroid_px_global = (float(rcx + x0), float(rcy + y0))
    cx_m, cz_m = grid.px_to_world(np.array([centroid_px_global[0]]),
                                  np.array([centroid_px_global[1]]))

    inst = ClusterInstance(
        id=0,                              # caller assigns
        prompt=detections[0]["prompt"],
        n_views=len(detections),
        median_score=float(np.median([d["score"] for d in detections])),
        centroid_px=centroid_px_global,
        centroid_xz_m=(float(cx_m[0]), float(cz_m[0])),
        bbox_px=(x0, y0, x1, y1),
        rotated_rect=dict(
            cx_px=centroid_px_global[0], cy_px=centroid_px_global[1],
            w_px=width_px, h_px=length_px, angle_deg=float(angle),
        ),
        width_m=float(width_px / px_per_m),
        length_m=float(length_px / px_per_m),
        area_cm2=area_cm2,
        mean_mask_diameter_cm=mean_mask_diameter_cm,
        image_ids=[d["image_id"] for d in detections],
        per_frame_polygons=[
            {
                "image_id": d["image_id"],
                "polygon_src": d["polygon_src"].astype(np.float32).tolist()
                                if "polygon_src" in d else [],
            }
            for d in detections
        ],
    )
    return inst, fused, (x0, y0)


__all__ = [
    "Detection", "ClusterInstance",
    "load_keyframe_detections",
    "project_polygons_to_ortho",
    "dbscan_cluster", "fuse_cluster",
]
