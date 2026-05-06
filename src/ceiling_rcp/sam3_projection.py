"""Project per-keyframe SAM 3 masks onto a top-down ortho canvas and fuse
the votes into one ortho-space mask per concept.

The math, in one place so future readers don't have to re-derive it:

* **Two coordinate spaces.** Polycam exports the OBJ mesh in *mesh space*
  and the per-keyframe cameras in *ARKit space*. ``mesh_info.json`` ships
  an ``alignmentTransform`` (16-element column-major) that bridges the
  two. The main-app ortho (``ceiling.jpg``, ``height.npy``,
  ``PlanGrid``) is built directly from ``mesh.V`` so it lives in
  **mesh space**. The cameras are in **ARKit space**. To project an
  ortho pixel into a camera you must first bring the world point into
  ARKit space:

      M_align     = np.array(info["alignmentTransform"]).reshape(4, 4, 'F')
      M_align_inv = np.linalg.inv(M_align)
      P_arkit     = M_align_inv @ [P_mesh, 1]

  Empirically the JSON matrix maps ARKit→mesh; the inverse brings the
  mesh into ARKit space (the briefing's prose got the direction wrong;
  the code in the briefing is right). Skipping this step rotates every
  projection by the alignment angle (~43.6° for Lachy's scan) — the
  symptom is masks that land on the wrong area of the ortho with a
  rotational error proportional to the camera's distance from the room
  origin. See ``sanity_align_check.py``.

* **Camera convention.** Polycam keyframes use ARKit convention: the
  camera looks down its ``-Z`` axis, world ``+Y`` is gravity-up. The
  ``corrected_cameras/<id>.json`` file holds intrinsics for the
  *original* (un-rotated, 1024x768 landscape) RGB image and a row-major
  3x3 rotation ``R`` (camera-to-world) plus the camera position
  ``t = (t_03, t_13, t_23)``. World-to-camera is therefore
  ``P_cam = R^T (P_arkit - t)``; the depth in front of the camera is
  ``depth = -P_cam.z`` (positive when ``P`` is visible).

* **Projection.** ARKit ``+Y_cam`` is up but image ``v`` increases
  *downward*, so the ``v`` formula flips:

      u =  fx * (x_cam / depth) + cx
      v = -fy * (y_cam / depth) + cy

  This was verified by overlaying the back-rotated mask of keyframe
  ``43118590133`` onto the ortho — the blue (vent) mask lands precisely
  on a square diffuser only with both the alignment transform and the v
  flip in place; either alone leaves the projection visibly off.

* **Mask back-rotation.** SAM 3 was fed images rotated 90° clockwise
  from the original orientation (768x1024 portrait), so the per-keyframe
  masks come back in rotated coords. We rotate them 90° counter-clockwise
  back to original coords *before* projection, since the intrinsics
  belong to the original frame.

* **Per-pixel pipeline.** For each ortho pixel ``(X_o, Y_o)`` whose
  ``height.npy`` value is finite we form the mesh-space world point
  ``P_mesh = (X_w, Y_w, Z_w)`` (with ``Y_w`` from ``height.npy`` and
  ``X_w, Z_w`` from :class:`PlanGrid.px_to_world`), bring it into
  ARKit space, project it into every keyframe, and accumulate a
  per-pixel weighted vote — one vote per (keyframe, prompt, ortho-pixel)
  triple. The fusion math + per-region cross-reference are in
  :func:`fuse` and :func:`per_region_counts`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import json
import numpy as np

from .planes import PlanGrid

PROMPTS = ("ceiling item", "light", "vent")
PROMPT_SLUG = {p: p.replace(" ", "_") for p in PROMPTS}

# BGR — matches the existing app palette for "ceiling object" highlights.
PROMPT_COLORS = {
    "ceiling item": (64, 64, 255),
    "light":        (64, 220, 255),
    "vent":         (255, 200, 64),
}


# ─── DATA ──────────────────────────────────────────────────────────────────

@dataclass
class PolycamCamera:
    R: np.ndarray             # 3x3 camera-to-world rotation
    t: np.ndarray             # 3-vec camera position in world
    fx: float; fy: float
    cx: float; cy: float
    W: int; H: int            # original (un-rotated) image dimensions

    @classmethod
    def from_json(cls, path: Path) -> "PolycamCamera":
        j = json.loads(path.read_text())
        R = np.array([
            [j["t_00"], j["t_01"], j["t_02"]],
            [j["t_10"], j["t_11"], j["t_12"]],
            [j["t_20"], j["t_21"], j["t_22"]],
        ], dtype=np.float64)
        t = np.array([j["t_03"], j["t_13"], j["t_23"]], dtype=np.float64)
        return cls(
            R=R, t=t,
            fx=float(j["fx"]), fy=float(j["fy"]),
            cx=float(j["cx"]), cy=float(j["cy"]),
            W=int(j["width"]), H=int(j["height"]),
        )

    def facing_up(self) -> float:
        """Cosine-like score in [-1, 1]: +1 when forward axis points to +Y."""
        # Forward is -R[:,2] (OpenGL); forward.y = -t_12.
        return float(-self.R[1, 2])


@dataclass
class KeyframeMaskBundle:
    """Per-keyframe masks + instance maps in *original* image orientation."""
    image_id: str
    cam: PolycamCamera
    masks: dict[str, np.ndarray]      # prompt -> (H, W) uint8 (0/255)
    instances: dict[str, np.ndarray]  # prompt -> (H, W) uint16 (0..N)
    scores: dict[str, dict[int, float]]  # prompt -> {instance_id: score}
    depth_mm: np.ndarray | None       # (h_d, w_d) uint16 in millimetres or None
    confidence: np.ndarray | None     # (h_d, w_d) uint8 0..255 or None


def back_rotate(mask_or_inst: np.ndarray) -> np.ndarray:
    """rotated CW-90 (HxW = 1024x768) -> original (768x1024)."""
    return cv2.rotate(mask_or_inst, cv2.ROTATE_90_COUNTERCLOCKWISE)


def cull_ceiling_items_against_specifics(
    masks: dict[str, np.ndarray],
    instances: dict[str, np.ndarray],
    scores: dict[str, dict[int, float]],
    *,
    overlap_threshold: float = 0.5,
    specific_prompts: tuple[str, ...] = ("light", "vent"),
    catchall_prompt: str = "ceiling item",
) -> int:
    """Drop ``catchall_prompt`` instances whose mask is mostly covered by
    a more-specific concept in the same keyframe. Mutates the dicts in
    place. Returns the number of instances culled.

    "Ceiling item" is the catch-all SAM 3 prompt and routinely re-detects
    the same fixture that ``light`` / ``vent`` already named more
    descriptively. Letting both survive double-counts those fixtures in
    the per-region totals. Apply this filter once per keyframe — *before*
    projection, so the redundant masks never reach the accumulator.
    """
    if catchall_prompt not in masks or catchall_prompt not in instances:
        return 0
    other = np.zeros(masks[catchall_prompt].shape, dtype=bool)
    for p in specific_prompts:
        if p in masks:
            other |= masks[p] > 0
    if not other.any():
        return 0

    inst_map = instances[catchall_prompt]
    score_lookup = scores.get(catchall_prompt, {})
    new_inst = np.zeros_like(inst_map)
    new_mask = np.zeros_like(masks[catchall_prompt])
    new_scores: dict[int, float] = {}
    next_id = 0
    culled = 0
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        sel = inst_map == iid
        area = int(sel.sum())
        if area == 0:
            continue
        overlap = int((sel & other).sum())
        if overlap / area >= overlap_threshold:
            culled += 1
            continue
        next_id += 1
        new_inst[sel] = next_id
        new_mask[sel] = 255
        if int(iid) in score_lookup:
            new_scores[next_id] = score_lookup[int(iid)]

    instances[catchall_prompt] = new_inst
    masks[catchall_prompt] = new_mask
    scores[catchall_prompt] = new_scores
    return culled


def load_keyframe(image_id: str, *,
                  cameras_dir: Path,
                  results_dir: Path,
                  depth_dir: Path | None = None,
                  confidence_dir: Path | None = None,
                  prompts: Iterable[str] = PROMPTS,
                  cull_catchall: bool = True) -> KeyframeMaskBundle | None:
    cam_path = cameras_dir / f"{image_id}.json"
    res_dir = results_dir / image_id
    if not cam_path.exists() or not res_dir.exists():
        return None

    cam = PolycamCamera.from_json(cam_path)

    masks: dict[str, np.ndarray] = {}
    instances: dict[str, np.ndarray] = {}
    scores: dict[str, dict[int, float]] = {}

    det_path = res_dir / "detections.json"
    det = json.loads(det_path.read_text()) if det_path.exists() else {"prompts": []}
    score_lookup_all: dict[str, dict[int, float]] = {}
    for entry in det.get("prompts", []):
        sl: dict[int, float] = {}
        for d in entry.get("detections", []) or []:
            sl[int(d["instance_id"])] = float(d.get("score", 1.0))
        score_lookup_all[entry["prompt"]] = sl

    for p in prompts:
        slug = PROMPT_SLUG[p]
        mp = res_dir / f"{slug}_mask.png"
        ip = res_dir / f"{slug}_instances.png"
        if not mp.exists() or not ip.exists():
            continue
        m_rot = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        i_rot = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if m_rot is None or i_rot is None:
            continue
        masks[p] = back_rotate(m_rot)
        instances[p] = back_rotate(i_rot.astype(np.uint16))
        scores[p] = score_lookup_all.get(p, {})

    if cull_catchall:
        cull_ceiling_items_against_specifics(masks, instances, scores)

    depth_arr = None
    if depth_dir is not None:
        dp = depth_dir / f"{image_id}.png"
        if dp.exists():
            depth_arr = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.uint16)
    conf_arr = None
    if confidence_dir is not None:
        cp = confidence_dir / f"{image_id}.png"
        if cp.exists():
            conf_arr = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED).astype(np.uint8)

    return KeyframeMaskBundle(
        image_id=image_id, cam=cam, masks=masks, instances=instances,
        scores=scores, depth_mm=depth_arr, confidence=conf_arr,
    )


# ─── PROJECTION ───────────────────────────────────────────────────────────

def world_to_image(P_arkit: np.ndarray, cam: PolycamCamera):
    """Project N ARKit-space world points into ``cam``'s image plane.

    ``P_arkit`` must already be in ARKit space — i.e. mesh-space points
    must have ``M_align_inv`` applied first (see :func:`align_to_arkit`).

    Returns ``(u, v, depth, dist)``. ``depth = -z_cam`` is positive for
    points in front of the camera.
    """
    Pc = (P_arkit - cam.t) @ cam.R    # equiv to R^T (P - t)
    depth = -Pc[:, 2]
    inv_d = np.where(depth > 0, 1.0 / depth, 0.0)
    u = cam.fx * (Pc[:, 0] * inv_d) + cam.cx
    # ARKit +y_cam is up; image v goes down → flip.
    v = -cam.fy * (Pc[:, 1] * inv_d) + cam.cy
    dist = np.linalg.norm(Pc, axis=1)
    return u, v, depth, dist


def load_alignment(mesh_info_path: Path) -> np.ndarray:
    """Return the M_align_inv matrix that maps mesh-space → ARKit space.

    The Polycam ``mesh_info.json`` ships a column-major 4x4 matrix in
    ``alignmentTransform``. Despite the docstring in the ARKit briefing
    claiming the matrix maps mesh→ARKit, empirically the JSON matrix is
    the *inverse* of that — the inverse is what brings mesh points into
    the cameras' coordinate frame.
    """
    info = json.loads(mesh_info_path.read_text())
    M = np.array(info["alignmentTransform"], dtype=np.float64).reshape(4, 4, order='F')
    return np.linalg.inv(M)


def align_to_arkit(P_mesh: np.ndarray, M_align_inv: np.ndarray) -> np.ndarray:
    """Apply ``M_align_inv`` to a (N, 3) array of mesh-space points."""
    N = P_mesh.shape[0]
    P_h = np.hstack([P_mesh, np.ones((N, 1), dtype=P_mesh.dtype)])
    return (M_align_inv @ P_h.T).T[:, :3]


# ─── ACCUMULATION ─────────────────────────────────────────────────────────

@dataclass
class Accumulators:
    """Per-prompt scatter buffers for a single fusion pass.

    At read time three things matter:

      * ``pos_w`` / ``tot_w`` — weighted positive votes vs total visible
        weight. Their ratio is the weighted-majority score in [0, 1].
      * ``n_vis`` — how many keyframes saw this ortho pixel at all
        (used for the "min contributing keyframes" gate).
      * ``max_pw`` — the max single-keyframe weighted-positive score
        (used by the max-confidence fusion rule).
    """
    pos_w: dict[str, np.ndarray]
    tot_w: dict[str, np.ndarray]
    n_vis: dict[str, np.ndarray]
    max_pw: dict[str, np.ndarray]

    @classmethod
    def empty(cls, shape: tuple[int, int], prompts: Iterable[str] = PROMPTS):
        return cls(
            pos_w={p: np.zeros(shape, np.float32) for p in prompts},
            tot_w={p: np.zeros(shape, np.float32) for p in prompts},
            n_vis={p: np.zeros(shape, np.uint16) for p in prompts},
            max_pw={p: np.zeros(shape, np.float32) for p in prompts},
        )


def accumulate(
    *,
    grid: PlanGrid,
    height: np.ndarray,                   # (H, W) float32 with NaN outside
    keyframes: Iterable[KeyframeMaskBundle],
    M_align_inv: np.ndarray,              # mesh → ARKit (REQUIRED; see module doc)
    n_keyframes_estimate: int | None = None,
    p_angle: float = 2.0,
    p_dist: float = 1.0,
    d_ref_m: float = 1.5,
    cos_min: float = 0.20,
    max_dist_m: float | None = None,      # drop votes farther than this
    with_occlusion: bool = False,
    depth_tol_m: float = 0.20,
    use_score: bool = True,
    progress: bool = True,
) -> Accumulators:
    """Run one accumulation pass over ``keyframes`` for every prompt.

    Each keyframe contributes one weighted vote per prompt at every ortho
    pixel that (a) projects inside its image, (b) sits in front of the
    camera, (c) has ``cos(theta) >= cos_min`` (i.e. the camera isn't
    looking at the ceiling at a grazing angle), and (d) — if
    ``with_occlusion`` — passes the depth-buffer test against the
    keyframe's ``corrected_depth/<id>.png`` map.
    """
    H, W = height.shape
    valid = np.isfinite(height)
    vy, vx = np.nonzero(valid)
    z_world = height[valid].astype(np.float64)
    x_world, z_w = grid.px_to_world(
        vx.astype(np.float32) + 0.5, vy.astype(np.float32) + 0.5,
    )
    # Mesh-space points (PlanGrid + height.npy live in mesh space).
    P_mesh = np.stack([x_world.astype(np.float64), z_world, z_w.astype(np.float64)], axis=1)
    # ARKit-space points — what every camera projection below operates on.
    P_arkit = align_to_arkit(P_mesh, M_align_inv)

    accums = Accumulators.empty((H, W))

    n_kf = n_keyframes_estimate if n_keyframes_estimate is not None else -1
    for k_idx, kf in enumerate(keyframes):
        cam = kf.cam
        u, v, depth, dist = world_to_image(P_arkit, cam)
        in_front = depth > 0
        # cos(angle) between view dir (point->camera) and ceiling normal
        # (world -Y). Using ARKit-space y so the camera-to-ceiling vector
        # is in the same frame as `cam.t`.
        delta_y = (cam.t[1] - P_arkit[:, 1])
        cos_th = np.where(dist > 1e-9, np.abs(delta_y) / np.maximum(dist, 1e-9), 0.0)

        # When zc==0 the projection is undefined (NaN); clip to keep the
        # cast safe — those entries are masked off by `in_front` anyway.
        ui = np.round(np.where(np.isfinite(u), u, -1.0)).astype(np.int32)
        vi = np.round(np.where(np.isfinite(v), v, -1.0)).astype(np.int32)
        sel = in_front & (cos_th >= cos_min) & \
              (ui >= 0) & (ui < cam.W) & (vi >= 0) & (vi < cam.H)
        if max_dist_m is not None:
            sel &= dist <= max_dist_m
        if not sel.any():
            if progress:
                print(f"  [{k_idx+1}/{n_kf if n_kf > 0 else '?'}] {kf.image_id}: 0 visible (skip)")
            continue

        if with_occlusion and kf.depth_mm is not None:
            dh, dw = kf.depth_mm.shape
            # depth buffer is at lower resolution than RGB
            ud = (u * (dw / cam.W)).astype(np.int32)
            vd = (v * (dh / cam.H)).astype(np.int32)
            in_d = (ud >= 0) & (ud < dw) & (vd >= 0) & (vd < dh)
            depth_mask = sel & in_d
            if depth_mask.any():
                expected_m = depth  # already positive (== |z_cam|)
                d_at_uv = np.zeros_like(expected_m)
                d_at_uv[depth_mask] = kf.depth_mm[vd[depth_mask], ud[depth_mask]] / 1000.0
                # treat depth==0 as "no measurement, don't reject"
                has_meas = d_at_uv > 0
                # A point is occluded if the recorded depth is significantly
                # closer than the expected ceiling depth.
                occluded = depth_mask & has_meas & (d_at_uv < (expected_m - depth_tol_m))
                if kf.confidence is not None:
                    cf_at_uv = np.zeros_like(expected_m)
                    cf_at_uv[depth_mask] = kf.confidence[vd[depth_mask], ud[depth_mask]]
                    # only trust depth when confidence is high (255 = best)
                    occluded &= cf_at_uv >= 200
                sel &= ~occluded

        if not sel.any():
            if progress:
                print(f"  [{k_idx+1}/{n_kf if n_kf > 0 else '?'}] {kf.image_id}: 0 visible after occlusion")
            continue

        idx = np.where(sel)[0]
        ui_s = ui[idx]; vi_s = vi[idx]
        oy = vy[idx];   ox = vx[idx]

        w_dist = (d_ref_m / np.maximum(dist[idx], 1e-3)) ** p_dist
        w_angle = np.maximum(0.0, cos_th[idx]) ** p_angle
        w_base = (w_dist * w_angle).astype(np.float32)

        for prompt in PROMPTS:
            m = kf.masks.get(prompt)
            if m is None:
                continue
            mask_vals = m[vi_s, ui_s] > 0
            if use_score and (inst := kf.instances.get(prompt)) is not None:
                inst_vals = inst[vi_s, ui_s]
                sl = kf.scores.get(prompt, {})
                if sl:
                    max_id = max(sl)
                    sl_arr = np.ones(int(max_id) + 2, dtype=np.float32)
                    for iid, sc in sl.items():
                        if 0 <= int(iid) < len(sl_arr):
                            sl_arr[int(iid)] = float(sc)
                    inst_clip = np.clip(inst_vals, 0, len(sl_arr) - 1)
                    w_score = sl_arr[inst_clip]
                else:
                    w_score = np.ones_like(w_base)
            else:
                w_score = np.ones_like(w_base)

            w_pos = w_base * w_score * mask_vals.astype(np.float32)

            accums.tot_w[prompt][oy, ox] += w_base
            accums.pos_w[prompt][oy, ox] += w_pos
            accums.n_vis[prompt][oy, ox] += 1
            cur = accums.max_pw[prompt][oy, ox]
            np.maximum(cur, w_pos, out=cur)
            accums.max_pw[prompt][oy, ox] = cur

        if progress and (k_idx % 20 == 0 or k_idx == (n_kf - 1)):
            print(f"  [{k_idx+1}/{n_kf if n_kf > 0 else '?'}] {kf.image_id}: visible={int(sel.sum())} pts")

    return accums


# ─── FUSION ───────────────────────────────────────────────────────────────

def fuse(
    accums: Accumulators,
    *,
    rule: str = "weighted_majority",
    t_fuse: float = 0.5,
    min_keyframes: int = 3,
    max_pw_floor: float = 0.05,
) -> dict[str, np.ndarray]:
    """Reduce accumulator buffers to one binary mask per prompt.

    ``rule`` is one of ``"mean"``, ``"weighted_majority"``,
    ``"max_confidence"``. Common-sense:

    * ``mean``: ``pos_w / tot_w >= t_fuse``. Most sensitive to a single
      noisy mask when the ortho pixel was only seen by a couple of frames.
    * ``weighted_majority``: ``mean`` but also requires at least
      ``min_keyframes`` frames to have seen the pixel at all. The default,
      and the rule the experiment matrix sweeps ``t_fuse`` against.
    * ``max_confidence``: keep the pixel if the best single-keyframe
      weighted-positive vote exceeds ``max_pw_floor``. Cheaper, noisier.
    """
    out: dict[str, np.ndarray] = {}
    for p in accums.pos_w:
        pos = accums.pos_w[p]
        tot = accums.tot_w[p]
        n = accums.n_vis[p]
        if rule == "max_confidence":
            mask = accums.max_pw[p] >= max_pw_floor
        else:
            ratio = np.where(tot > 1e-6, pos / np.maximum(tot, 1e-6), 0.0)
            mask = ratio >= t_fuse
            if rule == "weighted_majority":
                mask &= n >= min_keyframes
        out[p] = mask.astype(np.uint8) * 255
    return out


# ─── INSTANCES + RENDER ───────────────────────────────────────────────────

def extract_instances(
    mask: np.ndarray,
    grid: PlanGrid,
    *,
    min_area_cm2: float = 80.0,
) -> tuple[np.ndarray, list[dict]]:
    """Connected-components → per-instance descriptor list.

    ``min_area_cm2`` drops noise blobs in real-world units. 80 cm² is
    roughly an 9 cm × 9 cm patch — smaller than any real ceiling fixture.
    """
    px_per_m = grid.pixels_per_metre
    px_per_cm2 = (px_per_m / 100.0) ** 2  # px² per cm²
    min_px = max(1, int(round(min_area_cm2 * px_per_cm2)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances = []
    keep_mask = np.zeros_like(mask)
    inst_id = 0
    for lab in range(1, n):
        x, y, w, h, area_px = stats[lab]
        if area_px < min_px:
            continue
        inst_id += 1
        keep_mask[labels == lab] = inst_id
        cx, cy = centroids[lab]
        # convert centroid to world XZ
        wx, wz = grid.px_to_world(np.array([cx]), np.array([cy]))
        instances.append(dict(
            id=int(inst_id),
            area_px=int(area_px),
            area_cm2=float(area_px / px_per_cm2),
            bbox_px=[int(x), int(y), int(x + w), int(y + h)],
            centroid_px=[float(cx), float(cy)],
            centroid_xz_m=[float(wx[0]), float(wz[0])],
        ))
    return keep_mask, instances


def overlay_concept(base_bgr: np.ndarray, mask: np.ndarray, color, alpha=0.45):
    """Tint ``mask>0`` pixels of ``base_bgr`` with ``color`` (BGR) at ``alpha``."""
    h = mask > 0
    if not h.any():
        return base_bgr.copy()
    out = base_bgr.copy()
    pix = out[h].astype(np.float32)
    blend = (1 - alpha) * pix + alpha * np.array(color, dtype=np.float32)
    out[h] = blend.astype(np.uint8)
    return out


# ─── PER-REGION CROSS-REFERENCE ───────────────────────────────────────────

def _polygon_to_px(poly_xz: list[list[float]], grid: PlanGrid) -> np.ndarray:
    arr = np.array(poly_xz, dtype=np.float32)
    u, v = grid.world_to_px(arr[:, 0], arr[:, 1])
    return np.stack([u, v], axis=1)


def per_region_counts(
    plan: dict,
    instances_by_concept: dict[str, list[dict]],
    grid: PlanGrid,
) -> dict:
    """Cross-reference per-prompt instance centroids against ``plan["regions"]``.

    Returns the user-facing summary: for the main ceiling and for each
    user-traced region polygon, the number of light/vent/ceiling-item
    instances whose centroid sits inside.
    """
    H = grid.height; W = grid.width

    def make_mask(poly: list[list[float]], holes: list[list[list[float]]] | None) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        if not poly:
            return mask
        contour = _polygon_to_px(poly, grid).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        for h in holes or []:
            if not h:
                continue
            cv2.fillPoly(mask, [_polygon_to_px(h, grid).astype(np.int32)], 0)
        return mask

    polys: list[tuple[str, str, np.ndarray]] = []
    main = plan.get("main") or {}
    if main.get("polygon"):
        polys.append(("main", main.get("label", "main"),
                      make_mask(main["polygon"], main.get("holes_polygons"))))
    for r in plan.get("regions", []):
        polys.append((f"region_{r['id']}", r.get("label", str(r["id"])),
                      make_mask(r["polygon"], r.get("holes_polygons"))))

    out_polys: list[dict] = []
    for key, label, mask in polys:
        entry: dict = dict(key=key, label=label)
        for concept, insts in instances_by_concept.items():
            n = 0
            for inst in insts:
                cx, cy = inst["centroid_px"]
                xi = int(round(cx)); yi = int(round(cy))
                if 0 <= xi < W and 0 <= yi < H and mask[yi, xi] > 0:
                    n += 1
            entry[concept] = n
        out_polys.append(entry)

    return dict(per_polygon=out_polys)


__all__ = [
    "PROMPTS", "PROMPT_SLUG", "PROMPT_COLORS",
    "PolycamCamera", "KeyframeMaskBundle", "Accumulators",
    "back_rotate", "cull_ceiling_items_against_specifics",
    "load_keyframe", "world_to_image",
    "load_alignment", "align_to_arkit",
    "accumulate", "fuse", "extract_instances", "overlay_concept",
    "per_region_counts",
]
