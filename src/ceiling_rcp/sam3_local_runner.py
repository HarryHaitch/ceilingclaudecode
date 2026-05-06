"""Local SAM 3 → ortho projection pipeline, end-to-end.

Glues the four existing pieces — keyframe selection, SAM 3 inference,
polygon back-projection + DBSCAN clustering, symbol classification —
into one orchestrator that ``ceiling-rcp-process --sam3 local`` calls
on a Polycam scan folder.

Layout written under the scan's ``Processed Outputs/sam3/``::

    sam3/
    ├── working_set/
    │   ├── _manifest.json              filter result + rotation note
    │   └── <keyframe_id>.jpg           rotated CW90, ceiling on top
    ├── per_frame/
    │   └── <keyframe_id>/
    │       ├── input.jpg               rotated input
    │       ├── ceiling_item_mask.png   binary union  (raw, labelled)
    │       ├── ceiling_item_instances.png  uint16 label map
    │       ├── ceiling_item_overlay.jpg    coloured QC view
    │       ├── light_*.png ...
    │       ├── vent_*.png ...
    │       └── detections.json         per-instance score + box + area
    ├── clusters/
    │   ├── clusters_combined.png
    │   ├── clusters_<concept>.png
    │   ├── clusters_<concept>_mask.png
    │   ├── per_region_counts.json
    │   └── summary.json                cluster-level dump (input to symbols)
    └── _pipeline.json                  per-stage timing + counts

The user-visible canonical output (``Processed Outputs/symbols.json``)
is built from ``clusters/summary.json`` via the existing
:mod:`ceiling_rcp.sam3_symbols` rules. Both files survive — ``symbols.json``
is what the viewer reads; ``sam3/`` is the audit trail you can come
back to when classification rules change without re-running SAM 3.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .planes import PlanGrid
from .sam3_symbols import (
    SYMBOL_CLASSES, symbols_from_clusters, serialise,
)
from .sam3_projection import (
    PROMPTS, PROMPT_COLORS, load_keyframe, per_region_counts,
)
from .sam3_clusters import (
    dbscan_cluster, fuse_cluster, load_keyframe_detections,
    project_polygons_to_ortho,
)


# ─── KEYFRAME FILTER + ROTATE ────────────────────────────────────────────

CEILING_FACING_FORWARD_Y = 0.20  # min -t_12 (= world-Y of cam forward) to keep


def _select_ceiling_facing(cam_dir: Path, min_forward_y: float) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    for cam_path in sorted(cam_dir.glob("*.json")):
        try:
            cam = json.loads(cam_path.read_text())
        except Exception:
            continue
        forward_y = -float(cam.get("t_12", 0.0))   # OpenGL: forward = -Z
        if forward_y > min_forward_y:
            kept.append((cam_path.stem, forward_y))
    return kept


def _build_working_set(
    keyframes_dir: Path, working_set_dir: Path, *, min_forward_y: float,
) -> dict:
    """Filter keyframes to ceiling-facing + rotate CW 90° so world-up sits
    at the top of the image. Idempotent — reuses an existing working set
    when its manifest matches the current params."""
    cam_dir = keyframes_dir / "corrected_cameras"
    img_dir = keyframes_dir / "corrected_images"
    if not cam_dir.exists() or not img_dir.exists():
        raise SystemExit(
            f"missing corrected_cameras/ or corrected_images/ under "
            f"{keyframes_dir}. The Polycam raw export is required."
        )
    manifest_path = working_set_dir / "_manifest.json"
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
            if (abs(old.get("min_forward_y", -1) - min_forward_y) < 1e-9
                    and old.get("count", 0) > 0):
                print(f"[sam3:filter] reusing working set ({old['count']} frames)")
                return old
        except Exception:
            pass
    working_set_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_ceiling_facing(cam_dir, min_forward_y)
    total = len(list(cam_dir.glob("*.json")))
    print(f"[sam3:filter] {len(selected)}/{total} ceiling-facing frames "
          f"(min -t_12 > {min_forward_y})")

    items = []
    for stem, forward_y in selected:
        src = img_dir / f"{stem}.jpg"
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        # PIL ROTATE_270 == counter-clockwise 270° == clockwise 90°
        rot = img.transpose(Image.Transpose.ROTATE_270)
        dst = working_set_dir / f"{stem}.jpg"
        rot.save(dst, quality=92)
        items.append({"id": stem, "forward_y": forward_y,
                      "rotated_size": list(rot.size)})

    manifest = {
        "min_forward_y": min_forward_y,
        "convention": "OpenGL (forward = -Z)",
        "rotation": "clockwise 90° (PIL ROTATE_270)",
        "count": len(items),
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# ─── SAM 3 INFERENCE OVER WORKING SET ────────────────────────────────────

def _run_sam3_inference(
    working_set_dir: Path,
    per_frame_dir: Path,
    *,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    n_limit: int = 0,
) -> dict:
    """Walk every JPG in the working set, run SAM 3 with the three
    text prompts, and dump per-frame outputs. Skips images that
    already have a ``detections.json`` (idempotent)."""
    # Late import — torch + transformers are heavy and most people running
    # the rest of the app don't have them installed.
    import torch
    from transformers import Sam3Model, Sam3Processor

    # The experiment script's own helpers do exactly what we want —
    # reuse them so the on-disk format matches the existing results.
    # Add the repo root to sys.path so ``experiments.*`` resolves when
    # the runner is invoked through the installed package entry point
    # (which doesn't put the repo on the path automatically).
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from experiments.segmentation_sam3.run_sam3 import (
        process_image as _sam3_process_image, MODEL_ID,
    )

    images = sorted(p for p in working_set_dir.glob("*.jpg")
                    if not p.name.startswith("_"))
    if n_limit and n_limit > 0:
        images = images[:n_limit]
    if not images:
        raise SystemExit(f"no images to segment in {working_set_dir}")

    # Skip images that already have a detections.json — useful when a
    # previous run was interrupted and you re-run without --reprocess.
    todo = [p for p in images
            if not (per_frame_dir / p.stem / "detections.json").exists()]
    print(f"[sam3:infer] {len(todo)}/{len(images)} frames need inference")

    if not todo:
        return {"images_total": len(images), "images_run": 0,
                "device": "skipped", "model": MODEL_ID}

    # Pick device. MPS is the slow path; print a loud warning so the user
    # isn't surprised by a multi-hour run.
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device == "mps":
        print(f"[sam3:infer] WARNING: running on MPS — first-frame load "
              f"is ~2 min, then ~5–10 s per frame. {len(todo)} frames "
              f"≈ {len(todo) * 8 / 60:.0f} min. Use --sam3-n to cap "
              f"if you just want a smoke test.")
    print(f"[sam3:infer] loading {MODEL_ID} on {device}…")
    t_load = time.time()
    model = Sam3Model.from_pretrained(MODEL_ID).to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    print(f"[sam3:infer] model loaded in {time.time() - t_load:.1f}s")

    per_frame_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    t_all = time.time()
    for i, p in enumerate(todo, 1):
        print(f"[sam3:infer] [{i}/{len(todo)}] {p.stem}")
        try:
            res = _sam3_process_image(
                p, per_frame_dir, model, processor, device,
                threshold, mask_threshold,
            )
            counts = {item["prompt"]: item["count"] for item in res["prompts"]}
            print(f"            {counts}  ({res['elapsed_s']:.1f}s)")
            summary.append(res)
        except Exception as e:
            print(f"            ERROR: {e}")
            summary.append({"image_id": p.stem, "error": str(e)})

    return {
        "model": MODEL_ID,
        "device": device,
        "prompts": list(PROMPTS),
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "images_total": len(images),
        "images_run": len(todo),
        "total_s": time.time() - t_all,
        "per_image": summary,
    }


# ─── PROJECTION + DBSCAN CLUSTERING ──────────────────────────────────────
#
# Geometry now comes from the LiDAR depth maps Polycam ships per
# keyframe (see sam3_depth_projection.py). Earlier versions ray-cast
# polygons through a constant ceiling height + height.npy refinement
# and gated the result with cos_min / max_dist_m filters; that
# combination silently squashes scans whose ceilings sit close to the
# camera. Depth-map projection has no per-scan tuning knobs.

EPS_M = 0.15        # DBSCAN cluster radius — same units as ortho metres
MIN_SAMPLES = 3
MIN_CONFIDENCE = 1  # ARKit: 0=low, 1=medium, 2=high. 1 drops only low.


def _run_projection(
    *,
    grid: PlanGrid, height: np.ndarray, ceiling: np.ndarray, plan: dict,
    cameras_dir: Path, results_dir: Path, mesh_info_path: Path,
    depth_dir: Path, confidence_dir: Path,
    out_dir: Path,
) -> dict:
    """Project every per-keyframe SAM 3 detection onto the ortho via
    LiDAR depth, DBSCAN-cluster across keyframes, OR-fuse cluster
    polygons into one footprint per fixture. ``height`` is no longer
    used for projection — it stays in the signature only so callers
    can pass it through to the other downstream consumers (e.g. heatmap
    rendering). Schema of the per-instance dicts is unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from .sam3_depth_projection import load_depth

    # mesh_info_path is no longer needed for projection — depth maps
    # give us ARKit-space world points directly, and the grid is already
    # in ARKit space (load_mesh applies apply_alignment_inv before
    # make_grid runs). Argument is kept for signature stability and
    # potential future per-scan diagnostics.
    _ = mesh_info_path

    ids = sorted(d.name for d in results_dir.iterdir()
                 if d.is_dir() and (cameras_dir / f"{d.name}.json").exists())
    print(f"[sam3:project] {len(ids)} keyframes have masks + cameras")

    px_per_m = grid.pixels_per_metre
    eps_px = EPS_M * px_per_m

    all_proj: dict[str, list[dict]] = {p: [] for p in PROMPTS}
    n_kept = {p: 0 for p in PROMPTS}
    n_seen = {p: 0 for p in PROMPTS}
    n_no_depth = 0
    t0 = time.time()
    for idx, image_id in enumerate(ids):
        kf = load_keyframe(
            image_id, cameras_dir=cameras_dir,
            results_dir=results_dir, cull_catchall=True,
        )
        if kf is None:
            continue
        # Load the per-keyframe LiDAR depth (auto-upscaled to the camera's
        # full resolution). Skip the keyframe entirely if the LiDAR data
        # is missing or too noisy to use.
        depth_m = load_depth(
            image_id,
            depth_dir=depth_dir, confidence_dir=confidence_dir,
            target_size=(kf.cam.W, kf.cam.H),
            min_confidence=MIN_CONFIDENCE,
        )
        if depth_m is None:
            n_no_depth += 1
            continue
        det_by_prompt = load_keyframe_detections(
            image_id, results_dir=results_dir,
            masks_for_cull=kf.masks, cull_overlap=0.5,
        )
        for prompt, dets in det_by_prompt.items():
            n_seen[prompt] += len(dets)
            proj = project_polygons_to_ortho(
                dets, kf.cam,
                grid=grid, depth_m=depth_m,
            )
            n_kept[prompt] += len(proj)
            all_proj[prompt].extend(proj)
        if (idx + 1) % 30 == 0 or idx == len(ids) - 1:
            print(f"[sam3:project]   [{idx+1}/{len(ids)}] kept "
                  f"{ {p: n_kept[p] for p in PROMPTS} }")
    if n_no_depth:
        print(f"[sam3:project] {n_no_depth} keyframe(s) had no usable depth — skipped")
    print(f"[sam3:project] done in {time.time()-t0:.1f}s; kept {n_kept}")

    overlay = ceiling.copy()
    instances_by_concept: dict[str, list[dict]] = {p: [] for p in PROMPTS}
    summary = {
        "settings": dict(eps_m=EPS_M, min_samples=MIN_SAMPLES,
                         min_confidence=MIN_CONFIDENCE,
                         projection="depth_map"),
        "concepts": {},
    }
    for prompt in PROMPTS:
        proj = all_proj[prompt]
        if not proj:
            summary["concepts"][prompt] = dict(n_instances=0, n_detections_kept=0,
                                                n_detections_noise=0,
                                                instances=[])
            continue
        centroids = np.stack([d["ortho_centroid_px"] for d in proj])
        labels = dbscan_cluster(centroids, eps=eps_px, min_samples=MIN_SAMPLES)
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        n_noise = int((labels == -1).sum())
        print(f"[sam3:project]   {prompt}: {len(proj)} detections → "
              f"{n_clusters} clusters ({n_noise} noise)")

        full_mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
        prompt_overlay = ceiling.copy()
        color = np.array(PROMPT_COLORS[prompt], dtype=np.uint8)
        instance_descs = []
        next_id = 0
        for c in range(n_clusters):
            cluster_dets = [proj[i] for i in np.where(labels == c)[0]]
            if len(cluster_dets) < MIN_SAMPLES:
                continue
            res = fuse_cluster(cluster_dets, grid=grid)
            if res is None:
                continue
            inst, fused, (x0, y0) = res
            next_id += 1
            inst.id = next_id
            h_, w_ = fused.shape
            sub = full_mask[y0:y0+h_, x0:x0+w_]
            sub[fused > 0] = 255
            full_mask[y0:y0+h_, x0:x0+w_] = sub

            for canvas in (prompt_overlay, overlay):
                pix_mask = np.zeros_like(canvas[..., 0])
                pix_mask[y0:y0+h_, x0:x0+w_] = fused
                hit = pix_mask > 0
                base = canvas[hit].astype(np.float32)
                canvas[hit] = (0.45 * base + 0.55 * color.astype(np.float32)
                               ).astype(np.uint8)

            rr = inst.rotated_rect
            box = cv2.boxPoints((
                (rr["cx_px"], rr["cy_px"]),
                (rr["w_px"], rr["h_px"]),
                rr["angle_deg"],
            )).astype(np.int32)
            cv2.drawContours(prompt_overlay, [box], -1,
                             tuple(int(x) for x in color), 2)

            cx, cy = int(round(inst.centroid_px[0])), int(round(inst.centroid_px[1]))
            cv2.circle(prompt_overlay, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(prompt_overlay, (cx, cy), 6,
                       tuple(int(x) for x in color), 2)
            label = f"{inst.id} ({inst.n_views})"
            cv2.putText(prompt_overlay, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(prompt_overlay, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        tuple(int(x) for x in color), 1, cv2.LINE_AA)

            d = asdict(inst)
            d["centroid_px"] = list(inst.centroid_px)
            d["centroid_xz_m"] = list(inst.centroid_xz_m)
            d["bbox_px"] = list(inst.bbox_px)
            d["mean_mask_diameter_cm"] = inst.mean_mask_diameter_cm
            instance_descs.append(d)
            instances_by_concept[prompt].append(dict(
                id=inst.id,
                centroid_px=list(inst.centroid_px),
                centroid_xz_m=list(inst.centroid_xz_m),
                bbox_px=list(inst.bbox_px),
                area_cm2=inst.area_cm2,
            ))

        cv2.imwrite(str(out_dir / f"clusters_{prompt.replace(' ', '_')}.png"),
                    prompt_overlay)
        cv2.imwrite(str(out_dir / f"clusters_{prompt.replace(' ', '_')}_mask.png"),
                    full_mask)
        summary["concepts"][prompt] = dict(
            n_instances=len(instance_descs),
            n_detections_kept=len(proj),
            n_detections_noise=n_noise,
            instances=instance_descs,
        )

    cv2.imwrite(str(out_dir / "clusters_combined.png"), overlay)

    per_region = per_region_counts(plan, instances_by_concept, grid)
    (out_dir / "per_region_counts.json").write_text(
        json.dumps(per_region, indent=2)
    )
    summary["per_region"] = per_region["per_polygon"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    return summary


# ─── ENTRY POINT ──────────────────────────────────────────────────────────

def run_sam3_local(
    *,
    extracted_dir: Path,         # Polycam Outputs/_extracted/
    processed_outputs: Path,     # the scan's Processed Outputs/
    session_id: str,
    n_limit: int = 0,            # 0 = all images; >0 = first N for smoke test
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_forward_y: float = CEILING_FACING_FORWARD_Y,
    skip_inference: bool = False,
) -> dict:
    """Run the full SAM 3 ceiling-services pipeline on a scan and write
    ``Processed Outputs/symbols.json`` plus the audit folder
    ``Processed Outputs/sam3/``.

    Returns a per-stage timing + counts dict that the CLI prints."""
    keyframes_dir = extracted_dir / "keyframes"
    if not keyframes_dir.exists():
        raise SystemExit(
            f"missing 'keyframes/' under {extracted_dir}. The raw "
            "Polycam zip is required (the OBJ-only zip doesn't ship "
            "the per-frame imagery SAM 3 needs)."
        )

    sam3_root = processed_outputs / "sam3"
    working_set_dir = sam3_root / "working_set"
    per_frame_dir = sam3_root / "per_frame"
    clusters_dir = sam3_root / "clusters"
    sam3_root.mkdir(parents=True, exist_ok=True)

    pipeline: dict[str, Any] = {"session_id": session_id}

    # 1. Filter ceiling-facing + rotate.
    t0 = time.time()
    manifest = _build_working_set(
        keyframes_dir, working_set_dir, min_forward_y=min_forward_y,
    )
    pipeline["working_set"] = {
        "n_frames": manifest["count"],
        "elapsed_s": time.time() - t0,
    }

    # 2. SAM 3 inference (per-frame raw outputs).
    if skip_inference:
        print("[sam3:infer] skipped (--skip-inference); reusing existing per-frame results")
        pipeline["inference"] = {"skipped": True}
    else:
        t0 = time.time()
        infer = _run_sam3_inference(
            working_set_dir, per_frame_dir,
            threshold=threshold, mask_threshold=mask_threshold,
            n_limit=n_limit,
        )
        infer["elapsed_s"] = time.time() - t0
        pipeline["inference"] = infer

    # 3. Polygon back-projection + DBSCAN clusters → cluster summary.
    cameras_dir = keyframes_dir / "corrected_cameras"
    mesh_info_path = extracted_dir / "mesh_info.json"
    if not mesh_info_path.exists():
        raise SystemExit(
            f"missing mesh_info.json under {extracted_dir}; required for "
            "alignment between the OBJ mesh and the keyframe cameras."
        )

    plan = json.loads((processed_outputs / "plan.json").read_text())
    g = plan["grid"]
    grid = PlanGrid(
        min_x=g["min_x"], max_x=g["max_x"],
        min_z=g["min_z"], max_z=g["max_z"],
        pixels_per_metre=g["pixels_per_metre"],
    )
    height = np.load(processed_outputs / "height.npy")
    ceiling = cv2.imread(str(processed_outputs / "ceiling.jpg"))

    t0 = time.time()
    cluster_summary = _run_projection(
        grid=grid, height=height, ceiling=ceiling, plan=plan,
        cameras_dir=cameras_dir,
        results_dir=per_frame_dir,
        mesh_info_path=mesh_info_path,
        depth_dir=keyframes_dir / "depth",
        confidence_dir=keyframes_dir / "confidence",
        out_dir=clusters_dir,
    )
    pipeline["projection"] = {
        "elapsed_s": time.time() - t0,
        "concepts": {p: c["n_instances"]
                     for p, c in cluster_summary["concepts"].items()},
    }

    # 4. Symbols (apply Diffuser/Downlight/LED/Sprinkler/Misc rules).
    symbols = symbols_from_clusters(cluster_summary)
    sym_doc = serialise(
        symbols, grid=grid, session_id=session_id,
        ceiling_image_relpath="ceiling.jpg",
    )
    (processed_outputs / "symbols.json").write_text(json.dumps(sym_doc, indent=2))
    by_class: dict[str, int] = {}
    for s in symbols:
        by_class[s.cls] = by_class.get(s.cls, 0) + 1
    pipeline["symbols"] = {
        "n_total": len(symbols),
        "by_class": by_class,
    }

    (sam3_root / "_pipeline.json").write_text(json.dumps(pipeline, indent=2))
    return pipeline


__all__ = ["run_sam3_local", "CEILING_FACING_FORWARD_Y"]
