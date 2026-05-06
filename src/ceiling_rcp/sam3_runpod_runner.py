"""RunPod-backed SAM 3 → ortho projection pipeline.

Same shape as :mod:`ceiling_rcp.sam3_local_runner` but the per-frame
inference runs on a RunPod GPU pod instead of the local machine. This
is the recommended backend — H100 / 4090-class GPUs finish a
~270-frame scan in 5–10 min where MPS would take a couple of hours,
and the rest of the laptop stays cool.

Pod lifecycle is paranoid: atexit + SIGINT/SIGTERM handlers terminate
the pod before this process exits, even on Ctrl+C or a crash. If you
ever see a "[cleanup] WARNING" line, log into the RunPod dashboard
and confirm the pod is gone.

Layout written under the scan's ``Processed Outputs/sam3/`` matches
what the local runner produces — see that module's docstring for the
on-disk format. Only the inference engine differs.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .planes import PlanGrid
from .sam3_symbols import symbols_from_clusters, serialise
from .sam3_projection import (
    PROMPTS, PROMPT_COLORS, load_keyframe, load_alignment, per_region_counts,
)
from .sam3_clusters import (
    dbscan_cluster, fuse_cluster, load_keyframe_detections,
    project_polygons_to_ortho,
)
# Reuse the keyframe-filter + projection passes from the local runner —
# only the inference step differs across backends.
from .sam3_local_runner import (
    CEILING_FACING_FORWARD_Y, EPS_M, MIN_SAMPLES, MIN_CONFIDENCE,
    _build_working_set, _run_projection,
)


# ─── POD LIFECYCLE ────────────────────────────────────────────────────────

DOCKER_IMAGE = "roboflow/roboflow-inference-server-gpu:latest"
PORT = 9001

# Tried in order; first match wins. H100 first (the user's preferred
# SKU); 4090 + 3090 as cost-effective fallbacks; A-series + L4 as a
# last resort when consumer cards are oversubscribed. Naming follows
# RunPod's GPU type ids — keep these strings exact.
GPU_FALLBACKS: tuple[str, ...] = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A4000",
    "NVIDIA L4",
)

_pod_id_to_cleanup: Optional[str] = None
_terminate_on_exit: bool = True


def _cleanup_pod() -> None:
    """Stop or terminate the active pod. Safe to call multiple times."""
    global _pod_id_to_cleanup
    if _pod_id_to_cleanup is None:
        return
    pid = _pod_id_to_cleanup
    _pod_id_to_cleanup = None
    try:
        import runpod
        if _terminate_on_exit:
            print(f"[runpod:cleanup] terminating pod {pid}", flush=True)
            runpod.terminate_pod(pid)
        else:
            print(f"[runpod:cleanup] stopping pod {pid}", flush=True)
            runpod.stop_pod(pid)
    except Exception as e:
        print(f"[runpod:cleanup] WARNING: failed to clean up pod {pid}: {e}",
              flush=True)
        print(f"[runpod:cleanup] verify in https://console.runpod.io/pods that "
              f"{pid} is no longer running.", flush=True)


def _signal_handler(signum, frame):
    print(f"\n[runpod] received signal {signum}, cleaning up pod…", flush=True)
    _cleanup_pod()
    sys.exit(128 + signum)


def _arm_cleanup_hooks() -> None:
    atexit.register(_cleanup_pod)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _create_pod_with_fallback(
    *,
    runpod_api_key: str,
    roboflow_api_key: str,
    cloud_type: str,
    container_disk_in_gb: int = 20,
) -> dict:
    import runpod
    runpod.api_key = runpod_api_key

    last_err: Optional[Exception] = None
    for gpu in GPU_FALLBACKS:
        print(f"[runpod] requesting {gpu} on {cloud_type} cloud…", flush=True)
        try:
            pod = runpod.create_pod(
                name="sam3-ceiling-rcp",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu,
                cloud_type=cloud_type,
                container_disk_in_gb=container_disk_in_gb,
                ports=f"{PORT}/http",
                env={"ROBOFLOW_API_KEY": roboflow_api_key},
            )
            print(f"[runpod] got {gpu}: id={pod.get('id')}", flush=True)
            return pod
        except Exception as e:
            print(f"[runpod] {gpu} unavailable: {e}", flush=True)
            last_err = e
            time.sleep(2)
    raise RuntimeError(
        f"no GPU available across {GPU_FALLBACKS}; last error: {last_err}"
    )


def _wait_for_health(url: str, *, timeout_s: int = 600,
                     interval_s: float = 5.0) -> None:
    """Poll until SAM 3 is *actually* reachable on the pod.

    RunPod's proxy answers requests with 404 even before the container
    starts listening, so we probe two endpoints:
      1. /info — answers 200 once the inference server's HTTP layer is up.
      2. /sam3/concept_segment with empty body — any non-404 means the
         SAM3 route is registered (we expect 422 for the empty body).
    """
    import requests
    start = time.time()
    info_ok_at: Optional[float] = None
    last_obs = "no response yet"
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{url}/info", timeout=10)
            if r.status_code == 200:
                if info_ok_at is None:
                    info_ok_at = time.time()
                    print(f"[runpod:health] /info OK after "
                          f"{info_ok_at - start:.0f}s; waiting for SAM3 route…",
                          flush=True)
                try:
                    p = requests.post(
                        f"{url}/sam3/concept_segment", json={}, timeout=10,
                    )
                    if p.status_code != 404:
                        print(f"[runpod:health] /sam3/concept_segment ready "
                              f"after {time.time() - start:.0f}s "
                              f"(probe status={p.status_code})", flush=True)
                        return
                    last_obs = "SAM3 route still 404"
                except Exception as e:
                    last_obs = f"SAM3 probe: {type(e).__name__}"
            else:
                last_obs = f"/info HTTP {r.status_code}"
        except Exception as e:
            last_obs = f"/info: {type(e).__name__}"
        time.sleep(interval_s)
    raise TimeoutError(
        f"pod never became healthy in {timeout_s}s (last: {last_obs})"
    )


# ─── INFERENCE (mirrors run_runpod._process_image, on-disk format-compat) ─

def _polygon_to_mask(poly: list[list[float]], h: int, w: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def _polygon_bbox(poly: list[list[float]]) -> list[float]:
    arr = np.asarray(poly, dtype=float)
    return [
        float(arr[:, 0].min()), float(arr[:, 1].min()),
        float(arr[:, 0].max()), float(arr[:, 1].max()),
    ]


def _overlay_masks(base: Image.Image, instances: np.ndarray,
                   color: tuple[int, int, int],
                   alpha: float = 0.45) -> Image.Image:
    rgba = base.convert("RGBA")
    h, w = instances.shape
    if instances.max() > 0:
        tint = np.zeros((h, w, 4), dtype=np.uint8)
        tint[instances > 0] = (*color, int(255 * alpha))
        rgba = Image.alpha_composite(rgba, Image.fromarray(tint, "RGBA"))
    return rgba.convert("RGB")


def _annotate_overlay(img: Image.Image,
                      boxes_scores: list[tuple[list[float], float]],
                      color: tuple[int, int, int]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    for box, score in boxes_scores:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{score:.2f}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + th + 4], fill=color)
        draw.text((x1 + 3, y1 + 2), label, fill=(0, 0, 0), font=font)
    return out


def _save_results_for_prompt(
    out_dir: Path, prompt: str, base_img: Image.Image,
    predictions: list[dict], max_area_frac: float,
) -> dict:
    """Convert RunPod's polygon-format predictions into the same on-disk
    layout the local runner produces (mask.png + instances.png +
    overlay.jpg + entries in detections.json)."""
    pslug = prompt.replace(" ", "_")
    color = PROMPT_COLORS[prompt]
    w, h = base_img.size
    img_area = float(w * h)

    kept: list[dict] = []
    dropped_giant: list[dict] = []
    for p in predictions:
        polys = p.get("masks") or []
        if not polys:
            continue
        poly = polys[0]
        if len(poly) < 3:
            continue
        mask = _polygon_to_mask(poly, h, w)
        area_px = int(mask.sum())
        if area_px / img_area > max_area_frac:
            dropped_giant.append({
                "score": float(p.get("confidence", 0.0)),
                "area_px": area_px, "frac": area_px / img_area,
            })
            continue
        kept.append({
            "polygon": poly, "mask": mask,
            "score": float(p.get("confidence", 0.0)),
            "area_px": area_px,
        })

    if not kept:
        empty = base_img.copy()
        msg = f"{prompt}: no detections" + (
            f"  ({len(dropped_giant)} dropped > {max_area_frac*100:.0f}%)"
            if dropped_giant else ""
        )
        ImageDraw.Draw(empty).text((10, 10), msg, fill=color)
        empty.save(out_dir / f"{pslug}_overlay.jpg", quality=88)
        return {"prompt": prompt, "count": 0,
                "dropped_giant": dropped_giant, "detections": []}

    instances = np.zeros((h, w), dtype=np.uint16)
    detections: list[dict] = []
    for idx, k in enumerate(kept, start=1):
        instances[k["mask"]] = idx
        detections.append({
            "instance_id": idx,
            "score": k["score"],
            "polygon": k["polygon"],
            "box_xyxy": _polygon_bbox(k["polygon"]),
            "area_px": k["area_px"],
        })

    union = (instances > 0).astype(np.uint8) * 255
    Image.fromarray(union, "L").save(out_dir / f"{pslug}_mask.png")
    Image.fromarray(instances).save(out_dir / f"{pslug}_instances.png")
    overlay = _overlay_masks(base_img, instances, color)
    overlay = _annotate_overlay(
        overlay, [(d["box_xyxy"], d["score"]) for d in detections], color,
    )
    overlay.save(out_dir / f"{pslug}_overlay.jpg", quality=88)
    return {"prompt": prompt, "count": len(detections),
            "dropped_giant": dropped_giant, "detections": detections}


def _infer_one_image(
    img_path: Path, per_frame_dir: Path, client,
    *, threshold: float, max_area_frac: float,
) -> dict:
    image_id = img_path.stem
    out_dir = per_frame_dir / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    img.save(out_dir / "input.jpg", quality=92)

    per_prompt = []
    t0 = time.time()
    for prompt in PROMPTS:
        try:
            result = client.sam3_concept_segment(
                inference_input=str(img_path),
                prompts=[{"type": "text", "text": prompt}],
                output_prob_thresh=threshold,
                format="polygon",
            )
        except Exception as e:
            per_prompt.append({"prompt": prompt, "error": str(e)})
            continue
        if isinstance(result, list):
            result = result[0] if result else {}
        preds: list[dict] = []
        for pr in result.get("prompt_results", []) or []:
            preds.extend(pr.get("predictions", []) or [])
        per_prompt.append(
            _save_results_for_prompt(out_dir, prompt, img, preds, max_area_frac)
        )
    elapsed = time.time() - t0

    (out_dir / "detections.json").write_text(json.dumps({
        "image_id": image_id, "elapsed_s": elapsed,
        "prompts": per_prompt,
    }, indent=2))
    return {"image_id": image_id, "elapsed_s": elapsed, "prompts": per_prompt}


def _read_key_file(path: Path, what: str) -> str:
    if not path.exists():
        raise SystemExit(
            f"{what} key file missing: {path}. Drop your API key in there "
            f"(plain text, no newline-sensitive — leading/trailing "
            f"whitespace is stripped).\n"
            f"  • RunPod key: https://console.runpod.io/user/settings\n"
            f"  • Roboflow key: https://app.roboflow.com/settings/api"
        )
    return path.read_text().strip()


def _run_inference_runpod(
    working_set_dir: Path, per_frame_dir: Path,
    *,
    runpod_key_path: Path,
    roboflow_key_path: Path,
    cloud_type: str,
    threshold: float,
    max_area_frac: float,
    n_limit: int,
    no_terminate: bool,
    dry_run_id: str | None,
) -> dict:
    """Spin up a RunPod GPU pod, segment every working-set image with
    SAM 3 concept_segment, and tear the pod down — even on Ctrl+C or
    crash. Skips images that already have a detections.json so a
    re-run picks up where the last run was interrupted."""
    runpod_key = _read_key_file(runpod_key_path, "RunPod API")
    roboflow_key = _read_key_file(roboflow_key_path, "Roboflow API")

    images = sorted(p for p in working_set_dir.glob("*.jpg")
                    if not p.name.startswith("_"))
    if n_limit and n_limit > 0:
        images = images[:n_limit]
    if not images:
        raise SystemExit(f"no images to segment in {working_set_dir}")

    todo = [p for p in images
            if not (per_frame_dir / p.stem / "detections.json").exists()]
    print(f"[runpod] {len(todo)}/{len(images)} frames need inference")
    if not todo:
        return {"backend": "runpod", "images_total": len(images),
                "images_run": 0, "skipped": True}

    global _pod_id_to_cleanup, _terminate_on_exit
    _terminate_on_exit = not no_terminate
    _arm_cleanup_hooks()

    pod = _create_pod_with_fallback(
        runpod_api_key=runpod_key,
        roboflow_api_key=roboflow_key,
        cloud_type=cloud_type,
    )
    pod_id = pod["id"]
    _pod_id_to_cleanup = pod_id
    proxy_url = f"https://{pod_id}-{PORT}.proxy.runpod.net"
    print(f"[runpod] proxy: {proxy_url}", flush=True)

    print("[runpod] waiting for inference server…", flush=True)
    _wait_for_health(proxy_url, timeout_s=900)

    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=proxy_url, api_key=roboflow_key)

    per_frame_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    t_total = time.time()

    # Optional dry-run on a known-good frame first — also warms SAM3
    # weights inside the container so subsequent calls are fast.
    if dry_run_id:
        dry_path = next((p for p in todo if p.stem == dry_run_id), None)
        if dry_path is not None:
            print(f"[runpod] dry-run: {dry_path.stem}", flush=True)
            res = _infer_one_image(
                dry_path, per_frame_dir, client,
                threshold=threshold, max_area_frac=max_area_frac,
            )
            counts = {p["prompt"]: p.get("count", "ERR") for p in res["prompts"]}
            print(f"          {counts}  ({res['elapsed_s']:.1f}s)", flush=True)
            summary.append(res)
            todo = [p for p in todo if p.stem != dry_run_id]

    for i, p in enumerate(todo, 1):
        print(f"[runpod] [{i}/{len(todo)}] {p.stem}", flush=True)
        try:
            res = _infer_one_image(
                p, per_frame_dir, client,
                threshold=threshold, max_area_frac=max_area_frac,
            )
        except Exception as e:
            print(f"          ERROR: {e}", flush=True)
            summary.append({"image_id": p.stem, "error": str(e)})
            continue
        counts = {pp["prompt"]: pp.get("count", "ERR") for pp in res["prompts"]}
        print(f"          {counts}  ({res['elapsed_s']:.1f}s)", flush=True)
        summary.append(res)

    total_s = time.time() - t_total
    return {
        "backend": "runpod",
        "model": "roboflow/inference-server (sam3_final, concept_segment)",
        "pod_id": pod_id,
        "gpu": pod.get("machine", {}).get("gpuTypeId") or pod.get("gpuTypeId"),
        "cloud": cloud_type,
        "prompts": list(PROMPTS),
        "threshold": threshold,
        "max_area_frac": max_area_frac,
        "images_total": len(images),
        "images_run": len(summary),
        "total_s": total_s,
        "per_image": summary,
    }


# ─── ENTRY POINT ──────────────────────────────────────────────────────────

DEFAULT_RUNPOD_KEY = Path.home() / ".runpod_api_key"
DEFAULT_ROBOFLOW_KEY = Path.home() / ".roboflow_api_key"


def run_sam3_runpod(
    *,
    extracted_dir: Path,
    processed_outputs: Path,
    session_id: str,
    n_limit: int = 0,
    threshold: float = 0.5,
    max_area_frac: float = 0.50,
    min_forward_y: float = CEILING_FACING_FORWARD_Y,
    skip_inference: bool = False,
    cloud_type: str = "COMMUNITY",
    runpod_key_path: Path | None = None,
    roboflow_key_path: Path | None = None,
    no_terminate: bool = False,
    dry_run_id: str | None = "43179113618",
) -> dict:
    """End-to-end SAM 3 pipeline on a RunPod GPU. Mirrors
    :func:`ceiling_rcp.sam3_local_runner.run_sam3_local` shape for shape;
    only the inference step differs."""
    keyframes_dir = extracted_dir / "keyframes"
    if not keyframes_dir.exists():
        raise SystemExit(
            f"missing 'keyframes/' under {extracted_dir}. The raw "
            "Polycam zip is required (the OBJ-only zip doesn't ship "
            "the per-frame imagery SAM 3 needs)."
        )

    runpod_key_path = runpod_key_path or DEFAULT_RUNPOD_KEY
    roboflow_key_path = roboflow_key_path or DEFAULT_ROBOFLOW_KEY

    sam3_root = processed_outputs / "sam3"
    working_set_dir = sam3_root / "working_set"
    per_frame_dir = sam3_root / "per_frame"
    clusters_dir = sam3_root / "clusters"
    sam3_root.mkdir(parents=True, exist_ok=True)

    pipeline: dict[str, Any] = {"session_id": session_id, "backend": "runpod"}

    # 1. Filter ceiling-facing + rotate.
    t0 = time.time()
    manifest = _build_working_set(
        keyframes_dir, working_set_dir, min_forward_y=min_forward_y,
    )
    pipeline["working_set"] = {
        "n_frames": manifest["count"],
        "elapsed_s": time.time() - t0,
    }

    # 2. SAM 3 inference on RunPod.
    if skip_inference:
        print("[runpod] skipped inference (--skip-inference); reusing per-frame results")
        pipeline["inference"] = {"skipped": True}
    else:
        infer = _run_inference_runpod(
            working_set_dir, per_frame_dir,
            runpod_key_path=runpod_key_path,
            roboflow_key_path=roboflow_key_path,
            cloud_type=cloud_type,
            threshold=threshold,
            max_area_frac=max_area_frac,
            n_limit=n_limit,
            no_terminate=no_terminate,
            dry_run_id=dry_run_id,
        )
        pipeline["inference"] = infer

    # 3. Polygon back-projection + DBSCAN clustering.
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

    # 4. Symbol classification → Processed Outputs/symbols.json.
    symbols = symbols_from_clusters(cluster_summary)
    sym_doc = serialise(
        symbols, grid=grid, session_id=session_id,
        ceiling_image_relpath="ceiling.jpg",
    )
    (processed_outputs / "symbols.json").write_text(json.dumps(sym_doc, indent=2))
    by_class: dict[str, int] = {}
    for s in symbols:
        by_class[s.cls] = by_class.get(s.cls, 0) + 1
    pipeline["symbols"] = {"n_total": len(symbols), "by_class": by_class}

    (sam3_root / "_pipeline.json").write_text(json.dumps(pipeline, indent=2))
    return pipeline


__all__ = [
    "run_sam3_runpod",
    "GPU_FALLBACKS",
    "DEFAULT_RUNPOD_KEY",
    "DEFAULT_ROBOFLOW_KEY",
]
