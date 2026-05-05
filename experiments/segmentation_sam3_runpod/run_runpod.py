"""Spin up a RunPod GPU pod running roboflow/inference-server, batch-segment
the working set with SAM3 concept_segment, and tear the pod down again.

Hard guarantee: the pod is always cleaned up before this script exits, even
on Ctrl+C, an unexpected exception, or `kill <pid>`. We register both an
atexit handler and SIGINT/SIGTERM handlers, and the main loop runs inside
a try/finally with the pod-id captured. If everything works, you should
see a final "[cleanup] terminated pod ..." line and the RunPod dashboard
should show the pod as terminated within ~30 seconds.

Per-image output (under `results/<image_id>/`) matches the layout of the
local `experiments/segmentation_sam3/results_sample/` folders:

    input.jpg                   the rotated input image
    <prompt>_mask.png           binary union mask  (0 / 255)
    <prompt>_instances.png      uint16 label map   (0=bg, 1..N=instance)
    <prompt>_overlay.jpg        coloured overlay + boxes + score labels
    detections.json             per-prompt list of {score, polygon, box, area_px}
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import runpod
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont


PROMPTS: list[str] = ["ceiling item", "light", "vent"]
PROMPT_COLORS: dict[str, tuple[int, int, int]] = {
    "ceiling item": (255, 64, 64),
    "light": (255, 220, 64),
    "vent": (64, 200, 255),
}

DOCKER_IMAGE = "roboflow/roboflow-inference-server-gpu:latest"
PORT = 9001

# Tried in order; falls back to the next if the first is unavailable.
GPU_FALLBACKS: list[str] = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A4000",
    "NVIDIA L4",
]


# ---------------------------------------------------------------------------
# Pod lifecycle — paranoid cleanup
# ---------------------------------------------------------------------------

_pod_id_to_cleanup: Optional[str] = None
_terminate_on_exit: bool = True


def _cleanup() -> None:
    """Stop or terminate the pod. Safe to call multiple times."""
    global _pod_id_to_cleanup
    if _pod_id_to_cleanup is None:
        return
    pid = _pod_id_to_cleanup
    _pod_id_to_cleanup = None
    try:
        if _terminate_on_exit:
            print(f"[cleanup] terminating pod {pid}", flush=True)
            runpod.terminate_pod(pid)
        else:
            print(f"[cleanup] stopping pod {pid}", flush=True)
            runpod.stop_pod(pid)
    except Exception as e:
        print(f"[cleanup] WARNING: failed to clean up pod {pid}: {e}", flush=True)
        print(
            f"[cleanup] please verify in https://console.runpod.io/pods that {pid} "
            "is no longer running.",
            flush=True,
        )


def _signal_handler(signum, frame):
    print(f"\n[signal] received signal {signum}, cleaning up pod...", flush=True)
    _cleanup()
    sys.exit(128 + signum)


def _arm_cleanup_hooks() -> None:
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _create_pod_with_fallback(
    name: str,
    image: str,
    ports: str,
    container_disk_in_gb: int,
    env: dict,
    cloud_type: str,
) -> dict:
    last_err: Optional[Exception] = None
    for gpu in GPU_FALLBACKS:
        print(f"[pod] requesting {gpu} on {cloud_type} cloud...", flush=True)
        try:
            pod = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu,
                cloud_type=cloud_type,
                container_disk_in_gb=container_disk_in_gb,
                ports=ports,
                env=env,
            )
            print(f"[pod] got {gpu}: id={pod.get('id')}", flush=True)
            return pod
        except Exception as e:
            print(f"[pod] {gpu} unavailable: {e}", flush=True)
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"no GPU available across {GPU_FALLBACKS}; last error: {last_err}")


def _wait_for_health(url: str, timeout_s: int, interval_s: float = 5.0) -> None:
    """Poll until the SAM 3 endpoint is *actually* registered.

    RunPod's proxy returns 404 to anything reaching the host even before
    the container starts listening, so a plain "any 2xx/4xx" health check
    passes far too early. We instead probe two endpoints in sequence:

    1. /info — answers 200 once the inference server's HTTP layer is up.
    2. /sam3/concept_segment with empty body — any response other than
       404 means the SAM 3 route is registered (we expect 422 for the
       missing body). 404 means the foundation-model handlers haven't
       loaded yet.
    """
    start = time.time()
    info_ok_at: Optional[float] = None
    last_obs = "no response yet"
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{url}/info", timeout=10)
            if r.status_code == 200:
                if info_ok_at is None:
                    info_ok_at = time.time()
                    print(
                        f"[health] /info OK after {info_ok_at-start:.0f}s "
                        f"({len(r.content)}B); now waiting for SAM3 route...",
                        flush=True,
                    )
                # /info up — probe the SAM3 route
                try:
                    p = requests.post(
                        f"{url}/sam3/concept_segment", json={}, timeout=10
                    )
                    if p.status_code != 404:
                        print(
                            f"[health] /sam3/concept_segment registered after "
                            f"{time.time()-start:.0f}s (probe status={p.status_code})",
                            flush=True,
                        )
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


# ---------------------------------------------------------------------------
# Inference + rendering (mirrors run_sam3.py output layout)
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    return s.replace(" ", "_")


def _polygon_to_mask(poly: list[list[float]], h: int, w: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def _polygon_bbox(poly: list[list[float]]) -> list[float]:
    arr = np.asarray(poly, dtype=float)
    return [
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    ]


def _overlay_masks(
    base: Image.Image,
    instances: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    rgba = base.convert("RGBA")
    h, w = instances.shape
    if instances.max() > 0:
        tint = np.zeros((h, w, 4), dtype=np.uint8)
        tint[instances > 0] = (*color, int(255 * alpha))
        layer = Image.fromarray(tint, "RGBA")
        rgba = Image.alpha_composite(rgba, layer)
    return rgba.convert("RGB")


def _annotate_overlay(
    img: Image.Image,
    boxes_scores: list[tuple[list[float], float]],
    color: tuple[int, int, int],
) -> Image.Image:
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
    out_dir: Path,
    prompt: str,
    base_img: Image.Image,
    predictions: list[dict],
    max_area_frac: float,
) -> dict:
    pslug = _slug(prompt)
    color = PROMPT_COLORS[prompt]
    w, h = base_img.size
    img_area = float(w * h)

    kept: list[dict] = []
    dropped_giant: list[dict] = []
    for p in predictions:
        polys = p.get("masks") or []
        if not polys:
            continue
        # Roboflow returns one polygon per detection. Take the first.
        poly = polys[0]
        if len(poly) < 3:
            continue
        mask = _polygon_to_mask(poly, h, w)
        area_px = int(mask.sum())
        if area_px / img_area > max_area_frac:
            dropped_giant.append({
                "score": float(p.get("confidence", 0.0)),
                "area_px": area_px,
                "frac": area_px / img_area,
            })
            continue
        kept.append({
            "polygon": poly,
            "mask": mask,
            "score": float(p.get("confidence", 0.0)),
            "area_px": area_px,
        })

    if not kept:
        empty = base_img.copy()
        ImageDraw.Draw(empty).text(
            (10, 10),
            f"{prompt}: no detections" + (
                f"  ({len(dropped_giant)} dropped > {max_area_frac*100:.0f}%)"
                if dropped_giant else ""
            ),
            fill=color,
        )
        empty.save(out_dir / f"{pslug}_overlay.jpg", quality=88)
        return {
            "prompt": prompt,
            "count": 0,
            "dropped_giant": dropped_giant,
            "detections": [],
        }

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
        overlay,
        [(d["box_xyxy"], d["score"]) for d in detections],
        color,
    )
    overlay.save(out_dir / f"{pslug}_overlay.jpg", quality=88)

    return {
        "prompt": prompt,
        "count": len(detections),
        "dropped_giant": dropped_giant,
        "detections": detections,
    }


def _process_image(
    img_path: Path,
    out_root: Path,
    client: InferenceHTTPClient,
    threshold: float,
    max_area_frac: float,
) -> dict:
    image_id = img_path.stem
    out_dir = out_root / image_id
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

    with (out_dir / "detections.json").open("w") as f:
        json.dump(
            {"image_id": image_id, "elapsed_s": elapsed, "prompts": per_prompt},
            f,
            indent=2,
        )
    return {"image_id": image_id, "elapsed_s": elapsed, "prompts": per_prompt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--working-set",
        type=Path,
        default=Path(__file__).parent.parent
        / "segmentation_sam3"
        / "working_set",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    ap.add_argument(
        "--runpod-key-file",
        type=Path,
        default=Path.home() / ".runpod_api_key",
    )
    ap.add_argument(
        "--roboflow-key-file",
        type=Path,
        default=Path.home() / ".roboflow_api_key",
    )
    ap.add_argument("--n", type=int, default=0, help="0 = all images")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated image ids; overrides --n")
    ap.add_argument("--cloud", default="COMMUNITY",
                    choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-area-frac", type=float, default=0.50,
                    help="Drop any detection whose mask covers more than this "
                         "fraction of the image (sanity gate against the "
                         "borderline 'whole ceiling = ceiling item' case).")
    ap.add_argument("--no-terminate", action="store_true",
                    help="Stop the pod (preserves disk) instead of terminating "
                         "(default terminates so disk costs go to zero).")
    ap.add_argument("--dry-run-id", type=str, default="43179113618",
                    help="Image id to process first as a single-image sanity "
                         "check before the rest. Empty string disables.")
    args = ap.parse_args()

    global _terminate_on_exit
    _terminate_on_exit = not args.no_terminate

    runpod_key = args.runpod_key_file.read_text().strip()
    roboflow_key = args.roboflow_key_file.read_text().strip()
    runpod.api_key = runpod_key

    images = sorted(args.working_set.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {args.working_set}")
    if args.ids:
        wanted = set(args.ids.split(","))
        images = [p for p in images if p.stem in wanted]
    elif args.n and args.n > 0:
        images = images[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"will process {len(images)} images, prompts={PROMPTS}", flush=True)
    print(f"output -> {args.out}", flush=True)

    _arm_cleanup_hooks()

    pod = _create_pod_with_fallback(
        name="sam3-batch",
        image=DOCKER_IMAGE,
        ports=f"{PORT}/http",
        container_disk_in_gb=20,
        env={"ROBOFLOW_API_KEY": roboflow_key},
        cloud_type=args.cloud,
    )
    pod_id = pod["id"]
    global _pod_id_to_cleanup
    _pod_id_to_cleanup = pod_id
    proxy_url = f"https://{pod_id}-{PORT}.proxy.runpod.net"
    print(f"[pod] proxy URL: {proxy_url}", flush=True)

    print("[health] waiting for inference server...", flush=True)
    _wait_for_health(proxy_url, timeout_s=600)

    client = InferenceHTTPClient(api_url=proxy_url, api_key=roboflow_key)

    summary: list[dict] = []
    t_total = time.time()

    # 1) optional dry-run on a known image first (also warms the SAM3 weights)
    if args.dry_run_id:
        dry_path = next((p for p in images if p.stem == args.dry_run_id), None)
        if dry_path is not None:
            print(f"\n[dry-run] {dry_path.stem}", flush=True)
            res = _process_image(
                dry_path, args.out, client, args.threshold, args.max_area_frac
            )
            counts = {p["prompt"]: p.get("count", "ERR") for p in res["prompts"]}
            print(f"  {counts}  ({res['elapsed_s']:.1f}s)", flush=True)
            summary.append(res)
            images = [p for p in images if p.stem != args.dry_run_id]

    # 2) the rest
    for i, p in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {p.stem}", flush=True)
        try:
            res = _process_image(
                p, args.out, client, args.threshold, args.max_area_frac
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            summary.append({"image_id": p.stem, "error": str(e)})
            continue
        counts = {pp["prompt"]: pp.get("count", "ERR") for pp in res["prompts"]}
        print(f"  {counts}  ({res['elapsed_s']:.1f}s)", flush=True)
        summary.append(res)

    total_s = time.time() - t_total
    with (args.out / "_summary.json").open("w") as f:
        json.dump(
            {
                "model": "roboflow/inference-server (SAM3 / sam3_final)",
                "pod_id": pod_id,
                "gpu": pod.get("machine", {}).get("gpuTypeId") or pod.get("gpuTypeId"),
                "cloud": args.cloud,
                "prompts": PROMPTS,
                "threshold": args.threshold,
                "max_area_frac": args.max_area_frac,
                "image_count": len(summary),
                "total_s": total_s,
                "per_image": summary,
            },
            f,
            indent=2,
        )
    print(f"\n[done] processed {len(summary)} images in {total_s:.1f}s", flush=True)
    print(f"[done] results in {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
