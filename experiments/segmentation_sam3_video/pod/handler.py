"""FastAPI handler for SAM 3 video tracking on a RunPod pod.

The driver runs locally; this server runs inside the pod. The model
loads once (lazily on first request), then every call pivots through
the warm session. All heavy I/O — frames, masks — moves over HTTP.

Endpoints:
  GET  /info                  health probe + model status
  POST /upload_frames         multipart upload of all working-set
                              JPEGs. Returns an upload_id; subsequent
                              calls reference it instead of re-sending
                              raw JPEG bytes per chunk.
  POST /sam3/video_segment    payload {upload_id, frame_indices,
                              prompt, anchor_idx, threshold}; returns
                              per-track polygons + scores per frame.

Track-id namespacing is the caller's responsibility — the handler
returns SAM 3's per-session integer ``object_ids``; the driver
prefixes them with mode/chunk to keep them globally unique.
"""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image


# ─── Globals ──────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
UPLOAD_ROOT = Path("/tmp/sam3-video-uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

_model = None
_processor = None


def _ensure_model_loaded() -> None:
    global _model, _processor
    if _model is not None:
        return
    print(f"[handler] loading SAM 3 video model on {DEVICE}/{DTYPE}…",
          flush=True)
    t0 = time.time()
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    _processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    _model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
        device=DEVICE, dtype=DTYPE,
    )
    _model.eval()
    # SAM 3 video's "hotstart" buffers the first N frames and rejects
    # any object whose track isn't consistent across them. That's the
    # right behaviour for dense video where every frame is ~30 ms
    # apart — but Polycam keyframes have multi-second viewpoint jumps
    # between consecutive frames, so EVERY object gets filtered out.
    # Disable it so detections always reach the output.
    print(f"[handler] hotstart_delay was {_model.hotstart_delay}; "
          "setting to 0 for sparse-keyframe video", flush=True)
    _model.hotstart_delay = 0
    # Detection thresholds default to {0.5, 0.7, 0.8} which were tuned
    # for dense COCO-style video. On Polycam ceiling shots, objects
    # are smaller and concept matching less confident — drop these so
    # we actually get detections to look at. The driver's --threshold
    # filter further cuts low-quality matches in post.
    print(f"[handler] thresholds: score_threshold_detection="
          f"{_model.score_threshold_detection} -> 0.10, "
          f"new_det_thresh={_model.new_det_thresh} -> 0.20, "
          f"high_conf_thresh={_model.high_conf_thresh} -> 0.40",
          flush=True)
    _model.score_threshold_detection = 0.10
    _model.new_det_thresh = 0.20
    _model.high_conf_thresh = 0.40
    print(f"[handler] model loaded in {time.time() - t0:.1f}s",
          flush=True)


# ─── Mask → polygon helpers (same convention as image-mode pipeline) ──────

def _mask_to_polygon(mask: np.ndarray) -> tuple[list[list[float]], int]:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return [], 0
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return [], 0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return [], int(mask.sum())
    poly = [[float(p[0][0]), float(p[0][1])] for p in largest]
    return poly, int(mask.sum())


def _polygon_bbox(poly: list[list[float]]) -> list[float]:
    arr = np.asarray(poly, dtype=float)
    return [
        float(arr[:, 0].min()), float(arr[:, 1].min()),
        float(arr[:, 0].max()), float(arr[:, 1].max()),
    ]


# ─── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="sam3-video-handler", version="0.1")


@app.on_event("startup")
async def _preload_model() -> None:
    """Load SAM 3 weights at server boot, not lazily on the first
    chunk POST. Otherwise the first chunk pays ~60–90 s of weight
    download on top of inference, and 268-frame propagations exceed
    Cloudflare's 100 s proxy timeout.
    """
    try:
        _ensure_model_loaded()
    except Exception as e:
        # Log but don't abort startup — /info still answers, and
        # we can surface the error on the first chunk attempt.
        import traceback
        print(f"[handler] STARTUP MODEL LOAD FAILED:\n{traceback.format_exc()}",
              flush=True)


@app.exception_handler(Exception)
async def _exc_handler(request, exc):
    """Log the traceback AND return it in the response body so the
    driver can surface it. Default FastAPI hides exceptions behind a
    bare 500 with no body, which makes remote debugging painful."""
    import traceback
    tb = traceback.format_exc()
    print("[handler] EXCEPTION:\n" + tb, flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__,
                 "traceback": tb},
    )


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "model": "facebook/sam3",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_loaded": _model is not None,
        "uploads": [p.name for p in UPLOAD_ROOT.iterdir() if p.is_dir()],
    }


@app.post("/upload_frames")
async def upload_frames(
    frame_ids_json: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Upload all working-set JPEGs at once.

    ``frame_ids_json`` is a JSON list of frame ids (filename stems)
    in chronological order. ``files`` are the corresponding JPEG
    bytes — must be in the same order as ``frame_ids_json``.
    """
    frame_ids = json.loads(frame_ids_json)
    if len(frame_ids) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"frame_ids has {len(frame_ids)} entries but "
                   f"{len(files)} files were uploaded",
        )
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True)
    for fid, f in zip(frame_ids, files):
        out = upload_dir / f"{fid}.jpg"
        out.write_bytes(await f.read())
    (upload_dir / "frame_ids.json").write_text(json.dumps(frame_ids))
    return {"upload_id": upload_id, "n_frames": len(frame_ids)}


def _propagate(
    pil_chunk: list[Image.Image],
    prompt: str,
    anchor_local_idx: int,
    threshold: float,
    diag: dict | None = None,
) -> list[dict]:
    """Run forward + backward propagation from the anchor frame."""
    _ensure_model_loaded()
    session = _processor.init_video_session(
        video=pil_chunk,
        inference_device=DEVICE,
        dtype=DTYPE,
    )
    _processor.add_text_prompt(session, text=prompt)

    per_frame: dict[int, list[dict]] = {}
    diag_first_logged = [False]

    def _consume(model_output) -> None:
        fidx = int(getattr(model_output, "frame_idx", -1))
        if fidx < 0:
            return
        # processor.postprocess_outputs returns a DICT (per the model
        # card on facebook/sam3): keys are object_ids, scores, boxes,
        # masks. Earlier code was reading attribute names off the raw
        # Sam3VideoSegmentationOutput which silently returned defaults
        # of empty list/dict — that's why we got 0 detections.
        processed = _processor.postprocess_outputs(
            inference_session=session,
            model_outputs=model_output,
            original_sizes=[
                [pil_chunk[fidx].size[1], pil_chunk[fidx].size[0]]
            ],
        )
        object_ids = processed.get("object_ids", [])
        scores = processed.get("scores", [])
        boxes = processed.get("boxes", [])
        masks = processed.get("masks", [])

        # Convert tensors → python types up-front.
        if hasattr(object_ids, "tolist"):
            object_ids = object_ids.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        # Masks stay as tensor for now — will move to numpy per-mask.

        if not diag_first_logged[0]:
            diag_first_logged[0] = True
            try:
                mask_shape = (
                    list(masks.shape) if hasattr(masks, "shape") else None
                )
                mask_dtype = (
                    str(masks.dtype) if hasattr(masks, "dtype") else None
                )
                mask_minmax = None
                if hasattr(masks, "min") and len(masks) > 0:
                    mt = masks if not hasattr(masks, "detach") else masks.detach().cpu()
                    mask_minmax = [float(mt.min()), float(mt.max())]
                payload = {
                    "frame_idx": fidx,
                    "n_object_ids": len(object_ids),
                    "object_ids": object_ids[:10],
                    "scores": scores[:10],
                    "boxes_shape": (
                        list(boxes.shape)
                        if hasattr(boxes, "shape")
                        else (None if boxes is None
                              else f"list[{len(boxes)}]")
                    ),
                    "masks_shape": mask_shape,
                    "masks_dtype": mask_dtype,
                    "masks_minmax": mask_minmax,
                }
                print(f"[handler] DIAG first-frame post: "
                      f"{json.dumps(payload, default=str)}", flush=True)
                if diag is not None and "first_frame" not in diag:
                    diag["first_frame"] = payload
            except Exception as e:
                import traceback
                print(f"[handler] DIAG dump failed: {e}\n"
                      f"{traceback.format_exc()}", flush=True)

        instances = []
        for i, oid in enumerate(object_ids):
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < threshold:
                continue
            # masks is a tensor of shape (N, H, W) at original resolution.
            mask_t = masks[i] if i < len(masks) else None
            if mask_t is None:
                continue
            if hasattr(mask_t, "detach"):
                mask_np = mask_t.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask_t)
            while mask_np.ndim > 2 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            mask_bin = (mask_np > 0).astype(np.uint8)
            poly, area_px = _mask_to_polygon(mask_bin)
            if not poly or area_px < 8:
                continue
            # Boxes come back already in XYXY absolute coords.
            bbox: list[float]
            if hasattr(boxes, "shape") and i < len(boxes):
                bbox = [float(x) for x in boxes[i]]
            else:
                bbox = _polygon_bbox(poly)
            instances.append({
                "obj_id": int(oid),
                "score": score,
                "polygon": poly,
                "bbox": bbox,
                "area_px": area_px,
            })
        if instances:
            per_frame[fidx] = instances

    with torch.no_grad():
        for out in _model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=anchor_local_idx,
            reverse=False,
        ):
            _consume(out)
        if anchor_local_idx > 0:
            for out in _model.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=anchor_local_idx,
                reverse=True,
            ):
                _consume(out)

    return [
        {"frame_idx_local": fidx, "instances": insts}
        for fidx, insts in sorted(per_frame.items())
    ]


@app.post("/sam3/video_segment")
async def video_segment(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one video-mode propagation pass on a chunk of frames.

    Payload:
      {
        "upload_id": "...",
        "frame_indices": [int, ...],     # global frame indices in the
                                          # uploaded chronological list
        "prompt": "ceiling item",
        "anchor_idx_local": int,          # which entry in frame_indices
                                          # to use as the prompt anchor
        "threshold": 0.5
      }

    Response:
      {
        "n_frames": int,
        "elapsed_s": float,
        "per_frame": [{"frame_idx_local": int,
                       "instances": [{"obj_id": int, "score": float,
                                      "polygon": [...],
                                      "bbox": [...], "area_px": int}]}]
      }
    """
    upload_id = payload["upload_id"]
    frame_indices = payload["frame_indices"]
    prompt = payload["prompt"]
    anchor_local = int(payload.get("anchor_idx_local", 0))
    threshold = float(payload.get("threshold", 0.5))

    upload_dir = UPLOAD_ROOT / upload_id
    if not upload_dir.exists():
        raise HTTPException(404, f"unknown upload_id {upload_id}")
    frame_ids = json.loads((upload_dir / "frame_ids.json").read_text())

    pil_chunk = [
        Image.open(upload_dir / f"{frame_ids[i]}.jpg").convert("RGB")
        for i in frame_indices
    ]
    t0 = time.time()
    diag: dict = {}
    per_frame = _propagate(pil_chunk, prompt, anchor_local, threshold,
                           diag=diag)
    n_kept = sum(len(r["instances"]) for r in per_frame)
    return JSONResponse({
        "n_frames": len(frame_indices),
        "elapsed_s": time.time() - t0,
        "n_kept_instances": n_kept,
        "diag": diag,
        "per_frame": per_frame,
    })
