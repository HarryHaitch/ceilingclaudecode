"""Run SAM3 (facebook/sam3) text-prompted segmentation on the working set.

Three concept prompts per image: "ceiling item", "light", "vent".

Per-image output layout (under <out_dir>/<image_id>/):
    input.jpg                        the rotated input
    <prompt>_mask.png                union of instance masks for that prompt
                                     (binary 0/255, may be absent if no hits)
    <prompt>_instances.png           label map (0=bg, 1..N=instances)
    <prompt>_overlay.jpg             coloured overlay for visual review
    detections.json                  per-prompt list of {score, box, area_px}

Top-level <out_dir>/_summary.json records prompt counts and runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor


PROMPTS: list[str] = ["ceiling item", "light", "vent"]
PROMPT_COLORS: dict[str, tuple[int, int, int]] = {
    "ceiling item": (255, 64, 64),
    "light": (255, 220, 64),
    "vent": (64, 200, 255),
}
MODEL_ID = "facebook/sam3"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slug(s: str) -> str:
    return s.replace(" ", "_")


def load_model(device: str):
    model = Sam3Model.from_pretrained(MODEL_ID).to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    return model, processor


def run_prompt(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    threshold: float,
    mask_threshold: float,
):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = inputs.get("original_sizes")
    target_sizes = sizes.tolist() if sizes is not None else [image.size[::-1]]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]
    return results


def overlay_masks(
    base: Image.Image,
    instances: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    """Tint pixels covered by any instance with `color`, draw outlines."""
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h, w = instances.shape
    if instances.max() > 0:
        tint = np.zeros((h, w, 4), dtype=np.uint8)
        mask_any = instances > 0
        tint[mask_any] = (*color, int(255 * alpha))
        overlay = Image.alpha_composite(overlay, Image.fromarray(tint, "RGBA"))
        draw = ImageDraw.Draw(overlay)
    composited = Image.alpha_composite(rgba, overlay)
    return composited.convert("RGB")


def annotate_overlay(
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


def save_results_for_prompt(
    out_dir: Path,
    prompt: str,
    base_img: Image.Image,
    results: dict,
) -> dict:
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    n = 0 if masks is None else int(len(masks))
    detections: list[dict] = []
    pslug = slug(prompt)
    color = PROMPT_COLORS[prompt]
    w, h = base_img.size

    if n == 0:
        # Still drop an empty overlay so reviewers see "no detections"
        empty = base_img.copy()
        ImageDraw.Draw(empty).text((10, 10), f"{prompt}: no detections", fill=color)
        empty.save(out_dir / f"{pslug}_overlay.jpg", quality=88)
        return {"prompt": prompt, "count": 0, "detections": []}

    masks_np = masks.cpu().numpy().astype(bool) if hasattr(masks, "cpu") else np.asarray(masks).astype(bool)
    boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

    instances = np.zeros((h, w), dtype=np.uint16)
    for idx, m in enumerate(masks_np, start=1):
        instances[m] = idx
        detections.append(
            {
                "instance_id": idx,
                "score": float(scores_np[idx - 1]),
                "box_xyxy": [float(v) for v in boxes_np[idx - 1].tolist()],
                "area_px": int(m.sum()),
            }
        )

    union = (instances > 0).astype(np.uint8) * 255
    Image.fromarray(union, "L").save(out_dir / f"{pslug}_mask.png")
    Image.fromarray(instances).save(out_dir / f"{pslug}_instances.png")

    overlay = overlay_masks(base_img, instances, color)
    overlay = annotate_overlay(overlay, [(d["box_xyxy"], d["score"]) for d in detections], color)
    overlay.save(out_dir / f"{pslug}_overlay.jpg", quality=88)

    return {"prompt": prompt, "count": n, "detections": detections}


def process_image(
    img_path: Path,
    out_root: Path,
    model,
    processor,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> dict:
    image_id = img_path.stem
    out_dir = out_root / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    img.save(out_dir / "input.jpg", quality=92)

    per_prompt = []
    t0 = time.time()
    for prompt in PROMPTS:
        results = run_prompt(model, processor, img, prompt, device, threshold, mask_threshold)
        per_prompt.append(save_results_for_prompt(out_dir, prompt, img, results))
    elapsed = time.time() - t0

    with (out_dir / "detections.json").open("w") as f:
        json.dump(
            {"image_id": image_id, "elapsed_s": elapsed, "prompts": per_prompt},
            f,
            indent=2,
        )
    return {"image_id": image_id, "elapsed_s": elapsed, "prompts": per_prompt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--working-set",
        type=Path,
        default=Path(__file__).parent / "working_set",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results_sample",
    )
    ap.add_argument("--n", type=int, default=10, help="Sample size; 0 means all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--ids", type=str, default=None, help="Comma-separated image ids; overrides --n")
    args = ap.parse_args()

    images = sorted(args.working_set.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {args.working_set}")

    if args.ids:
        wanted = set(args.ids.split(","))
        images = [p for p in images if p.stem in wanted]
    elif args.n and args.n > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, k=min(args.n, len(images)))
        images.sort()

    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device} model={MODEL_ID}")
    print(f"processing {len(images)} images, prompts={PROMPTS}")
    model, processor = load_model(device)

    summary = []
    t_all = time.time()
    for i, p in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {p.stem}")
        res = process_image(p, args.out, model, processor, device, args.threshold, args.mask_threshold)
        counts = {item["prompt"]: item["count"] for item in res["prompts"]}
        print(f"   {counts}  ({res['elapsed_s']:.1f}s)")
        summary.append(res)

    total_s = time.time() - t_all
    with (args.out / "_summary.json").open("w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "device": device,
                "prompts": PROMPTS,
                "threshold": args.threshold,
                "mask_threshold": args.mask_threshold,
                "image_count": len(images),
                "total_s": total_s,
                "per_image": summary,
            },
            f,
            indent=2,
        )
    print(f"done in {total_s:.1f}s. results -> {args.out}")


if __name__ == "__main__":
    main()
