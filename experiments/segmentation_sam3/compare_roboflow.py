"""Hit Roboflow's direct SAM3 endpoint on one image and diff vs our local run.

Saves the raw response (masks/boxes/scores) and prints a side-by-side
summary against our `detections.json` for the same image_id.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from inference_sdk import InferenceHTTPClient


PROMPTS = ["ceiling item", "light", "vent"]


def summarise(name: str, dets: list[dict]) -> None:
    if not dets:
        print(f"  {name}: 0 detections")
        return
    scores = sorted([d["score"] for d in dets], reverse=True)
    boxes = [d.get("box") or d.get("box_xyxy") for d in dets]
    print(f"  {name}: n={len(dets)}  top scores={[round(s,2) for s in scores[:6]]}")
    if boxes:
        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        print(
            f"     box w: min={min(widths):.0f} median={sorted(widths)[len(widths)//2]:.0f} max={max(widths):.0f}"
            f"   h: min={min(heights):.0f} median={sorted(heights)[len(heights)//2]:.0f} max={max(heights):.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent
        / "results_sample"
        / "43179113618"
        / "input.jpg",
    )
    ap.add_argument(
        "--image-id",
        type=str,
        default="43179113618",
        help="Folder name under results_sample/ for our local run",
    )
    ap.add_argument(
        "--key-file", type=Path, default=Path.home() / ".roboflow_api_key"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "roboflow_compare",
    )
    args = ap.parse_args()

    api_key = args.key_file.read_text().strip()
    out_dir = args.out / args.image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    rf: dict[str, list[dict]] = {}
    for prompt in PROMPTS:
        print(f"sam3_concept_segment(text={prompt!r}) on {args.image.name}")
        t0 = time.time()
        try:
            result = client.sam3_concept_segment(
                inference_input=str(args.image),
                prompts=[{"type": "text", "text": prompt}],
                output_prob_thresh=0.5,
                format="polygon",
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rf[prompt] = []
            continue
        print(f"  {time.time()-t0:.1f}s")

        if isinstance(result, list):
            result = result[0] if result else {}
        with (out_dir / f"rf_{prompt.replace(' ', '_')}_raw.json").open("w") as f:
            try:
                json.dump(result, f, indent=2, default=str)
            except Exception as e:
                f.write(f"<unable to serialise: {e}>\n")
        # Roboflow returns prompt_results: list, each with predictions: list
        preds: list[dict] = []
        if isinstance(result, dict):
            for pr in result.get("prompt_results", []) or []:
                preds.extend(pr.get("predictions", []) or [])
            if not preds:
                preds = result.get("predictions", []) or []
        rf[prompt] = preds
        if isinstance(result, dict):
            print(f"  keys: {list(result.keys())}")
        if preds:
            print(f"  first prediction keys: {list(preds[0].keys())[:12]}")

    # local run for the same image
    local_path = (
        Path(__file__).parent
        / "results_sample"
        / args.image_id
        / "detections.json"
    )
    local = json.loads(local_path.read_text()) if local_path.exists() else None

    print("\n=== Roboflow SAM3 (concept_segment) ===")
    for prompt in PROMPTS:
        summarise(prompt, [
            {"score": d.get("confidence", d.get("score", 0)),
             "box": [d.get("x", 0) - d.get("width", 0)/2,
                     d.get("y", 0) - d.get("height", 0)/2,
                     d.get("x", 0) + d.get("width", 0)/2,
                     d.get("y", 0) + d.get("height", 0)/2]}
            for d in rf[prompt]
        ])

    if local:
        print("\n=== Our local SAM3 (transformers Sam3Model on MPS) ===")
        for p in local["prompts"]:
            summarise(
                p["prompt"],
                [{"score": d["score"], "box": d["box_xyxy"]}
                 for d in p["detections"]],
            )

    print(f"\nraw responses written to {out_dir}/rf_*_raw.json")


if __name__ == "__main__":
    main()
