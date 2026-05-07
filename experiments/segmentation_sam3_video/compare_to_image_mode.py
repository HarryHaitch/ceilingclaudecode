"""Compare video-mode tracks against the existing image-mode results.

For each (mode, prompt):
  - load video-mode tracks
  - load per-frame image-mode detections
  - for every (track_id, frame) pair, IoU-match the video-mode polygon
    against image-mode polygons on the same frame; greedy assign
  - count fragmentation (one image-mode fixture mapped to >1 track)
    and collapse (one track mapped to >1 image-mode fixture)

Output: ``comparison.json`` with raw numbers + ``comparison.md`` with
a Markdown table grouped by prompt.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw


def _polygon_to_mask(poly: list[list[float]], h: int, w: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    pts = [(int(round(x)), int(round(y))) for x, y in poly]
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def _load_image_mode_per_frame(
    image_mode_root: Path, prompt: str,
) -> dict[str, list[dict]]:
    """Return ``{image_id: [{polygon, score, ...}]}`` filtered to ``prompt``."""
    out: dict[str, list[dict]] = {}
    for det in image_mode_root.glob("*/detections.json"):
        image_id = det.parent.name
        try:
            doc = json.loads(det.read_text())
        except Exception:
            continue
        for pp in doc.get("prompts", []) or []:
            if pp.get("prompt") == prompt:
                out[image_id] = pp.get("detections", []) or []
                break
    return out


def _track_id_field(track: dict, *, prefer_stitched: bool) -> str:
    if prefer_stitched and "stitched_track_id" in track:
        return track["stitched_track_id"]
    return track["track_id"]


def compare(
    *,
    tracks_doc: dict,
    image_mode_per_frame: dict[str, list[dict]],
    iou_threshold: float = 0.3,
    canvas_h: int = 1024,
    canvas_w: int = 768,
    prefer_stitched: bool = True,
) -> dict:
    tracks = tracks_doc.get("tracks", [])
    # Track id (stitched if present) → set of (image_id, image_mode_instance_idx)
    track_to_imgmode: dict[str, set[tuple[str, int]]] = defaultdict(set)
    # Inverse: (image_id, image_mode_instance_idx) → set of track ids.
    imgmode_to_tracks: dict[tuple[str, int], set[str]] = defaultdict(set)

    track_lifespans: dict[str, int] = defaultdict(int)
    for t in tracks:
        tid = _track_id_field(t, prefer_stitched=prefer_stitched)
        track_lifespans[tid] += t["lifespan"]
        for f in t["frames"]:
            image_id = f["image_id"]
            t_poly = f["polygon"]
            if len(t_poly) < 3:
                continue
            t_mask = _polygon_to_mask(t_poly, canvas_h, canvas_w)
            best_idx = -1
            best_iou = 0.0
            for j, det in enumerate(image_mode_per_frame.get(image_id, [])):
                d_poly = det.get("polygon") or []
                if len(d_poly) < 3:
                    continue
                d_mask = _polygon_to_mask(d_poly, canvas_h, canvas_w)
                v = _iou(t_mask, d_mask)
                if v > best_iou:
                    best_iou = v
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                track_to_imgmode[tid].add((image_id, best_idx))
                imgmode_to_tracks[(image_id, best_idx)].add(tid)

    # Aggregate stats.
    n_tracks = len(track_lifespans)
    matched_tracks = sum(1 for tid in track_lifespans if track_to_imgmode[tid])
    n_image_instances_total = sum(
        len(v) for v in image_mode_per_frame.values()
    )
    matched_imgmode = len(imgmode_to_tracks)

    # Identity confusion (one track maps to multiple image-mode fixtures
    # from different image_ids — a track-collapse).
    n_track_collapse = 0
    for tid, pairs in track_to_imgmode.items():
        # Group by image-mode-instance identity inferred from polygon
        # similarity *across* frames is hard; we conservatively count
        # tracks where image_mode instances span multiple image_ids
        # AND have >1 distinct image-mode index per image_id (rare but
        # indicates the tracker chained two real fixtures into one).
        per_image: dict[str, set[int]] = defaultdict(set)
        for image_id, idx in pairs:
            per_image[image_id].add(idx)
        if any(len(v) > 1 for v in per_image.values()):
            n_track_collapse += 1

    # Fragmentation: an image-mode (image_id, idx) maps to >1 track.
    n_imgmode_fragmented = sum(1 for tids in imgmode_to_tracks.values()
                               if len(tids) > 1)

    lifespans = list(track_lifespans.values()) or [0]
    return {
        "n_tracks": n_tracks,
        "n_tracks_matched_to_image_mode": matched_tracks,
        "n_image_instances_total": n_image_instances_total,
        "n_image_instances_matched": matched_imgmode,
        "n_track_collapse": n_track_collapse,
        "n_imgmode_fragmented_across_tracks": n_imgmode_fragmented,
        "mean_track_len": float(np.mean(lifespans)),
        "median_track_len": float(median(lifespans)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-results", required=True, type=Path,
                    help="dir containing tracks_<mode>_<prompt>.json files")
    ap.add_argument("--image-results", required=True, type=Path,
                    help="dir containing per-image-id detections.json")
    ap.add_argument("--prompts", default="ceiling item,light,vent")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--canvas-h", type=int, default=1024)
    ap.add_argument("--canvas-w", type=int, default=768)
    ap.add_argument("--prefer-stitched", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: comparison.json + comparison.md "
                         "alongside tracks files")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    # Pre-load image-mode detections per prompt (heavy I/O — do once).
    print("[compare] loading image-mode detections…", flush=True)
    image_mode_by_prompt = {
        p: _load_image_mode_per_frame(args.image_results, p)
        for p in prompts
    }
    for p, m in image_mode_by_prompt.items():
        n = sum(len(v) for v in m.values())
        print(f"          {p!r}: {n} detections across {len(m)} frames")

    # Find tracks_<mode>_<prompt>.json files. Prefer stitched variants
    # when both are present.
    track_files = sorted(args.video_results.glob("tracks_*.json"))
    # If a __stitched.json exists, prefer it over the raw.
    by_stem: dict[str, Path] = {}
    for tf in track_files:
        base = tf.stem.replace("__stitched", "")
        if "__stitched" in tf.stem or base not in by_stem:
            by_stem[base] = tf

    rows: list[dict] = []
    for stem, tf in sorted(by_stem.items()):
        doc = json.loads(tf.read_text())
        mode = doc.get("mode", stem)
        prompt = doc.get("prompt")
        if prompt not in image_mode_by_prompt:
            print(f"[compare] skipping {tf.name} — prompt {prompt!r} "
                  "not in --prompts", flush=True)
            continue
        print(f"[compare] {mode}  {prompt!r}  ({tf.name})", flush=True)
        stats = compare(
            tracks_doc=doc,
            image_mode_per_frame=image_mode_by_prompt[prompt],
            iou_threshold=args.iou_threshold,
            canvas_h=args.canvas_h, canvas_w=args.canvas_w,
            prefer_stitched=args.prefer_stitched,
        )
        rows.append({"mode": mode, "prompt": prompt, **stats,
                     "tracks_file": str(tf.name)})

    out_dir = args.out or args.video_results
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps({
        "iou_threshold": args.iou_threshold,
        "rows": rows,
    }, indent=2))

    # Markdown table.
    md_lines: list[str] = ["# SAM 3 video-mode vs image-mode comparison",
                           "",
                           f"IoU match threshold: {args.iou_threshold}",
                           ""]
    for prompt in prompts:
        md_lines.append(f"## prompt: `{prompt}`")
        md_lines.append("")
        md_lines.append("| mode | tracks | matched | img-mode | "
                        "matched | collapse | fragmented | "
                        "mean len | median len |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["prompt"] != prompt:
                continue
            md_lines.append(
                f"| `{r['mode']}` | {r['n_tracks']} | "
                f"{r['n_tracks_matched_to_image_mode']} | "
                f"{r['n_image_instances_total']} | "
                f"{r['n_image_instances_matched']} | "
                f"{r['n_track_collapse']} | "
                f"{r['n_imgmode_fragmented_across_tracks']} | "
                f"{r['mean_track_len']:.1f} | "
                f"{r['median_track_len']:.1f} |"
            )
        md_lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(md_lines))
    print(f"[compare] wrote {out_dir / 'comparison.md'}")
    print(f"[compare] wrote {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
