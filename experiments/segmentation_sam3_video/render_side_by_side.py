"""Side-by-side overlay: image-mode masks (left) vs video-mode tracks (right).

The image-mode pipeline produces per-frame masks with no cross-frame
identity — different mask colours each frame. The video-mode pipeline
produces persistent track ids — same fixture keeps its colour across
frames. Watching them side-by-side makes the identity question
visually obvious.

Usage:
    python render_side_by_side.py \\
        --tracks-json results/tracks_<mode>_light.json \\
        --frames-dir <working-set> \\
        --image-results <image-mode-results> \\
        --prompt "light" \\
        --out side_by_side_light.mp4
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _color_for_track(track_id: str) -> tuple[int, int, int]:
    h_int = int(hashlib.md5(track_id.encode()).hexdigest()[:6], 16)
    hue = (h_int % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _color_per_frame_instance(image_id: str, idx: int) -> tuple[int, int, int]:
    """Distinct color per (image_id, idx) — emphasises that image-mode has
    no cross-frame identity."""
    return _color_for_track(f"imgmode__{image_id}__{idx}")


def _font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_polygons(
    img: Image.Image,
    polygons: list[tuple[list[list[float]], tuple[int, int, int], str]],
    *,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od_main = ImageDraw.Draw(out)
    for poly, color, label in polygons:
        if len(poly) < 3:
            continue
        pts = [(int(round(x)), int(round(y))) for x, y in poly]
        od.polygon(pts, fill=(*color, 70))
        od_main.line(pts + [pts[0]], fill=color, width=3)
        # Label at centroid.
        arr = np.asarray(poly, dtype=float)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        bbox = od_main.textbbox((0, 0), label, font=label_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        od_main.rectangle(
            [int(cx) - tw // 2 - 3, int(cy) - th // 2 - 2,
             int(cx) + tw // 2 + 3, int(cy) + th // 2 + 2],
            fill=(0, 0, 0, 200),
        )
        od_main.text(
            (int(cx) - tw // 2, int(cy) - th // 2),
            label, fill=color, font=label_font,
        )
    out = Image.alpha_composite(out, overlay)
    return out.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks-json", required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--image-results", required=True, type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--prefer-stitched", action="store_true", default=True)
    args = ap.parse_args()

    tracks_doc = json.loads(args.tracks_json.read_text())

    # Pivot video-mode tracks → per-frame.
    video_per_frame: dict[int, list[tuple[list[list[float]], str, float]]] = \
        defaultdict(list)
    chronological: dict[int, str] = {}
    for t in tracks_doc.get("tracks", []):
        tid = t.get("stitched_track_id") if args.prefer_stitched and "stitched_track_id" in t else t["track_id"]
        for f in t["frames"]:
            chronological[f["frame_idx_global"]] = f["image_id"]
            video_per_frame[f["frame_idx_global"]].append(
                (f["polygon"], tid, f["score"])
            )

    sorted_frames = sorted(chronological.items())
    if not sorted_frames:
        raise SystemExit(f"no tracks in {args.tracks_json}")

    label_font = _font(20)
    header_font = _font(22)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8) as writer:
        for fidx, fid in sorted_frames:
            img_path = args.frames_dir / f"{fid}.jpg"
            if not img_path.exists():
                continue
            base = Image.open(img_path).convert("RGB")
            w, h = base.size

            # LEFT: image-mode masks (every instance gets a per-frame color).
            left_polys: list = []
            det_path = args.image_results / fid / "detections.json"
            if det_path.exists():
                doc = json.loads(det_path.read_text())
                for pp in doc.get("prompts", []) or []:
                    if pp.get("prompt") == args.prompt:
                        for j, det in enumerate(pp.get("detections", []) or []):
                            poly = det.get("polygon") or []
                            color = _color_per_frame_instance(fid, j)
                            label = f"{j+1}  {det.get('score', 0):.2f}"
                            left_polys.append((poly, color, label))
                        break
            left = _draw_polygons(base, left_polys, label_font=label_font)

            # RIGHT: video-mode tracks (persistent color per track id).
            right_polys: list = []
            for poly, tid, score in video_per_frame.get(fidx, []):
                color = _color_for_track(tid)
                # Show last token of track id for compactness.
                short = tid.split("_")[-1] if "_" in tid else tid
                label = f"T{short}  {score:.2f}"
                right_polys.append((poly, color, label))
            right = _draw_polygons(base, right_polys, label_font=label_font)

            # Compose side-by-side.
            sbs = Image.new("RGB", (w * 2 + 20, h + 60), (16, 16, 16))
            sbs.paste(left, (0, 60))
            sbs.paste(right, (w + 20, 60))
            d = ImageDraw.Draw(sbs)
            d.text((10, 10), f"image-mode  ({len(left_polys)} masks)",
                   fill=(255, 255, 255), font=header_font)
            d.text((w + 30, 10), f"video-mode  ({len(right_polys)} tracks)",
                   fill=(255, 255, 255), font=header_font)
            d.text((10, 36),
                   f"{tracks_doc.get('mode', '?')}  "
                   f"{args.prompt!r}  frame {fidx}  ({fid})",
                   fill=(180, 180, 180), font=_font(16))
            writer.append_data(np.asarray(sbs))

    print(f"[sbs] wrote {args.out}")


if __name__ == "__main__":
    main()
