"""Render an overlay MP4 from a tracks JSON + working-set frames.

One MP4 per (mode, prompt). Each track gets a persistent color (hash
of track id → HSV → RGB). Polygon outlined, track-id label drawn at
the polygon centroid. Frames without any tracks still appear so the
video stays in sync with the working set.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _color_for_track(track_id: str) -> tuple[int, int, int]:
    h_int = int(hashlib.md5(track_id.encode()).hexdigest()[:6], 16)
    hue = (h_int % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _polygon_centroid(poly: list[list[float]]) -> tuple[float, float]:
    arr = np.asarray(poly, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _track_id_field(track: dict, *, prefer_stitched: bool) -> str:
    if prefer_stitched and "stitched_track_id" in track:
        return track["stitched_track_id"]
    return track["track_id"]


def render(
    tracks_doc: dict,
    *,
    frames_dir: Path,
    out_path: Path,
    fps: int = 6,
    label_short: bool = True,
    prefer_stitched: bool = True,
) -> None:
    tracks = tracks_doc.get("tracks", [])
    if not tracks:
        print(f"[overlay] no tracks in {tracks_doc.get('mode')}; "
              f"writing empty mp4")
    # Pivot tracks → per-frame list of (track_id, polygon, score).
    per_frame: dict[int, list[tuple[str, list[list[float]], float]]] = {}
    chronological_frames: dict[int, str] = {}
    for t in tracks:
        tid = _track_id_field(t, prefer_stitched=prefer_stitched)
        for f in t["frames"]:
            fidx = f["frame_idx_global"]
            chronological_frames[fidx] = f["image_id"]
            per_frame.setdefault(fidx, []).append(
                (tid, f["polygon"], f["score"])
            )

    if not chronological_frames:
        # No tracks at all — still emit a header frame so the file exists.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8) as w:
            placeholder = Image.new("RGB", (768, 1024), (32, 32, 32))
            ImageDraw.Draw(placeholder).text(
                (40, 40), f"no tracks for {tracks_doc.get('prompt')}",
                fill=(255, 255, 255), font=_font(28),
            )
            w.append_data(np.asarray(placeholder))
        return

    sorted_frames = sorted(chronological_frames.items())
    font_label = _font(20)
    font_header = _font(22)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264",
                            quality=8) as writer:
        for fidx, fid in sorted_frames:
            img_path = frames_dir / f"{fid}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            for tid, poly, score in per_frame.get(fidx, []):
                if len(poly) < 3:
                    continue
                color = _color_for_track(tid)
                pts = [(int(round(x)), int(round(y))) for x, y in poly]
                # Translucent fill for the polygon body.
                draw.polygon(pts, fill=(*color, 70), outline=color)
                # Outline (thicker, fully opaque).
                draw.line(pts + [pts[0]], fill=color, width=3)
                cx, cy = _polygon_centroid(poly)
                label = (tid.split("_")[-1] if label_short else tid) + \
                        f" {score:.2f}"
                # Label background + text.
                tb = draw.textbbox((0, 0), label, font=font_label)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                draw.rectangle(
                    [int(cx) - tw // 2 - 3, int(cy) - th // 2 - 2,
                     int(cx) + tw // 2 + 3, int(cy) + th // 2 + 2],
                    fill=(0, 0, 0, 180),
                )
                draw.text(
                    (int(cx) - tw // 2, int(cy) - th // 2),
                    label, fill=color, font=font_label,
                )
            # Header strip.
            header = (
                f"{tracks_doc.get('mode', '?')}  "
                f"{tracks_doc.get('prompt', '?')}  "
                f"frame {fidx}/{sorted_frames[-1][0]}  ({fid})"
            )
            tb = draw.textbbox((0, 0), header, font=font_header)
            draw.rectangle([0, 0, img.size[0], tb[3] + 8], fill=(0, 0, 0, 200))
            draw.text((6, 4), header, fill=(255, 255, 255), font=font_header)
            writer.append_data(np.asarray(img))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks-json", required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--no-stitched", action="store_true",
                    help="use raw track_id even when stitched_track_id "
                         "is available")
    args = ap.parse_args()

    doc = json.loads(args.tracks_json.read_text())
    render(
        doc,
        frames_dir=args.frames_dir,
        out_path=args.out,
        fps=args.fps,
        prefer_stitched=not args.no_stitched,
    )
    print(f"[overlay] wrote {args.out}")


if __name__ == "__main__":
    main()
