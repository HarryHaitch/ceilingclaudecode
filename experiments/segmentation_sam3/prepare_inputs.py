"""Build the SAM3 working set from a Polycam keyframes folder.

Polycam keyframe transforms use the OpenGL camera convention: the
camera looks down its **negative** Z axis. So the world-Y component of
the camera's forward direction is ``-t_12``. A frame faces the ceiling
when ``-t_12 > 0.20`` (forward tilted at least ~11.5 deg above
horizontal). On Lachy's session that leaves 268 of 549 frames.

Each kept frame is rotated 90 deg clockwise so the ceiling sits at the
top of the image (in the corrected landscape image the phone is held
in portrait, so world-up corresponds to image-left).

Outputs:
    <out_dir>/<image_id>.jpg     rotated RGB, 768x1024
    <out_dir>/_manifest.json     ordered list of selected ids and forward_y
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


# Min world-Y component of camera-forward direction (forward = -Z under OpenGL)
CEILING_FACING_FORWARD_Y = 0.20


def load_cam(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def select_ceiling_facing(cam_dir: Path, min_forward_y: float) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    for cam_path in sorted(cam_dir.glob("*.json")):
        cam = load_cam(cam_path)
        forward_y = -cam["t_12"]  # OpenGL: forward = -Z
        if forward_y > min_forward_y:
            kept.append((cam_path.stem, forward_y))
    return kept


def rotate_cw90(image: Image.Image) -> Image.Image:
    # PIL ROTATE_270 == counter-clockwise 270 deg == clockwise 90 deg
    return image.transpose(Image.Transpose.ROTATE_270)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keyframes",
        type=Path,
        default=Path(
            "/Users/harishusic/Documents/Claude Code/Space Room Plan GS/"
            "Scan data/Lachys Polycam/Lachys line/keyframes"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "working_set",
    )
    ap.add_argument(
        "--min-forward-y",
        type=float,
        default=CEILING_FACING_FORWARD_Y,
        help="Min world-Y component of camera-forward (= -t_12) to keep a frame.",
    )
    args = ap.parse_args()

    cam_dir = args.keyframes / "corrected_cameras"
    img_dir = args.keyframes / "corrected_images"
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_ceiling_facing(cam_dir, args.min_forward_y)
    print(f"selected {len(selected)} of {len(list(cam_dir.glob('*.json')))} frames")

    manifest = []
    for stem, forward_y in selected:
        src = img_dir / f"{stem}.jpg"
        if not src.exists():
            print(f"  skip (missing image): {stem}")
            continue
        img = Image.open(src).convert("RGB")
        rot = rotate_cw90(img)
        dst = out_dir / f"{stem}.jpg"
        rot.save(dst, quality=92)
        manifest.append({"id": stem, "forward_y": forward_y, "rotated_size": list(rot.size)})

    manifest_path = out_dir / "_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "min_forward_y": args.min_forward_y,
                "convention": "OpenGL (forward = -Z)",
                "rotation": "clockwise 90 deg (PIL ROTATE_270)",
                "count": len(manifest),
                "items": manifest,
            },
            f,
            indent=2,
        )
    print(f"wrote {len(manifest)} images + manifest to {out_dir}")


if __name__ == "__main__":
    main()
