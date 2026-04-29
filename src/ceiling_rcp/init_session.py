"""``ceiling-rcp-init <scan_dir>`` — create a server-side session from a
local Polycam folder and run the full segmentation pipeline.

Skips the browser upload entirely. After this finishes, point your
browser at::

    http://127.0.0.1:8765/?session=<id>

and the frontend loads the processed plan straight into the editor.

The session lives under ``./sessions/<id>/`` (same layout the FastAPI
upload path produces), so the existing edit / export endpoints work
without any changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from .server import SESSIONS_DIR, process_session


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ceiling-rcp-init")
    p.add_argument("scan_dir", type=Path,
                   help="Local folder containing the Polycam .obj + .mtl + textures + mesh_info.json")
    p.add_argument("--ppm", type=int, default=150, help="pixels per metre")
    p.add_argument("--port", type=int, default=8765, help="port the server runs on (for the printed URL)")
    args = p.parse_args(argv)

    scan_dir = args.scan_dir.expanduser().resolve()
    if not scan_dir.is_dir():
        print(f"Not a directory: {scan_dir}", file=sys.stderr)
        return 2

    sid = uuid.uuid4().hex[:12]
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    upload = SESSIONS_DIR / sid / "upload"
    upload.mkdir(parents=True, exist_ok=True)

    # Skip the same heavyweights the web upload filters out: keyframes,
    # depth maps, point clouds, the source video, .zip archives, etc.
    KEEP_EXT = {".obj", ".mtl", ".jpg", ".jpeg", ".png", ".json"}
    SKIP_DIRS = {
        "keyframes", "depth", "confidence", "cameras",
        "corrected_cameras", "images", "corrected_images",
    }
    SKIP_NAMES = {
        ".DS_Store", "thumbnail.jpg", "polycam.mp4",
        "ceiling_geometry.json", "ceiling_geometry_preview.png",
        "roomplan.json",
    }

    print(f"Session {sid}: staging scan …")
    n_kept = 0
    for src in scan_dir.rglob("*"):
        if not src.is_file():
            continue
        rel_parts = src.relative_to(scan_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if src.name in SKIP_NAMES:
            continue
        if src.suffix.lower() not in KEEP_EXT:
            continue
        target = upload / Path(*rel_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        n_kept += 1
    print(f"Session {sid}: copied {n_kept} files")

    print(f"Session {sid}: processing …")
    plan = process_session(sid, ppm=args.ppm)

    rep = plan.get("report", {})
    print(f"  ok: {rep.get('ok')}")
    for w in rep.get("warnings", []):
        print(f"  warn: {w}")
    for e in rep.get("errors", []):
        print(f"  err:  {e}")
    if rep.get("ok"):
        print(f"  clusters: {len(plan.get('clusters', []))}, "
              f"regions: {len(plan.get('regions', []))}, "
              f"composites: {len(plan.get('composites', []))}")
        print()
        print(f"  → http://127.0.0.1:{args.port}/?session={sid}")
    return 0 if rep.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
