"""``python -m debug.segment_lab <session_id> [--algo X | --compare a,b,c]``

Iterates segmentation algorithms against a cached session's height map
without going near FastAPI or the browser. Output is a single PNG (one
algo) or a horizontal grid (compare mode), written to ``debug_out/``.

Examples::

    python -m debug.segment_lab abc123def456 --algo wall_constrained
    python -m debug.segment_lab abc123def456 --compare histogram,felzenszwalb,wall_constrained
    python -m debug.segment_lab abc123def456 --algo felzenszwalb --algo-arg scale=400

After the first run on a session, ``out/wall_edges.npy`` is cached;
subsequent runs are fast.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .algos import ALGOS
from .data import load_session
from .output import render_overlay, stack_horizontal


def _parse_kwargs(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--algo-arg expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="segment_lab")
    ap.add_argument("session_id")
    ap.add_argument("--algo", default="histogram",
                    help=f"one of: {', '.join(ALGOS)}")
    ap.add_argument("--compare", default=None,
                    help="comma-separated algo names for side-by-side output")
    ap.add_argument("--algo-arg", action="append", default=[],
                    help="algorithm-specific kwargs, e.g. scale=400")
    ap.add_argument("--out_dir", type=Path, default=Path("debug_out"))
    ap.add_argument("--refresh-walls", action="store_true",
                    help="re-rasterise the wall-edge mask from the mesh")
    ap.add_argument("--no-walls-overlay", action="store_true",
                    help="hide the white wall-edge overlay on output PNGs")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = _parse_kwargs(args.algo_arg)

    print(f"Loading session {args.session_id}…")
    data = load_session(args.session_id, refresh_walls=args.refresh_walls)
    H, W = data.height_map.shape
    print(f"  height map: {W}×{H}, "
          f"{int((~np.isnan(data.height_map)).sum())} valid pixels")
    print(f"  room mask: {int(data.room_mask.sum())} pixels")
    print(f"  wall edges: {int(data.wall_edges.sum())} pixels")

    algo_names = (args.compare.split(",")
                  if args.compare else [args.algo])
    images = []
    for name in algo_names:
        name = name.strip()
        if name not in ALGOS:
            print(f"unknown algo: {name}; available: {list(ALGOS)}",
                  file=sys.stderr)
            return 2
        print(f"Running {name}…")
        t0 = time.perf_counter()
        labels = ALGOS[name](data, **kwargs) if kwargs else ALGOS[name](data)
        dt = time.perf_counter() - t0
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        print(f"  {name}: {n_clusters} clusters in {dt:.2f}s")
        title = f"{name}  ({dt:.2f}s)"
        img = render_overlay(data, labels, title=title,
                             show_walls=not args.no_walls_overlay)
        images.append(img)

    out = stack_horizontal(images)
    if len(algo_names) == 1:
        path = args.out_dir / f"{args.session_id}_{algo_names[0]}.png"
    else:
        path = (args.out_dir
                / f"{args.session_id}_compare_{'_'.join(algo_names)}.png")
    cv2.imwrite(str(path), out)
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
