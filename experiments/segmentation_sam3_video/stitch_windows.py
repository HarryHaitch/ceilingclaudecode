"""Stitch overlapping-window tracks into globally-unified track ids.

The in-pod sweep emits per-window tracks with ids of the form
``sliding_w32_s16_c<window_idx>_<prompt>_<obj_id>``. Adjacent windows
share frames (default window=32, stride=16 → 16 frames overlap), so a
single physical fixture appears under different track ids in adjacent
windows. We resolve identity by IoU-matching tracks across overlap
regions and merging the resulting components.

Reads ``tracks_<sliding_mode>_<prompt-slug>.json`` (the in-pod sweep
output) and writes ``tracks_<mode>_<prompt-slug>__stitched.json`` with
a ``stitched_track_id`` field added per track. Original ``track_id``
is preserved for traceability.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _polygon_to_bin_mask(poly: list[list[float]], h: int, w: int) -> np.ndarray:
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


class _UF:
    """Tiny union-find over hashable items."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stitch_tracks(
    tracks: list[dict], *,
    iou_threshold: float = 0.3,
    canvas_h: int = 1024, canvas_w: int = 768,
) -> list[dict]:
    """Group tracks into stitched components by IoU on shared frames.

    Two tracks merge if there's at least one frame where both are
    present and their masks have IoU ≥ ``iou_threshold``.
    """
    # Index tracks by chunk and by frame.
    tracks_by_id = {t["track_id"]: t for t in tracks}
    by_frame: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for t in tracks:
        for f in t["frames"]:
            by_frame[f["frame_idx_global"]].append((t["track_id"], f))

    uf = _UF()

    # For each frame, IoU-match every pair of tracks (cheap — chunk
    # frames typically have <10 instances each).
    for fidx, entries in by_frame.items():
        if len(entries) < 2:
            continue
        masks: list[tuple[str, np.ndarray]] = []
        for tid, f in entries:
            poly = f.get("polygon") or []
            if len(poly) < 3:
                continue
            mask = _polygon_to_bin_mask(poly, canvas_h, canvas_w)
            masks.append((tid, mask))
        for i in range(len(masks)):
            tid_i, m_i = masks[i]
            for j in range(i + 1, len(masks)):
                tid_j, m_j = masks[j]
                # Don't merge tracks that come from the same chunk —
                # SAM 3's per-session tracker already decided they were
                # separate objects.
                ti = tracks_by_id[tid_i]
                tj = tracks_by_id[tid_j]
                if ti["chunk_idx"] == tj["chunk_idx"]:
                    continue
                if _iou(m_i, m_j) >= iou_threshold:
                    uf.union(tid_i, tid_j)

    # Assign stitched ids: <component-root>_S
    component_seq: dict[str, int] = {}
    seq = 0
    out: list[dict] = []
    for t in tracks:
        root = uf.find(t["track_id"])
        if root not in component_seq:
            component_seq[root] = seq
            seq += 1
        new_t = dict(t)
        new_t["stitched_track_id"] = (
            f"stitched_{component_seq[root]:04d}_{t['prompt'].replace(' ', '_')}"
        )
        out.append(new_t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-tracks", required=True, type=Path,
                    help="Path to tracks_<mode>_<prompt>.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: alongside input with "
                         "__stitched suffix)")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--canvas-h", type=int, default=1024)
    ap.add_argument("--canvas-w", type=int, default=768)
    args = ap.parse_args()

    doc = json.loads(args.in_tracks.read_text())
    tracks = doc.get("tracks", [])
    stitched = stitch_tracks(
        tracks,
        iou_threshold=args.iou_threshold,
        canvas_h=args.canvas_h, canvas_w=args.canvas_w,
    )
    out_doc = dict(doc)
    out_doc["tracks"] = stitched
    out_doc["n_stitched_tracks"] = len({t["stitched_track_id"] for t in stitched})

    out_path = args.out or args.in_tracks.with_name(
        args.in_tracks.stem + "__stitched.json"
    )
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"[stitch] {len(tracks)} raw tracks → "
          f"{out_doc['n_stitched_tracks']} stitched (IoU≥{args.iou_threshold})")
    print(f"[stitch] wrote {out_path}")


if __name__ == "__main__":
    main()
