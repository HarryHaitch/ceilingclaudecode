"""``python -m debug.experiments <session_id>`` — sweep algorithms +
parameters against your ground truth and print a leaderboard.

Reads ``debug/ground_truth/<sid>.json`` (capture it first with
``debug.snapshot_truth``). Runs every entry in :data:`EXPERIMENTS`
against the cached height map, scores each with Hungarian-matched IoU,
sorts the leaderboard by ``mean_iou`` descending, and writes:

  debug/results/<sid>.json    full per-experiment record
  debug/results/<sid>.csv     leaderboard, easy to diff between runs
  debug_out/<sid>_top<k>_<algo>.png   side-by-side prediction vs truth

Edit :data:`EXPERIMENTS` to add new configurations — each entry just
needs a unique ``name``, an algorithm key from
:data:`debug.algos.ALGOS`, and a kwargs dict.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .algos import ALGOS
from .data import load_session, SegInput
from .output import render_overlay, stack_horizontal
from .scoring import polygons_to_label_image, score, ScoreResult


# ─── EXPERIMENT GRID ─────────────────────────────────────────────────────────
# Tweak this list to add / remove configurations. Each entry must have:
#   name  : unique identifier shown on the leaderboard
#   algo  : key in debug.algos.ALGOS
#   kwargs: dict passed to the algorithm

EXPERIMENTS: list[dict[str, Any]] = [
    # baseline
    {"name": "histogram",                 "algo": "histogram",        "kwargs": {}},

    # graph segmentation, scale sweep
    {"name": "felzenszwalb_s100",         "algo": "felzenszwalb",     "kwargs": {"scale": 100}},
    {"name": "felzenszwalb_s200",         "algo": "felzenszwalb",     "kwargs": {"scale": 200}},
    {"name": "felzenszwalb_s400",         "algo": "felzenszwalb",     "kwargs": {"scale": 400}},
    {"name": "felzenszwalb_s800",         "algo": "felzenszwalb",     "kwargs": {"scale": 800}},
    {"name": "felzenszwalb_s1500",        "algo": "felzenszwalb",     "kwargs": {"scale": 1500}},
    {"name": "felzenszwalb_s400_sigma0",  "algo": "felzenszwalb",     "kwargs": {"scale": 400, "sigma": 0.0}},

    # gradient-gated region growing, threshold sweep
    {"name": "region_growing_1cm",        "algo": "region_growing",   "kwargs": {"threshold_m": 0.01}},
    {"name": "region_growing_2cm",        "algo": "region_growing",   "kwargs": {"threshold_m": 0.02}},
    {"name": "region_growing_3cm",        "algo": "region_growing",   "kwargs": {"threshold_m": 0.03}},
    {"name": "region_growing_5cm",        "algo": "region_growing",   "kwargs": {"threshold_m": 0.05}},

    # wall-edge constrained, min-cell sweep
    {"name": "wall_constrained_010",      "algo": "wall_constrained", "kwargs": {"min_cell_px_m2": 0.10}},
    {"name": "wall_constrained_030",      "algo": "wall_constrained", "kwargs": {"min_cell_px_m2": 0.30}},
    {"name": "wall_constrained_060",      "algo": "wall_constrained", "kwargs": {"min_cell_px_m2": 0.60}},

    # round 2 — fine-tune around the s200 winner
    {"name": "felzenszwalb_s150",         "algo": "felzenszwalb",     "kwargs": {"scale": 150}},
    {"name": "felzenszwalb_s175",         "algo": "felzenszwalb",     "kwargs": {"scale": 175}},
    {"name": "felzenszwalb_s225",         "algo": "felzenszwalb",     "kwargs": {"scale": 225}},
    {"name": "felzenszwalb_s250",         "algo": "felzenszwalb",     "kwargs": {"scale": 250}},

    # round 2 — felzenszwalb + same-height merge post-pass
    {"name": "felz_merged_s200_2cm",      "algo": "felzenszwalb_merged", "kwargs": {"scale": 200, "merge_threshold_m": 0.02}},
    {"name": "felz_merged_s200_3cm",      "algo": "felzenszwalb_merged", "kwargs": {"scale": 200, "merge_threshold_m": 0.03}},
    {"name": "felz_merged_s200_5cm",      "algo": "felzenszwalb_merged", "kwargs": {"scale": 200, "merge_threshold_m": 0.05}},
    {"name": "felz_merged_s400_3cm",      "algo": "felzenszwalb_merged", "kwargs": {"scale": 400, "merge_threshold_m": 0.03}},

    # round 2 — multi-channel watershed (height + texture + walls)
    {"name": "watershed_default",         "algo": "watershed_multi",  "kwargs": {}},
    {"name": "watershed_high_wall",       "algo": "watershed_multi",  "kwargs": {"wall_weight": 4.0}},
    {"name": "watershed_no_texture",      "algo": "watershed_multi",  "kwargs": {"texture_weight": 0.0}},
    {"name": "watershed_high_texture",    "algo": "watershed_multi",  "kwargs": {"texture_weight": 1.5}},

    # round 2 — region growing with texture-edge gating
    {"name": "region_growing_tex_3cm",    "algo": "region_growing_texture", "kwargs": {"threshold_m": 0.03}},

    # round 2 — SLIC superpixels merged by height
    {"name": "slic_merge_n200_2cm",       "algo": "slic_merge",       "kwargs": {"n_segments": 200, "merge_threshold_m": 0.02}},
    {"name": "slic_merge_n500_2cm",       "algo": "slic_merge",       "kwargs": {"n_segments": 500, "merge_threshold_m": 0.02}},
    {"name": "slic_merge_n1000_3cm",      "algo": "slic_merge",       "kwargs": {"n_segments": 1000, "merge_threshold_m": 0.03}},
]


# ─── LOADING TRUTH ───────────────────────────────────────────────────────────

def load_truth(session_id: str, *, path: Path | None = None) -> dict:
    p = path or (Path("debug") / "ground_truth" / f"{session_id}.json")
    if not p.exists():
        raise SystemExit(
            f"no ground-truth file at {p}.\n"
            f"draw the truth in the web UI, then run "
            f"`python -m debug.snapshot_truth {session_id}`."
        )
    return json.loads(p.read_text())


def truth_label_image(truth: dict, data: SegInput) -> np.ndarray:
    """Rasterise the ground-truth polygons onto ``data.grid``. Main ceiling
    is label 0; regions get labels 1..N in their stored order. Order
    doesn't matter for IoU scoring (Hungarian handles label permutations)
    but staying consistent makes the side-by-side overlay readable.
    """
    polys: list[list[list[float]]] = [truth["main"]["polygon"]]
    for r in truth.get("regions", []):
        polys.append(r["polygon"])
    return polygons_to_label_image(polys, data.grid)


# ─── REPORTING ───────────────────────────────────────────────────────────────

def print_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print("(no results)"); return
    cols = ["rank", "name", "mean_iou", "weighted", "min_iou",
            "n_pred", "n_truth", "time_s"]
    widths = [4, 32, 8, 8, 8, 6, 7, 7]
    bar = "-" * (sum(widths) + len(widths) * 2)
    print(bar)
    print("  ".join(f"{c:>{w}}" if c not in {"name"} else f"{c:<{w}}"
                    for c, w in zip(cols, widths)))
    print(bar)
    for i, r in enumerate(rows, 1):
        print(f"{i:>4}  "
              f"{r['name']:<32}  "
              f"{r['mean_iou']:>8.3f}  "
              f"{r['weighted_iou']:>8.3f}  "
              f"{r['min_iou']:>8.3f}  "
              f"{r['n_pred']:>6d}  "
              f"{r['n_truth']:>7d}  "
              f"{r['time_s']:>7.2f}")
    print(bar)


def save_results(session_id: str, rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{session_id}.json"
    csv_path = out_dir / f"{session_id}.csv"
    json_path.write_text(json.dumps({"session_id": session_id, "results": rows},
                                    indent=2))
    with csv_path.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "name", "algo", "mean_iou", "weighted_iou",
                    "min_iou", "n_pred", "n_truth", "time_s", "kwargs"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["name"], r["algo"], f"{r['mean_iou']:.4f}",
                        f"{r['weighted_iou']:.4f}", f"{r['min_iou']:.4f}",
                        r["n_pred"], r["n_truth"], f"{r['time_s']:.3f}",
                        json.dumps(r["kwargs"])])
    return csv_path


def save_top_pngs(
    session_id: str, rows: list[dict], data: SegInput,
    truth_labels: np.ndarray, *, top_k: int = 3, out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    truth_overlay = render_overlay(
        data, truth_labels, title="ground truth", show_walls=False,
    )
    for i, r in enumerate(rows[:top_k], 1):
        labels = r["labels"]
        title = f"#{i}  {r['name']}  IoU={r['mean_iou']:.3f}"
        pred_overlay = render_overlay(data, labels, title=title)
        combined = stack_horizontal([truth_overlay, pred_overlay])
        path = out_dir / f"{session_id}_top{i}_{r['name']}.png"
        cv2.imwrite(str(path), combined)
        paths.append(path)
    return paths


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments")
    ap.add_argument("session_id")
    ap.add_argument("--truth", type=Path, default=None,
                    help="path to ground-truth JSON")
    ap.add_argument("--top_k", type=int, default=3,
                    help="how many top experiments to dump as PNGs")
    ap.add_argument("--results_dir", type=Path, default=Path("debug/results"))
    ap.add_argument("--out_dir", type=Path, default=Path("debug_out"))
    ap.add_argument("--filter", default=None,
                    help="run only experiments whose name contains this substring")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(args.session_id, path=args.truth)
    print(f"truth: {truth.get('name', args.session_id)} — "
          f"main + {len(truth.get('regions', []))} regions")

    data = load_session(args.session_id)
    truth_labels = truth_label_image(truth, data)
    print(f"truth label image: "
          f"{int(truth_labels.max() + 1)} polygons rasterised")

    rows: list[dict] = []
    selected = [e for e in EXPERIMENTS
                if not args.filter or args.filter in e["name"]]
    print(f"running {len(selected)} experiments…\n")

    for exp in selected:
        algo_fn = ALGOS.get(exp["algo"])
        if algo_fn is None:
            print(f"  skip {exp['name']}: unknown algo {exp['algo']}")
            continue
        t0 = time.perf_counter()
        try:
            labels = (algo_fn(data, **exp["kwargs"])
                      if exp["kwargs"] else algo_fn(data))
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  {exp['name']}: FAILED in {dt:.1f}s ({type(e).__name__}: {e})")
            continue
        dt = time.perf_counter() - t0
        s = score(labels, truth_labels)
        n_pred = int(labels.max() + 1) if labels.max() >= 0 else 0
        print(f"  {exp['name']}: IoU={s.mean_iou:.3f} "
              f"(weighted {s.weighted_iou:.3f}, min {s.min_iou:.3f}) "
              f"pred={n_pred} truth={s.n_truth} {dt:.1f}s")
        rows.append({
            "name": exp["name"],
            "algo": exp["algo"],
            "kwargs": exp["kwargs"],
            "mean_iou": s.mean_iou,
            "weighted_iou": s.weighted_iou,
            "min_iou": s.min_iou,
            "n_pred": n_pred,
            "n_truth": s.n_truth,
            "time_s": dt,
            "score": s.as_dict(),
            "labels": labels,
        })

    rows.sort(key=lambda r: -r["mean_iou"])
    print()
    print_leaderboard(rows)

    csv_path = save_results(
        args.session_id,
        [{k: v for k, v in r.items() if k != "labels"} for r in rows],
        args.results_dir,
    )
    print(f"\nresults → {csv_path}")

    pngs = save_top_pngs(
        args.session_id, rows, data, truth_labels,
        top_k=args.top_k, out_dir=args.out_dir,
    )
    for p in pngs:
        print(f"   PNG → {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
