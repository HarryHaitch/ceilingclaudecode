"""Read sweep + comparison outputs and emit a single chat-friendly summary.

Picks a winning mode per prompt by minimising fragmentation+collapse,
breaking ties on track count closest to ``n_image_clusters_estimate``
(taken as the mode of image-mode-instance counts across modes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _score_row(row: dict) -> float:
    """Lower is better. Penalises fragmentation and collapse equally,
    bonuses for high match rate."""
    n = max(1, row["n_tracks"])
    matched_frac = row["n_tracks_matched_to_image_mode"] / n
    frag = row["n_imgmode_fragmented_across_tracks"]
    collapse = row["n_track_collapse"]
    return (frag + collapse) - 5.0 * matched_frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path,
                    help="results dir with comparison.json + summary.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <results>/SUMMARY.md")
    args = ap.parse_args()

    comparison = json.loads((args.results / "comparison.json").read_text())
    sweep = json.loads((args.results / "summary.json").read_text())
    rows = comparison["rows"]

    by_prompt: dict[str, list[dict]] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], []).append(r)

    lines: list[str] = []
    lines.append("# SAM 3 video-tracking experiment — results")
    lines.append("")
    lines.append(f"Frames: **{sweep['n_frames']}** chronologically-ordered Polycam keyframes  ")
    lines.append(f"Device: **{sweep['device']}** ({sweep['dtype']})  ")
    lines.append(f"Total wall time on pod: **{sweep['total_elapsed_s']:.0f}s**  ")
    lines.append(f"IoU match threshold (vs image-mode): **{comparison['iou_threshold']}**")
    lines.append("")
    lines.append("## Verdict per prompt")
    lines.append("")

    winners: dict[str, dict] = {}
    for prompt, prompt_rows in by_prompt.items():
        best = min(prompt_rows, key=_score_row)
        winners[prompt] = best
        lines.append(f"### `{prompt}`")
        lines.append("")
        lines.append(f"**Best chunking: `{best['mode']}`** — "
                     f"{best['n_tracks']} tracks, "
                     f"{best['n_tracks_matched_to_image_mode']} matched to "
                     f"image-mode, "
                     f"{best['n_imgmode_fragmented_across_tracks']} image-mode "
                     f"fixtures fragmented across multiple tracks, "
                     f"{best['n_track_collapse']} tracks that collapsed >1 fixture, "
                     f"mean track lifespan {best['mean_track_len']:.1f} frames.")
        lines.append("")
        lines.append(f"All modes for `{prompt}`:")
        lines.append("")
        lines.append("| mode | tracks | matched | img-mode | fragmented | collapse | mean len |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(prompt_rows, key=_score_row):
            star = " ★" if r is best else ""
            lines.append(
                f"| `{r['mode']}`{star} | "
                f"{r['n_tracks']} | "
                f"{r['n_tracks_matched_to_image_mode']} | "
                f"{r['n_image_instances_total']} | "
                f"{r['n_imgmode_fragmented_across_tracks']} | "
                f"{r['n_track_collapse']} | "
                f"{r['mean_track_len']:.1f} |"
            )
        lines.append("")

    # Global recommendation.
    lines.append("## Recommendation")
    lines.append("")
    if len({w["mode"] for w in winners.values()}) == 1:
        m = next(iter(winners.values()))["mode"]
        lines.append(
            f"All three prompts agree on **`{m}`** — that's the candidate to "
            "ship as default if we want to replace the image-mode + DBSCAN "
            "clustering pipeline with video-mode tracking."
        )
    else:
        lines.append("Different prompts prefer different chunking modes:")
        lines.append("")
        for p, w in winners.items():
            lines.append(f"- `{p}` → `{w['mode']}`")
        lines.append("")
        lines.append(
            "Inspect the overlay MP4s for each before deciding. If the "
            "differences are small the cheaper mode (more chunks → smaller "
            "video sessions → less GPU memory) wins on cost."
        )

    out = args.out or (args.results / "SUMMARY.md")
    out.write_text("\n".join(lines))
    print(f"[summary] wrote {out}")
    print(f"[summary] winners: " +
          ", ".join(f"{p}={w['mode']}" for p, w in winners.items()))


if __name__ == "__main__":
    main()
