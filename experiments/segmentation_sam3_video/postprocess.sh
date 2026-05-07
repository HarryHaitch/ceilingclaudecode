#!/bin/bash
# Post-process the sweep outputs once the pod has returned results.
# Stitches sliding-window tracks, renders one overlay MP4 per
# (mode, prompt), runs comparison vs image-mode, and produces a
# top-level summary the chat can pick the winning mode from.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${1:-$HERE/results}"
WORKING_SET="${2:-/Users/harishusic/Documents/Claude Code/Space Room Plan GS/.claude/worktrees/trusting-lovelace-911506/experiments/segmentation_sam3/working_set}"
IMAGE_RESULTS="${3:-/Users/harishusic/Documents/Claude Code/Space Room Plan GS/.claude/worktrees/trusting-lovelace-911506/experiments/segmentation_sam3_runpod/results}"

echo "[post] results dir:    $RESULTS"
echo "[post] working set:    $WORKING_SET"
echo "[post] image results:  $IMAGE_RESULTS"

# 1. Stitch sliding-window tracks (no-op if no sliding mode ran).
for f in "$RESULTS"/tracks_sliding_*.json; do
    [ -f "$f" ] || continue
    [[ "$f" == *__stitched.json ]] && continue
    echo "[post] stitching $f"
    python3 "$HERE/stitch_windows.py" --in-tracks "$f"
done

# 2. Render an overlay MP4 per (mode, prompt). Stitched variants take priority.
for f in "$RESULTS"/tracks_*.json; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    # If a stitched twin exists, skip the raw.
    raw_stem="${base%.json}"
    stitched="$RESULTS/${raw_stem}__stitched.json"
    if [[ "$base" == *__stitched.json ]]; then
        :  # always render stitched
    elif [ -f "$stitched" ]; then
        echo "[post] skip raw $base (stitched twin exists)"
        continue
    fi
    out="${f%.json}.mp4"
    [ -f "$out" ] && { echo "[post] skip $out (exists)"; continue; }
    echo "[post] rendering $out"
    python3 "$HERE/render_overlay.py" \
        --tracks-json "$f" \
        --frames-dir "$WORKING_SET" \
        --out "$out"
done

# 3. Comparison table.
echo "[post] running comparison"
python3 "$HERE/compare_to_image_mode.py" \
    --video-results "$RESULTS" \
    --image-results "$IMAGE_RESULTS" \
    --out "$RESULTS"

echo "[post] done. See:"
echo "  $RESULTS/comparison.md"
echo "  $RESULTS/comparison.json"
ls "$RESULTS"/*.mp4 2>/dev/null | sed 's/^/  /'
