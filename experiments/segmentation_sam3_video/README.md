# SAM 3 video-mode tracking experiment

Stitch the 268 ceiling-facing keyframes into a chronological "stop-motion"
video, run SAM 3 **video mode** with concept (text) prompts on a RunPod
GPU pod, and see whether persistent track ids let us skip the per-frame
DBSCAN clustering that the image-mode pipeline currently relies on.

## What this folder is

This is an **experiment**, not production code. The image-mode SAM 3 +
projection pipeline under `src/ceiling_rcp/` is unchanged. If video mode
turns out to do a better job of fixture identity than per-frame masks +
clustering, *then* we'd promote some of this back into `src/`. Until
then, treat everything here as throwaway.

## Inputs

The experiment reads its inputs from a sibling worktree by default
(saves copying ~800 MB of imagery between worktrees):

- `--working-set` — directory of chronologically-named JPEGs. Default
  points at `.claude/worktrees/trusting-lovelace-911506/experiments/segmentation_sam3/working_set/`.
- `--image-mode-results` — per-frame `detections.json` tree from the
  existing image-mode SAM 3 pipeline. Default points at the sibling
  worktree's `experiments/segmentation_sam3_runpod/results/`.

Both are overridable. The image-mode results are *optional* — the
densest-frame anchor strategy uses them, but if absent it falls back to
chunk-midpoint anchoring.

## Chunking sweep

Default sweep is the six chunk-counts requested by the user plus one
sliding-window mode:

| Mode | Chunks | Frames each |
|---|---:|---:|
| `disjoint_1` | 1 | 268 |
| `disjoint_4` | 4 | 67 |
| `disjoint_8` | 8 | 33–34 |
| `disjoint_16` | 16 | 16–17 |
| `disjoint_32` | 32 | 8–9 |
| `disjoint_64` | 64 | 4–5 |
| `sliding_w32_s16` | 16 windows | 32 each, 16-frame overlap |

Within each chunk, the **anchor frame** (where the text prompt is
attached) is the chunk-local frame with the most detections of that
prompt in the existing image-mode results. SAM 3 is asked to propagate
forward AND backward from the anchor.

For sliding windows, post-processing IoU-stitches tracks across
overlap regions so a single fixture gets one global id rather than one
per window.

## How to run

The driver uses a custom Docker image (built once by GitHub Actions
and pushed to `ghcr.io/<owner>/sam3-video:latest`) to skip the SSH
key/auth dance that the stock RunPod images make painful. The pod
exposes a small FastAPI handler on port 9001 — the driver POSTs the
working-set frames once, then issues one request per (chunk, prompt).

**First time only** — push to the branch to trigger the build:

```bash
git push origin <branch>
# wait ~5–7 min for .github/workflows/sam3-video-pod.yml to build
# and push to ghcr.io. Then make the package public via the GitHub
# package settings UI (or wire up registry auth in the driver).
```

**Run the sweep:**

```bash
# Full sweep — ~20–30 min on a 4090, ~$0.20–$0.40
python experiments/segmentation_sam3_video/run_video_sam3.py

# Override the image (e.g. a SHA-tagged build)
python experiments/segmentation_sam3_video/run_video_sam3.py \
    --docker-image ghcr.io/harryhaitch/sam3-video:abc1234

# Smoke test — 4 disjoint chunks only, keep pod alive for inspection
python experiments/segmentation_sam3_video/run_video_sam3.py \
    --chunks 4 --no-sliding --keep-pod

# Cap to 32 frames for a fast sanity check (one chunk only)
python experiments/segmentation_sam3_video/run_video_sam3.py \
    --chunks 1 --no-sliding --max-frames 32 --keep-pod
```

After the driver finishes, render overlay videos and run the
comparison vs image-mode:

```bash
# One overlay per (mode, prompt). Loop over the JSONs.
for f in experiments/segmentation_sam3_video/results/tracks_*.json; do
    out="${f%.json}.mp4"
    python experiments/segmentation_sam3_video/render_overlay.py \
        --tracks-json "$f" \
        --frames-dir "$WORKING_SET" \
        --out "$out"
done

# Sliding-window stitching (only needed for sliding modes)
python experiments/segmentation_sam3_video/stitch_windows.py \
    --in-tracks experiments/segmentation_sam3_video/results/tracks_sliding_w32_s16_light.json

# Aggregate comparison table
python experiments/segmentation_sam3_video/compare_to_image_mode.py \
    --video-results experiments/segmentation_sam3_video/results \
    --image-results <path-to-image-mode-results>
```

## Files

```
experiments/segmentation_sam3_video/
├── README.md                       (this file)
├── run_video_sam3.py               local driver — pod lifecycle + HTTP
├── chunking.py                     split_chunks / sliding_windows / pick_densest
├── stitch_windows.py               IoU-merge tracks across overlap regions
├── render_overlay.py               tracks.json + frames → overlay.mp4
├── render_side_by_side.py          image-mode (left) vs video-mode (right) MP4
├── compare_to_image_mode.py        IoU comparison vs image-mode pipeline
├── summarise.py                    pick winner per prompt → SUMMARY.md
├── postprocess.sh                  stitch + render + compare + summarise
├── results/                        all outputs land here
└── pod/
    ├── Dockerfile                  base = runpod/pytorch + transformers + handler
    └── handler.py                  FastAPI: /info, /upload_frames, /sam3/video_segment

.github/workflows/sam3-video-pod.yml   build & push the pod image to ghcr.io
```

## Pod runtime

Custom image `ghcr.io/<owner>/sam3-video:latest` (built from
[pod/Dockerfile](pod/Dockerfile) by the GHA workflow). Base is
`runpod/pytorch:2.4.0-…-devel-ubuntu22.04` plus transformers 5.7,
FastAPI, and our [pod/handler.py](pod/handler.py).

The pod exposes port 9001 over RunPod's HTTPS proxy at
`https://<pod-id>-9001.proxy.runpod.net`. The driver waits for
`GET /info` to return 200, then POSTs to `/upload_frames` once and
`/sam3/video_segment` per (chunk, prompt).

GPU fallback chain reused from
[src/ceiling_rcp/sam3_runpod_runner.py:59](../../src/ceiling_rcp/sam3_runpod_runner.py:59)
(H100 → 4090 → 3090 → A5000 → A4000 → L4). Cleanup is paranoid:
atexit + SIGINT/SIGTERM handlers terminate the pod even on Ctrl+C.

SAM 3 weights are NOT baked into the image (saves ~3 GB and ~10 min
of build time). The handler downloads them on the first POST
(~60–90 s) and caches them in `/opt/hf-cache` for the rest of the
pod's life.

## Output format

For each `(mode, prompt)`:

```
results/tracks_<mode>_<prompt>.json
{
  "mode": "disjoint_8",
  "prompt": "light",
  "n_chunks": 8,
  "n_tracks": 17,
  "elapsed_s": 31.2,
  "tracks": [
    {"track_id": "disjoint_8_c2_light_3",
     "prompt": "light", "chunk_idx": 2, "obj_id": 3,
     "lifespan": 12, "mean_score": 0.78,
     "frames": [{"frame_idx_global": 67, "image_id": "43179...",
                 "bbox": [...], "polygon": [[x,y],...],
                 "score": 0.81, "area_px": 4231}, ...]},
    ...
  ]
}
```

Plus a top-level `summary.json` and (after `compare_to_image_mode.py`)
a `comparison.md` table.

## Known unknowns

1. **Concept tracking and new mid-video instances.** SAM 3 video
   tracks objects matched at the prompt frame. Whether new instances
   entering the scene later get auto-discovered depends on the model's
   concept-prompt re-detection behaviour, which we'll learn from
   the first run. If they don't, the densest-anchor strategy is doing
   most of the heavy lifting and small chunks will dominate.
2. **Polycam temporal sparsity.** Adjacent keyframes can have large
   viewpoint deltas. Smaller chunks reduce drift but lose cross-chunk
   identity — the whole point of the experiment. Sliding windows are
   the hedge.
3. **`Sam3VideoModel` API churn.** Method names in transformers 5.7
   have already settled (`init_video_session`, `add_text_prompt`,
   `propagate_in_video_iterator`), but if a future pin lands on a
   different version the in-pod script may need adjustment.
