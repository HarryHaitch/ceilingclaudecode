# SAM 3 → ortho projection

Project per-keyframe SAM 3 masks (`ceiling item`, `light`, `vent`)
back onto the main app's top-down ortho image, fuse votes from all
contributing keyframes under a view-quality weighting, and report
which instances land in which user-traced ceiling region.

**Status**: not yet implemented. The driving prompt is
[`NEXT_SESSION.md`](NEXT_SESSION.md) in this folder. Read that
end-to-end before coding.

## Inputs

- `experiments/segmentation_sam3_runpod/results/<keyframe_id>/` —
  per-keyframe masks + detections (gitignored bulk; full path in
  `NEXT_SESSION.md`).
- `Scan data/Lachys Polycam/Lachys line/keyframes/corrected_cameras/<keyframe_id>.json`
  — per-keyframe camera intrinsics + extrinsics (OpenGL, world-Y
  gravity-up).
- `sessions/0beced53c9df/out/{ceiling.jpg, height.npy, plan.json}`
  — ortho canvas, per-pixel ceiling height, and the user-traced
  region polygons.

## Outputs (under `outputs/`, gitignored)

- One fused ortho-space mask per concept × per-experiment-setting.
- A coloured overlay PNG on `ceiling.jpg`.
- `summary.json` — instance count, mean per-instance area cm²,
  median per-instance contributing-keyframe count.
- `per_region_counts.json` — how many lights / vents / items land
  in each region of `plan.json["regions"]`.
- Contact-sheet `index.html` showing all panels side-by-side.

## Re-run

Once `project_masks.py` lands, the canonical invocation is:

```bash
python -m ceiling_rcp.sam3_projection \
  --session 0beced53c9df \
  --results-dir <abs path to segmentation_sam3_runpod/results> \
  --output experiments/segmentation_projection/outputs/<setting-id> \
  [--p-angle 2] [--t-fuse 0.5] [--rule weighted-majority] [--occlusion]
```

Parameters mean what `NEXT_SESSION.md` says they mean.
