# Segmentation experiment — SAM 3 (text prompts)

Run Meta's SAM 3 (`facebook/sam3`) over the upward-facing keyframes of
Lachy's Polycam scan to localise three concepts on the ceiling:

- `ceiling item`
- `light`
- `vent`

## Inputs

Keyframes from
`Scan data/Lachys Polycam/Lachys line/keyframes/`. Two files matter:

- `corrected_images/<id>.jpg` — 1024×768 colour-corrected RGB
- `corrected_cameras/<id>.json` — 3×4 camera-to-world transform with
  `t_ij` = row `i`, col `j`. Third column `(t_02, t_12, t_22)` is the
  camera's +Z axis (forward) in world coords; world Y is gravity-up.

## Pipeline

### 1. Filter to ceiling-facing frames

Polycam keyframes use the **OpenGL** camera convention — the lens
looks down −Z. The world-Y component of the camera-forward direction
is therefore `−t_12`. `prepare_inputs.py` keeps a frame when that
value is `> 0.20` (forward tilted ≥11.5° above horizontal). On
Lachy's session this leaves **268 of 549 frames**.

### 2. Rotate to put ceiling on top

In the corrected landscape image the phone is held in portrait, so
world-up sits on the **left** edge. Rotating clockwise 90° (PIL
`ROTATE_270`) brings the ceiling to the top — final size 768×1024. The
manifest `working_set/_manifest.json` records every selected id and its
`t_12` value.

### 3. SAM 3 inference

`run_sam3.py` loads `facebook/sam3` once, then for every image runs
three text prompts and post-processes with
`Sam3Processor.post_process_instance_segmentation` at default
thresholds (`threshold=0.5`, `mask_threshold=0.5`).

Device autoselect: CUDA → MPS → CPU. On Apple Silicon (MPS) expect a
few seconds per (image × prompt) after the one-off ~30 s model load.

## Output layout

```
experiments/segmentation_sam3/
  prepare_inputs.py
  run_sam3.py
  README.md
  working_set/                 268 rotated 768×1024 JPGs + _manifest.json
  results_sample/              10-image sanity check (seed 42)
    <image_id>/
      input.jpg                rotated input
      ceiling_item_mask.png    binary union of all instances (0/255)
      ceiling_item_instances.png   uint16 label map (0=bg, 1..N=instance)
      ceiling_item_overlay.jpg     coloured overlay + boxes + score labels
      light_mask.png ...
      vent_mask.png ...
      detections.json          per-prompt list of {score, box_xyxy, area_px}
    _summary.json              global record (model, device, totals)
  results_full/                same layout for all 268 frames
```

Overlay colours: `ceiling item` = red, `light` = yellow, `vent` = blue.

## Reproduce

```bash
# 1. build the working set (idempotent)
python3 experiments/segmentation_sam3/prepare_inputs.py

# 2. 10-image sanity check (default --n 10 --seed 42)
python3 experiments/segmentation_sam3/run_sam3.py \
    --out experiments/segmentation_sam3/results_sample

# 3. full run on all 268
python3 experiments/segmentation_sam3/run_sam3.py \
    --n 0 \
    --out experiments/segmentation_sam3/results_full

# Re-segment specific images by id (comma-separated)
python3 experiments/segmentation_sam3/run_sam3.py \
    --ids 43107985971,43111470676 \
    --out experiments/segmentation_sam3/results_adhoc
```

Both scripts have `--threshold` and `--mask-threshold` flags if you
want to lower them to chase recall on small/dim objects.

## Notes

- First run downloads `facebook/sam3` weights (~few hundred MB) into
  `~/.cache/huggingface`. Requires `huggingface-cli login`.
- The threshold `−t_12 > 0.20` was chosen to match the user's
  expected ~268 kept frames. A finer or coarser cut is one number
  change via `--min-forward-y`.
- Frames where the phone was held in landscape (rare in this scan) come
  out rotated incorrectly — they'd need a per-frame "which axis is
  world-up?" branch in `prepare_inputs.py`. Not implemented; eyeball
  the working set if you want to confirm.
