# Handoff — project SAM 3 ceiling-item masks onto the ortho view

You are the next agent. The previous session built a SAM 3 segmentation
pipeline that produced per-keyframe masks for three concepts on Lachy's
Polycam scan: `ceiling item`, `light`, `vent`. Your job is to take those
per-keyframe masks, project them back onto the top-down (ortho) ceiling
render that the main app already produces, fuse the votes from all 268
keyframes into one ortho-space mask per concept with a sensible
view-quality weighting, then sweep a handful of fusion parameters and
save side-by-side renders for me to review.

**Read this file end-to-end before writing any code.** Read the
existing source it references too. Don't re-derive what's already in
the repo — wire into it.

## What you have

### Per-image SAM 3 results — the input to your work

`experiments/segmentation_sam3_runpod/results/<image_id>/`
- `input.jpg` — 768 × 1024 (W × H), rotated **clockwise 90°** from the
  original 1024 × 768 Polycam frame
- `<prompt>_instances.png` — uint16 label map in rotated-image coords
  (0 = background, 1..N = instance id), one file per prompt
- `<prompt>_mask.png` — binary union (0 / 255), rotated coords
- `detections.json` — list per prompt, each detection has
  `score`, `box_xyxy`, `polygon` (rotated coords), `area_px`
- 268 image folders total. The id matches the keyframe filename used
  by the rest of the project.

### Polycam keyframes — what the masks correspond to

`Scan data/Lachys Polycam/Lachys line/keyframes/`
- `corrected_images/<id>.jpg` — 1024 × 768 colour-corrected RGB,
  *original* orientation (the un-rotated input that was rotated CW 90
  to feed SAM 3)
- `corrected_cameras/<id>.json` — 3×4 camera-to-world transform with
  intrinsics. Fields:
  ```
  cx, cy, fx, fy            intrinsics for the ORIGINAL 1024×768 image
  width, height              1024, 768
  t_00 .. t_22               3×3 rotation, row-major (t_ij = row i col j)
  t_03, t_13, t_23           camera position in world coords
  ```
  **Convention is OpenGL: camera looks down its −Z axis.** World Y is
  gravity-up (a horizontal ceiling has `normal.y = -1` in `mesh.py`).
  See `experiments/segmentation_sam3/prepare_inputs.py` and
  `~/.claude/projects/.../memory/polycam_camera_convention.md`.
- `corrected_depth/<id>.png` — per-keyframe depth (16-bit, mm or
  similar; verify the unit by reading any existing depth-loading code
  in the repo before assuming).

### The mesh and the ortho pipeline you must integrate with

`src/ceiling_rcp/`
- `mesh.py` — load the OBJ + textures, world-Y range,
  `downward_face_mask` for picking ceiling triangles
- `raster.py` — the existing top-down rasteriser; this is what
  produces `ceiling.jpg` and `ceiling_planes.jpg`. **Read this first
  to learn the canvas dimensions, pixel-to-world transform, and how
  per-pixel ceiling height (Z) is sampled.**
- `planes.py`, `polygons.py`, `analyse.py`, `cli.py`, `server.py` —
  the rest of the ceiling-RCP app; you don't need to edit these for
  this experiment, but if you can, factor your projection so it could
  be added as a sibling module under `src/ceiling_rcp/`.

`Scan data/Lachys Polycam/Lachys line mesh/`
- `15_4_2026.obj` + `.mtl` + `textures/` — the canonical mesh

`out_Lachys_Polycam/` (or a session under `sessions/`)
- `ceiling.jpg`, `ceiling_planes.jpg`, `plan.json` — what the app
  currently renders. Use the same canvas / pixel-to-world transform
  in your output so a viewer can flip between the existing
  segmentation overlay and your fused-masks overlay.

### Background reading

- `experiments/segmentation_sam3_runpod/README.md` — explains the
  Roboflow-vs-local calibration finding and why the masks we have
  are clean (no whole-ceiling false positive)
- `experiments/segmentation_sam3/README.md` — explains the keyframe
  filter (`-t_12 > 0.20` ⇒ 268 of 549 frames face up) and the CW 90°
  rotation
- `STATUS.md` — current MVP direction

## The task

### 1. Rotate masks back to original-image orientation

The masks are in 768×1024 rotated-image coords. The intrinsics in the
camera JSON are for the 1024×768 original. **Rotate the binary
mask / instance map / polygon coords CCW 90°** before projecting.

For polygons: a point `(u_rot, v_rot)` in the rotated image maps to
`(v_rot, W_rot - 1 - u_rot)` = `(v_rot, 767 - u_rot)` in the original
image. Verify by overlaying a back-rotated mask on the
`corrected_images/<id>.jpg` for one keyframe and visually checking it
lines up with the fixture.

### 2. Project each mask onto the ortho canvas

For every ortho pixel `(X_o, Y_o)`:

1. **Look up the ceiling height** `Z(X_o, Y_o)` at that pixel — the
   raster pipeline already builds a height map; reuse it. The world
   point is `P = (X_w, Z_w, Y_w)` where the convention swap is what
   `raster.py` uses (read it). If the ortho is X-Z (top-down) then
   `P = (X_w, Z_ceiling, Y_w)` in world coords.
2. **For each of the 268 keyframes**, project `P` into that camera:
   ```
   P_cam = R^T (P − t)        # world → camera (R is 3×3, t is camera
                              # position; OpenGL convention so the
                              # camera looks down −Z_cam)
   if P_cam.z >= 0: skip       # behind the camera
   u = fx * (P_cam.x / -P_cam.z) + cx
   v = fy * (P_cam.y / -P_cam.z) + cy
   ```
   If `(u, v)` is inside `[0, W) × [0, H)` and the back-rotated mask
   is 1 at `(round(u), round(v))`, this keyframe contributes a vote
   for that ortho pixel.
3. **Optionally consult depth** to drop occluded contributions: if
   `P_cam.z < -depth(u, v) − tolerance`, the camera can't see this
   ceiling point — skip. Worth it for any ceiling that has bulkheads
   or pendants.

### 3. View-quality weighting

Each contributing keyframe casts a vote with weight

```
w = w_dist × w_angle × w_score
```

where, for that ortho pixel `(X_o, Y_o)`:

- `w_dist`   = `(d_ref / d)^p_dist`, with `d` = distance from the
              camera to the ceiling point, `d_ref` = some reference
              distance (e.g. 1.5 m), `p_dist` ∈ [1, 2]. Closer is
              more reliable.
- `w_angle`  = `max(0, cos θ)^p_angle`, where `θ` is the angle
              between the camera-to-ceiling-point ray and the ceiling
              normal (≈ world −Y for a flat ceiling; use the per-pixel
              normal if you have it). Less oblique = more reliable.
              `p_angle` ∈ [1, 4]. Use the cosine, not the angle, to
              keep it cheap and continuous. Drop any vote with cos θ
              below ~0.2 (very grazing).
- `w_score`  = SAM 3 detection score for the instance whose mask the
              pixel belongs to. (Available in `detections.json`.)

### 4. Fuse the votes per ortho pixel

Pick a fusion rule and let me sweep over the choice. Three obvious
ones, in order of robustness:

- **Mean vote**: `sum(w · 1) / sum(w)` over all contributing
  keyframes, with the pixel kept if the mean exceeds a threshold
  `t_fuse`. Sensitive to outliers when one camera's mask is wrong.
- **Weighted majority**: same, but threshold the weighted ratio of
  positive-to-total votes (positive = mask = 1, total = pixel was
  visible to that camera) and require a minimum number of contributing
  keyframes (e.g. ≥ 3). This is the one I expect to win.
- **Max-confidence vote**: keep if any single keyframe with high score
  *and* good `w_angle * w_dist` votes positive. Cheaper, noisier.

After fusion you have one binary or float-confidence ortho mask per
prompt. Run connected components, drop blobs below some
min-area-cm² (in world units, not pixels), assign instance ids, and
write out PNGs in the same coordinate space as `ceiling.jpg`.

## Experiments to run for me

Spin out four panels per concept, each panel showing the fused ortho
mask under different settings:

1. **Vary `p_angle`** ∈ {1, 2, 4}, fix everything else.
2. **Vary `t_fuse`** (the fusion threshold) on the weighted-majority
   rule ∈ {0.3, 0.5, 0.7}.
3. **Compare fusion rules**: mean / weighted-majority / max-conf at
   their best per-rule defaults.
4. **Ablate occlusion**: with vs without the depth-based occlusion
   skip in step 2.

For each setting, save:
- `ortho_<concept>_<setting-id>.png` — coloured overlay on
  `ceiling.jpg` (red / yellow / blue per concept, alpha 0.45 same as
  the existing local pipeline)
- `ortho_<concept>_<setting-id>_mask.png` — pure mask 0/255
- A `summary.json` per experiment with: settings, instance count,
  mean per-instance area cm², median per-instance contributing
  keyframe count

Build a contact-sheet `index.html` like
`experiments/segmentation_sam3_runpod/index.html` so I can scroll the
panels side by side. Same UI shape (filterable, sortable). The CSS
in that file is fine to copy.

Save everything under
`experiments/segmentation_projection/`. Each experiment gets its own
subfolder. Drop a `README.md` at the top of that folder explaining
the projection math, the weighting, and the fusion rules in one
page (don't restate this handoff).

## Pitfalls captured from the previous session

- **Don't re-derive the camera convention**. It's OpenGL (forward =
  −Z), world-Y is gravity-up. Several of our debugging detours came
  from assuming OpenCV.
- **Don't trust intrinsics on the rotated image**. The fx/fy/cx/cy in
  the camera JSON are for the *original* 1024 × 768. Rotate masks
  back before projecting.
- **The ceiling isn't a single plane**. There are bulkheads. The
  existing raster builds a per-pixel height; use it. Projecting
  against a single mean ceiling height will smear masks.
- **Weight by `cos θ`, not by the angle in radians**. The cosine maps
  cleanly to "fraction of the texel that's visible from the camera",
  which is what you actually want for an angle-quality term.
- **The same instance id in different keyframes is meaningless**.
  Don't try to track instances across keyframes from the SAM 3 ids;
  fuse pixel-level masks and re-run connected components in ortho
  space to recover instances.
- **Don't burn GPU time re-running SAM 3**. The 268-image sweep cost
  about $0.07 on a RunPod 3090. The masks you need are already on
  disk. If you do need to re-run, the script at
  `experiments/segmentation_sam3_runpod/run_runpod.py` handles pod
  spin-up + always-cleanup; pass `--ids a,b,c` for a subset.
- **The Roboflow-hosted endpoint is on a `-preview-` workflow id**
  that they can yank. Don't depend on it for the main app. The
  RunPod self-host path is the durable one — see
  `experiments/segmentation_sam3_runpod/README.md`.

## Definition of done for this session

- A working `project_masks.py` (or equivalent module under
  `src/ceiling_rcp/`) that takes the working-set masks + cameras +
  ortho canvas and emits one fused ortho-space mask per concept.
- The four experiment panels above, saved with overlays + masks +
  summary.
- Contact-sheet `index.html` for browsing.
- README explaining the math and how to re-run / change parameters.
- One sentence in your final reply telling me which fusion rule and
  which `p_angle` / `t_fuse` you'd ship as the default — based on
  what you saw, not what you assumed.

Read the code, run something small first to validate the projection
maths on a single keyframe (overlay one keyframe's back-rotated mask
on `ceiling.jpg` and check it lands on the right fixture), *then*
batch the rest. The single-frame sanity check will save you an hour
of debugging.
