# Next session — project SAM 3 ceiling-item masks onto session `0beced53c9df`'s ortho

You are the next agent. The repo's `experiments/segmentation_sam3_runpod/`
already holds the scripts and per-keyframe SAM 3 mask outputs for
Lachy's Polycam scan (three concepts: `ceiling item`, `light`, `vent`,
across 268 keyframes). Your job is to project those per-keyframe masks
onto the **top-down ortho image of session `0beced53c9df`** that the
main app produces, fuse the votes from all 268 keyframes into one
ortho-space mask per concept under a sensible view-quality weighting,
sweep a handful of fusion parameters, and save side-by-side renders
for review.

This file replaces the more general handoff at
`experiments/segmentation_sam3_runpod/HANDOFF.md` — read both, but
when they conflict, the paths and target session below win.

**Read this file end-to-end before writing any code.** Read the
existing source it references too. Don't re-derive what's already in
the repo — wire into it.

## What you have

### Target session — what you're projecting onto

`sessions/0beced53c9df/out/`
- `ceiling.jpg` — top-down ceiling render at the grid resolution
  defined in `plan.json["grid"]`. This is the canvas you draw the
  fused ortho mask onto.
- `height.npy` — per-pixel ceiling height map, world-Y in metres,
  NaN outside any down-face. **This is what gives you Z(X_o, Y_o)
  for each ortho pixel — read this, not a single mean ceiling.**
- `plan.json` — read `grid`, `room`, `main`, `regions`,
  `obstructions` to understand the canvas + the user-traced
  region polygons. The grid block tells you the world↔pixel
  transform: `min_x`, `max_x`, `min_z`, `max_z`,
  `pixels_per_metre`, `width`, `height`.

The main app's coordinate convention lives in
`src/ceiling_rcp/planes.py` (`PlanGrid`). Use that class — don't
re-implement world↔pixel arithmetic.

### Per-keyframe SAM 3 results — the input to your projection

`experiments/segmentation_sam3_runpod/results/<image_id>/` (this
directory is **gitignored** — bulk PNGs live at
`/Users/harishusic/Documents/Claude Code/Space Room Plan GS/.claude/worktrees/trusting-lovelace-911506/experiments/segmentation_sam3_runpod/results/`
on disk. Either symlink or `cp -a` it into your worktree, or pass
its absolute path as `--results-dir`. Don't re-run RunPod unless
you have a reason — the existing 268-image sweep was clean.)

Per result folder:
- `input.jpg` — 768 × 1024 (W × H), rotated **clockwise 90°** from
  the original 1024 × 768 Polycam frame
- `<prompt>_instances.png` — uint16 label map in rotated-image
  coords (0 = background, 1..N = instance id), one file per prompt
- `<prompt>_mask.png` — binary union (0 / 255), rotated coords
- `detections.json` — list per prompt; each detection has `score`,
  `box_xyxy`, `polygon` (rotated coords), `area_px`
- 268 image folders total. The id matches the keyframe filename used
  by the rest of the project.

### Polycam keyframes — what the masks correspond to

`Scan data/Lachys Polycam/Lachys line/keyframes/` (gitignored;
already on disk):
- `corrected_images/<id>.jpg` — 1024 × 768 colour-corrected RGB,
  *original* orientation (the un-rotated input that was rotated CW
  90° to feed SAM 3)
- `corrected_cameras/<id>.json` — 3×4 camera-to-world transform
  with intrinsics. Fields:
  ```
  cx, cy, fx, fy            intrinsics for the ORIGINAL 1024×768 image
  width, height              1024, 768
  t_00 .. t_22               3×3 rotation, row-major (t_ij = row i col j)
  t_03, t_13, t_23           camera position in world coords
  ```
  **Convention is OpenGL: camera looks down its −Z axis.** World Y
  is gravity-up (a horizontal ceiling has `normal.y = -1` in
  `mesh.py`).
- `corrected_depth/<id>.png` — per-keyframe depth (16-bit, mm or
  similar — verify by reading any existing depth-loading code in
  the repo before assuming).

### The ortho pipeline you must integrate with

`src/ceiling_rcp/`
- `mesh.py` — load the OBJ + textures, world-Y range,
  `ceiling_face_mask(min_ceiling_height_m=2.0)` for picking
  ceiling triangles. **The semantics changed in WIP 4** — it's
  now an absolute world-Y floor, not a band-below-top. Don't
  recreate the old behaviour.
- `raster.py` — the existing top-down rasteriser. Read this first
  to learn the canvas dimensions, pixel-to-world transform, and how
  per-pixel ceiling height is sampled.
- `planes.py` — `PlanGrid` (the shared XZ↔pixel coordinate
  transform that also implements the RCP X-mirror). **Use this
  for every world↔pixel conversion** — don't roll your own.
- `polylabel.py`, `polygons.py`, `analyse.py`, `topology.py`,
  `server.py` — the rest of the app; you don't need to edit these
  for this experiment, but factor your projection so it could be
  added as a sibling module under `src/ceiling_rcp/` (e.g.
  `src/ceiling_rcp/sam3_projection.py`).

### Background reading

- `experiments/segmentation_sam3_runpod/README.md` — explains the
  Roboflow-vs-local calibration finding and why the masks we have
  are clean (no whole-ceiling false positive).
- `experiments/segmentation_sam3/README.md` — explains the
  keyframe filter (`-t_12 > 0.20` ⇒ 268 of 549 frames face up)
  and the CW 90° rotation.
- `STATUS.md` — current MVP direction; current build is
  `good-wip-4-05052026` on `main`.

## The task

### 1. Rotate masks back to original-image orientation

The masks are in 768 × 1024 rotated-image coords. The intrinsics in
the camera JSON are for the 1024 × 768 original. **Rotate the binary
mask / instance map / polygon coords CCW 90°** before projecting.

For polygons: a point `(u_rot, v_rot)` in the rotated image maps to
`(v_rot, W_rot − 1 − u_rot)` = `(v_rot, 767 − u_rot)` in the original
image. Verify by overlaying a back-rotated mask on the
`corrected_images/<id>.jpg` for one keyframe and visually checking it
lines up with the fixture.

### 2. Project each mask onto the ortho canvas

For every ortho pixel `(X_o, Y_o)`:

1. **Look up the ceiling height** `Z(X_o, Y_o)` at that pixel by
   reading `height.npy[Y_o, X_o]`. Skip NaN pixels — those are
   outside any down-face, so no projection makes sense.
2. Convert ortho pixel → world via `PlanGrid.px_to_world` (or
   whatever the equivalent helper is — read `planes.py`). The
   resulting world point is `P = (X_w, Z_world, Z_w)` where
   `Z_world` comes from `height.npy`. (Polycam world Y is up.)
3. **For each of the 268 keyframes**, project `P` into that camera:
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
4. **Optionally consult depth** to drop occluded contributions: if
   `P_cam.z < -depth(u, v) − tolerance`, the camera can't see this
   ceiling point — skip. Worth it for any ceiling that has bulkheads
   or pendants (this scan does — there's a clear bulkhead on the
   right side of the room, see the existing topology in
   `sessions/0beced53c9df/out/plan.json`).

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
              between the camera-to-ceiling-point ray and the
              ceiling normal (≈ world −Y for a flat ceiling; use the
              per-pixel normal from `mesh.py` if you have it).
              Less oblique = more reliable. `p_angle` ∈ [1, 4]. Use
              the cosine, not the angle, to keep it cheap and
              continuous. Drop any vote with cos θ below ~0.2 (very
              grazing).
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
  visible to that camera) and require a minimum number of
  contributing keyframes (e.g. ≥ 3). This is the one I expect to
  win.
- **Max-confidence vote**: keep if any single keyframe with high
  score *and* good `w_angle * w_dist` votes positive. Cheaper,
  noisier.

After fusion you have one binary or float-confidence ortho mask per
prompt. Run connected components, drop blobs below some
min-area-cm² (in world units, not pixels), assign instance ids, and
write out PNGs in the same coordinate space as `ceiling.jpg`.

### 5. Cross-reference against the user's region polygons

This is the bit specific to session `0beced53c9df`: the user has
already traced ceiling regions in this session. After you have one
fused ortho-space mask per concept, **also** report per-region
counts: for each region in `plan.json["regions"]` (and the main
ceiling), count how many `light` / `vent` / `ceiling item` instances
land inside. Output to `outputs/per_region_counts.json` like:

```json
{
  "main": {"lights": 4, "vents": 2, "items": 1},
  "regions": [
    {"id": 0, "label": "Ceiling Region (2)", "lights": 2, "vents": 0, "items": 0},
    ...
  ]
}
```

This is the user-facing signal for the experiment — "given the
zones I've drawn, what's actually in each one."

## Experiments to run

Spin out four panels per concept, each panel showing the fused
ortho mask under different settings:

1. **Vary `p_angle`** ∈ {1, 2, 4}, fix everything else.
2. **Vary `t_fuse`** (the fusion threshold) on the
   weighted-majority rule ∈ {0.3, 0.5, 0.7}.
3. **Compare fusion rules**: mean / weighted-majority / max-conf at
   their best per-rule defaults.
4. **Ablate occlusion**: with vs without the depth-based occlusion
   skip in step 2.

For each setting, save under
`experiments/segmentation_projection/outputs/<setting-id>/`:
- `ortho_<concept>.png` — coloured overlay on `ceiling.jpg` (red /
  yellow / blue per concept, alpha 0.45 same as the main app's
  fill alpha — pull `PLAN_FILL_ALPHA` from `server.py` if you want
  to match exactly)
- `ortho_<concept>_mask.png` — pure mask 0/255
- `summary.json` — settings, instance count, mean per-instance
  area cm², median per-instance contributing keyframe count
- `per_region_counts.json` — the cross-reference above

Build a contact-sheet `index.html` like
`experiments/segmentation_sam3_runpod/index.html` so I can scroll
the panels side by side. Same UI shape (filterable, sortable). The
CSS in that file is fine to copy.

Drop a `README.md` at the top of `experiments/segmentation_projection/`
explaining the projection math, the weighting, and the fusion rules
in one page (don't restate this handoff).

## Pitfalls captured from prior sessions

- **Don't re-derive the camera convention**. It's OpenGL (forward =
  −Z), world-Y is gravity-up. Several of our debugging detours came
  from assuming OpenCV.
- **Don't trust intrinsics on the rotated image**. The fx/fy/cx/cy
  in the camera JSON are for the *original* 1024 × 768. Rotate
  masks back before projecting.
- **The ceiling isn't a single plane**. There are bulkheads. The
  existing raster builds a per-pixel height; use it. Projecting
  against a single mean ceiling height will smear masks.
- **Use `PlanGrid` from `planes.py`** for every world↔pixel
  conversion. It implements the RCP X-mirror; rolling your own is
  how you get a horizontally-flipped result that works on geometry
  but not on the rendered ortho.
- **Weight by `cos θ`, not by the angle in radians**. The cosine
  maps cleanly to "fraction of the texel that's visible from the
  camera", which is what you actually want for an angle-quality
  term.
- **The same instance id in different keyframes is meaningless**.
  Don't try to track instances across keyframes from the SAM 3 ids;
  fuse pixel-level masks and re-run connected components in ortho
  space to recover instances.
- **Don't burn GPU time re-running SAM 3**. The 268-image sweep
  cost about $0.07 on a RunPod 3090. The masks you need are
  already on disk at the path called out under "Per-keyframe SAM
  3 results" above. If you do need to re-run, the script at
  `experiments/segmentation_sam3_runpod/run_runpod.py` handles pod
  spin-up + always-cleanup; pass `--ids a,b,c` for a subset.
- **The Roboflow-hosted endpoint is on a `-preview-` workflow id**
  that they can yank. Don't depend on it for the main app. The
  RunPod self-host path is the durable one — see
  `experiments/segmentation_sam3_runpod/README.md`.
- **The scan-settings semantic changed in WIP 4.**
  `ceiling_face_mask` now takes `min_ceiling_height_m` (absolute
  world-Y floor, default 2.0 m) instead of
  `max_ceiling_variance_m`. If you're rendering a fresh ortho for
  cross-checking, use the new name.

## Definition of done

- A working `project_masks.py` (or equivalent module under
  `src/ceiling_rcp/`) that takes the keyframe masks + cameras +
  ortho canvas and emits one fused ortho-space mask per concept.
- The four experiment panels above, saved with overlays + masks +
  summary.
- `per_region_counts.json` for each setting, cross-referenced
  against `sessions/0beced53c9df/out/plan.json`.
- Contact-sheet `index.html` for browsing.
- `README.md` explaining the math and how to re-run / change
  parameters.
- One sentence in your final reply telling me which fusion rule
  and which `p_angle` / `t_fuse` you'd ship as the default —
  based on what you saw, not what you assumed.

Read the code, run something small first to validate the
projection maths on a single keyframe (overlay one keyframe's
back-rotated mask on `ceiling.jpg` and check it lands on the
right fixture), *then* batch the rest. The single-frame sanity
check will save you an hour of debugging.
