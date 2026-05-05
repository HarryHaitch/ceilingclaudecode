# Next session

The 20-item WIP-4 punch list shipped on `main` (tag
`good-wip-4-05052026`). The next experiment is a fresh thread:

→ **Project SAM 3 ceiling-item masks onto the ortho.**

Read [`experiments/segmentation_projection/NEXT_SESSION.md`](experiments/segmentation_projection/NEXT_SESSION.md)
end-to-end before writing code. It gives you the data layout, the
projection math, the view-quality weighting, the fusion rules, the
experiment matrix, and the per-region cross-reference that's the
user-facing answer.

## TL;DR

You have:

- 268 per-keyframe SAM 3 mask outputs (three concepts:
  `ceiling item`, `light`, `vent`) from a previous session.
  Bulk lives at
  `/Users/harishusic/Documents/Claude Code/Space Room Plan GS/.claude/worktrees/trusting-lovelace-911506/experiments/segmentation_sam3_runpod/results/`.
  The scripts and READMEs are in
  `experiments/segmentation_sam3_runpod/` (committed to main; bulk
  is gitignored).
- Polycam keyframes + per-frame intrinsics/extrinsics under
  `Scan data/Lachys Polycam/Lachys line/keyframes/`.
- The main app's ortho output for the target session at
  `sessions/0beced53c9df/out/{ceiling.jpg, height.npy, plan.json}`,
  including the user-traced region polygons.

You produce:

- One fused ortho-space mask per concept under various
  fusion-parameter settings, written to
  `experiments/segmentation_projection/outputs/<setting-id>/`.
- Per-region instance counts cross-referenced against
  `plan.json["regions"]` (the user-facing signal: "this region
  has 3 lights, 1 vent").
- A contact-sheet `index.html` for browsing the parameter sweep.
- A `README.md` documenting the projection math + how to
  re-run / change parameters.

## Sanity-check first

Single-keyframe overlay before batching: rotate one mask CCW 90°,
project the ceiling-point cloud through that camera, draw the
back-projected mask onto `ceiling.jpg`. Visually verify the mask
lands on the right fixture. **Don't** start the 268-keyframe sweep
before this passes.

## Existing context worth a skim

- `STATUS.md` — what's on `main` (WIP 4: bug fixes + interface
  tracing finish-up + histogram + scan-settings + page-size).
- `ARCHITECTURE.md` — the v3+ plan JSON shape, coordinate
  convention, and where each main-app module lives.
- `experiments/segmentation_sam3_runpod/HANDOFF.md` — the original
  generic version of this prompt (this file overrides where they
  conflict, in particular the target session id).
- `experiments/segmentation_sam3_runpod/README.md` — calibration
  notes + RunPod self-host gotchas.
