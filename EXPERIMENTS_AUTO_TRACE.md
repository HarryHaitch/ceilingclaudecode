# Experiments session — auto-trace room outline + interface chords

Self-contained prompt for a **separate** Claude Code session whose only
job is research: detect the room outline and the ceiling interface
chords directly from the 3D mesh, well enough that the user finishes
by hand instead of starting from a blank canvas.

This is a **research session**, not feature work. No production code
needs to land in the main editor. The deliverable is a writeup, a
leaderboard, and overlay PNGs the user can eyeball.

Paste the block between the `---` lines into the new session.

---

This is the ceiling-rcp project at
`/Users/harishusic/Documents/Claude Code/Space Room Plan GS`.
GitHub: https://github.com/HarryHaitch/ceilingclaudecode

The active build is **Good WIP 3 05052026**. Read-only snapshot at
`/Users/harishusic/Documents/Claude Code/Good WIP 3 05052026` (do
NOT touch — `chmod a-w`). Tag `good-wip-3-05052026` marks the same
commit. The companion live branch is
`claude/admiring-fermi-4b18e7-impl` in worktree
`<repo>/.claude/worktrees/admiring-fermi-4b18e7`.

You should **not edit the main impl branch** in this session. Spin
up a fresh worktree off the milestone tag for all your experiment
code:

```
cd "/Users/harishusic/Documents/Claude Code/Space Room Plan GS"
git worktree add .claude/worktrees/auto-trace -b experiments/auto-trace good-wip-3-05052026
cd .claude/worktrees/auto-trace
python3 -m pip install -e .
```

Verify the install points at the new worktree:

```
python3 -c "import ceiling_rcp; print(ceiling_rcp.__file__)"
# → must end in .claude/worktrees/auto-trace/src/ceiling_rcp/__init__.py
```

## What you're trying to do

Right now the user starts from a blank canvas: trace the room
outline, trace each ceiling-zone interface chord, click "Define
ceilings". We want to **pre-populate** those linework hints from
the 3D mesh, so the user is correcting / completing rather than
drawing from scratch.

Two outputs per scan:

1. **Room outline** — the closed polygon of walls that meet the
   ceiling. Today the user clicks 4–20 vertices to define it.
2. **Interface chords** — line segments inside the room where
   adjacent ceiling zones meet (bulkhead edges, recess perimeters,
   tray-ceiling steps). Today the user traces each one as an
   interface polyline.

You don't need to be perfect. Recall is more important than
precision — better to over-detect (user deletes spurious lines)
than under-detect (user retraces missing ones). A 60–80 % recall
with low geometric error is a usable MVP.

## Why the previous experiments don't apply

A prior session ran a sweep on `felzenszwalb_*` and friends in
`debug/` to segment ceiling **regions** (faces). That solved a
different problem — partition the ceiling area into N labelled
regions. Best score was ~0.83 IoU with `felz_merged_s200_3cm`. We
keep that around for the legacy auto-detect endpoint but it is
**not** the right tool for line / edge detection.

This session attacks the line problem from first principles. Don't
just copy the old harness — model your own pipeline around it but
score against line-segment ground truth, not region masks.

## The user's hypothesis (this is the seed signal to investigate)

Walls and bulkhead lips are **vertical surfaces in the mesh**.
Specifically:

- A **room wall** that meets the ceiling has a tall stack of
  near-vertical mesh faces (floor → ceiling, typically 1.5–3 m).
  Even partial walls reach ~1 m before the LiDAR loses them.
- A **bulkhead lip** (interior interface chord) has the same
  signature, just shorter — vertical faces that **start at ceiling
  level and drop 5 cm to ~1 m** depending on the bulkhead depth.
  They share the same top-edge Y as the surrounding ceiling.

The user's intuition: filter the mesh for vertical-ish faces, group
them into vertical "stacks", project each stack's footprint to the
XZ floor plane, and you get line segments — full-height stacks
become room-outline edges; ceiling-anchored short stacks become
interface chords.

This is a hypothesis, not gospel. **If the data tells you a
different signal works better — say so and try it.** The user will
trust your judgement if you back it with experiment results.

## Suggested experiments (start here, branch out)

A handful of approaches worth comparing. Don't do all of them
sequentially; spike the cheapest ones first, then dig deeper into
whatever's working.

### A. Vertical-face mask raster (cheapest, strongest naive)

For each mesh triangle, compute its normal. Filter for "vertical-ish"
(`|normal.y| < cos(80°)` ≈ 0.17, or whatever angle the data wants).
Project each kept triangle onto the XZ plan grid (same grid as
`raster.render_textured_topdown`), accumulating face area or count
per pixel. The result is a heatmap where walls + bulkhead lips
light up. Threshold + skeletonise + Hough transform → line segments.

Variants to try:
- Weight by face area (default).
- Weight by face Y-extent (taller faces matter more).
- Bin faces by their **top Y** (ceiling-band membership) before
  projecting — only project faces whose top reaches the ceiling.
  This filters out random furniture verticals.

### B. Ceiling-anchored vertical stacks

For each near-vertical face, check whether its top edge sits within
the ceiling band (re-use `mesh.ceiling_face_mask`'s percentile
logic — top of ceiling envelope ± 5 cm). If yes, project the face's
bottom edge to XZ. Group adjacent projections into line segments.
This isolates **only** the surfaces that descend from the ceiling
— exactly what defines bulkhead lips and the inside face of walls.

The Y-extent of each accepted face also tells you wall vs bulkhead:

- Extent ≥ 1.0 m → wall fragment → contributes to room outline.
- 0.05 m ≤ extent < 1.0 m → bulkhead lip → contributes to interface
  chord.
- < 0.05 m → noise.

### C. Horizontal cross-section just below ceiling

Slice the full mesh at `Y = ceiling_top - 0.05 m` (about 50 mm
below the ceiling envelope). The intersection of the mesh with that
horizontal plane is a set of 2D line segments — every wall and
every bulkhead lip cuts that plane. This is a direct geometric
read of "what edges does the ceiling sit on", independent of face
counts or stack depth.

Library: `trimesh.intersections.mesh_plane` if `trimesh` is
available, otherwise a simple per-triangle plane-cut you can
implement in <50 lines (each triangle either misses, intersects in
a segment, or is fully on one side of the plane).

This may turn out to be the most robust signal — try it early.

### D. RANSAC line fit on vertical-face vertex point clouds

Take the vertices of every vertical-face triangle, project them to
XZ, and run RANSAC line fitting iteratively (each iteration fits
the line with most inliers, removes them, repeats until inlier
count drops below a threshold). Each accepted line is a candidate
wall / chord segment.

### E. Edge gradient on the existing height map

`sessions/<id>/out/height.npy` already exists per session. Sharp
height drops along an edge are ceiling-zone interfaces. Sobel +
non-max-suppression + Hough on `height.npy` gives you bulkhead
edges nearly for free. Compare against approach (A) — if (E)
matches at a fraction of the cost, ship it.

### F. Combine signals

A meta-experiment — fuse the best two approaches (e.g. (A)
vertical-face raster ∪ (E) height-gradient) and see if recall
improves without precision tanking.

## Available data

### Mesh helpers (already shipped)

`src/ceiling_rcp/mesh.py`:

- `inspect_folder(folder) -> FolderReport` — find `.obj`, `.mtl`,
  `mesh_info.json`, textures.
- `load_mesh(report, align=True) -> Mesh` — returns dataclass with
  vertex array, face index array, per-face normals, per-vertex Y,
  texture refs.
- `downward_face_mask(normals, max_tilt_deg=30)` — keeps faces with
  normal pointing roughly down (`normal.y < -cos(deg)`).
- `ceiling_face_mask(mesh, max_tilt_deg=60, max_ceiling_variance_m=1.5)`
  — combined cone + height-band ceiling filter (the "what the user
  sees as ceiling" mask).

You'll want a **vertical** sibling to `downward_face_mask` —
trivial: `np.abs(normals[:, 1]) < cos(deg)`. Build it on the fly.

### Geometry helpers

`src/ceiling_rcp/planes.py`:

- `PlanGrid(min_x, max_x, min_z, max_z, pixels_per_metre)` —
  XZ↔pixel converter. `world_to_px`, `px_to_world`. Mirrored X for
  RCP convention.
- `make_grid(...)` — builds a PlanGrid from world bounds.

### Existing per-session outputs

For each session under `<worktree>/sessions/<id>/`:

- `upload/` — raw Polycam folder (`.obj`, `.mtl`, `mesh_info.json`,
  textures).
- `out/ceiling.jpg` — top-down textured render at 150 ppm.
- `out/height.npy` — float32 H×W per-pixel world-Y, NaN outside
  ceiling-band coverage.
- `out/plan.json` — has `room`, `main`, `regions`, optionally
  `topology` and `interfaces`. **Use this as ground truth** —
  `plan["room"]` is the user-traced room outline; if a `topology`
  exists, `topology.edges` are de-facto interface chords (each
  edge with two adjacent faces is an interface; each edge with
  exactly one face is room outline).

### Sessions in scope

- `<worktree>/sessions/0beced53c9df/` — fully developed, has main +
  5 regions + topology + traced interfaces. Best test case.
- `<worktree>/sessions/c9c1d9033c14/` — same project pre-snap.
- The parent project at `~/Documents/Claude Code/Space Room Plan
  GS/sessions/cc367350d007/` has hand-traced ground truth at
  `debug/ground_truth/cc367350d007.json` (used by the legacy region
  harness). If you want a second test scan, copy or symlink that
  session into your worktree's `sessions/` folder.

### Existing debug harness pattern

`<impl>/debug/` is the FH-era harness. Its module shape is worth
copying (data loading, scoring, leaderboard, overlay PNGs) even
though the algorithms inside don't apply. Mirror it under
`debug/auto_trace/`:

```
debug/auto_trace/
├── __init__.py
├── README.md            # how to run, ground-truth shape, scoring
├── data.py              # session loading, mesh + ground-truth loaders
├── scoring.py           # line-segment IoU / Hausdorff / chamfer
├── algos/
│   ├── __init__.py
│   ├── vertical_raster.py     # approach A
│   ├── ceiling_anchored.py    # approach B
│   ├── ceiling_slice.py       # approach C
│   ├── ransac_lines.py        # approach D
│   ├── height_gradient.py     # approach E
│   └── combined.py            # approach F
├── ground_truth/
│   └── <id>.json        # derived from plan.json (committed)
├── results/             # CSVs (gitignored — regenerated)
└── visualisations/      # overlay PNGs (gitignored)
```

Wire the harness so a single command runs every algorithm against
every session, scores against the ground truth, and writes both
the leaderboard and the overlay PNGs. Mirror
`python -m debug.experiments <id>` from the legacy harness.

## Ground-truth derivation

Don't ask the user to retrace. Derive ground truth from the
existing `plan.json`:

- **Room outline ground truth** = `plan["room"]` as a closed polyline
  in world XZ.
- **Interface chord ground truth** = every `topology.edges[i]` whose
  `faces` field has two non-null face IDs (interior shared edges).
  Resolve each edge to a polyline via `topology.vertices[edge.vertices]`.
  Edges where one face is `null` are the room outline (already
  covered by the first bullet).

Cache the derived ground truth in `debug/auto_trace/ground_truth/<id>.json`
so re-runs are fast. Commit those files alongside the algorithms.

## Scoring

Line-segment ground truth doesn't fit the IoU framing the legacy
harness used for region masks. Suggested metrics, in priority order:

1. **Buffered IoU** — rasterise both the predicted and ground-truth
   line sets at a small buffer (say 5 cm) on the existing PlanGrid,
   compute mask IoU. Forgiving of small geometric jitter, easy to
   visualise as a difference image.
2. **Hausdorff distance** — for each predicted segment, max distance
   to nearest ground-truth segment, and vice versa. Catches outliers
   (a single bad detection drags the score). Report median +
   95th-percentile.
3. **Chamfer distance** — average nearest-neighbour distance, both
   directions. Smoother than Hausdorff; good for ranking algorithms
   that are roughly equivalent.
4. **Per-segment recall + precision** at a fixed tolerance (e.g.
   "ground-truth segment is matched if any predicted segment lies
   entirely within 10 cm of it"). Single number, easy to compare.

Pick one as the leaderboard's primary sort. Show the others as
columns. Always print **predicted segment count** alongside scores
— a "high IoU" with 200 lines is suspect.

Score room outline and interface chords **separately**. They have
different geometry (closed polygon vs disjoint segments) and
different cost-of-error (a missing interface chord costs the user
30 seconds; a missing wall is much worse).

## Visualisations

Three layers per overlay PNG, top to bottom:

1. **Base** — the session's `ceiling.jpg` at 50 % brightness so
   detected lines pop. Or `height.npy` rendered as a heatmap if
   you prefer geometric reading.
2. **Ground truth** — green polyline (room) + green dashed segments
   (interface chords).
3. **Predictions** — red solid lines, with line-thickness scaled by
   detection confidence if the algorithm produces one.

Save one PNG per `(session, algorithm)` pair under
`debug/auto_trace/visualisations/<algo>__<session>.png`. The user
will scroll through these; make them readable at A4 print size.

A side-by-side composite per session (`<session>__compare.png`)
showing the top 3 algorithms next to ground truth is gold.

## Deliverables

When you wrap up, write `debug/auto_trace/README.md` summarising:

- The hypothesis space and which approaches you tried.
- A leaderboard table sorted by your primary metric, separately
  for room outline and interface chords.
- One paragraph per algorithm: what it does, what worked, what
  didn't, recommended params.
- Top 3 overlay PNGs inline (relative paths) so the README reads
  end-to-end.
- A short "what to ship" recommendation — which algorithm (or
  combination) the next feature-work session should hook into the
  Trace-interface tool. Concrete enough that someone can implement
  from the recommendation alone.

Commit the README, the algorithm sources, the ground-truth JSONs,
and the harness on branch `experiments/auto-trace`. Don't merge
into the impl branch. The next feature session reads your README,
picks a winner, and writes a clean integration patch (with you
deleting the rest of the experimental code if requested).

Tag the wrap-up commit `experiments/auto-trace-<date>` so the user
can find it later.

## Files most likely useful (in priority order)

- `src/ceiling_rcp/mesh.py` — face / normal / ceiling-mask helpers.
- `src/ceiling_rcp/planes.py` — `PlanGrid` for XZ↔pixel mapping.
- `src/ceiling_rcp/raster.py` — reference top-down rendering pipeline.
- `src/ceiling_rcp/topology.py` — read-only here; useful for
  grounding the interface-chord ground truth from snapped sessions.
- `debug/data.py`, `debug/output.py`, `debug/scoring.py`,
  `debug/experiments.py` — pattern source for your own harness
  (the algorithms inside are the wrong domain — don't reuse).
- `debug/ground_truth/cc367350d007.json` — example legacy ground
  truth shape (region polygons, not line segments — yours will be
  different).

## What success looks like

A README the user can scroll through with images, ranking five to
ten algorithm variants, with a clear top-2 recommendation and a
sense of what's going to work in production. **Don't aim for a
shippable feature** — aim for a recommendation backed by data.

The user's expectation is recall-heavy detection of walls + chords
from one or two scans, with overlay images they can squint at and
say "yeah, that one". Get there fast, iterate on what works.

---
