# Next-session prompt

Paste the block below into a fresh Claude session when you pick this
project back up. It mirrors the briefing format used at the start of
the Good WIP 3 session and points at every artefact that matters.

---

This is the ceiling-rcp project at
`/Users/harishusic/Documents/Claude Code/Space Room Plan GS`.
GitHub: https://github.com/HarryHaitch/ceilingclaudecode

A read-only snapshot of the current "Good WIP 3 05052026" milestone
is at `/Users/harishusic/Documents/Claude Code/Good WIP 3 05052026`.
Do NOT touch that folder — it's `chmod a-w` and is the rollback
point. Tag `good-wip-3-05052026` marks the same commit in git.
Earlier milestones (`Good WIP 2 04052026`, `Good WIP 1 040526`) are
also read-only — leave them alone.

Active branch / starting point for new work:

  claude/admiring-fermi-4b18e7-impl  (HEAD = tag good-wip-3-05052026 + post-WIP polish)
  Worktree: <repo>/.claude/worktrees/admiring-fermi-4b18e7

The branch carries four clusters of work on top of `good-wip-2-04052026`:

  Cluster A (bc554d6)  Editor UX polish + PDF title-block tweaks
  Cluster B (83ba8c0)  Render PDF plan at a true architectural scale
  Cluster C (7b0b601)  Per-region height histograms + draggable height pick
  Cluster D (0e25a57)  Interface-tracing pipeline (replaces per-region tracing)

Read these in order before suggesting anything:

1. `STATUS.md` — "What shipped this session (Good WIP 3 05052026)"
   covers the current build (true-scale PDF at 1:20/50/100/200 metric
   or 1:24/48/96/192 imperial; SCALE in title block; per-region
   height histograms with draggable selected_y markers; interface-
   tracing pipeline that replaces per-region tracing; main-face
   radio toggle; right-side Regions panel; clean north-arrow drag;
   horizontal ceiling labels; aspect-correct ortho thumbnail).
2. `README.md`
3. `ARCHITECTURE.md` — has the v3 plan JSON shape; cluster C added
   `selected_y` + `histogram` (v4), cluster D added `interfaces` +
   `main_face_id` (v5). Schema migrations in `_migrate_plan` are
   additive.
4. `debug/README.md` — only relevant if touching segmentation.

Install gotcha (read this BEFORE running anything):

There used to be a stale editable install named "rpgs" that put the
parent path's src/ on sys.path ahead of the worktree's. The fix in
this repo is to install from the worktree:

```
cd <worktree>
python3 -m pip install -e .
```

Verify with:

```
python3 -c "import ceiling_rcp; print(ceiling_rcp.__file__)"
```

The path must end in
`.claude/worktrees/admiring-fermi-4b18e7/src/ceiling_rcp/__init__.py`.
Hard-refresh the browser (Cmd+Shift+R) so cached old `app.js` /
`style.css` don't haunt you. Current versions: `app.js?v=18`,
`style.css?v=17` — bump both whenever you ship frontend changes.

Run the app:

```
ceiling-rcp-server --port 8765                    # terminal 1
ceiling-rcp-init "Scan data/<your-folder>"        # terminal 2
```

Files most likely to need editing for follow-ups (with their roles):

- `src/ceiling_rcp/server.py` — endpoints (interfaces,
  define_ceilings, main_face, main/region selected_y), PDF builder,
  snap pipeline, schema migrations, `_choose_standard_scale`,
  `_height_histogram`, `_recompute_relatives`.
- `src/ceiling_rcp/topology.py` — planar graph, multi-ring faces,
  edits.
- `src/ceiling_rcp/mesh.py` — `ceiling_face_mask` (60° cone + Y
  band). Don't tighten this; the next session adds a separate
  outlier-trim filter on top.
- `src/ceiling_rcp/raster.py` — top-down render + z-buffer.
- `src/ceiling_rcp/units.py` — metric / imperial formatters
  (mirror in app.js).
- `src/ceiling_rcp/polylabel.py` — pole-of-inaccessibility,
  hole-aware.
- `src/ceiling_rcp/static/index.html`
- `src/ceiling_rcp/static/app.js` — canvas editor, no framework,
  no build step. Trace-interface state machine, histogram
  sparkline drag, Main radio.
- `src/ceiling_rcp/static/style.css`
- `STATUS.md` / `README.md` / `ARCHITECTURE.md` — keep up to date.

Plan-shape additions vs v3:

- `selected_y` — float metres absolute, on main + each region +
  each topology face. Defaults to `mean_y`. The PDF / sidebar
  height delta is `face.selected_y − main.selected_y`.
- `histogram` — `{bin_edges_m, counts, min_y, max_y, bin_w_m}`
  on each face, computed from `height.npy[mask]`.
- `interfaces` — list of `{id, polyline:[[x,z],…], closed:bool}`.
- `main_face_id` — int, default 0; which topology face is the
  datum.

Existing test sessions in the worktree's `sessions/`:

- `0beced53c9df` — fully-developed Lachlan's Line scan, room +
  main + 5 regions + topology, units = metric. Good for snap /
  PDF / cluster-C drag testing.
- `c9c1d9033c14` — same project pre-snap.

Don't merge to main without my say-so, and don't move the existing
tags. Each cluster of work below should land as one commit on
`claude/admiring-fermi-4b18e7-impl` (or a fresh feature branch off
it if the change is large).

---

## Punch list for this session

Five proposed clusters, ordered by my best read of priority. Confirm
the order with me before starting. Bugs in **Cluster E** are
top-priority; everything else is feature work and can be reshuffled.

### Cluster E — bug fixes

1. **Main-ceiling swap doesn't stick.** Clicking the Main radio on
   another row in the Regions panel reverts back to the original
   main after the API round-trip. Investigate
   `api_swap_main_face` (server.py) + `pushMainFace` (app.js) —
   likely the swap-then-resnap is undoing the change, or the radio
   `name="main-face"` group is being reset on plan reload.
2. **PDF legend colours are too vibrant** compared to the plan
   fills they label. The plan uses 0.45 alpha on the tints
   (`fill_alpha = 0.45` in `api_pdf`); the legend swatches don't.
   Use the same blended/muted colour in `_draw_legends` so the
   eye reads "this swatch = this fill".

### Cluster F — interface-tracing finish-up (cluster D follow-ups)

3. **Live snap-to-existing while tracing.** Today the trace tool
   has no cursor snap, so chord endpoints rarely land exactly on
   the room outline / another interface and `define_ceilings`
   silently leaves them as dangling chords. Snap the cursor to
   the nearest room-outline / interface vertex / midpoint within
   ~0.15 m of paper distance, with **Shift to override** (mirrors
   the existing `constrainShiftSnap` axis-lock pattern).
4. **Server-side end-extension in `define_ceilings`.** Belt-and-
   braces for #3: extend each chord's two endpoints to the
   nearest line in the union (Shapely `nearest_points` + a
   trim-to-tolerance step) so endpoints that miss by 1–5 mm
   still cut cleanly.
5. **Define ceilings missed a bunch and left chords dangling.**
   Same root cause as #3 / #4 — closing this out validates the
   fix.
6. **Interface vertex drag post-trace.** Today you can only delete
   + re-trace. Add the same drag affordance the topology vertex
   drag has: pick a vertex, drag, push back to
   `PUT /interface/{iid}` with the new polyline, and re-call
   `define_ceilings` if a topology already exists.
7. **Interface chords should disappear after Define.** Once a
   chord is consumed into a face boundary it's noise on the
   canvas. Hide rendered interfaces after a successful Define
   (or fade them); keep the rows in the Regions panel for delete
   so the user can still remove a chord they didn't want.

### Cluster G — histogram + region-row UX

The cluster-C sparkline shipped functional but undercooked — a real
project will live in this widget. The user's punch list:

8. **Notes input belongs above the histogram**, not below it.
9. **Histogram 2× height.** Wider sparkline reads better once the
   empty-tail axis space (see #11) is gone.
10. **Slider readout in relative mm**, not absolute m. Main is
    always `0`, regions are `+/− mm` against main's `selected_y`.
    Same convention as the row meta and PDF labels.
11. **Histogram axis goes from `−max(spread)` to `+max(spread)`
    with `0` marked.** Replaces the absolute-m axis (which gives
    huge empty tails on a 1.5 m range when only 5 cm of it has
    data). Marks: peak, 0, ±extents.
12. **Show peak % frequency as a number** (the tallest bar's
    fraction of pixels — gives the user a "how flat is this
    ceiling?" gut-check).
13. **Show what % of the ceiling falls above and below the
    current slider position** as live readouts on either side
    of the slider.
14. **Crop preview.** When the slider is moved, render any pixels
    in the polygon whose Y falls below the slider in a bright
    pink + black checker on the canvas, so the user can see what
    they'd be excluding from the "ceiling height" they pick. Read
    `height.npy[mask]` per face and overlay an on-the-fly bitmap
    above the existing heatmap layer.
15. **Per-row label format.** Replace the current
    `+12 mm  spread 23 mm` blob with two lines, vertically aligned:

        Height:  +12 mm
        Spread:   23 mm

    (Two-space indent on `Height:` so the numbers line up under
    `Spread:`.)

### Cluster H — scan settings rename + outlier trim

16. **Rename "Max ceiling height variance" → "Minimum ceiling
    height".** Same plumbing, friendlier framing. Default 2.0 m;
    hint copy something like "Increase if it's clipping
    furniture." Imperial unit support — current input is m-only.
17. **New 2 %-trim outlier filter on per-face stats + histogram.**
    Don't tighten `mesh.ceiling_face_mask`; instead trim the top
    2 % and bottom 2 % of valid pixel heights when computing
    `mean_y` / `std_y` / `min_y` / `max_y` and the histogram bins.
    Keep `valid_frac` / `n_valid_px` / `n_total_px` from the full
    sample (those are coverage metrics, not statistical ones).
    Drop into `_analyse_and_pack` as a single `_trim_outliers`
    helper. This will narrow the histogram axis automatically and
    helps with #11.

### Cluster I — page size + scale selectors

18. **Page-size selector** in the Project info panel. Default
    A1; options grouped:
    - Metric: A4, A3, A2, A1, A0
    - Imperial (US): Letter, Tabloid, Arch B, Arch C, Arch D,
      Arch E (or whichever set covers A4-equivalent through
      A0-equivalent cleanly)
    When the units toggle is set to imperial, default the size to
    the closest US equivalent of A1 (Arch D, 24″ × 36″). Stored
    on `plan.project.page_size` (or similar).
19. **Scale selector** alongside. Default = the current
    `_choose_standard_scale` auto-pick, but allow manual override.
    Options grouped metric / imperial. If the user picks a scale
    too small for the chosen page, warn but allow.
20. **Sheet size in the PDF's scale box.** Add an extra line in
    the title-block SCALE section showing `A1 (841 × 594 mm)` or
    `Arch D (24" × 36")` so the printer / drafter knows the
    intended sheet.

The page-size change cascades into `api_pdf`'s gridspec figure size
— A1 is hardcoded today (`A1_W_IN, A1_H_IN = 33.11, 23.39`). Pull
those from a small `PAGE_SIZES` table keyed on the selected size.

---

## Backlog (parked — not on the priority list this round)

These were on the prior STATUS.md "Next session priorities" but
the user hasn't asked for them this round. Park them here so they
don't get lost:

- Light segmentation + symbol placement (the empty Services
  legend cell is still waiting).
- Edge-drag tool for shared topology boundaries.
- Numbered column references (C1 / C2 / …) in the legend.
- Cleaner Cmd+Z undo (current edits push to the server
  immediately).

Pull anything from here only if the active list is done or the
user asks.
