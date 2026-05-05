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

  claude/admiring-fermi-4b18e7-impl  (HEAD = tag good-wip-3-05052026)
  Worktree: <repo>/.claude/worktrees/admiring-fermi-4b18e7

The branch carries four commits on top of `good-wip-2-04052026`:

  0e25a57  Cluster D: interface-tracing pipeline
  7b0b601  Cluster C: per-region height histograms + draggable height pick
  83ba8c0  Cluster B: render PDF plan at a true architectural scale
  bc554d6  Cluster A: editor UX polish + PDF title-block tweaks
  9e4b352  Good WIP 2 (tag good-wip-2-04052026)

Read these in order before suggesting anything:

1. `STATUS.md` — "What shipped this session (Good WIP 3 05052026)"
   covers everything in the current build (true-scale PDF at
   1:20/50/100/200 metric or 1:24/48/96/192 imperial; SCALE in title
   block; per-region height histograms with draggable selected_y
   markers; interface-tracing pipeline that replaces per-region
   tracing; main-face radio toggle; right-side Regions panel; clean
   north-arrow drag; horizontal ceiling labels; aspect-correct ortho
   thumbnail).
2. `README.md`
3. `ARCHITECTURE.md` — has the v3 plan JSON shape; cluster C added
   `selected_y` + `histogram` (v4), cluster D added `interfaces` +
   `main_face_id` (v5). Schema migrations in `_migrate_plan` are
   additive.
4. `debug/README.md` — only relevant if touching segmentation.

Install gotcha (read this BEFORE running anything):

  There used to be a stale editable install named "rpgs" that put
  the parent path's src/ on sys.path ahead of the worktree's. The
  fix in this repo is to install from the worktree:

      cd <worktree>
      python3 -m pip install -e .

  Verify with:

      python3 -c "import ceiling_rcp; print(ceiling_rcp.__file__)"

  The path must end in
  `.claude/worktrees/admiring-fermi-4b18e7/src/ceiling_rcp/__init__.py`
  — if it points anywhere else, the CLI is bound to old code and
  nothing you change will run. Hard-refresh the browser
  (Cmd+Shift+R) so the cached old `app.js` / `style.css` don't
  haunt you. Current versions: `app.js?v=18`, `style.css?v=17`.

Run the app:

  ceiling-rcp-server --port 8765                    # terminal 1
  ceiling-rcp-init "Scan data/<your-folder>"        # terminal 2

Files most likely to need editing for follow-ups (with their roles):

  src/ceiling_rcp/server.py        — endpoints (interfaces,
                                     define_ceilings, main_face,
                                     main/region selected_y), PDF
                                     builder, snap pipeline, schema
                                     migrations
  src/ceiling_rcp/topology.py      — planar graph, multi-ring faces,
                                     edits
  src/ceiling_rcp/mesh.py          — ceiling_face_mask (60° cone +
                                     Y band) — known data-quality
                                     gap; tighten for cleaner
                                     histograms
  src/ceiling_rcp/raster.py        — top-down render + z-buffer
  src/ceiling_rcp/units.py         — metric/imperial formatters
                                     (mirror in app.js)
  src/ceiling_rcp/polylabel.py     — pole-of-inaccessibility,
                                     hole-aware
  src/ceiling_rcp/static/index.html
  src/ceiling_rcp/static/app.js    — canvas editor, no framework,
                                     no build step. Trace-interface
                                     state machine, histogram
                                     sparkline drag, Main radio
  src/ceiling_rcp/static/style.css
  STATUS.md / README.md / ARCHITECTURE.md — keep up to date

Key plan-shape additions (vs v3):

  selected_y         — float metres absolute, on main + each region
                       + each topology face. Defaults to mean_y. The
                       PDF / sidebar height delta is
                       (face.selected_y - main.selected_y).
  histogram          — {bin_edges_m, counts, min_y, max_y, bin_w_m}
                       on each face, computed from height.npy[mask].
  interfaces         — list of {id, polyline:[[x,z],…], closed:bool}.
  main_face_id       — int, default 0; which topology face is the
                       datum.

Open carry-overs from STATUS.md "Next session priorities":

  1. Live snap-to-existing while tracing interfaces (cursor snaps
     to room-outline / existing-interface vertices and midpoints
     within a tolerance, Shift overrides). Plus server-side
     end-extension in define_ceilings so chords that miss the
     room outline by a few mm still cut cleanly.
  2. Interface vertex drag post-trace (same shape as the existing
     topology vertex drag — pick, drag, PUT /interface/{iid}, then
     re-define_ceilings if a topology exists).
  3. Light segmentation + symbol placement (the empty Services
     legend cell is still waiting for it).
  4. Edge-drag tool for shared topology boundaries.
  5. Tighten mesh.ceiling_face_mask so histograms don't include
     non-ceiling pixels (region 0 of session 0beced53c9df spans
     0.74–2.05 m which is too wide).
  6. Numbered column references (C1 / C2 / …) in the legend if a
     real project asks for it.
  7. Cleaner Cmd+Z undo.

Existing test sessions in the worktree's `sessions/`:

  0beced53c9df — fully-developed Lachlan's Line scan, room + main
                 + 5 regions + topology, units = metric. Good for
                 snap / PDF / cluster-C drag testing.
  c9c1d9033c14 — same project pre-snap.

Don't merge to main without my say-so, and don't move the existing
tags. Each cluster of work should land as one commit on
`claude/admiring-fermi-4b18e7-impl` (or a fresh feature branch off
it if the change is large).
