# Experiment: auto-trace room outline + interface chords

The full prompt + outputs live in a Finder-visible folder so the
user can browse the deliverables without diving into the repo:

  `~/Documents/Claude Code/Experiments/Auto-trace room outline and chords/`

```
Experiments/
└── Auto-trace room outline and chords/
    ├── prompt.md                  # paste between --- markers into a fresh session
    └── outputs/
        ├── README.md              # writeup (initially a stub)
        ├── leaderboard_room.csv
        ├── leaderboard_chords.csv
        └── *.png                  # overlay visualisations
```

Read `~/Documents/Claude Code/Experiments/Auto-trace room outline
and chords/prompt.md` for the full session brief. The block between
the `---` markers in that file is what gets pasted into the new
session.

## Quick ground rules (copied from the prompt for in-repo discoverability)

- This is a **research session**, not feature work. Land a
  recommendation backed by data, not a shipped feature.
- Spin up a fresh worktree off `good-wip-3-05052026`:

```
cd "/Users/harishusic/Documents/Claude Code/Space Room Plan GS"
git worktree add .claude/worktrees/auto-trace -b experiments/auto-trace good-wip-3-05052026
```

- **Algorithm code** → in that new worktree under
  `debug/auto_trace/` (committed on `experiments/auto-trace`).
- **Human-visible writeup + overlay PNGs** → in the visible
  Experiments folder above. Don't put final images inside the
  worktree.
- Don't touch the impl branch (`claude/admiring-fermi-4b18e7-impl`).
