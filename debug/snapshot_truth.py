"""``python -m debug.snapshot_truth <session_id>`` — capture the polygons
you've drawn in the web UI as ground truth for the experiment harness.

Reads ``sessions/<id>/out/plan.json`` and writes
``debug/ground_truth/<id>.json`` containing just the polygon shapes
(no heatmaps, no stats — just the geometry the experiments score
against).

Workflow:

1. Start ``ceiling-rcp-server``, open the session URL.
2. Trace the room, run auto-detect, refine each region's polygon
   with the insert/delete vertex tools and Snap until you're happy.
3. Run ``python -m debug.snapshot_truth <session_id>``.
4. Now the experiment harness has something to score against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


GROUND_TRUTH_DIR = Path("debug") / "ground_truth"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="snapshot_truth")
    ap.add_argument("session_id")
    ap.add_argument("--name", default=None,
                    help="optional friendly name to store alongside the truth")
    ap.add_argument("--out_dir", type=Path, default=GROUND_TRUTH_DIR)
    args = ap.parse_args(argv)

    plan_path = Path("sessions") / args.session_id / "out" / "plan.json"
    if not plan_path.exists():
        print(f"no plan.json under {plan_path}", file=sys.stderr)
        return 2
    plan = json.loads(plan_path.read_text())

    if not plan.get("room"):
        print("no room outline drawn yet — set one in the web UI first.",
              file=sys.stderr)
        return 3
    if not plan.get("main"):
        print("no main ceiling drawn yet.", file=sys.stderr)
        return 3

    truth = {
        "session_id": args.session_id,
        "name": args.name or f"session_{args.session_id}",
        "grid": plan["grid"],
        "room": plan["room"],
        "main": {
            "polygon": plan["main"]["polygon"],
            "label": plan["main"].get("label", "Main Ceiling (1)"),
            "notes": plan["main"].get("notes", ""),
        },
        "regions": [
            {
                "id": r["id"],
                "polygon": r["polygon"],
                "label": r.get("label", f"region {r['id']}"),
                "relative_y": r.get("relative_y"),
                "notes": r.get("notes", ""),
            }
            for r in plan.get("regions", [])
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.session_id}.json"
    out_path.write_text(json.dumps(truth, indent=2))

    n_regions = len(truth["regions"])
    print(f"→ {out_path}")
    print(f"   room: {len(truth['room'])} verts")
    print(f"   main: {len(truth['main']['polygon'])} verts")
    print(f"   regions: {n_regions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
