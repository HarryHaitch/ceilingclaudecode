"""``ceiling-rcp-process <scan_folder>`` — process a Polycam scan folder
end-to-end into the user-visible Scans layout.

Expected input layout::

    <scan_folder>/
    └── Polycam Outputs/
        ├── *.zip               (the OBJ export AND/OR the raw scan zip
        │                        from Polycam — order doesn't matter,
        │                        both are unpacked into the same dir)
        └── (optional loose .obj/.mtl/textures if you've already unpacked)

Produces::

    <scan_folder>/
    ├── Polycam Outputs/
    │   ├── *.zip               (kept as-is)
    │   └── _extracted/         (unzipped contents — mesh + keyframes)
    └── Processed Outputs/
        ├── plan.json           (empty plan — fill in by hand-tracing
        │                        regions in the web app)
        ├── ceiling.jpg         (top-down ortho)
        ├── height.npy          (per-pixel ceiling Y, NaN where uncovered)
        ├── (symbols.json — written later by the SAM 3 pipeline)

Then point a browser at::

    http://127.0.0.1:8765/?session=<scan folder name>

with the server running in scans-dir mode (``--scans-dir <demo/Scans>``
or ``CEILING_RCP_SCANS_DIR=<demo/Scans>``).

The SAM 3 segmentation step is stubbed for now (matches the Rev.1
commit): ``--sam3 local`` and ``--sam3 runpod`` print the manual
script invocation rather than running the pipeline. ``--sam3 skip``
(default) leaves the symbols out entirely. The bundled example
``0beced53c9df_symbols.json`` is the only fully-detected scan today.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable


# Set the env var BEFORE importing the server module — the path globals
# at the top of server.py read CEILING_RCP_SCANS_DIR at import time. The
# scan folder's parent IS the scans dir.

def _resolve_scans_dir(scan_folder: Path) -> Path:
    return scan_folder.parent.resolve()


def _bootstrap_env(scan_folder: Path) -> None:
    """Tell server.py to use the scans-dir layout for this run."""
    scans_dir = _resolve_scans_dir(scan_folder)
    os.environ["CEILING_RCP_SCANS_DIR"] = str(scans_dir)


def _is_safe_member(member_name: str) -> bool:
    """Reject zip members that try to escape via ``..`` or absolute paths."""
    p = Path(member_name)
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    return True


def _extract_zip(zip_path: Path, dest: Path) -> int:
    """Unpack a zip into ``dest``, skipping unsafe members. Returns the
    number of files actually written."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            if not _is_safe_member(member.filename):
                continue
            target = dest / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def _stage_polycam_outputs(scan_folder: Path) -> Path:
    """Unzip every ``*.zip`` in ``Polycam Outputs/`` into a co-located
    ``_extracted/`` dir, *then* return that dir. Idempotent — if the
    extracted dir already exists with content, it's reused as-is."""
    polycam = scan_folder / "Polycam Outputs"
    if not polycam.exists():
        raise SystemExit(
            f"missing 'Polycam Outputs/' in {scan_folder}.\n"
            "Drop the Polycam .zip exports there first."
        )
    extracted = polycam / "_extracted"
    if extracted.exists() and any(extracted.iterdir()):
        print(f"[stage] reusing existing {extracted}")
        return extracted

    extracted.mkdir(parents=True, exist_ok=True)
    zips = sorted(p for p in polycam.iterdir()
                  if p.is_file() and p.suffix.lower() == ".zip")
    if not zips:
        raise SystemExit(
            f"no .zip files in {polycam}. Drop the Polycam exports "
            "(OBJ + raw) there and re-run."
        )
    print(f"[stage] unpacking {len(zips)} zip(s) → {extracted}")
    for z in zips:
        n = _extract_zip(z, extracted)
        print(f"  • {z.name}: {n} files")
    return extracted


def _filter_for_processing(extracted_dir: Path,
                           processing_upload_dir: Path) -> int:
    """Mirror what ``init_session.py`` does: copy only the files the mesh
    processor + SAM 3 actually consume, into a flat ``upload/``-shaped
    layout the processor expects.

    Skips the same heavyweight bits the web upload filters out (depth
    maps, raw images, point cloud, mp4) so the on-disk Polycam Outputs/
    can be left intact for archival but the processor doesn't choke on
    them.
    """
    KEEP_EXT = {".obj", ".mtl", ".jpg", ".jpeg", ".png", ".json"}
    SKIP_DIRS = {
        "keyframes", "depth", "confidence", "cameras",
        "corrected_cameras", "images", "corrected_images",
    }
    SKIP_NAMES = {
        ".DS_Store", "thumbnail.jpg", "polycam.mp4",
        "ceiling_geometry.json", "ceiling_geometry_preview.png",
        "roomplan.json",
    }
    n = 0
    for src in extracted_dir.rglob("*"):
        if not src.is_file():
            continue
        rel_parts = src.relative_to(extracted_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if src.name in SKIP_NAMES:
            continue
        if src.suffix.lower() not in KEEP_EXT:
            continue
        target = processing_upload_dir / Path(*rel_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        n += 1
    return n


def _ensure_processing_upload(scan_folder: Path, extracted: Path) -> None:
    """``inspect_folder`` walks Polycam Outputs/ recursively and finds the
    OBJ + MTL + textures inside ``_extracted/`` already, so we don't
    need to promote anything to the top level. (Earlier versions did,
    but that produced duplicate-OBJ warnings.) This function is kept as
    a hook so future formats — e.g. Polycam exports without a top-level
    ``mesh_info.json`` — have a single place to add staging logic."""
    return


def _auto_min_ceiling_height(scan_folder: Path) -> float | None:
    """Pick a sensible ceiling-height threshold by reading the mesh's
    own bounding box out of ``mesh_info.json``.

    Strategy: keep just the **top ~0.5 m** of the Y range. That's wide
    enough to catch bulkhead steps and service zones above a dropped
    ceiling, but narrow enough that the resulting ``height.npy`` only
    encodes real ceiling surfaces — critical for SAM 3 projection
    accuracy, where every ortho pixel's stored Y is the surface a
    keyframe ray gets snapped onto. A wider band leaks furniture /
    wall lips into the height field and bends each ray sideways.

    Earlier versions used a 1.5 m band; that worked on Lachy's 4 m
    scan (band fell entirely above 2 m, no walls captured) but broke
    Scan example 2 (3.5 m room — band reached the floor and dragged
    every projection sideways).

    Returns None if mesh_info isn't readable (caller falls back to
    the server's 2.0 m default).
    """
    import json
    candidates = [
        scan_folder / "Polycam Outputs" / "mesh_info.json",
        scan_folder / "Polycam Outputs" / "_extracted" / "mesh_info.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            cy = float(data.get("bboxCenter", [0, 0, 0])[1])
            sy = float(data.get("bboxSize", [0, 0, 0])[1])
            y_max = cy + sy / 2.0
            # Keep the top 0.5 m. Clamp the floor at y_max - 0.5 so
            # short rooms don't end up with a negative threshold.
            return float(y_max - 0.5)
        except Exception:
            continue
    return None


def _run_mesh_processing(session_id: str, *,
                         ppm: int,
                         min_ceiling_height_m: float | None = None) -> dict:
    """Invoke the existing process_session pipeline. Returns the
    plan dict for caller-side reporting."""
    from .server import process_session, DEFAULT_MIN_CEILING_HEIGHT_M
    h = min_ceiling_height_m if min_ceiling_height_m is not None \
        else DEFAULT_MIN_CEILING_HEIGHT_M
    print(f"[mesh] min_ceiling_height_m = {h:.2f}")
    return process_session(session_id, ppm=ppm, min_ceiling_height_m=h)


def _run_sam3_step(
    backend: str,
    *,
    extracted_dir: Path,
    processed_outputs: Path,
    session_id: str,
    n_limit: int,
    skip_inference: bool,
    cloud_type: str,
    no_terminate: bool,
) -> None:
    """Dispatch to the RunPod runner (default) or the local CPU/MPS/CUDA
    runner. Both write the same on-disk artefact layout under
    ``Processed Outputs/sam3/`` and the same canonical ``symbols.json``
    next to plan.json."""
    if backend == "skip":
        return
    if backend == "runpod":
        from .sam3_runpod_runner import run_sam3_runpod
        print()
        print("[sam3] running RunPod pipeline (filter → SAM 3 on H100 → project → symbols)")
        result = run_sam3_runpod(
            extracted_dir=extracted_dir,
            processed_outputs=processed_outputs,
            session_id=session_id,
            n_limit=n_limit,
            skip_inference=skip_inference,
            cloud_type=cloud_type,
            no_terminate=no_terminate,
        )
    elif backend == "local":
        from .sam3_local_runner import run_sam3_local
        print()
        print("[sam3] running LOCAL pipeline (filter → SAM 3 on this machine → project → symbols)")
        print("       Tip: --sam3 runpod is much faster on H100; only use 'local' if you can't reach RunPod.")
        result = run_sam3_local(
            extracted_dir=extracted_dir,
            processed_outputs=processed_outputs,
            session_id=session_id,
            n_limit=n_limit,
            skip_inference=skip_inference,
        )
    else:
        print(f"[sam3] unknown backend '{backend}', skipping")
        return

    sym = result.get("symbols", {})
    print()
    print(f"[sam3] done. {sym.get('n_total', 0)} symbols → "
          f"{processed_outputs / 'symbols.json'}")
    by_class = sym.get("by_class") or {}
    for k in sorted(by_class):
        print(f"        • {k}: {by_class[k]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ceiling-rcp-process",
        description=__doc__.split("\n\n", 1)[0] if __doc__ else None,
    )
    p.add_argument(
        "scan_folder", type=Path,
        help="Path to a scan folder containing 'Polycam Outputs/'."
    )
    p.add_argument(
        "--ppm", type=int, default=150,
        help="Top-down ortho resolution in pixels per metre (default 150).",
    )
    p.add_argument(
        "--sam3", choices=["skip", "runpod", "local"], default="runpod",
        help=("SAM 3 ceiling-services pipeline backend. "
              "'runpod' (default) spins up an H100-class GPU pod via "
              "RunPod, segments every ceiling-facing keyframe, and tears "
              "the pod down on exit. 'local' runs SAM 3 on this machine "
              "(MPS/CUDA/CPU) — slow on Apple Silicon. 'skip' leaves "
              "ceiling-services symbols out of this run."),
    )
    p.add_argument(
        "--cloud", default="COMMUNITY",
        choices=["COMMUNITY", "SECURE", "ALL"],
        help=("RunPod cloud tier. COMMUNITY is the cheapest H100; "
              "SECURE is more reliable but pricier. (default: COMMUNITY)"),
    )
    p.add_argument(
        "--no-terminate", action="store_true",
        help=("After the SAM 3 stage finishes, *stop* the RunPod pod "
              "(preserves the disk so a re-run avoids the SAM3-weights "
              "download) instead of terminating it. Defaults to "
              "terminate so disk cost goes to zero."),
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Port the printed open URL points at (default 8765).",
    )
    p.add_argument(
        "--reprocess", action="store_true",
        help=("Force re-extraction of the Polycam zips even if "
              "Polycam Outputs/_extracted/ already exists."),
    )
    p.add_argument(
        "--min-ceiling-height", type=float, default=None,
        help=("World-Y threshold (metres) below which mesh triangles are "
              "ignored when rendering the ortho. If omitted, auto-pick "
              "from mesh_info.json (top 1.5 m of the bbox), falling back "
              "to the 2.0 m server default."),
    )
    p.add_argument(
        "--sam3-n", type=int, default=0,
        help=("Cap the number of keyframes SAM 3 processes (smoke-test "
              "knob). 0 = process every ceiling-facing frame."),
    )
    p.add_argument(
        "--sam3-skip-inference", action="store_true",
        help=("Skip the SAM 3 inference stage and re-run only the "
              "projection + clustering + symbols stages against existing "
              "per-frame results. Lets you iterate on the rules without "
              "re-running the GPU step."),
    )
    args = p.parse_args(argv)

    scan_folder = args.scan_folder.expanduser().resolve()
    if not scan_folder.is_dir():
        print(f"Not a directory: {scan_folder}", file=sys.stderr)
        return 2

    session_id = scan_folder.name

    # Configure the server module BEFORE importing it.
    _bootstrap_env(scan_folder)

    if args.reprocess:
        extracted = scan_folder / "Polycam Outputs" / "_extracted"
        if extracted.exists():
            print(f"[reprocess] removing {extracted}")
            shutil.rmtree(extracted)

    print(f"[scan] {scan_folder}")
    extracted = _stage_polycam_outputs(scan_folder)
    _ensure_processing_upload(scan_folder, extracted)

    print("[mesh] running ceiling-mesh processor (this can take a minute)…")
    h = args.min_ceiling_height
    if h is None:
        h = _auto_min_ceiling_height(scan_folder)
    plan = _run_mesh_processing(session_id, ppm=args.ppm,
                                min_ceiling_height_m=h)

    rep = plan.get("report", {})
    print(f"[mesh] ok: {rep.get('ok')}")
    for w in rep.get("warnings") or []:
        print(f"  warn: {w}")
    for e in rep.get("errors") or []:
        print(f"  err:  {e}")
    if not rep.get("ok"):
        print("[mesh] processing failed — see warnings/errors above.",
              file=sys.stderr)
        return 3

    _run_sam3_step(
        args.sam3,
        extracted_dir=extracted,
        processed_outputs=scan_folder / "Processed Outputs",
        session_id=session_id,
        n_limit=args.sam3_n,
        skip_inference=args.sam3_skip_inference,
        cloud_type=args.cloud,
        no_terminate=args.no_terminate,
    )

    encoded = session_id.replace(" ", "%20")
    print()
    print("─" * 60)
    print(f" Done. Open in browser:")
    print(f"   http://127.0.0.1:{args.port}/?session={encoded}")
    print()
    print(f" Make sure the server was launched with the matching scans dir:")
    print(f"   ceiling-rcp-server --scans-dir '{scan_folder.parent}'")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
