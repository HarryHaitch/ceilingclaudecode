"""FastAPI server for the manual-draw RCP workflow.

Sessions live under ``./sessions/<id>/`` containing:

  upload/         the user's raw drop (zip extracted or files copied)
  out/
    ceiling.jpg   textured top-down render of all downward-facing geometry
    height.npy    float32 H×W per-pixel world-Y, NaN where no LiDAR coverage
    plan.json     authoritative plan object, mutated by edit endpoints

The plan structure:

    {
      "session_id": str,
      "report": {…validator output…},
      "grid": {min_x, max_x, min_z, max_z, pixels_per_metre, width, height},
      "room": [[x, z], …]     | null,    # user-drawn outer boundary
      "main": {                          # user-drawn main ceiling, datum
          "polygon": [[x, z], …],
          "mean_y": float, "std_y": float, "valid_frac": float, …
      } | null,
      "regions": [{                      # user-drawn extra planes
          "id": int, "label": str,
          "polygon": [[x, z], …],
          "mean_y": float, "relative_y": float (mean_y - main.mean_y), …
      }, …]
    }
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .analyse import (
    analyse_polygon,
    deviation_heatmap_png,
    height_map_to_storage,
)
from .mesh import inspect_folder, load_mesh, ceiling_face_mask, downward_face_mask
from .planes import PlanGrid, make_grid
from .polylabel import polylabel
from .raster import render_textured_topdown
from .units import (
    DEFAULT_UNITS,
    UNIT_SYSTEMS,
    format_height_delta,
    format_length,
)


# ─── PATHS ────────────────────────────────────────────────────────────────────

SESSIONS_DIR = Path.cwd() / "sessions"
STATIC_DIR = Path(__file__).parent / "static"


def _session_dir(session_id: str) -> Path:
    p = SESSIONS_DIR / session_id
    if not p.exists():
        raise HTTPException(404, f"unknown session {session_id}")
    return p


# Schema version for plan.json. Bump when the on-disk shape changes; add a
# migration step in _migrate_plan. Files written by older versions are
# upgraded transparently on first read.
PLAN_SCHEMA_VERSION = 5


def _migrate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an on-disk plan to PLAN_SCHEMA_VERSION in place."""
    v = int(plan.get("schema_version", 1))
    if v < 2:
        # v1 → v2: introduces plan["topology"] (built fresh by api_snap). v1
        # files that were already snapped lose their per-polygon shared-edge
        # property until the user re-snaps; the polygons themselves are
        # preserved so editing still works.
        plan.setdefault("topology", None)
    if v < 3:
        # v2 → v3: introduces plan["units"], plan["scan_settings"],
        # plan["project"]. All optional with sensible defaults.
        plan.setdefault("units", "metric")
        plan.setdefault("scan_settings", {"min_ceiling_height_m": 2.0})
    # H16 rename: scan_settings.max_ceiling_variance_m → min_ceiling_height_m.
    # Different semantic (absolute floor instead of band-below-top), so we
    # don't try to convert the old value — replace with the new default.
    ss = plan.setdefault("scan_settings", {})
    if "max_ceiling_variance_m" in ss and "min_ceiling_height_m" not in ss:
        ss["min_ceiling_height_m"] = 2.0
        ss.pop("max_ceiling_variance_m", None)
    ss.setdefault("min_ceiling_height_m", 2.0)
    if v < 4:
        # v3 → v4: introduces face-level ``selected_y`` and ``histogram``.
        # Both back-fill lazily from existing stats — selected_y defaults
        # to stats.mean_y and the histogram is regenerated next time
        # _analyse_and_pack runs for that face. The frontend tolerates
        # missing histograms (renders a "no data" sparkline) so an
        # unedited migrated plan is still usable.
        pass
    if v < 5:
        # v4 → v5: introduces plan["interfaces"] (user-traced ceiling-zone
        # boundary polylines / rings) and plan["main_face_id"] (which
        # face is the height datum, default 0). Both have safe defaults
        # so old plans keep working.
        plan.setdefault("interfaces", [])
        plan.setdefault("main_face_id", 0)
    plan["schema_version"] = PLAN_SCHEMA_VERSION
    # Field-level defaults that don't warrant a schema bump.
    plan.setdefault("obstructions", [])
    plan.setdefault("interfaces", [])
    plan.setdefault("main_face_id", 0)
    proj = plan.get("project") or {}
    defaults = {
        "name": "", "address": "", "client": "", "company": "",
        "drawing_number": "", "north_deg": 0.0, "print_north": True,
        "drawing_register": [],
    }
    for k, v_default in defaults.items():
        proj.setdefault(k, v_default)
    plan["project"] = proj
    # selected_y back-fills from stats.mean_y; relative_y derives from
    # selected_y everywhere. Safe to call on every load — it only
    # writes selected_y if missing, and re-derives relative_y always.
    _recompute_relatives(plan)
    return plan


def _backfill_selected_y(face: dict | None) -> None:
    """If ``face.selected_y`` is missing, copy it from ``face.stats.mean_y``.
    NaN means we have no LiDAR for that face — leave selected_y absent
    so callers know the height is unknown."""
    if not face:
        return
    if face.get("selected_y") is not None:
        return
    mean_y = (face.get("stats") or {}).get("mean_y")
    if mean_y is None:
        return
    try:
        my = float(mean_y)
    except (TypeError, ValueError):
        return
    import math as _math
    if _math.isnan(my):
        return
    face["selected_y"] = my


def _recompute_relatives(plan: dict) -> None:
    """Walk main + regions + topology faces, back-fill ``selected_y`` from
    ``stats.mean_y`` where missing, and re-derive ``relative_y`` as
    ``face.selected_y - main.selected_y``.

    The main ceiling's ``relative_y`` is always 0 (it *is* the datum).
    Faces with no selected_y (no LiDAR coverage) carry ``relative_y =
    None`` so the UI can show "—" rather than a misleading "0 mm"."""
    main = plan.get("main")
    _backfill_selected_y(main)
    if main:
        main["relative_y"] = 0.0

    main_sy = (main or {}).get("selected_y")

    def _rel_for(face: dict) -> float | None:
        sy = face.get("selected_y")
        if sy is None or main_sy is None:
            return None
        return float(sy) - float(main_sy)

    for r in (plan.get("regions") or []):
        _backfill_selected_y(r)
        r["relative_y"] = _rel_for(r)
    topo = plan.get("topology")
    if topo:
        for f in (topo.get("faces") or []):
            _backfill_selected_y(f)
            f["relative_y"] = _rel_for(f)


def _load_plan(session_id: str) -> dict[str, Any]:
    sd = _session_dir(session_id)
    plan_path = sd / "out" / "plan.json"
    if not plan_path.exists():
        raise HTTPException(409, "session not processed yet")
    return _migrate_plan(json.loads(plan_path.read_text()))


def _save_plan(session_id: str, plan: dict[str, Any]) -> None:
    plan["schema_version"] = PLAN_SCHEMA_VERSION
    sd = _session_dir(session_id)
    (sd / "out" / "plan.json").write_text(json.dumps(plan, indent=2))


def _load_height_map(session_id: str) -> tuple[np.ndarray, PlanGrid]:
    sd = _session_dir(session_id)
    h = np.load(sd / "out" / "height.npy")
    plan = _load_plan(session_id)
    g = plan["grid"]
    grid = PlanGrid(
        min_x=g["min_x"], max_x=g["max_x"],
        min_z=g["min_z"], max_z=g["max_z"],
        pixels_per_metre=g["pixels_per_metre"],
    )
    return h, grid


# ─── UPLOAD HANDLING ──────────────────────────────────────────────────────────

def _extract_upload(
    files: list[UploadFile], dest: Path, paths: list[str] | None = None,
) -> None:
    """Write uploaded files (and unzip any zip) into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    use_sidecar = paths is not None and len(paths) == len(files)
    for i, f in enumerate(files):
        rel = paths[i] if use_sidecar else (f.filename or "")
        rel = rel.lstrip("/").replace("\\", "/")
        if not rel or ".." in rel.split("/"):
            rel = f"unnamed_{uuid.uuid4().hex[:6]}"
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        if target.suffix.lower() == ".zip":
            with zipfile.ZipFile(target) as zf:
                zf.extractall(target.parent)
            target.unlink()


# ─── PROCESSING ───────────────────────────────────────────────────────────────

DEFAULT_MIN_CEILING_HEIGHT_M = 2.0


# ─── SCALE LADDERS ────────────────────────────────────────────────────────────
# Standard architectural scales — ordered largest → smallest. The PDF picks
# the largest (most detail) that still fits the room within the plan area.
SCALE_LADDER_METRIC = (20, 50, 100, 200)
SCALE_LADDER_IMPERIAL = (24, 48, 96, 192)   # 1/2", 1/4", 1/8", 1/16" = 1'

# Tick-mark distances for the graphic scale bar (one entry per ratio).
# Marks express world distance (metres / feet); the bar's paper length is
# computed at PDF time as marks[-1] / scale.
SCALE_BAR_MARKS_METRIC = {
    20:  [0, 1, 2],
    50:  [0, 1, 2, 5],
    100: [0, 1, 5, 10],
    200: [0, 5, 10, 20],
}
SCALE_BAR_MARKS_IMPERIAL = {
    24:  [0, 1, 2, 5],
    48:  [0, 1, 5, 10],
    96:  [0, 5, 10, 20],
    192: [0, 10, 20, 40],
}

# Paper margin around the room outline at the chosen scale. 10 mm is
# tight enough to let close-fit rooms keep their preferred scale (e.g.
# 21 m in the plan area at 1:50) while still leaving room for the
# heavier room outline stroke and edge dimension labels.
PLAN_PAPER_PAD_M = 0.010


def _choose_standard_scale(
    room_w_m: float, room_h_m: float,
    plan_w_in: float, plan_h_in: float,
    units: str,
) -> int:
    """Return the largest standard scale ratio that fits ``room_w_m × room_h_m``
    inside the plan area (``plan_w_in × plan_h_in``) with ``PLAN_PAPER_PAD_M``
    of paper margin per side. Falls back to the coarsest ratio if nothing
    fits — the room would overspill but the user gets *some* output."""
    pad_in = PLAN_PAPER_PAD_M / 0.0254
    avail_w_m = max(0.0, plan_w_in - 2 * pad_in) * 0.0254
    avail_h_m = max(0.0, plan_h_in - 2 * pad_in) * 0.0254
    ladder = (SCALE_LADDER_IMPERIAL
              if units == "imperial" else SCALE_LADDER_METRIC)
    for s in ladder:
        if room_w_m <= avail_w_m * s and room_h_m <= avail_h_m * s:
            return s
    return ladder[-1]


def _default_project() -> dict[str, Any]:
    """Empty project metadata. The user fills these in via the project
    panel; the PDF reads them straight off."""
    return {
        "name": "",
        "address": "",
        "client": "",
        "company": "",
        "drawing_number": "",
        "north_deg": 0.0,           # rotation in degrees, CCW from +Z (page up)
        "print_north": True,        # if False, the arrow is omitted
        "drawing_register": [],     # list of {rev, date, by, note}
    }


def _do_render(
    session_id: str,
    *,
    ppm: int,
    min_ceiling_height_m: float,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None] | None:
    """Render ceiling.jpg + height.npy for a session.

    Returns ``(folder_report, grid_dict, height_summary)`` on success, or
    ``None`` if the upload folder is broken (caller writes the report).
    Doesn't touch plan.json — caller is responsible for persisting.
    """
    sd = _session_dir(session_id)
    upload = sd / "upload"
    out = sd / "out"
    out.mkdir(parents=True, exist_ok=True)

    rep = inspect_folder(upload)
    if not rep.ok:
        return None

    mesh = load_mesh(rep)
    keep = ceiling_face_mask(
        mesh,
        max_tilt_deg=60.0,
        min_ceiling_height_m=float(min_ceiling_height_m),
    )
    keep_idx = np.where(keep)[0]

    grid = make_grid(mesh, keep_idx, pixels_per_metre=ppm)
    canvas, zbuf = render_textured_topdown(mesh, keep_idx, grid)
    cv2.imwrite(str(out / "ceiling.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    height_map = height_map_to_storage(zbuf)
    np.save(out / "height.npy", height_map)

    grid_dict = {
        "min_x": grid.min_x, "max_x": grid.max_x,
        "min_z": grid.min_z, "max_z": grid.max_z,
        "pixels_per_metre": grid.pixels_per_metre,
        "width": grid.width, "height": grid.height,
    }
    height_summary: dict[str, Any] | None = None
    valid = ~np.isnan(height_map)
    if valid.any():
        height_summary = {
            "min_y": float(np.nanmin(height_map)),
            "max_y": float(np.nanmax(height_map)),
            "median_y": float(np.nanmedian(height_map)),
            "valid_px": int(valid.sum()),
            "total_px": int(height_map.size),
        }
    return rep, grid_dict, height_summary


def _report_dict(rep: Any) -> dict[str, Any]:
    return {
        "ok": rep.ok,
        "obj": str(rep.obj) if rep.obj else None,
        "mtl": str(rep.mtl) if rep.mtl else None,
        "mesh_info": str(rep.mesh_info) if rep.mesh_info else None,
        "textures_found": len(rep.textures_found),
        "textures_missing": rep.textures_missing,
        "warnings": rep.warnings,
        "errors": rep.errors,
    }


def process_session(
    session_id: str,
    *,
    ppm: int = 150,
    min_ceiling_height_m: float = DEFAULT_MIN_CEILING_HEIGHT_M,
) -> dict[str, Any]:
    """Initial render — fresh plan, fresh ceiling.jpg + height.npy.

    No automatic segmentation — the user draws polygons by hand.
    """
    sd = _session_dir(session_id)
    out = sd / "out"
    out.mkdir(parents=True, exist_ok=True)
    upload = sd / "upload"
    rep = inspect_folder(upload)
    plan: dict[str, Any] = {
        "session_id": session_id,
        "schema_version": PLAN_SCHEMA_VERSION,
        "report": _report_dict(rep),
        "room": None,
        "main": None,
        "regions": [],
        "obstructions": [],
        "interfaces": [],
        "main_face_id": 0,
        "topology": None,
        "units": DEFAULT_UNITS,
        "scan_settings": {
            "min_ceiling_height_m": float(min_ceiling_height_m),
        },
        "project": _default_project(),
    }

    if not rep.ok:
        (out / "plan.json").write_text(json.dumps(plan, indent=2))
        return plan

    result = _do_render(
        session_id, ppm=ppm, min_ceiling_height_m=min_ceiling_height_m,
    )
    if result is None:
        (out / "plan.json").write_text(json.dumps(plan, indent=2))
        return plan
    _, grid_dict, height_summary = result
    plan["grid"] = grid_dict
    if height_summary is not None:
        plan["height_summary"] = height_summary
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    return plan


# ─── ANALYSIS HELPERS ─────────────────────────────────────────────────────────

def _analyse_and_pack(
    session_id: str, polygon: list[list[float]] | None = None,
    *, mask: np.ndarray | None = None,
    stats_mask: np.ndarray | None = None,
    holes: list[list[list[float]]] | None = None,
    range_m: float = 0.05, tint: str = "#80cbc4",
) -> dict:
    """Compute mean Y + tinted deviation heatmap.

    Either ``polygon`` or ``mask`` defines the *visual* coverage. If a
    separate ``stats_mask`` is supplied, the mean / σ / range stats are
    computed from it instead — useful for auto-detect where we want
    stats to ignore wrongly-absorbed cross-cluster pixels even though
    they remain in the visual polygon.

    ``holes``, if given, is a list of inner rings (typically column
    polygons) whose pixels are subtracted from the mask. This keeps the
    heatmap and stats from including column-wall LiDAR noise inside a
    face that contains an obstruction.
    """
    import base64
    from .analyse import polygon_to_mask
    height_map, grid = _load_height_map(session_id)

    if mask is None:
        if polygon is None:
            raise ValueError("polygon or mask required")
        mask = polygon_to_mask([tuple(p) for p in polygon], grid) > 0
    else:
        mask = mask > 0
    if holes:
        for hole in holes:
            if not hole or len(hole) < 3:
                continue
            hmask = polygon_to_mask([tuple(p) for p in hole], grid) > 0
            mask = mask & ~hmask
    if stats_mask is None:
        stats_mask = mask
    else:
        stats_mask = stats_mask > 0

    n_total = int(mask.sum())
    n_stats = int(stats_mask.sum())
    sample = height_map[stats_mask] if n_stats > 0 else np.array([], dtype=np.float32)
    valid = ~np.isnan(sample) if sample.size else np.array([], dtype=bool)
    if n_total == 0 or not valid.any():
        from .analyse import PolygonAnalysis
        stats = PolygonAnalysis(
            mean_y=float("nan"), std_y=float("nan"),
            valid_frac=0.0, n_valid_px=0, n_total_px=n_total,
            min_y=float("nan"), max_y=float("nan"),
        )
        png_bytes = b""
        bbox = (0, 0, 0, 0)
    else:
        vals = sample[valid].astype(np.float64)
        # 2 % outlier trim — drop the lowest 2 % and the highest 2 % of
        # pixel heights before computing mean / std / min / max so a
        # handful of stray pixels at the extremes don't drag the stats
        # (and the histogram) into a wide range. Coverage metrics
        # (valid_frac / n_valid_px / n_total_px) stay based on the full
        # untrimmed sample — they're honesty about how much of the
        # polygon had LiDAR data.
        trimmed = _trim_outliers(vals, frac=0.02)
        from .analyse import PolygonAnalysis
        stats = PolygonAnalysis(
            mean_y=float(trimmed.mean()) if trimmed.size else float(vals.mean()),
            std_y=float(trimmed.std()) if trimmed.size else float(vals.std()),
            valid_frac=float(valid.sum()) / float(n_total),
            n_valid_px=int(valid.sum()),
            n_total_px=n_total,
            min_y=float(trimmed.min()) if trimmed.size else float(vals.min()),
            max_y=float(trimmed.max()) if trimmed.size else float(vals.max()),
        )
        png_bytes, bbox = _heatmap_from_mask(
            mask, height_map, mean_y=stats.mean_y,
            range_m=range_m, tint=tint,
        )

    histogram = _height_histogram(height_map, stats_mask)

    return {
        "stats": stats.as_dict(),
        "heatmap_bbox_px": list(bbox),
        "heatmap_png_b64": base64.b64encode(png_bytes).decode("ascii") if png_bytes else "",
        "heatmap_range_m": range_m,
        "tint": tint,
        "histogram": histogram,
    }


# Histogram bin width — ~5 mm matches the cluster-C UX (markers
# move in millimetre-feel increments). The cone-band-filtered height
# map is already free of floor / furniture noise, so the histogram
# represents the ceiling-area-vs-height distribution directly.
HEIGHT_HISTOGRAM_BIN_M = 0.005

# Fraction trimmed from each tail before computing per-face stats and
# building the histogram. A handful of stray LiDAR pixels at either
# extreme (e.g. a glint that registered as a single high pixel, a
# shadow that registered as low) would otherwise stretch min_y/max_y
# and bury the bulk of the bins in narrow space.
HEIGHT_TRIM_FRAC = 0.02


def _trim_outliers(values: np.ndarray, *, frac: float = HEIGHT_TRIM_FRAC) -> np.ndarray:
    """Drop the lowest ``frac`` and highest ``frac`` of ``values``.
    Returns a new (sorted-ish) array; tiny inputs pass through
    untouched so a face with only 50 valid pixels doesn't get gutted."""
    if values.size < 50 or frac <= 0:
        return values
    n = values.size
    k = int(round(n * frac))
    if k <= 0:
        return values
    sorted_vals = np.sort(values)
    return sorted_vals[k : n - k]


def _height_histogram(
    height_map: np.ndarray, mask: np.ndarray,
    *, bin_w_m: float = HEIGHT_HISTOGRAM_BIN_M,
) -> dict | None:
    """Return ``{bin_edges_m, counts, min_y, max_y, bin_w_m}`` for the
    valid (non-NaN) ceiling heights inside ``mask``. ``None`` if the
    mask is empty or every pixel is NaN.

    Bins are 5 mm by default and span exactly the polygon's local range
    so each region's sparkline fills its width regardless of overall
    room spread."""
    if not mask.any():
        return None
    sample = height_map[mask]
    valid = sample[~np.isnan(sample)]
    if valid.size == 0:
        return None
    # Trim 2% tails so the histogram axis tracks the bulk of the data
    # rather than a few stray pixels at the extremes.
    trimmed = _trim_outliers(valid.astype(np.float64))
    if trimmed.size == 0:
        return None
    lo = float(trimmed.min())
    hi = float(trimmed.max())
    if hi - lo < bin_w_m:
        # Perfectly flat polygon — single bin, centre on the value.
        edges = [lo - bin_w_m / 2, lo + bin_w_m / 2]
        counts = [int(trimmed.size)]
    else:
        n_bins = max(2, int(np.ceil((hi - lo) / bin_w_m)))
        edges_arr = np.linspace(lo, hi, n_bins + 1)
        # np.histogram drops anything outside [lo, hi], which is what we
        # want — the trim already removed the tails, so binning the
        # untrimmed array against the trimmed range gives a clean
        # in-range count without smearing the tails into edge bins.
        counts_arr, _ = np.histogram(valid, bins=edges_arr)
        edges = [float(e) for e in edges_arr]
        counts = [int(c) for c in counts_arr]
    return {
        "bin_edges_m": edges,
        "counts": counts,
        "min_y": lo,
        "max_y": hi,
        "bin_w_m": bin_w_m,
    }


def _heatmap_from_mask(
    mask: np.ndarray, height_map: np.ndarray, *,
    mean_y: float, range_m: float, tint: str, alpha: float = 0.55,
) -> tuple[bytes, tuple[int, int, int, int]]:
    from .analyse import _hex_to_rgb
    if not mask.any():
        return b"", (0, 0, 0, 0)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub_mask = mask[y0:y1, x0:x1]
    sub_height = height_map[y0:y1, x0:x1]
    valid = sub_mask & ~np.isnan(sub_height)

    dev = (sub_height - mean_y) / max(range_m, 1e-6)
    dev = np.clip(np.where(valid, dev, 0.0), -1.0, 1.0)

    base_r, base_g, base_b = _hex_to_rgb(tint)
    lightness = np.clip(0.70 + 0.30 * dev, 0.35, 1.0)
    boost = dev.clip(0, 1)
    r = (base_r * lightness) + (255 - base_r * lightness) * (boost * 0.5)
    g = (base_g * lightness) + (255 - base_g * lightness) * (boost * 0.5)
    b = (base_b * lightness) + (255 - base_b * lightness) * (boost * 0.5)

    h, w = dev.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = (valid * int(round(alpha * 255))).astype(np.uint8)
    rgba_bgra = rgba[..., [2, 1, 0, 3]]
    ok, buf = cv2.imencode(".png", rgba_bgra)
    if not ok:
        return b"", (0, 0, 0, 0)
    return buf.tobytes(), (x0, y0, x1, y1)


def _below_overlay_from_mask(
    mask: np.ndarray, height_map: np.ndarray, *, threshold_y: float,
) -> tuple[bytes, tuple[int, int, int, int]]:
    """Return an RGBA PNG sized to the mask's bbox where pixels in the
    mask AND with Y < ``threshold_y`` render as a 4-px pink/black
    checker, everything else transparent. The histogram slider hits
    this on every drag tick so the user sees what would be excluded if
    they pick the marker's current position as the face's height."""
    if not mask.any():
        return b"", (0, 0, 0, 0)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub_mask = mask[y0:y1, x0:x1]
    sub_height = height_map[y0:y1, x0:x1]
    valid = sub_mask & ~np.isnan(sub_height)
    below = valid & (sub_height < threshold_y)
    if not below.any():
        return b"", (x0, y0, x1, y1)

    h, w = below.shape
    yy, xx = np.indices((h, w))
    checker = ((yy // 4) + (xx // 4)) % 2 == 0  # 4-px squares
    pink = (236, 71, 167)   # #ec47a7 — saturated, hard to confuse with any tint
    black = (10, 10, 10)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    pick = below & checker
    fall = below & ~checker
    rgba[..., 0] = np.where(pick, pink[0], np.where(fall, black[0], 0))
    rgba[..., 1] = np.where(pick, pink[1], np.where(fall, black[1], 0))
    rgba[..., 2] = np.where(pick, pink[2], np.where(fall, black[2], 0))
    rgba[..., 3] = (below * 220).astype(np.uint8)
    rgba_bgra = rgba[..., [2, 1, 0, 3]]
    ok, buf = cv2.imencode(".png", rgba_bgra)
    if not ok:
        return b"", (x0, y0, x1, y1)
    return buf.tobytes(), (x0, y0, x1, y1)


# Stable palette used both for the colour stripe in the polygon list and
# as the heatmap tint passed back into _analyse_and_pack.
MAIN_TINT = "#80cbc4"
REGION_PALETTE = [
    "#ff7043",  # orange
    "#9c27b0",  # purple
    "#ffeb3b",  # yellow
    "#03a9f4",  # cyan
    "#e91e63",  # pink
    "#cddc39",  # lime
    "#3f51b5",  # indigo
    "#ff9800",  # amber
    "#009688",  # teal
    "#f44336",  # red
]


def _region_tint(region_id: int) -> str:
    return REGION_PALETTE[region_id % len(REGION_PALETTE)]


# Alpha applied to every face fill in the PDF plan area. The legend
# swatches blend to white at the same alpha so the swatch reads as
# the same colour the eye sees on the plan.
PLAN_FILL_ALPHA = 0.45


def _muted_fill(hex_color: str, alpha: float = PLAN_FILL_ALPHA) -> tuple[float, float, float]:
    """Pre-blend ``hex_color`` with a white background at ``alpha`` so the
    resulting RGB triple is what an alpha=alpha fill would look like over
    white. Used for legend swatches so they match the muted plan fills
    without relying on alpha rendering through PDF backends."""
    h = (hex_color or "#ffffff").lstrip("#")
    if len(h) != 6:
        return (1.0, 1.0, 1.0)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (
        r * alpha + (1 - alpha),
        g * alpha + (1 - alpha),
        b * alpha + (1 - alpha),
    )


# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ceiling-rcp", version="0.2.0")


@app.post("/api/sessions")
async def api_create_session(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
) -> dict:
    sid = uuid.uuid4().hex[:12]
    sd = SESSIONS_DIR / sid
    sd.mkdir(parents=True, exist_ok=True)
    _extract_upload(files, sd / "upload", paths=paths or None)
    rep = inspect_folder(sd / "upload")
    return {
        "session_id": sid,
        "report": {
            "ok": rep.ok,
            "obj": rep.obj.name if rep.obj else None,
            "mtl": rep.mtl.name if rep.mtl else None,
            "mesh_info": rep.mesh_info.name if rep.mesh_info else None,
            "textures_found": len(rep.textures_found),
            "textures_missing": rep.textures_missing,
            "warnings": rep.warnings,
            "errors": rep.errors,
        },
    }


@app.post("/api/sessions/{session_id}/process")
async def api_process(session_id: str, ppm: int = Form(150)) -> dict:
    return process_session(session_id, ppm=ppm)


@app.get("/api/sessions/{session_id}/project")
async def api_get_project(session_id: str) -> dict:
    plan = _load_plan(session_id)
    return plan.get("project") or _default_project()


@app.put("/api/sessions/{session_id}/project")
async def api_set_project(session_id: str, payload: dict = Body(...)) -> dict:
    """Update project metadata (title-block fields + drawing register +
    north arrow). All fields are optional; missing keys keep their old
    values."""
    plan = _load_plan(session_id)
    proj = plan.get("project") or _default_project()
    for key in ("name", "address", "client", "company", "drawing_number"):
        if key in payload:
            proj[key] = str(payload[key])[:200]
    if "north_deg" in payload:
        try:
            proj["north_deg"] = float(payload["north_deg"]) % 360.0
        except (TypeError, ValueError):
            raise HTTPException(400, "north_deg must be numeric")
    if "print_north" in payload:
        proj["print_north"] = bool(payload["print_north"])
    if "drawing_register" in payload:
        register = payload["drawing_register"] or []
        if not isinstance(register, list):
            raise HTTPException(400, "drawing_register must be a list")
        proj["drawing_register"] = [
            {
                "rev": str((row or {}).get("rev", ""))[:8],
                "date": str((row or {}).get("date", ""))[:32],
                "by": str((row or {}).get("by", ""))[:32],
                "note": str((row or {}).get("note", ""))[:200],
            }
            for row in register
        ]
    plan["project"] = proj
    _save_plan(session_id, plan)
    return {"ok": True, "project": proj}


@app.put("/api/sessions/{session_id}/units")
async def api_set_units(session_id: str, payload: dict = Body(...)) -> dict:
    """Pick metric or imperial display. World coords stay in metres
    everywhere — only display strings (PDF labels, side panel) change."""
    plan = _load_plan(session_id)
    value = payload.get("value")
    if value not in UNIT_SYSTEMS:
        raise HTTPException(
            400, f"value must be one of {UNIT_SYSTEMS!r}; got {value!r}",
        )
    plan["units"] = value
    _save_plan(session_id, plan)
    return {"ok": True, "units": value}


@app.put("/api/sessions/{session_id}/scan_settings")
async def api_set_scan_settings(session_id: str, payload: dict = Body(...)) -> dict:
    """Update scan-settings (currently just ``min_ceiling_height_m``) and
    re-render the ortho image + height map.

    User-drawn polygons (room, main, regions, obstructions) are preserved —
    their stats and heatmaps are refreshed against the new height map.
    Any topology is dropped (snapped state is invalidated by re-render);
    the user re-snaps when ready.
    """
    plan = _load_plan(session_id)
    # Accept the old key for one cycle of clients-loading-cached-app.js,
    # but the new key wins if both are present.
    raw = payload.get("min_ceiling_height_m")
    if raw is None:
        raw = payload.get("max_ceiling_variance_m")
    if raw is None:
        raise HTTPException(400, "min_ceiling_height_m required")
    try:
        new_h = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "min_ceiling_height_m must be numeric")
    if not (0.5 <= new_h <= 6.0):
        raise HTTPException(400, "min_ceiling_height_m must be between 0.5 and 6.0 metres")

    ppm = int((plan.get("grid") or {}).get("pixels_per_metre") or 150)
    result = _do_render(
        session_id, ppm=ppm, min_ceiling_height_m=new_h,
    )
    if result is None:
        raise HTTPException(400, "scan upload is no longer valid; reprocess from CLI")
    rep, grid_dict, height_summary = result
    plan["report"] = _report_dict(rep)
    plan["grid"] = grid_dict
    if height_summary is not None:
        plan["height_summary"] = height_summary
    ss = plan.setdefault("scan_settings", {})
    ss["min_ceiling_height_m"] = new_h
    ss.pop("max_ceiling_variance_m", None)

    # Refresh per-polygon stats + heatmaps against the new height map.
    if plan.get("room"):
        plan["room_heatmap"] = _analyse_and_pack(
            session_id, plan["room"], range_m=0.15, tint="#ffffff",
        )
    if plan.get("main"):
        m = plan["main"]
        analysis = _analyse_and_pack(
            session_id, m["polygon"],
            range_m=float(m.get("heatmap_range_m", 0.05)),
            tint=m.get("tint", MAIN_TINT),
        )
        m.update(analysis)
        m.setdefault("relative_y", 0.0)
    datum = (plan.get("main") or {}).get("stats", {}).get("mean_y")
    for r in plan.get("regions", []):
        analysis = _analyse_and_pack(
            session_id, r["polygon"],
            range_m=float(r.get("heatmap_range_m", 0.05)),
            tint=r.get("tint", _region_tint(r.get("id", 0))),
        )
        r.update(analysis)
        m_y = r.get("stats", {}).get("mean_y")
        r["relative_y"] = (
            float(m_y - datum) if (m_y is not None and datum is not None) else None
        )

    # Topology bbox / heatmap pixel coords are tied to the old grid; drop
    # the topology so the user re-snaps against the new render.
    if plan.get("topology") is not None:
        plan["topology"] = None
    plan["snapped"] = False

    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return plan


@app.get("/api/sessions/{session_id}/plan")
async def api_get_plan(session_id: str) -> dict:
    return _load_plan(session_id)


@app.get("/api/sessions/{session_id}/image/ceiling.jpg")
async def api_ceiling_image(session_id: str) -> FileResponse:
    sd = _session_dir(session_id)
    p = sd / "out" / "ceiling.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ─── ROOM POLYGON ─────────────────────────────────────────────────────────────

@app.put("/api/sessions/{session_id}/room")
async def api_set_room(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    poly = payload.get("polygon")
    if poly is None:
        plan["room"] = None
        plan["room_heatmap"] = None
    else:
        if len(poly) < 3:
            raise HTTPException(400, "room polygon must have at least 3 vertices")
        plan["room"] = [list(p) for p in poly]
        # Build a white-tinted variance preview for the whole room so the
        # user can see where heights differ before drawing main / regions.
        # A wider range_m (15 cm) keeps the preview readable across mixed
        # ceilings — small bulkheads still pop without clipping recesses.
        plan["room_heatmap"] = _analyse_and_pack(
            session_id, poly, range_m=0.15, tint="#ffffff",
        )
    _save_plan(session_id, plan)
    return {"ok": True, "room": plan["room"], "room_heatmap": plan.get("room_heatmap")}


# ─── OBSTRUCTIONS (columns / structural negative space) ──────────────────────
#
# Obstructions are user-drawn polygons inside the room outline that the
# ceiling can't span — typically structural columns. They live alongside
# `regions` but never become ceiling faces; the snap pipeline subtracts
# their area from the room mask so adjacent regions stop cleanly at their
# boundary, and the canvas + PDF render them as a hatched overlay.
#
# Schema: ``plan["obstructions"] = [{id, polygon, label, kind: "column"}]``.

OBSTRUCTION_KINDS = {"column"}


@app.post("/api/sessions/{session_id}/obstruction")
async def api_add_obstruction(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    poly = payload.get("polygon", [])
    if len(poly) < 3:
        raise HTTPException(400, "obstruction polygon must have at least 3 vertices")
    kind = str(payload.get("kind", "column"))
    if kind not in OBSTRUCTION_KINDS:
        raise HTTPException(400, f"unknown obstruction kind {kind!r}")
    obs_id = (max((o["id"] for o in plan.get("obstructions", [])), default=-1)) + 1
    label = str(payload.get("label") or f"Column ({obs_id + 1})")
    obstruction = {
        "id": obs_id,
        "kind": kind,
        "label": label,
        "polygon": [list(p) for p in poly],
    }
    plan.setdefault("obstructions", []).append(obstruction)
    had_topology = plan.get("topology") is not None
    _save_plan(session_id, plan)
    if had_topology:
        # Topology was built without this obstruction in the room mask; rebuild
        # so the new column is honoured.
        await api_snap(session_id)
        return {"ok": True, "obstruction": obstruction, "snapped": True,
                "plan": _load_plan(session_id)}
    return {"ok": True, "obstruction": obstruction}


@app.put("/api/sessions/{session_id}/obstruction/{obstruction_id}")
async def api_update_obstruction(
    session_id: str, obstruction_id: int, payload: dict = Body(...),
) -> dict:
    plan = _load_plan(session_id)
    obs = next(
        (o for o in plan.get("obstructions", []) if o["id"] == obstruction_id),
        None,
    )
    if obs is None:
        raise HTTPException(404, f"unknown obstruction {obstruction_id}")
    if "polygon" in payload:
        if len(payload["polygon"]) < 3:
            raise HTTPException(400, "polygon must have at least 3 vertices")
        obs["polygon"] = [list(p) for p in payload["polygon"]]
    if "label" in payload:
        obs["label"] = str(payload["label"])[:120]
    had_topology = plan.get("topology") is not None
    _save_plan(session_id, plan)
    if had_topology and "polygon" in payload:
        await api_snap(session_id)
        return {"ok": True, "obstruction": obs, "snapped": True,
                "plan": _load_plan(session_id)}
    return {"ok": True, "obstruction": obs}


@app.delete("/api/sessions/{session_id}/obstruction/{obstruction_id}")
async def api_delete_obstruction(session_id: str, obstruction_id: int) -> dict:
    plan = _load_plan(session_id)
    before = len(plan.get("obstructions", []))
    plan["obstructions"] = [
        o for o in plan.get("obstructions", []) if o["id"] != obstruction_id
    ]
    if len(plan["obstructions"]) == before:
        raise HTTPException(404, f"unknown obstruction {obstruction_id}")
    had_topology = plan.get("topology") is not None
    _save_plan(session_id, plan)
    if had_topology:
        await api_snap(session_id)
        return {"ok": True, "snapped": True, "plan": _load_plan(session_id)}
    return {"ok": True}


# ─── INTERFACES (ceiling-zone boundaries) ─────────────────────────────────────
#
# An interface is a user-traced polyline that bounds two adjacent ceilings.
# Open polylines are *chords* — both endpoints land on the room outline or
# another interface, and they subdivide whichever face contains them.
# Closed polylines are *island rings* — they sit inside another face and
# create a new face wholly contained within. ``define_ceilings`` polygonises
# room outline + interfaces (subtracting columns) and rebuilds main +
# regions + topology from the result.
#
# Schema: ``plan["interfaces"] = [{id, polyline: [[x, z], ...], closed: bool}]``.


@app.post("/api/sessions/{session_id}/interface")
async def api_add_interface(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    poly = payload.get("polyline") or []
    if len(poly) < 2:
        raise HTTPException(400, "interface polyline must have at least 2 vertices")
    closed = bool(payload.get("closed", False))
    if closed and len(poly) < 3:
        raise HTTPException(400, "closed interface (ring) must have at least 3 vertices")
    iid = (max((i["id"] for i in plan.get("interfaces", [])), default=-1)) + 1
    iface = {
        "id": iid,
        "polyline": [[float(p[0]), float(p[1])] for p in poly],
        "closed": closed,
    }
    plan.setdefault("interfaces", []).append(iface)
    _save_plan(session_id, plan)
    return {"ok": True, "interface": iface}


@app.put("/api/sessions/{session_id}/interface/{interface_id}")
async def api_update_interface(
    session_id: str, interface_id: int, payload: dict = Body(...),
) -> dict:
    """Update an interface's polyline / closed flag.

    If a topology has already been derived (i.e. ``define_ceilings`` was
    run), moving a vertex invalidates the polygonisation. Auto re-run
    ``define_ceilings`` so the user's edit is reflected immediately —
    same shape of auto-resnap that ``api_delete_region`` /
    ``api_update_obstruction`` already implement."""
    plan = _load_plan(session_id)
    iface = next(
        (i for i in plan.get("interfaces", []) if i["id"] == interface_id),
        None,
    )
    if iface is None:
        raise HTTPException(404, f"unknown interface {interface_id}")
    if "polyline" in payload:
        poly = payload["polyline"] or []
        if len(poly) < 2:
            raise HTTPException(400, "interface polyline must have at least 2 vertices")
        iface["polyline"] = [[float(p[0]), float(p[1])] for p in poly]
    if "closed" in payload:
        iface["closed"] = bool(payload["closed"])
    if iface.get("closed") and len(iface["polyline"]) < 3:
        raise HTTPException(400, "closed interface needs at least 3 vertices")
    had_topology = plan.get("topology") is not None
    _save_plan(session_id, plan)
    if had_topology:
        # api_define_ceilings returns the full plan dict (via its own
        # internal hand-off to api_snap). Wrap it in {plan: ...} so the
        # frontend can branch on the redefined flag without sniffing the
        # response shape.
        new_plan = await api_define_ceilings(session_id)
        return {"ok": True, "interface": iface, "redefined": True,
                "plan": new_plan}
    return {"ok": True, "interface": iface}


@app.delete("/api/sessions/{session_id}/interface/{interface_id}")
async def api_delete_interface(session_id: str, interface_id: int) -> dict:
    plan = _load_plan(session_id)
    before = len(plan.get("interfaces", []))
    plan["interfaces"] = [
        i for i in plan.get("interfaces", []) if i["id"] != interface_id
    ]
    if len(plan["interfaces"]) == before:
        raise HTTPException(404, f"unknown interface {interface_id}")
    _save_plan(session_id, plan)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/define_ceilings")
async def api_define_ceilings(session_id: str) -> dict:
    """Polygonise the room outline + every interface into a tiling of N
    faces, populate ``plan.main`` (largest face by default) +
    ``plan.regions`` (the rest) with auto-tinted polygons, then run the
    snap pipeline so the topology / heatmaps / histograms are built
    against the new polygons.

    The user picks which face is the ceiling datum afterwards via
    ``PUT /main_face`` — the largest face is just a sensible default.

    Belt-and-braces for the cursor-snap during tracing: every open
    chord's two endpoints get extended onto the nearest point on the
    union of all *other* lines (room outline + every other interface),
    if that nearest point is within ``END_EXTENSION_TOL_M``. This
    rescues chords whose endpoints land 1-5 mm shy of an existing line
    — those wouldn't share a common vertex with anything in
    ``unary_union``, so ``polygonize`` would silently leave the chord
    dangling rather than cut a face out of the room."""
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union, polygonize, nearest_points

    plan = _load_plan(session_id)
    room_pts = plan.get("room")
    if not room_pts or len(room_pts) < 3:
        raise HTTPException(400, "room outline required before defining ceilings")

    END_EXTENSION_TOL_M = 0.05  # 5 cm — generous on top of the 14 px live snap

    raw_interfaces = list(plan.get("interfaces", []))
    room_ring = list(room_pts) + [room_pts[0]]
    room_line = LineString(room_ring)

    # Pre-build LineStrings for every interface (or None for those too
    # short to use), so the per-chord "every other line" union is just a
    # filter rather than re-parsing.
    iface_lines: list[LineString | None] = []
    for iface in raw_interfaces:
        pts = iface.get("polyline") or []
        if len(pts) < 2:
            iface_lines.append(None)
            continue
        if iface.get("closed"):
            if len(pts) < 3:
                iface_lines.append(None)
                continue
            iface_lines.append(LineString(list(pts) + [pts[0]]))
        else:
            iface_lines.append(LineString(pts))

    # End-extend each open chord. Closed rings don't have endpoints to
    # extend — they already close back to themselves.
    snapped_endpoints: list[list[list[float]] | None] = []
    for i, iface in enumerate(raw_interfaces):
        if iface_lines[i] is None or iface.get("closed"):
            snapped_endpoints.append(None)
            continue
        pts = list(iface.get("polyline") or [])
        if len(pts) < 2:
            snapped_endpoints.append(None)
            continue
        others = [room_line] + [
            ln for j, ln in enumerate(iface_lines)
            if j != i and ln is not None
        ]
        union = unary_union(others)
        new_pts = [list(p) for p in pts]
        for k in (0, -1):
            ep = Point(new_pts[k])
            try:
                target, _ = nearest_points(union, ep)
            except Exception:
                continue
            if ep.distance(target) <= END_EXTENSION_TOL_M:
                new_pts[k] = [float(target.x), float(target.y)]
        snapped_endpoints.append(new_pts)

    # Build the final linework using the (possibly extended) chord pts.
    lines: list[LineString] = [room_line]
    for i, iface in enumerate(raw_interfaces):
        if iface_lines[i] is None:
            continue
        if iface.get("closed"):
            lines.append(iface_lines[i])
            continue
        pts = snapped_endpoints[i] or list(iface.get("polyline") or [])
        if len(pts) < 2:
            continue
        lines.append(LineString(pts))

    merged = unary_union(lines)
    polys = list(polygonize(merged))

    # Keep only polygons of meaningful area whose representative point
    # falls inside the room (polygonize on degenerate linework can leak
    # a few sliver polygons).
    room_poly = Polygon(room_pts)
    min_face_area = 0.05  # 0.05 m² = a 22 cm × 22 cm patch
    filtered = [
        p for p in polys
        if p.area >= min_face_area and room_poly.contains(p.representative_point())
    ]
    if not filtered:
        raise HTTPException(
            409,
            "no faces produced — chords must terminate on the room outline "
            "or another interface, and rings must close back to their start",
        )

    filtered.sort(key=lambda p: p.area, reverse=True)

    def _coords(p: Polygon) -> list[list[float]]:
        return [[float(x), float(z)] for x, z in list(p.exterior.coords)[:-1]]

    # Seed the legacy main / regions views with the polygonisation, then
    # run the snap pipeline — it builds the planar topology + per-face
    # height analysis from these polygons. (The Voronoi step is a no-op
    # on already-tiling polygons, but the topology + analysis it
    # performs after is what we want.)
    main_pts = _coords(filtered[0])
    plan["main"] = {
        "polygon": main_pts,
        "label": "Main Ceiling (1)",
        "tint": MAIN_TINT,
        "notes": (plan.get("main") or {}).get("notes", ""),
    }
    plan["regions"] = []
    for i, p in enumerate(filtered[1:]):
        plan["regions"].append({
            "id": i,
            "label": f"Ceiling Region ({i + 2})",
            "polygon": _coords(p),
            "tint": _region_tint(i),
            "notes": "",
        })
    plan["main_face_id"] = 0
    plan["topology"] = None
    plan["snapped"] = False
    _save_plan(session_id, plan)

    # Hand off to the snap pipeline. It re-derives plan.main + plan.regions
    # from the topology, runs height analysis, and saves.
    return await api_snap(session_id)


@app.put("/api/sessions/{session_id}/main_face")
async def api_swap_main_face(session_id: str, payload: dict = Body(...)) -> dict:
    """Pick which face is the ceiling-height datum.

    Body: ``{"key": "main"}`` (no-op) or ``{"key": "region:<id>"}``.

    Post-snap, the topology already tiles the room cleanly and we just
    need to relabel which face plays the "main" role. We do that by
    flipping ``kind`` / ``region_id`` / ``label`` / ``tint`` on the two
    affected faces and updating ``plan.main_face_id`` to point at the new
    main. The vertices, edges and rings — i.e. the actual face shapes —
    don't move.

    Re-Voronoi'ing here was wrong: the new main's polygon is nominally
    inside an existing region's polygon (the previously-main face), so
    stage-1 region claims would steal every pixel of the new main and
    leave face 0 with zero coverage."""
    plan = _load_plan(session_id)
    key = str(payload.get("key", ""))
    if key == "main":
        return {"ok": True, "noop": True, "plan": plan}
    if not key.startswith("region:"):
        raise HTTPException(400, "key must be 'main' or 'region:<id>'")
    try:
        rid = int(key.split(":", 1)[1])
    except (ValueError, IndexError):
        raise HTTPException(400, f"bad region key {key!r}")

    topo = plan.get("topology")
    if topo is not None:
        old_main_id = int(plan.get("main_face_id", 0))
        old_main_face = next(
            (f for f in topo["faces"] if int(f["id"]) == old_main_id),
            None,
        )
        new_main_face = next(
            (f for f in topo["faces"]
             if int(f["id"]) != old_main_id
             and int(f.get("region_id", -1)) == rid),
            None,
        )
        if old_main_face is None:
            raise HTTPException(409, "topology has no main face — re-snap")
        if new_main_face is None:
            raise HTTPException(404, f"unknown region {rid}")
        if int(new_main_face["id"]) == old_main_id:
            return {"ok": True, "noop": True, "plan": plan}

        old_main_label = old_main_face.get("label", "Main Ceiling (1)")
        old_main_face["kind"] = "region"
        old_main_face["region_id"] = rid
        old_main_face["label"] = (
            old_main_label
            if old_main_label != "Main Ceiling (1)"
            else f"Ceiling Region ({rid + 2})"
        )
        old_main_face["tint"] = _region_tint(rid)

        new_main_face["kind"] = "main"
        new_main_face.pop("region_id", None)
        new_main_face["label"] = "Main Ceiling (1)"
        new_main_face["tint"] = MAIN_TINT

        plan["main_face_id"] = int(new_main_face["id"])
        # Re-render heatmaps + recompute stats with the new role tints,
        # and refresh the legacy main / regions views.
        _refresh_topology_polygons(plan, session_id)
        _save_plan(session_id, plan)
        return {"ok": True, "plan": _load_plan(session_id)}

    # Pre-snap: simple polygon swap on the legacy views.
    target = next((r for r in plan.get("regions", []) if r["id"] == rid), None)
    if target is None:
        raise HTTPException(404, f"unknown region {rid}")
    cur_main = plan.get("main")
    if cur_main is None:
        raise HTTPException(409, "no main ceiling set yet")

    target_poly = [list(p) for p in target["polygon"]]
    target_notes = target.get("notes", "")
    old_main_poly = [list(p) for p in cur_main["polygon"]]
    old_main_notes = cur_main.get("notes", "")
    old_main_label = cur_main.get("label", "Main Ceiling (1)")

    plan["main"] = {
        "polygon": target_poly,
        "label": "Main Ceiling (1)",
        "tint": MAIN_TINT,
        "notes": target_notes,
    }
    plan["regions"] = [r for r in plan["regions"] if r["id"] != rid]
    plan["regions"].append({
        "id": rid,
        "label": old_main_label if old_main_label != "Main Ceiling (1)"
                 else f"Ceiling Region ({rid + 2})",
        "polygon": old_main_poly,
        "tint": _region_tint(rid),
        "notes": old_main_notes,
    })
    plan["regions"].sort(key=lambda r: r["id"])
    plan["main_face_id"] = 0

    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return {"ok": True, "plan": _load_plan(session_id)}


# ─── MAIN CEILING POLYGON ─────────────────────────────────────────────────────

@app.put("/api/sessions/{session_id}/main")
async def api_set_main(session_id: str, payload: dict = Body(...)) -> dict:
    """Set the main ceiling polygon. The mean Y of valid pixels inside it
    becomes the room's height datum (relative_y = 0). All other regions'
    relative heights are recomputed against the new datum.

    If the session is snapped, any update auto-clears the topology — the
    user is editing the underlying main polygon, so the derived topology
    is invalidated until they re-snap."""
    plan = _load_plan(session_id)
    poly = payload.get("polygon")
    if plan.get("topology") is not None:
        plan["topology"] = None
        plan["snapped"] = False
    if poly is None:
        plan["main"] = None
    else:
        if len(poly) < 3:
            raise HTTPException(400, "main polygon must have at least 3 vertices")
        analysis = _analyse_and_pack(
            session_id, poly,
            range_m=float(payload.get("range_m", 0.05)),
            tint=MAIN_TINT,
        )
        plan["main"] = {
            "polygon": [list(p) for p in poly],
            "label": "Main Ceiling (1)",
            **analysis,
        }
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return {"ok": True, "main": plan["main"]}


# ─── REGION POLYGONS ──────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/region")
async def api_add_region(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    poly = payload.get("polygon", [])
    if len(poly) < 3:
        raise HTTPException(400, "region polygon must have at least 3 vertices")
    region_id = (max((r["id"] for r in plan.get("regions", [])), default=-1)) + 1
    tint = _region_tint(region_id)
    analysis = _analyse_and_pack(
        session_id, poly,
        range_m=float(payload.get("range_m", 0.05)),
        tint=tint,
    )
    datum = (plan.get("main") or {}).get("stats", {}).get("mean_y")
    # Numbering: Main Ceiling = 1, regions start at 2.
    label = payload.get("label") or f"Ceiling Region ({region_id + 2})"
    region = {
        "id": region_id,
        "label": label,
        "polygon": [list(p) for p in poly],
        "relative_y": (analysis["stats"]["mean_y"] - datum)
            if (analysis["stats"]["mean_y"] is not None and datum is not None)
            else None,
        **analysis,
    }
    plan.setdefault("regions", []).append(region)
    had_topology = plan.get("topology") is not None
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    if had_topology:
        # Topology was already built — rebuild it so the new region
        # participates with shared edges instead of dangling outside.
        await api_snap(session_id)
        plan = _load_plan(session_id)
        return {"ok": True, "region": region, "snapped": True, "plan": plan}
    return {"ok": True, "region": region}


@app.put("/api/sessions/{session_id}/region/{region_id}")
async def api_update_region(
    session_id: str, region_id: int, payload: dict = Body(...),
) -> dict:
    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if "polygon" in payload and topo is not None:
        # After snap, polygons are derived from the topology — direct
        # polygon edits would silently desync. Force callers through the
        # topology endpoints so shared edges stay shared.
        raise HTTPException(
            409,
            "this session is snapped — edit vertices via "
            "PUT /topology/vertices instead of PUT /region/{id}",
        )
    for r in plan.get("regions", []):
        if r["id"] == region_id:
            if "polygon" in payload:
                analysis = _analyse_and_pack(
                    session_id, payload["polygon"],
                    range_m=float(payload.get("range_m", r.get("heatmap_range_m", 0.05))),
                    tint=_validate_tint(payload.get("tint"), r.get("tint", _region_tint(region_id))),
                )
                r["polygon"] = [list(p) for p in payload["polygon"]]
                r.update(analysis)
                datum = (plan.get("main") or {}).get("stats", {}).get("mean_y")
                m = r["stats"]["mean_y"]
                r["relative_y"] = (m - datum) if (m is not None and datum is not None) else None
            if "label" in payload:
                r["label"] = str(payload["label"])
            if "notes" in payload:
                r["notes"] = str(payload["notes"])[:500]
                _mirror_notes_to_topology(plan, region_id=region_id, notes=r["notes"])
            if "tint" in payload and "polygon" not in payload:
                # Tint-only update: re-pack the heatmap so its tint matches.
                tint = _validate_tint(payload["tint"], r.get("tint", _region_tint(region_id)))
                analysis = _analyse_and_pack(
                    session_id, r["polygon"],
                    range_m=float(r.get("heatmap_range_m", 0.05)),
                    tint=tint,
                )
                r.update(analysis)
                _mirror_tint_to_topology(plan, region_id=region_id, tint=tint)
            _recompute_relatives(plan)
            _save_plan(session_id, plan)
            return {"ok": True, "region": r}
    raise HTTPException(404, f"unknown region {region_id}")


@app.put("/api/sessions/{session_id}/main/notes")
async def api_main_notes(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    if not plan.get("main"):
        raise HTTPException(409, "main not set")
    notes = str(payload.get("notes", ""))[:500]
    plan["main"]["notes"] = notes
    _mirror_notes_to_topology(plan, region_id=None, notes=notes)
    _save_plan(session_id, plan)
    return {"ok": True}


@app.put("/api/sessions/{session_id}/main/tint")
async def api_main_tint(session_id: str, payload: dict = Body(...)) -> dict:
    """Recolour the main ceiling (legacy + topology-mirrored) and refresh
    its tinted heatmap PNG. Doesn't touch geometry or stats."""
    plan = _load_plan(session_id)
    if not plan.get("main"):
        raise HTTPException(409, "main not set")
    tint = _validate_tint(payload.get("tint"), plan["main"].get("tint", MAIN_TINT))
    analysis = _analyse_and_pack(
        session_id, plan["main"]["polygon"],
        range_m=float(plan["main"].get("heatmap_range_m", 0.05)),
        tint=tint,
    )
    plan["main"].update(analysis)
    _mirror_tint_to_topology(plan, region_id=None, tint=tint)
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return {"ok": True, "main": plan["main"]}


@app.put("/api/sessions/{session_id}/main/selected_y")
async def api_main_selected_y(session_id: str, payload: dict = Body(...)) -> dict:
    """Set the main ceiling's reported height. The histogram lets the
    user pick a peak (e.g. the upper of two surfaces in a stepped
    main) instead of the mean. Every region's ``relative_y`` is
    re-derived against the new datum."""
    plan = _load_plan(session_id)
    if not plan.get("main"):
        raise HTTPException(409, "main not set")
    raw = payload.get("value")
    if raw is None:
        raise HTTPException(400, "value (selected_y, metres) required")
    try:
        sy = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "value must be numeric")
    plan["main"]["selected_y"] = sy
    _mirror_selected_y_to_topology(plan, region_id=None, selected_y=sy)
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return {"ok": True, "main": plan["main"], "regions": plan.get("regions", [])}


@app.put("/api/sessions/{session_id}/region/{region_id}/selected_y")
async def api_region_selected_y(
    session_id: str, region_id: int, payload: dict = Body(...),
) -> dict:
    """Set a region's reported height. The PDF / sidebar use
    ``selected_y - main.selected_y`` for the height delta — this is
    the user's escape hatch when the polygon's mean Y doesn't match
    what they want to call out."""
    plan = _load_plan(session_id)
    raw = payload.get("value")
    if raw is None:
        raise HTTPException(400, "value (selected_y, metres) required")
    try:
        sy = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "value must be numeric")
    target = next(
        (r for r in plan.get("regions", []) if r["id"] == region_id), None,
    )
    if target is None:
        raise HTTPException(404, f"unknown region {region_id}")
    target["selected_y"] = sy
    _mirror_selected_y_to_topology(plan, region_id=region_id, selected_y=sy)
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return {"ok": True, "region": target}


@app.get("/api/sessions/{session_id}/face_below")
async def api_face_below_overlay(
    session_id: str, key: str, y: float,
) -> Response:
    """Return a PNG overlay (RGBA) sized to the named face's bbox,
    rendering pixels with Y < ``y`` as a pink/black 4-px checker.

    The histogram slider hits this on every drag tick to preview which
    pixels would be excluded if the user committed the marker's
    position as the face's height. ``key`` is ``"main"`` or
    ``"region:<id>"``. The PNG's bbox in pixel coordinates is returned
    in the ``X-Bbox`` header as ``"x0,y0,x1,y1"`` so the canvas can
    composite it at the right place."""
    from .analyse import polygon_to_mask
    plan = _load_plan(session_id)
    if key == "main":
        face_data = plan.get("main")
    elif key.startswith("region:"):
        try:
            rid = int(key.split(":", 1)[1])
        except (ValueError, IndexError):
            raise HTTPException(400, f"bad face key {key!r}")
        face_data = next(
            (r for r in plan.get("regions", []) if r["id"] == rid), None,
        )
    else:
        raise HTTPException(400, f"key must be 'main' or 'region:<id>'")
    if face_data is None:
        raise HTTPException(404, f"unknown face {key!r}")
    poly = face_data.get("polygon") or []
    if len(poly) < 3:
        raise HTTPException(400, "face has no polygon")

    height_map, grid = _load_height_map(session_id)
    mask = polygon_to_mask([tuple(p) for p in poly], grid) > 0
    for hole in (face_data.get("holes_polygons") or []):
        if len(hole) >= 3:
            mask &= ~(polygon_to_mask([tuple(p) for p in hole], grid) > 0)

    png_bytes, bbox = _below_overlay_from_mask(
        mask, height_map, threshold_y=float(y),
    )
    headers = {
        "X-Bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "Cache-Control": "no-store",
    }
    if not png_bytes:
        # No pixels below threshold — return a 1×1 transparent PNG so
        # the client doesn't have to special-case 404s in the drag loop.
        ok, buf = cv2.imencode(
            ".png", np.zeros((1, 1, 4), dtype=np.uint8),
        )
        png_bytes = buf.tobytes() if ok else b""
    return Response(
        content=png_bytes, media_type="image/png", headers=headers,
    )


def _mirror_selected_y_to_topology(
    plan: dict, *, region_id: int | None, selected_y: float,
) -> None:
    """Keep ``plan['topology'].faces[*].selected_y`` in sync with the
    legacy ``main`` / ``regions`` views. ``region_id=None`` means main."""
    topo = plan.get("topology")
    if not topo:
        return
    for f in topo.get("faces") or []:
        if region_id is None and f.get("id") == 0:
            f["selected_y"] = selected_y
            return
        if region_id is not None and f.get("region_id") == region_id:
            f["selected_y"] = selected_y
            return


_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _validate_tint(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    if not _HEX_RE.match(s):
        raise HTTPException(400, f"tint must be #rrggbb hex; got {value!r}")
    return s


def _mirror_tint_to_topology(plan: dict, *, region_id: int | None, tint: str) -> None:
    """Keep ``plan['topology'].faces[*].tint`` in sync with the legacy
    ``main`` / ``regions`` tint. ``region_id=None`` means main."""
    topo = plan.get("topology")
    if topo is None:
        return
    for face in topo.get("faces", []):
        if region_id is None and face.get("kind") == "main":
            face["tint"] = tint
        elif region_id is not None and face.get("region_id") == region_id:
            face["tint"] = tint


def _mirror_notes_to_topology(plan: dict, *, region_id: int | None, notes: str) -> None:
    """Keep ``plan['topology'].faces[*].notes`` in sync with the legacy
    ``main`` / ``regions`` notes inputs. ``region_id=None`` means main."""
    topo = plan.get("topology")
    if topo is None:
        return
    for face in topo.get("faces", []):
        if region_id is None and face.get("kind") == "main":
            face["notes"] = notes
            return
        if region_id is not None and face.get("region_id") == region_id:
            face["notes"] = notes
            return


@app.delete("/api/sessions/{session_id}/region/{region_id}")
async def api_delete_region(session_id: str, region_id: int) -> dict:
    plan = _load_plan(session_id)
    before = len(plan.get("regions", []))
    plan["regions"] = [r for r in plan.get("regions", []) if r["id"] != region_id]
    if len(plan["regions"]) == before:
        raise HTTPException(404, f"unknown region {region_id}")
    had_topology = plan.get("topology") is not None
    _save_plan(session_id, plan)
    if had_topology:
        # Rebuild topology so the deleted region's area is re-absorbed by
        # its neighbours via Voronoi assignment.
        if plan.get("regions"):
            await api_snap(session_id)
            plan = _load_plan(session_id)
        else:
            # No regions left — clear topology too; leaves only main.
            plan["topology"] = None
            plan["snapped"] = False
            _save_plan(session_id, plan)
        return {"ok": True, "snapped": True, "plan": plan}
    return {"ok": True}


# ─── SNAP TOPOLOGY ────────────────────────────────────────────────────────────

# ─── AUTO-DETECT ──────────────────────────────────────────────────────────────

def _enforce_min_edge(pts: np.ndarray, min_edge_px: float) -> np.ndarray:
    """Drop vertices whose edge to the previous kept vertex is shorter than
    ``min_edge_px``. Operates on a closed ring."""
    if len(pts) < 4:
        return pts
    kept = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - kept[-1])) >= min_edge_px:
            kept.append(p)
    if len(kept) > 3 and float(np.linalg.norm(kept[-1] - kept[0])) < min_edge_px:
        kept.pop()
    return np.asarray(kept, dtype=pts.dtype)


@app.post("/api/sessions/{session_id}/auto_detect")
async def api_auto_detect(session_id: str, payload: dict = Body(default={})) -> dict:
    """Auto-fill the ceiling with a clean partition of the room polygon.

    Pipeline:

    1. Histogram peaks on the height map within the room (capped at
       ``max_clusters`` heights, default 5; uses fewer if fewer real
       peaks exist).
    2. Each room pixel is assigned to its nearest peak in Y. Pixels with
       no LiDAR coverage are assigned by spatial nearest-neighbour, so
       coverage is 100%.
    3. Connected components per cluster; tiny CCs (< ``min_cc_area_m2``)
       are absorbed by the nearest surviving CC (any cluster).
    4. Each surviving CC is contoured and Douglas-Peucker simplified at
       ``min_edge_m`` tolerance, then short edges are collapsed.
    5. The largest polygon by area becomes "Main Ceiling (1)" (defines
       the height datum); the rest become regions ordered by Y.

    Replaces ``plan.main`` and ``plan.regions``. Notes / labels on
    user-drawn polygons are NOT preserved (this is a fresh starting
    partition the user can then refine).
    """
    from scipy.ndimage import distance_transform_edt, median_filter
    from scipy.signal import find_peaks
    from .analyse import polygon_to_mask

    plan = _load_plan(session_id)
    room_pts = plan.get("room")
    if not room_pts or len(room_pts) < 3:
        raise HTTPException(400, "draw the room outline first")

    max_clusters = int(payload.get("max_clusters", 5))
    min_cc_area_m2 = float(payload.get("min_cc_area_m2", 0.3))
    min_edge_m = float(payload.get("min_edge_m", 0.5))
    median_kernel_m = float(payload.get("median_kernel_m", 0.10))
    bin_size_m = 0.005
    min_peak_separation_m = 0.10

    height_map, grid = _load_height_map(session_id)
    H, W = grid.height, grid.width
    px_per_m = grid.pixels_per_metre

    room_mask = polygon_to_mask([tuple(p) for p in room_pts], grid) > 0
    if not room_mask.any():
        raise HTTPException(409, "room polygon doesn't overlap the rendered area")

    heights = height_map.copy()
    heights[~room_mask] = np.nan
    valid = ~np.isnan(heights)
    if int(valid.sum()) < 1000:
        raise HTTPException(409, "not enough LiDAR coverage in the room")

    # 1. Histogram peaks (≤ max_clusters)
    valid_y = heights[valid].astype(np.float64)
    edges = np.arange(
        float(valid_y.min()) - bin_size_m,
        float(valid_y.max()) + 2 * bin_size_m,
        bin_size_m,
    )
    hist, _ = np.histogram(valid_y, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.array([0.25, 0.5, 0.25])
    hist_s = np.convolve(hist.astype(np.float64), smooth, mode="same")

    sep_bins = max(1, int(round(min_peak_separation_m / bin_size_m)))
    peaks, _props = find_peaks(
        hist_s, distance=sep_bins,
        prominence=0.05 * float(hist_s.max()),
    )
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(hist_s))])
    # Keep top-N by histogram height, then re-sort by Y for determinism.
    peaks = peaks[np.argsort(-hist_s[peaks])][:max_clusters]
    peaks.sort()
    cluster_ys = centres[peaks]

    # 2. Per-pixel nearest-peak assignment
    flat = heights[valid]
    assign_flat = np.argmin(
        np.abs(flat[:, None] - cluster_ys[None, :]), axis=1,
    ).astype(np.int32)
    assignment = np.full((H, W), -1, dtype=np.int32)
    assignment[valid] = assign_flat

    # 2b. Fill NaN-in-room pixels by spatial nearest valid neighbour
    nan_in_room = room_mask & ~valid
    if nan_in_room.any():
        _, idx = distance_transform_edt(~valid, return_indices=True)
        ys, xs = np.where(nan_in_room)
        assignment[ys, xs] = assignment[idx[0, ys, xs], idx[1, ys, xs]]

    assignment[~room_mask] = -1

    # 2c. Median-filter the assignment to kill salt-and-pepper noise at
    # cluster boundaries. Without this, the per-cluster connected-components
    # step explodes into dozens of tiny features.
    k = max(3, int(round(median_kernel_m * px_per_m)) | 1)
    if k > 1:
        # Treat -1 (outside room) as a sentinel by temporarily replacing it
        # with a large value so the median filter doesn't pull it into the room.
        sentinel = int(assignment.max() + 100)
        a_padded = np.where(assignment >= 0, assignment, sentinel).astype(np.int32)
        a_padded = median_filter(a_padded, size=k, mode="nearest")
        assignment = np.where(room_mask, a_padded, -1)

    # 3. Per-cluster CCs → globally-labelled survivors. A CC survives if
    # it's big enough AND backed by enough real LiDAR coverage (not just
    # NaN-fill). Each surviving CC gets a unique global label so we can
    # output one polygon per CC, even multiple per cluster.
    min_cc_px = max(50, int(min_cc_area_m2 * px_per_m * px_per_m))
    min_cc_coverage = float(payload.get("min_cc_coverage", 0.4))

    global_label = np.full((H, W), -1, dtype=np.int32)
    cluster_for_label: list[int] = []
    for ci in range(len(cluster_ys)):
        cmask = (assignment == ci).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cmask, connectivity=8)
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_cc_px:
                continue
            cc_pixels = (labels == lab)
            cov = float((cc_pixels & valid).sum()) / max(1, area)
            if cov < min_cc_coverage:
                continue
            global_label[cc_pixels] = len(cluster_for_label)
            cluster_for_label.append(int(ci))

    if not cluster_for_label:
        raise HTTPException(409, "no clusters survived the coverage / area filters")

    # 3a. Same-cluster loser absorption: for each loser pixel originally in
    # cluster ci, if any survivor of ci exists, copy the nearest one's
    # global label (so the absorbed pixels become part of that CC).
    for ci in range(len(cluster_ys)):
        same_cluster_survivor = np.zeros((H, W), dtype=bool)
        for gid, c in enumerate(cluster_for_label):
            if c == ci:
                same_cluster_survivor |= (global_label == gid)
        if not same_cluster_survivor.any():
            continue
        same_loser = room_mask & (assignment == ci) & (global_label == -1)
        if not same_loser.any():
            continue
        _, idx = distance_transform_edt(~same_cluster_survivor, return_indices=True)
        ys, xs = np.where(same_loser)
        global_label[ys, xs] = global_label[idx[0, ys, xs], idx[1, ys, xs]]

    # 3b. Orphans: any room pixel still unlabelled → nearest survivor of any cluster.
    orphans = room_mask & (global_label == -1)
    if orphans.any():
        survivor_mask = global_label != -1
        _, idx = distance_transform_edt(~survivor_mask, return_indices=True)
        ys, xs = np.where(orphans)
        global_label[ys, xs] = global_label[idx[0, ys, xs], idx[1, ys, xs]]
    global_label[~room_mask] = -1

    # 4. Per-cluster contour → DP simplify @ 50 cm → enforce min edge
    min_edge_px = max(2.0, min_edge_m * px_per_m)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(0.06 * px_per_m)) | 1),) * 2,
    )

    polygons_out: list[dict] = []
    for gid in range(len(cluster_for_label)):
        cc_pixel_mask = (global_label == gid)
        if not cc_pixel_mask.any():
            continue
        ci = cluster_for_label[gid]
        cc_u8 = cc_pixel_mask.astype(np.uint8) * 255
        cc_u8 = cv2.morphologyEx(cc_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        contours, _ = cv2.findContours(cc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, min_edge_px, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue
        approx = _enforce_min_edge(approx.astype(np.float64), min_edge_px)
        if len(approx) < 3:
            continue
        u = approx[:, 0]; v = approx[:, 1]
        x, z = grid.px_to_world(u, v)
        polygons_out.append({
            "cluster_idx": ci,
            "y": float(cluster_ys[ci]),
            "polygon": [[float(xx), float(zz)] for xx, zz in zip(x, z)],
            "area_px": int(cc_pixel_mask.sum()),
            "pixel_mask": cc_pixel_mask,
        })

    if not polygons_out:
        raise HTTPException(409, "no clusters detected after merging tiny components")

    # Largest by area becomes "main"; rest sort by relative height.
    polygons_out.sort(key=lambda p: -p["area_px"])
    main_data = polygons_out[0]
    rest = sorted(polygons_out[1:], key=lambda p: p["y"])

    # Stats-mask: only pixels whose actual height matches the cluster's
    # peak (within ±band_stats_m). This prevents wrongly-absorbed
    # cross-cluster orphans from skewing mean / σ — they still appear in
    # the visual polygon (and as off-tint patches in the heatmap), but
    # don't pollute the datum or the relative heights.
    band_stats_m = float(payload.get("band_stats_m", 0.10))

    def _stats_mask_for(visual_mask: np.ndarray, ci: int) -> np.ndarray:
        in_band = np.abs(height_map - cluster_ys[ci]) <= band_stats_m
        return visual_mask & valid & in_band

    main_ci = main_data["cluster_idx"]
    main_analysis = _analyse_and_pack(
        session_id,
        mask=main_data["pixel_mask"],
        stats_mask=_stats_mask_for(main_data["pixel_mask"], main_ci),
        tint=MAIN_TINT,
    )
    plan["main"] = {
        "polygon": main_data["polygon"],
        "label": "Main Ceiling (1)",
        "notes": "",
        **main_analysis,
    }
    datum = main_analysis["stats"]["mean_y"]

    new_regions = []
    for i, p in enumerate(rest):
        rid = i
        analysis = _analyse_and_pack(
            session_id,
            mask=p["pixel_mask"],
            stats_mask=_stats_mask_for(p["pixel_mask"], p["cluster_idx"]),
            tint=_region_tint(rid),
        )
        m = analysis["stats"]["mean_y"]
        new_regions.append({
            "id": rid,
            "label": f"Ceiling Region ({rid + 2})",
            "notes": "",
            "polygon": p["polygon"],
            "relative_y": (m - datum) if (m is not None and datum is not None) else None,
            **analysis,
        })
    plan["regions"] = new_regions
    plan["auto_detected"] = True
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return plan


@app.post("/api/sessions/{session_id}/snap")
async def api_snap(session_id: str) -> dict:
    """Push/pull polygon borders so they share clean edges and tile the
    room without gaps or overlap, and produce a planar topology where
    every shared boundary is stored as one edge referenced by both faces.

    Pipeline:

    1. Voronoi-style assignment of every room pixel to its nearest drawn
       polygon (regions win their drawn pixels first, then main, then
       distance-transform fill into the gaps).
    2. Morphological close on each face's claim to smooth jaggies, then
       merge back into a single label image.
    3. ``topology.build_from_assignment`` traces boundaries at corner
       resolution, simplifies once per shared edge, and emits
       ``{vertices, edges, faces}``.
    4. Re-run the height analysis on every face polygon so heatmaps and
       σ reflect the new shapes; populate ``main`` and ``regions`` with
       the derived polygons so the existing frontend renders unchanged
       while the topology underlies it.

    After snap, the *topology* is the source of truth — editing should go
    through the topology endpoints so shared edges move both faces
    together. ``main`` and ``regions`` are kept as a derived view.
    """
    from .analyse import polygon_to_mask
    from .topology import build_from_assignment, OUTSIDE

    plan = _load_plan(session_id)
    room_pts = plan.get("room")
    if not room_pts or len(room_pts) < 3:
        raise HTTPException(400, "room outline required before snapping")

    _, grid = _load_height_map(session_id)
    room_mask = polygon_to_mask([tuple(p) for p in room_pts], grid) > 0
    if not room_mask.any():
        raise HTTPException(409, "room polygon doesn't overlap the rendered area")

    # Subtract obstructions (columns) from the room mask so ceiling regions
    # can't claim those pixels. Voronoi-driven gap filling later in this
    # function then can't push into the column either, since the obstruction
    # area is treated as outside-the-room.
    for obs in plan.get("obstructions", []):
        if len(obs.get("polygon", [])) >= 3:
            obs_mask = polygon_to_mask(
                [tuple(p) for p in obs["polygon"]], grid,
            ) > 0
            room_mask = room_mask & ~obs_mask
    if not room_mask.any():
        raise HTTPException(409, "obstructions cover the entire room")

    polygons: list[dict] = []
    if plan.get("main"):
        polygons.append({
            "key": "main",
            "kind": "main",
            "polygon": plan["main"]["polygon"],
            "label": plan["main"].get("label", "Main Ceiling (1)"),
            "tint": plan["main"].get("tint") or MAIN_TINT,
            "notes": plan["main"].get("notes", ""),
        })
    for r in plan.get("regions", []):
        polygons.append({
            "key": f"region:{r['id']}",
            "kind": "region",
            "id": int(r["id"]),
            "polygon": r["polygon"],
            "label": r.get("label", f"Ceiling Region ({int(r['id']) + 2})"),
            "tint": r.get("tint") or _region_tint(int(r["id"])),
            "notes": r.get("notes", ""),
        })
    if not polygons:
        raise HTTPException(409, "draw a main ceiling and any regions before snapping")
    if polygons[0]["kind"] != "main":
        raise HTTPException(409, "main ceiling required before snapping")

    H, W = grid.height, grid.width

    # Per-polygon distance maps: 0 inside the polygon, increasing outside.
    dist_stack = np.empty((len(polygons), H, W), dtype=np.float32)
    drawn_masks: list[np.ndarray] = []
    for i, p in enumerate(polygons):
        m = polygon_to_mask([tuple(v) for v in p["polygon"]], grid) > 0
        drawn_masks.append(m)
        if not m.any():
            dist_stack[i] = np.float32(1e9)
            continue
        inv = (~m).astype(np.uint8) * 255
        dist_stack[i] = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    # Stage 1: each region keeps its drawn pixels; later-drawn wins overlap.
    assignment = np.full((H, W), -1, dtype=np.int32)
    for i, p in enumerate(polygons):
        if p["kind"] != "region":
            continue
        assignment[drawn_masks[i]] = i

    # Stage 2: main claims its drawn pixels that no region took.
    main_drawn = drawn_masks[0]
    free = main_drawn & (assignment == -1)
    assignment[free] = 0

    # Stage 3: any pixel still inside the room but unassigned grows the
    # closest polygon out to fill it (Voronoi between drawn shapes).
    gap = room_mask & (assignment == -1)
    if gap.any():
        nearest = np.argmin(dist_stack, axis=0)
        assignment[gap] = nearest[gap]

    # Pixels outside the room belong to nobody.
    assignment_room = np.where(room_mask, assignment, OUTSIDE)

    # Smooth the per-face boundaries, then re-merge into a label image
    # the topology builder can trace.
    px_per_m = grid.pixels_per_metre
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(0.06 * px_per_m)) | 1),) * 2,
    )
    smoothed = np.full_like(assignment_room, OUTSIDE)
    for i in range(len(polygons)):
        claim = (assignment_room == i).astype(np.uint8) * 255
        if claim.sum() == 0:
            continue
        claim = cv2.morphologyEx(claim, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        # Smoothing can grow the claim outside the room; clip back.
        claim = (claim > 0) & room_mask
        # Later-iterated faces overwrite earlier (rare overlap from dilation
        # in tight gaps). Net effect is small because closes are 1-2 px.
        smoothed[claim] = i

    # Any pixel that smoothing left unassigned: hand to nearest face.
    leftover = room_mask & (smoothed == OUTSIDE)
    if leftover.any():
        nearest = np.argmin(dist_stack, axis=0)
        smoothed[leftover] = nearest[leftover]

    topo = build_from_assignment(smoothed, grid)
    if not topo["faces"]:
        raise HTTPException(409, "snap produced no faces — try widening the room polygon")

    # Decorate each topology face with the matching polygon's metadata +
    # a fresh height analysis on the new polygon shape (with column holes
    # subtracted from the per-face mask).
    edges_list = topo["edges"]
    vert_list = topo["vertices"]
    main_face = next((f for f in topo["faces"] if f["id"] == 0), None)
    if main_face is None:
        raise HTTPException(409, "main ceiling lost all coverage after snapping")
    main_coords = main_face["polygon"]
    main_holes = _resolve_face_holes(main_face, edges_list, vert_list)
    main_face["holes_polygons"] = main_holes
    main_meta = polygons[0]
    main_tint = main_meta.get("tint") or MAIN_TINT
    main_analysis = _analyse_and_pack(
        session_id, main_coords, holes=main_holes, tint=main_tint,
    )
    main_face.update({
        "kind": "main",
        "label": main_meta["label"],
        "notes": main_meta["notes"],
        "tint": main_tint,
        "relative_y": 0.0,
        **main_analysis,
    })
    plan["main"] = {
        "polygon": main_coords,
        "holes_polygons": main_holes,
        "label": main_meta["label"],
        "notes": main_meta["notes"],
        "tint": main_tint,
        **main_analysis,
    }
    datum = main_analysis["stats"]["mean_y"]

    new_regions = []
    for face in topo["faces"]:
        if face["id"] == 0:
            continue
        meta = polygons[face["id"]]  # face id matches polygons-list index
        coords = face["polygon"]
        holes = _resolve_face_holes(face, edges_list, vert_list)
        face["holes_polygons"] = holes
        rid = int(meta["id"])
        face_tint = meta.get("tint") or _region_tint(rid)
        analysis = _analyse_and_pack(
            session_id, coords, holes=holes, tint=face_tint,
        )
        m = analysis["stats"]["mean_y"]
        relative_y = (m - datum) if (m is not None and datum is not None) else None
        face.update({
            "kind": "region",
            "region_id": rid,
            "label": meta["label"],
            "notes": meta["notes"],
            "tint": face_tint,
            "relative_y": relative_y,
            **analysis,
        })
        new_regions.append({
            "id": rid,
            "label": meta["label"],
            "notes": meta["notes"],
            "polygon": coords,
            "holes_polygons": holes,
            "relative_y": relative_y,
            "tint": face_tint,
            **analysis,
        })

    plan["topology"] = topo
    plan["regions"] = new_regions
    plan["main_face_id"] = 0  # fresh snap: face id 0 is always the main
    plan["snapped"] = True
    _recompute_relatives(plan)
    _save_plan(session_id, plan)
    return plan


# ─── TOPOLOGY EDITS (post-snap) ───────────────────────────────────────────────


def _resolve_ring(ring: list[dict], edges_by_id: dict, vertices: list[list[float]]) -> list[list[float]]:
    pts: list[list[float]] = []
    for h in ring:
        e = edges_by_id[h["edge"]]
        verts = list(reversed(e["vertices"])) if h["rev"] else e["vertices"]
        for vid in verts[:-1]:
            x, z = vertices[vid]
            pts.append([float(x), float(z)])
    return pts


def _resolve_face_polygon(face: dict, edges: list[dict], vertices: list[list[float]]) -> list[list[float]]:
    """Walk a face's ring of {edge, rev} entries against the given vertex pool."""
    edges_by_id = {e["id"]: e for e in edges}
    return _resolve_ring(face["ring"], edges_by_id, vertices)


def _resolve_face_holes(face: dict, edges: list[dict], vertices: list[list[float]]) -> list[list[list[float]]]:
    """Resolve every hole ring on a face into a polygon. Empty if no holes."""
    if not face.get("holes"):
        return []
    edges_by_id = {e["id"]: e for e in edges}
    return [_resolve_ring(ring, edges_by_id, vertices) for ring in face["holes"]]


def _refresh_topology_polygons(plan: dict, session_id: str) -> None:
    """After mutating ``plan['topology']['vertices']`` (or any face's ring),
    rebuild every face's ``polygon``, re-run the height analysis on it, and
    sync the derived ``main`` / ``regions`` views the frontend reads.

    Honours ``plan.main_face_id`` so a post-snap main-face swap (which
    relabels roles in place rather than re-Voronoi'ing) renders with the
    correct main / region split."""
    topo = plan["topology"]
    if topo is None:
        return

    vertices = topo["vertices"]
    edges = topo["edges"]
    main_id = int(plan.get("main_face_id", 0))

    main_face = next((f for f in topo["faces"] if f["id"] == main_id), None)
    datum = None
    if main_face is not None:
        coords = _resolve_face_polygon(main_face, edges, vertices)
        if len(coords) < 3:
            raise HTTPException(409, "main face collapsed to fewer than 3 vertices")
        holes = _resolve_face_holes(main_face, edges, vertices)
        tint = main_face.get("tint", MAIN_TINT)
        analysis = _analyse_and_pack(session_id, coords, holes=holes, tint=tint)
        main_face["polygon"] = coords
        main_face["holes_polygons"] = holes
        main_face.update({
            "kind": "main",
            "label": main_face.get("label", "Main Ceiling (1)"),
            "notes": main_face.get("notes", ""),
            "tint": tint,
            "relative_y": 0.0,
            **analysis,
        })
        # Strip any region_id left over from a previous role.
        main_face.pop("region_id", None)
        datum = analysis["stats"]["mean_y"]
        plan["main"] = {
            "polygon": coords,
            "holes_polygons": holes,
            "label": main_face.get("label", "Main Ceiling (1)"),
            "notes": main_face.get("notes", ""),
            "tint": tint,
            **analysis,
        }

    new_regions = []
    for face in topo["faces"]:
        if face["id"] == main_id:
            continue
        coords = _resolve_face_polygon(face, edges, vertices)
        if len(coords) < 3:
            # Face collapsed; drop it (topology stays but the region view skips).
            continue
        holes = _resolve_face_holes(face, edges, vertices)
        # region_id is the user-facing id and must persist across role swaps.
        # Fall back to (face_id - 1) for legacy plans where region_id wasn't
        # written; subtract an extra 1 if the face id is past the main slot.
        fallback_rid = face["id"] - 1 if face["id"] > main_id else face["id"]
        rid = int(face.get("region_id", fallback_rid))
        tint = face.get("tint", _region_tint(rid))
        analysis = _analyse_and_pack(session_id, coords, holes=holes, tint=tint)
        m = analysis["stats"]["mean_y"]
        relative_y = (m - datum) if (m is not None and datum is not None) else None
        face["polygon"] = coords
        face["holes_polygons"] = holes
        face.update({
            "kind": "region",
            "region_id": rid,
            "tint": tint,
            "relative_y": relative_y,
            **analysis,
        })
        new_regions.append({
            "id": rid,
            "label": face.get("label", f"Ceiling Region ({rid + 2})"),
            "notes": face.get("notes", ""),
            "polygon": coords,
            "holes_polygons": holes,
            "relative_y": relative_y,
            "tint": tint,
            **analysis,
        })
    plan["regions"] = new_regions
    _recompute_relatives(plan)


@app.put("/api/sessions/{session_id}/topology/vertices")
async def api_topology_set_vertices(session_id: str, payload: dict = Body(...)) -> dict:
    """Replace the topology's vertex pool. Edges and faces keep their
    structure (they reference indices into this pool), so moving a vertex
    that's shared by two faces shifts both face polygons in lockstep —
    which is the whole point of having shared edges."""
    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if topo is None:
        raise HTTPException(409, "no topology yet — snap first")
    new_verts = payload.get("vertices")
    if not isinstance(new_verts, list) or len(new_verts) != len(topo["vertices"]):
        raise HTTPException(
            400,
            f"expected {len(topo['vertices'])} vertices, got "
            f"{len(new_verts) if isinstance(new_verts, list) else 'invalid'}",
        )
    topo["vertices"] = [[float(p[0]), float(p[1])] for p in new_verts]
    _refresh_topology_polygons(plan, session_id)
    _save_plan(session_id, plan)
    return plan


@app.delete("/api/sessions/{session_id}/topology")
async def api_topology_clear(session_id: str) -> dict:
    """Un-snap. Drops the topology and leaves the legacy ``main`` /
    ``regions`` polygons intact at their last derived shape, so the user
    can edit them with the original tools and then re-snap."""
    plan = _load_plan(session_id)
    plan["topology"] = None
    plan["snapped"] = False
    _save_plan(session_id, plan)
    return plan


@app.post("/api/sessions/{session_id}/topology/edge/{edge_id}/insert_vertex")
async def api_topology_insert_vertex(
    session_id: str, edge_id: int, payload: dict = Body(...),
) -> dict:
    """Split a topology edge by inserting a new vertex at ``position``.

    Both faces incident to the edge gain the new vertex in their boundary
    rings — the headline ``shared edges stay shared`` property carries
    through the split.
    """
    from .topology import insert_vertex_on_edge

    pos = payload.get("position")
    if not (isinstance(pos, list) and len(pos) == 2):
        raise HTTPException(400, "position must be [x, z]")
    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if topo is None:
        raise HTTPException(409, "no topology yet — snap first")
    try:
        insert_vertex_on_edge(topo, edge_id, (float(pos[0]), float(pos[1])))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _refresh_topology_polygons(plan, session_id)
    _save_plan(session_id, plan)
    return plan


@app.delete("/api/sessions/{session_id}/topology/vertex/{vertex_id}")
async def api_topology_delete_vertex(session_id: str, vertex_id: int) -> dict:
    """Remove a topology vertex.

    Supported cases:

    - **Interior vertex** of a single edge's polyline → spliced out.
    - **Degree-2 endpoint** (two edges with the same face pair) → the
      two edges merge into one.

    Junction vertices (3+ incident edges with mixed face pairs) reject
    with 400 — that's a face-level edit, not a vertex edit.
    """
    from .topology import delete_vertex as topo_delete_vertex

    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if topo is None:
        raise HTTPException(409, "no topology yet — snap first")
    if not (0 <= vertex_id < len(topo["vertices"])):
        raise HTTPException(404, f"unknown vertex {vertex_id}")
    try:
        topo_delete_vertex(topo, vertex_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _refresh_topology_polygons(plan, session_id)
    _save_plan(session_id, plan)
    return plan


@app.put("/api/sessions/{session_id}/topology/face/{face_id}/notes")
async def api_topology_set_face_notes(
    session_id: str, face_id: int, payload: dict = Body(...),
) -> dict:
    """Update a face's notes (post-snap). Mirrors the value into the
    derived ``main`` / ``regions`` view so the frontend's existing notes
    inputs keep working without knowing about topology."""
    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if topo is None:
        raise HTTPException(409, "no topology yet — snap first")
    face = next((f for f in topo["faces"] if f["id"] == face_id), None)
    if face is None:
        raise HTTPException(404, f"unknown face {face_id}")
    notes = str(payload.get("notes", ""))
    face["notes"] = notes
    if face_id == 0 and plan.get("main"):
        plan["main"]["notes"] = notes
    else:
        rid = int(face.get("region_id", face_id - 1))
        for r in plan.get("regions", []):
            if r["id"] == rid:
                r["notes"] = notes
                break
    _save_plan(session_id, plan)
    return {"ok": True}


@app.put("/api/sessions/{session_id}/topology/face/{face_id}/tint")
async def api_topology_set_face_tint(
    session_id: str, face_id: int, payload: dict = Body(...),
) -> dict:
    """Recolour a face (post-snap). Mirrors the value into the legacy
    ``main`` / ``regions`` view and refreshes the heatmap PNG."""
    plan = _load_plan(session_id)
    topo = plan.get("topology")
    if topo is None:
        raise HTTPException(409, "no topology yet — snap first")
    face = next((f for f in topo["faces"] if f["id"] == face_id), None)
    if face is None:
        raise HTTPException(404, f"unknown face {face_id}")
    tint = _validate_tint(payload.get("tint"), face.get("tint", MAIN_TINT))
    face["tint"] = tint
    # Mirror to the legacy view + refresh heatmap, preserving any column holes.
    holes = face.get("holes_polygons") or _resolve_face_holes(
        face, topo["edges"], topo["vertices"],
    )
    if face_id == 0 and plan.get("main"):
        analysis = _analyse_and_pack(
            session_id, plan["main"]["polygon"],
            holes=holes,
            range_m=float(plan["main"].get("heatmap_range_m", 0.05)),
            tint=tint,
        )
        plan["main"]["tint"] = tint
        plan["main"].update(analysis)
    else:
        rid = int(face.get("region_id", face_id - 1))
        for r in plan.get("regions", []):
            if r["id"] == rid:
                analysis = _analyse_and_pack(
                    session_id, r["polygon"],
                    holes=holes,
                    range_m=float(r.get("heatmap_range_m", 0.05)),
                    tint=tint,
                )
                r["tint"] = tint
                r.update(analysis)
                break
    _save_plan(session_id, plan)
    return {"ok": True, "plan": plan}


# ─── PDF EXPORT ───────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/pdf")
async def api_pdf(session_id: str) -> Response:
    """Produce an architectural PDF: coloured fills per face, then each
    *edge* stroked exactly once. Pre-snap (no topology) falls back to the
    old per-polygon stroking — those plans don't have shared edges yet."""
    plan = _load_plan(session_id)
    if not plan.get("room"):
        raise HTTPException(400, "room required for PDF")

    import io
    import math as _math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import (
        Polygon as MplPolygon, PathPatch, Rectangle, FancyArrow,
    )
    from matplotlib.path import Path as MplPath

    room = plan["room"]
    main = plan.get("main")
    regions = plan.get("regions", [])
    topo = plan.get("topology")
    units = plan.get("units") or DEFAULT_UNITS
    project = plan.get("project") or _default_project()

    xs = [p[0] for p in room]
    zs = [p[1] for p in room]
    room_minx, room_maxx = min(xs), max(xs)
    room_minz, room_maxz = min(zs), max(zs)
    room_w_m = room_maxx - room_minx
    room_h_m = room_maxz - room_minz

    # ─── A1 landscape page (841 × 594 mm = 33.11 × 23.39 in) ──
    # gridspec: plan area (top-left) + title block (full-height right strip)
    # + legend strip (bottom-left).
    A1_W_IN, A1_H_IN = 33.11, 23.39
    fig = plt.figure(figsize=(A1_W_IN, A1_H_IN))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[3.2, 0.7],   # plan : title block (30 % narrower than v1)
        height_ratios=[5.0, 1.0],  # plan : legend
        left=0.02, right=0.98, top=0.98, bottom=0.02,
        hspace=0.03, wspace=0.03,
    )
    ax = fig.add_subplot(gs[0, 0])
    title_ax = fig.add_subplot(gs[:, 1])
    legend_ax = fig.add_subplot(gs[1, 0])

    # Plan-area paper dimensions, derived from the gridspec slot rather
    # than the ax (the latter only resolves once the figure renders).
    plan_bbox = gs[0, 0].get_position(fig)
    plan_w_in = plan_bbox.width * A1_W_IN
    plan_h_in = plan_bbox.height * A1_H_IN

    # Pick the largest standard scale that fits the room inside the plan
    # area with PLAN_PAPER_PAD_M of paper margin per side.
    scale_ratio = _choose_standard_scale(
        room_w_m, room_h_m, plan_w_in, plan_h_in, units,
    )

    # Compute the world window that the plan area represents at this
    # scale: paper width × paper-mm-per-world-mm. Centre on the room.
    window_w_m = plan_w_in * 0.0254 * scale_ratio
    window_h_m = plan_h_in * 0.0254 * scale_ratio
    cx = 0.5 * (room_minx + room_maxx)
    cz = 0.5 * (room_minz + room_maxz)
    minx, maxx = cx - window_w_m / 2, cx + window_w_m / 2
    minz, maxz = cz - window_h_m / 2, cz + window_h_m / 2

    ax.set_aspect("equal")
    # RCP convention: mirror X so the plan reads with floor-plan handedness.
    ax.set_xlim(maxx, minx)
    ax.set_ylim(minz, maxz)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ─── Fills + labels ──
    # When a topology is present, both `_fill` (no edge) and `_stroke_edges`
    # below are used so each shared boundary is drawn exactly once. Without
    # a topology, fall back to drawing each polygon outline as part of the
    # patch (the legacy double-stroke behaviour, accepted for pre-snap
    # plans).
    use_topology = topo is not None and bool(topo.get("edges"))
    fill_alpha = PLAN_FILL_ALPHA

    def _fill(poly_pts, *, face_color, edge_color, label, rel_text, notes,
              holes=None):
        if not poly_pts:
            return
        pts = [(p[0], p[1]) for p in poly_pts]
        if holes:
            # Build a holed Path: outer ring CCW, each hole CW. matplotlib
            # uses the even-odd / non-zero fill rule depending on
            # orientation, so we trust the topology builder's CCW/CW
            # convention rather than re-orienting here.
            verts = list(pts) + [pts[0]]
            codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
                     + [MplPath.CLOSEPOLY])
            for hole in holes:
                if not hole or len(hole) < 3:
                    continue
                hpts = [(p[0], p[1]) for p in hole]
                verts += hpts + [hpts[0]]
                codes += ([MplPath.MOVETO] + [MplPath.LINETO] * (len(hpts) - 1)
                          + [MplPath.CLOSEPOLY])
            path = MplPath(verts, codes)
            patch = PathPatch(
                path, facecolor=face_color,
                edgecolor="none" if use_topology else edge_color,
                linewidth=0 if use_topology else 1.5,
                alpha=fill_alpha,
            )
        elif use_topology:
            patch = MplPolygon(pts, closed=True, facecolor=face_color,
                               edgecolor="none", alpha=fill_alpha)
        else:
            patch = MplPolygon(pts, closed=True, facecolor=face_color,
                               edgecolor=edge_color, linewidth=1.5,
                               alpha=fill_alpha)
        ax.add_patch(patch)
        # Pole of inaccessibility — keeps the label inside the actual face
        # (centroid falls outside L-shapes and avoids column holes).
        outer_ring = [(p[0], p[1]) for p in pts]
        hole_rings = [
            [(p[0], p[1]) for p in (h or [])]
            for h in (holes or [])
            if h and len(h) >= 3
        ]
        try:
            cx, cz = polylabel(outer_ring, hole_rings, precision=0.02)
        except Exception:
            cx = sum(p[0] for p in pts) / len(pts)
            cz = sum(p[1] for p in pts) / len(pts)
        # Architectural convention: ceiling labels read horizontally,
        # matching the title block. Earlier versions rotated to the
        # polygon's longest edge — that read like CAD section markers
        # rather than RCP labels.
        text = f"{label}\n{rel_text}"
        if notes:
            text += f"\n{notes}"
        ax.text(cx, cz, text,
                ha="center", va="center", rotation=0,
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=edge_color, linewidth=0.6))

    # Hole rings come straight from the topology faces when present; the
    # legacy ``main`` / ``regions`` views also carry them for parity.
    def _holes_for(face_view: dict) -> list:
        return face_view.get("holes_polygons") or []

    if main:
        _fill(main["polygon"],
              face_color=main.get("tint", MAIN_TINT), edge_color="#00897b",
              label=main.get("label", "Main Ceiling (1)"),
              rel_text=format_height_delta(0.0, units),
              notes=main.get("notes", ""),
              holes=_holes_for(main))

    for r in regions:
        rel = r.get("relative_y")
        rel_text = "—" if rel is None else format_height_delta(float(rel), units)
        _fill(r["polygon"],
              face_color=r.get("tint", "#ff7043"),
              edge_color="#444",
              label=r.get("label", f"region {r['id']}"),
              rel_text=rel_text,
              notes=r.get("notes", ""),
              holes=_holes_for(r))

    # ─── Edges (one stroke per topology edge) ──
    # Interior edges (shared between two faces) get the standard line
    # weight. Boundary edges (against the room outside) get the heavier
    # outline weight that the room outline used to use.
    if use_topology:
        verts = topo["vertices"]
        for e in topo["edges"]:
            xs_e = [verts[vid][0] for vid in e["vertices"]]
            zs_e = [verts[vid][1] for vid in e["vertices"]]
            is_boundary = e["faces"][1] is None or e["faces"][0] is None
            ax.plot(xs_e, zs_e,
                    color="#222",
                    linewidth=2.0 if is_boundary else 1.2,
                    linestyle="-")

    # Room outline (always shown — it's the user-traced boundary, not
    # necessarily the same as the topology's outermost edges if the snap
    # fell short of the room's extent).
    rxs = [p[0] for p in room] + [room[0][0]]
    rzs = [p[1] for p in room] + [room[0][1]]
    ax.plot(rxs, rzs, color="#222",
            linewidth=2.0,
            linestyle="--" if use_topology else "--")

    # ─── Obstructions (columns) — hatched on top of fills ──
    # Drawn last so the cross-hatch reads above any region fill that
    # might otherwise be visible inside the column outline. No per-column
    # label on the drawing — the legend carries a single "Columns" entry.
    for obs in plan.get("obstructions", []):
        pts = [(p[0], p[1]) for p in obs.get("polygon", [])]
        if len(pts) < 3:
            continue
        patch = MplPolygon(pts, closed=True,
                           facecolor="white", edgecolor="#222",
                           linewidth=1.5, hatch="xx", alpha=1.0)
        ax.add_patch(patch)

    n = len(room)
    for i in range(n):
        ax_, az_ = room[i]
        bx, bz = room[(i + 1) % n]
        midx, midz = 0.5 * (ax_ + bx), 0.5 * (az_ + bz)
        dx, dz = bx - ax_, bz - az_
        L = (dx ** 2 + dz ** 2) ** 0.5
        if L < 1e-3:
            continue
        label = format_length(L, units)
        # Skip labels on edges too short to host them readably.
        # ~0.025 m of plot data per char at 7pt on the A1 page is a usable
        # rule of thumb; require the edge to be ≥ 1.5× the label width.
        est_label_width = 0.025 * len(label)
        if L < 1.5 * est_label_width:
            continue
        # Place the label *on* the edge midpoint — its white bbox cuts the
        # line, giving the architectural ——[label]—— look.
        angle = 0.0
        try:
            import math as _m
            angle = _m.degrees(_m.atan2(dz, dx))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
        except Exception:
            angle = 0.0
        ax.text(midx, midz, label,
                ha="center", va="center", rotation=angle,
                fontsize=7, color="#222",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="#aaa", linewidth=0.4))

    # ─── North arrow (overlay on the plan axes) ──
    # The scale bar lives in the title block now (true-scale rendering
    # makes the plan-area inset redundant — its job was a sanity check
    # at an arbitrary scale).
    if project.get("print_north", True):
        _draw_north_arrow(
            ax, minx, maxx, minz, maxz,
            north_deg=float(project.get("north_deg", 0.0)),
        )

    # ─── Title block (right strip) ──
    _draw_title_block(
        title_ax, project=project, session_id=session_id,
        ortho_path=_session_dir(session_id) / "out" / "ceiling.jpg",
        scale_ratio=scale_ratio, units=units,
    )

    # ─── Legends (bottom strip) ──
    _draw_legends(legend_ax, main=main, regions=regions,
                  obstructions=plan.get("obstructions", []), units=units)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="rcp_{session_id}.pdf"'},
    )


def _draw_north_arrow(ax, minx, maxx, minz, maxz, *, north_deg: float) -> None:
    """Top-right corner of the plan axes: 25 mm circle with a triangle
    pointing along ``north_deg`` (degrees CCW from page-up = +Z)."""
    from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon
    import numpy as _np
    # Place inset axes proportionally so the arrow doesn't move when the
    # plan resizes for different rooms.
    ix = ax.inset_axes([0.86, 0.86, 0.12, 0.12])
    ix.set_xlim(-1.2, 1.2)
    ix.set_ylim(-1.2, 1.2)
    ix.set_aspect("equal")
    ix.axis("off")
    theta = _np.deg2rad(90.0 - north_deg)
    # Outer circle.
    ix.add_patch(MplCircle((0, 0), 1.0, fill=False, edgecolor="#222",
                            linewidth=1.2))
    # Triangle pointing at the north direction.
    tip = (_np.cos(theta), _np.sin(theta))
    perp = (-_np.sin(theta) * 0.18, _np.cos(theta) * 0.18)
    base_back = (-tip[0] * 0.7, -tip[1] * 0.7)
    pts = [tip, (base_back[0] + perp[0], base_back[1] + perp[1]),
           (base_back[0] - perp[0], base_back[1] - perp[1])]
    ix.add_patch(MplPolygon(pts, closed=True, facecolor="#222"))
    ix.text(tip[0] * 1.15, tip[1] * 1.15, "N",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#222")


def _draw_title_scale(title_ax, *, scale_ratio: int, units: str,
                       y_top: float, y_bot: float, strip_w_in: float) -> None:
    """Draw the SCALE section inside the title strip: a label, the
    "1:N" ratio, and a graphic scale bar whose paper length matches the
    chosen scale exactly.

    ``y_top`` / ``y_bot`` bound the section in title-axis y-units.
    ``strip_w_in`` is the title strip's paper width in inches — needed
    to translate paper distance into title-axis x-fractions.
    """
    from matplotlib.patches import Rectangle
    section_h = y_top - y_bot
    # Heading row: "SCALE" label on the left, "1:N" value on the right.
    label_y = y_top - 0.005
    title_ax.text(0.06, label_y, "SCALE",
                   ha="left", va="top",
                   fontsize=7, fontweight="bold", color="#888")
    title_ax.text(0.94, label_y, f"1:{scale_ratio}",
                   ha="right", va="top",
                   fontsize=14, color="#222")

    # Graphic bar: marks chosen per ratio so the paper length is
    # roughly half-strip-wide and the unit increments read cleanly.
    if units == "imperial":
        marks = SCALE_BAR_MARKS_IMPERIAL.get(
            scale_ratio, SCALE_BAR_MARKS_IMPERIAL[48])
        unit_to_m = 0.3048
        unit_label = "ft"
    else:
        marks = SCALE_BAR_MARKS_METRIC.get(
            scale_ratio, SCALE_BAR_MARKS_METRIC[100])
        unit_to_m = 1.0
        unit_label = "m"
    total_world_m = marks[-1] * unit_to_m
    bar_paper_in = total_world_m / scale_ratio / 0.0254
    bar_w_frac = bar_paper_in / strip_w_in
    # Hard cap so very low ratios (1:20) don't overflow the strip.
    bar_w_frac = min(bar_w_frac, 0.88)
    bar_x0 = (1.0 - bar_w_frac) / 2
    bar_x1 = bar_x0 + bar_w_frac
    # Bar height: small fraction of the section, with a tick zone below.
    bar_h = section_h * 0.22
    bar_y0 = y_bot + section_h * 0.45
    bar_y1 = bar_y0 + bar_h

    # Alternating black/white segments between successive marks.
    for i in range(len(marks) - 1):
        seg_x0 = bar_x0 + bar_w_frac * marks[i] / marks[-1]
        seg_x1 = bar_x0 + bar_w_frac * marks[i + 1] / marks[-1]
        face = "#222" if i % 2 == 0 else "white"
        title_ax.add_patch(Rectangle(
            (seg_x0, bar_y0), seg_x1 - seg_x0, bar_h,
            facecolor=face, edgecolor="#222", linewidth=0.6,
        ))
    # Tick labels under each mark.
    for m_v in marks:
        x = bar_x0 + bar_w_frac * m_v / marks[-1]
        title_ax.plot([x, x], [bar_y0, bar_y0 - bar_h * 0.5],
                       color="#222", linewidth=0.6)
        title_ax.text(x, bar_y0 - bar_h * 0.7, f"{m_v}",
                       ha="center", va="top", fontsize=7, color="#222")
    # Unit suffix at the right end.
    title_ax.text(bar_x1 + 0.005, bar_y1 - bar_h * 0.5, unit_label,
                   ha="left", va="center", fontsize=7, color="#222")


def _draw_title_block(title_ax, *, project: dict, session_id: str,
                       ortho_path: Path, scale_ratio: int,
                       units: str) -> None:
    """Right-side title block: ortho thumbnail, project metadata, scale
    badge + bar, and drawing register table."""
    from matplotlib.patches import Rectangle
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    # Outer frame.
    title_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor="#222", linewidth=1.2))

    # Strip paper dimensions — needed by both the thumbnail aspect-fit
    # and the scale-bar paper-length calculation. Computed once.
    fig = title_ax.figure
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_bbox = title_ax.get_position()
    strip_w_in = ax_bbox.width * fig_w_in
    strip_h_in = ax_bbox.height * fig_h_in

    # Top: ortho thumbnail. The box is fixed; the image is letter-boxed
    # inside it so its natural aspect is preserved (no stretching).
    THUMB_TOP = 0.98
    THUMB_BOTTOM = 0.62
    THUMB_LEFT = 0.04
    THUMB_RIGHT = 0.96
    title_ax.add_patch(Rectangle(
        (THUMB_LEFT, THUMB_BOTTOM),
        THUMB_RIGHT - THUMB_LEFT, THUMB_TOP - THUMB_BOTTOM,
        fill=False, edgecolor="#888", linewidth=0.5,
    ))
    try:
        if ortho_path.exists():
            import matplotlib.image as mpimg
            img = mpimg.imread(str(ortho_path))
            ih, iw = img.shape[:2]
            box_w_in = strip_w_in * (THUMB_RIGHT - THUMB_LEFT)
            box_h_in = strip_h_in * (THUMB_TOP - THUMB_BOTTOM)
            img_aspect = iw / ih           # >1 = wide, <1 = tall
            box_aspect = box_w_in / box_h_in
            if img_aspect > box_aspect:
                # Image wider than box — fit by width, letterbox top/bottom.
                disp_w_ax = THUMB_RIGHT - THUMB_LEFT
                disp_h_in = box_w_in / img_aspect
                disp_h_ax = disp_h_in / strip_h_in
            else:
                # Image taller than box — fit by height, letterbox sides.
                disp_h_ax = THUMB_TOP - THUMB_BOTTOM
                disp_w_in = box_h_in * img_aspect
                disp_w_ax = disp_w_in / strip_w_in
            cx = (THUMB_LEFT + THUMB_RIGHT) / 2
            cy = (THUMB_BOTTOM + THUMB_TOP) / 2
            title_ax.imshow(
                img,
                extent=(cx - disp_w_ax / 2, cx + disp_w_ax / 2,
                        cy - disp_h_ax / 2, cy + disp_h_ax / 2),
                aspect="auto", zorder=1,
            )
    except Exception:
        pass
    title_ax.text(0.5, THUMB_BOTTOM - 0.015, "REFERENCE — ORTHO VIEW",
                  ha="center", va="top", fontsize=8, color="#888")

    # Project fields.
    fields = [
        ("PROJECT",        project.get("name", "") or "—"),
        ("ADDRESS",        project.get("address", "") or "—"),
        ("CLIENT",         project.get("client", "") or "—"),
        ("COMPANY",        project.get("company", "") or "—"),
        ("DRAWING NO.",    project.get("drawing_number", "") or "—"),
    ]
    field_top = 0.58
    field_h = 0.04
    for i, (k, v) in enumerate(fields):
        y_top = field_top - i * field_h
        y_bot = y_top - field_h
        title_ax.add_patch(Rectangle(
            (0.04, y_bot), 0.92, field_h, fill=False,
            edgecolor="#bbb", linewidth=0.5,
        ))
        title_ax.text(0.06, y_top - 0.005, k,
                       ha="left", va="top",
                       fontsize=7, fontweight="bold", color="#888")
        title_ax.text(0.06, y_bot + 0.008, v,
                       ha="left", va="bottom",
                       fontsize=14, color="#222")

    # Scale section (badge + graphic bar) — sits between the project
    # fields and the drawing register.
    scale_top = field_top - len(fields) * field_h - 0.02
    scale_bot = scale_top - 0.06
    title_ax.add_patch(Rectangle(
        (0.04, scale_bot), 0.92, scale_top - scale_bot,
        fill=False, edgecolor="#bbb", linewidth=0.5,
    ))
    _draw_title_scale(
        title_ax, scale_ratio=scale_ratio, units=units,
        y_top=scale_top, y_bot=scale_bot, strip_w_in=strip_w_in,
    )

    # Drawing register table.
    reg_top = scale_bot - 0.02
    reg_bot = 0.05
    title_ax.text(0.04, reg_top + 0.005, "DRAWING REGISTER",
                  ha="left", va="bottom",
                  fontsize=7, fontweight="bold", color="#888")
    cols = [("REV", 0.04, 0.14),
            ("DATE", 0.14, 0.34),
            ("BY", 0.34, 0.46),
            ("NOTE", 0.46, 0.96)]
    title_ax.add_patch(Rectangle(
        (0.04, reg_bot), 0.92, reg_top - reg_bot,
        fill=False, edgecolor="#bbb", linewidth=0.5,
    ))
    header_h = 0.025
    for label, x0, x1 in cols:
        title_ax.text((x0 + x1) / 2, reg_top - 0.005, label,
                       ha="center", va="top",
                       fontsize=7, fontweight="bold", color="#888")
    register = list(project.get("drawing_register") or [])
    rows = register[:14]  # whatever fits; rest spills off the page
    if not rows:
        title_ax.text(0.5, (reg_top + reg_bot) / 2,
                       "— no revisions recorded —",
                       ha="center", va="center",
                       fontsize=8, color="#aaa", style="italic")
    else:
        row_h = (reg_top - reg_bot - header_h) / max(len(rows), 1)
        for i, row in enumerate(rows):
            y_top = reg_top - header_h - i * row_h
            for label, x0, x1 in cols:
                v = (row.get(label.lower().rstrip(".") if label != "NOTE" else "note", "") or "")
                title_ax.text(
                    x0 + 0.005 if label != "REV" else (x0 + x1) / 2,
                    y_top - 0.005,
                    str(v),
                    ha="left" if label != "REV" else "center",
                    va="top",
                    fontsize=8, color="#222",
                )

    title_ax.text(0.5, 0.025,
                  f"session {session_id}",
                  ha="center", va="center",
                  fontsize=6, color="#aaa", style="italic")


def _draw_legends(legend_ax, *, main: dict | None, regions: list,
                  obstructions: list, units: str) -> None:
    """Bottom strip: ceiling-zone legend (one row per face) + structural
    legend (column hatch swatch) + services placeholder."""
    from matplotlib.patches import Rectangle
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                                   edgecolor="#222", linewidth=1.2))

    # Layout: zone column (60%) + structural column (20%) + services column (20%)
    legend_ax.text(0.005, 0.96, "CEILING ZONES",
                   ha="left", va="top",
                   fontsize=8, fontweight="bold", color="#888")

    rows = []
    if main:
        rows.append({
            "tint": main.get("tint", MAIN_TINT),
            "label": main.get("label", "Main Ceiling (1)"),
            "rel": format_height_delta(0.0, units),
            "notes": main.get("notes", ""),
        })
    for r in regions:
        rel = r.get("relative_y")
        rel_text = "—" if rel is None else format_height_delta(float(rel), units)
        rows.append({
            "tint": r.get("tint", "#ff7043"),
            "label": r.get("label", f"region {r['id']}"),
            "rel": rel_text,
            "notes": r.get("notes", ""),
        })

    # Two columns of zone rows so wide rooms with many zones still fit.
    col_x = [0.01, 0.31]
    row_h = 0.10
    for i, row in enumerate(rows):
        col = i // 7
        if col >= 2:
            break
        idx = i % 7
        x = col_x[col]
        y = 0.86 - idx * row_h
        legend_ax.add_patch(Rectangle(
            (x + 0.01, y - 0.04), 0.025, 0.05,
            facecolor=_muted_fill(row["tint"]),
            edgecolor="#222", linewidth=0.5,
        ))
        legend_ax.text(x + 0.045, y, f"{row['label']}",
                       ha="left", va="top", fontsize=8, color="#222")
        legend_ax.text(x + 0.045, y - 0.025,
                       f"{row['rel']}{('  ' + row['notes']) if row['notes'] else ''}",
                       ha="left", va="top", fontsize=7, color="#555")

    # Structural column (columns + future structural callouts).
    legend_ax.text(0.605, 0.96, "STRUCTURAL",
                   ha="left", va="top",
                   fontsize=8, fontweight="bold", color="#888")
    if obstructions:
        legend_ax.add_patch(Rectangle(
            (0.61, 0.78), 0.025, 0.05,
            facecolor="white", edgecolor="#222", linewidth=0.5,
            hatch="xx",
        ))
        legend_ax.text(0.645, 0.83,
                       f"Columns ({len(obstructions)})",
                       ha="left", va="top", fontsize=8, color="#222")
        legend_ax.text(0.645, 0.81,
                       "structural negative space",
                       ha="left", va="top", fontsize=7, color="#555")
    else:
        legend_ax.text(0.645, 0.83, "— no columns drawn —",
                       ha="left", va="top", fontsize=8, color="#aaa",
                       style="italic")

    # Services placeholder — we'll fill this in once light detection lands.
    legend_ax.text(0.805, 0.96, "SERVICES",
                   ha="left", va="top",
                   fontsize=8, fontweight="bold", color="#888")
    legend_ax.add_patch(Rectangle(
        (0.81, 0.05), 0.18, 0.85,
        fill=False, edgecolor="#bbb", linewidth=0.4,
    ))
    legend_ax.text(0.9, 0.475,
                   "TBD\n(lights, diffusers,\ndownlights…)",
                   ha="center", va="center",
                   fontsize=8, color="#aaa", style="italic")


# ─── EXPORT ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/export")
async def api_export(session_id: str) -> JSONResponse:
    return JSONResponse(_load_plan(session_id))


# ─── STATIC FRONTEND ──────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="ceiling-rcp-server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    import uvicorn
    uvicorn.run(
        "ceiling_rcp.server:app",
        host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":
    main()
