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
from .mesh import inspect_folder, load_mesh, downward_face_mask
from .planes import PlanGrid, make_grid
from .raster import render_textured_topdown


# ─── PATHS ────────────────────────────────────────────────────────────────────

SESSIONS_DIR = Path.cwd() / "sessions"
STATIC_DIR = Path(__file__).parent / "static"


def _session_dir(session_id: str) -> Path:
    p = SESSIONS_DIR / session_id
    if not p.exists():
        raise HTTPException(404, f"unknown session {session_id}")
    return p


def _load_plan(session_id: str) -> dict[str, Any]:
    sd = _session_dir(session_id)
    plan_path = sd / "out" / "plan.json"
    if not plan_path.exists():
        raise HTTPException(409, "session not processed yet")
    return json.loads(plan_path.read_text())


def _save_plan(session_id: str, plan: dict[str, Any]) -> None:
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

def process_session(session_id: str, *, ppm: int = 150) -> dict[str, Any]:
    """Render the textured top-down + height map from an uploaded scan.

    No automatic segmentation — the user draws polygons by hand.
    """
    sd = _session_dir(session_id)
    upload = sd / "upload"
    out = sd / "out"
    out.mkdir(parents=True, exist_ok=True)

    rep = inspect_folder(upload)
    plan: dict[str, Any] = {
        "session_id": session_id,
        "report": {
            "ok": rep.ok,
            "obj": str(rep.obj) if rep.obj else None,
            "mtl": str(rep.mtl) if rep.mtl else None,
            "mesh_info": str(rep.mesh_info) if rep.mesh_info else None,
            "textures_found": len(rep.textures_found),
            "textures_missing": rep.textures_missing,
            "warnings": rep.warnings,
            "errors": rep.errors,
        },
        "room": None,
        "main": None,
        "regions": [],
    }

    if not rep.ok:
        (out / "plan.json").write_text(json.dumps(plan, indent=2))
        return plan

    mesh = load_mesh(rep)
    normals = mesh.face_normals()
    down = downward_face_mask(normals, max_tilt_deg=30.0)
    down_idx = np.where(down)[0]

    grid = make_grid(mesh, down_idx, pixels_per_metre=ppm)
    canvas, zbuf = render_textured_topdown(mesh, down_idx, grid)
    cv2.imwrite(str(out / "ceiling.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    height_map = height_map_to_storage(zbuf)
    np.save(out / "height.npy", height_map)

    plan["grid"] = {
        "min_x": grid.min_x, "max_x": grid.max_x,
        "min_z": grid.min_z, "max_z": grid.max_z,
        "pixels_per_metre": grid.pixels_per_metre,
        "width": grid.width, "height": grid.height,
    }
    valid = ~np.isnan(height_map)
    if valid.any():
        plan["height_summary"] = {
            "min_y": float(np.nanmin(height_map)),
            "max_y": float(np.nanmax(height_map)),
            "median_y": float(np.nanmedian(height_map)),
            "valid_px": int(valid.sum()),
            "total_px": int(height_map.size),
        }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    return plan


# ─── ANALYSIS HELPERS ─────────────────────────────────────────────────────────

def _analyse_and_pack(
    session_id: str, polygon: list[list[float]] | None = None,
    *, mask: np.ndarray | None = None,
    stats_mask: np.ndarray | None = None,
    range_m: float = 0.05, tint: str = "#80cbc4",
) -> dict:
    """Compute mean Y + tinted deviation heatmap.

    Either ``polygon`` or ``mask`` defines the *visual* coverage. If a
    separate ``stats_mask`` is supplied, the mean / σ / range stats are
    computed from it instead — useful for auto-detect where we want
    stats to ignore wrongly-absorbed cross-cluster pixels even though
    they remain in the visual polygon.
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
        from .analyse import PolygonAnalysis
        stats = PolygonAnalysis(
            mean_y=float(vals.mean()),
            std_y=float(vals.std()),
            valid_frac=float(valid.sum()) / float(n_total),
            n_valid_px=int(valid.sum()),
            n_total_px=n_total,
            min_y=float(vals.min()),
            max_y=float(vals.max()),
        )
        png_bytes, bbox = _heatmap_from_mask(
            mask, height_map, mean_y=stats.mean_y,
            range_m=range_m, tint=tint,
        )

    return {
        "stats": stats.as_dict(),
        "heatmap_bbox_px": list(bbox),
        "heatmap_png_b64": base64.b64encode(png_bytes).decode("ascii") if png_bytes else "",
        "heatmap_range_m": range_m,
        "tint": tint,
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


# ─── MAIN CEILING POLYGON ─────────────────────────────────────────────────────

@app.put("/api/sessions/{session_id}/main")
async def api_set_main(session_id: str, payload: dict = Body(...)) -> dict:
    """Set the main ceiling polygon. The mean Y of valid pixels inside it
    becomes the room's height datum (relative_y = 0). All other regions'
    relative heights are recomputed against the new datum."""
    plan = _load_plan(session_id)
    poly = payload.get("polygon")
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
    # Recompute relative_y on regions
    datum = (plan.get("main") or {}).get("stats", {}).get("mean_y")
    for r in plan.get("regions", []):
        m = r.get("stats", {}).get("mean_y")
        r["relative_y"] = (m - datum) if (m is not None and datum is not None) else None
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
    _save_plan(session_id, plan)
    return {"ok": True, "region": region}


@app.put("/api/sessions/{session_id}/region/{region_id}")
async def api_update_region(
    session_id: str, region_id: int, payload: dict = Body(...),
) -> dict:
    plan = _load_plan(session_id)
    for r in plan.get("regions", []):
        if r["id"] == region_id:
            if "polygon" in payload:
                analysis = _analyse_and_pack(
                    session_id, payload["polygon"],
                    range_m=float(payload.get("range_m", r.get("heatmap_range_m", 0.05))),
                    tint=_region_tint(region_id),
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
            _save_plan(session_id, plan)
            return {"ok": True, "region": r}
    raise HTTPException(404, f"unknown region {region_id}")


@app.put("/api/sessions/{session_id}/main/notes")
async def api_main_notes(session_id: str, payload: dict = Body(...)) -> dict:
    plan = _load_plan(session_id)
    if not plan.get("main"):
        raise HTTPException(409, "main not set")
    plan["main"]["notes"] = str(payload.get("notes", ""))[:500]
    _save_plan(session_id, plan)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}/region/{region_id}")
async def api_delete_region(session_id: str, region_id: int) -> dict:
    plan = _load_plan(session_id)
    plan["regions"] = [r for r in plan.get("regions", []) if r["id"] != region_id]
    _save_plan(session_id, plan)
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
    _save_plan(session_id, plan)
    return plan


@app.post("/api/sessions/{session_id}/snap")
async def api_snap(session_id: str) -> dict:
    """Push/pull polygon borders so they share clean edges and tile the
    room without gaps or overlap.

    Voronoi-style nearest-polygon assignment:

    1. Rasterise the room outline + every polygon (main, regions) onto
       the same pixel grid as the height map.
    2. For each pixel inside the room, compute the Euclidean distance
       to each polygon and assign that pixel to the nearest one. This
       grows polygons into the black gaps between them and stops them
       at the midline between neighbours.
    3. Each polygon's claimed pixel set is morphologically closed to
       smooth jaggies, then traced and simplified back to a vertex
       polygon (≈5 cm tolerance).
    4. Re-run the height analysis on every snapped polygon so the
       heatmap and σ reflect the new shape.

    Polygon ids and labels are preserved.
    """
    from .analyse import polygon_to_mask
    from .polygons import mask_to_polygon

    plan = _load_plan(session_id)
    room_pts = plan.get("room")
    if not room_pts or len(room_pts) < 3:
        raise HTTPException(400, "room outline required before snapping")

    height_map, grid = _load_height_map(session_id)
    room_mask = polygon_to_mask([tuple(p) for p in room_pts], grid) > 0
    if not room_mask.any():
        raise HTTPException(409, "room polygon doesn't overlap the rendered area")

    polygons: list[dict] = []
    if plan.get("main"):
        polygons.append({
            "key": "main",
            "polygon": plan["main"]["polygon"],
            "label": plan["main"].get("label", "Main Ceiling (1)"),
            "tint": MAIN_TINT,
            "notes": plan["main"].get("notes", ""),
        })
    for r in plan.get("regions", []):
        polygons.append({
            "key": f"region:{r['id']}",
            "id": int(r["id"]),
            "polygon": r["polygon"],
            "label": r.get("label", f"Ceiling Region ({int(r['id']) + 2})"),
            "tint": _region_tint(int(r["id"])),
            "notes": r.get("notes", ""),
        })
    if not polygons:
        raise HTTPException(409, "draw a main ceiling and any regions before snapping")

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
        if not p["key"].startswith("region:"):
            continue
        assignment[drawn_masks[i]] = i

    # Stage 2: main claims its drawn pixels that no region took.
    main_idx = next(
        (i for i, p in enumerate(polygons) if p["key"] == "main"),
        -1,
    )
    if main_idx >= 0:
        main_drawn = drawn_masks[main_idx]
        free = main_drawn & (assignment == -1)
        assignment[free] = main_idx

    # Stage 3: any pixel still inside the room but unassigned grows the
    # closest polygon out to fill it (Voronoi between drawn shapes).
    gap = room_mask & (assignment == -1)
    if gap.any():
        nearest = np.argmin(dist_stack, axis=0)
        assignment[gap] = nearest[gap]

    # Pixels outside the room belong to nobody.
    assignment_room = np.where(room_mask, assignment, -1)

    px_per_m = grid.pixels_per_metre
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(0.06 * px_per_m)) | 1),) * 2,
    )

    snapped_polys: dict[str, list[list[float]]] = {}
    for i, p in enumerate(polygons):
        claim = (assignment_room == i).astype(np.uint8) * 255
        if claim.sum() == 0:
            continue
        claim = cv2.morphologyEx(claim, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        coords = mask_to_polygon(claim, grid, simplify_m=0.05)
        if len(coords) >= 3:
            snapped_polys[p["key"]] = [[float(x), float(z)] for x, z in coords]

    # Re-analyse and rebuild plan
    main_coords = snapped_polys.get("main")
    if main_coords is None:
        raise HTTPException(409, "main ceiling lost all coverage after snapping")
    main_analysis = _analyse_and_pack(session_id, main_coords, tint=MAIN_TINT)
    plan["main"] = {
        "polygon": main_coords,
        "label": polygons[0]["label"],
        "notes": polygons[0]["notes"],
        **main_analysis,
    }
    datum = main_analysis["stats"]["mean_y"]

    new_regions = []
    for p in polygons:
        if not p["key"].startswith("region:"):
            continue
        coords = snapped_polys.get(p["key"])
        if coords is None:
            continue
        rid = int(p["id"])
        analysis = _analyse_and_pack(session_id, coords, tint=_region_tint(rid))
        m = analysis["stats"]["mean_y"]
        new_regions.append({
            "id": rid,
            "label": p["label"],
            "notes": p["notes"],
            "polygon": coords,
            "relative_y": (m - datum) if (m is not None and datum is not None) else None,
            **analysis,
        })
    plan["regions"] = new_regions
    plan["snapped"] = True
    _save_plan(session_id, plan)
    return plan


# ─── PDF EXPORT ───────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/pdf")
async def api_pdf(session_id: str) -> Response:
    """Produce an architectural PDF: coloured masks (no heatmap), each tagged
    with its label and relative height, and dimension lines along every
    edge of the room outline."""
    plan = _load_plan(session_id)
    if not plan.get("room"):
        raise HTTPException(400, "room required for PDF")

    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    room = plan["room"]
    main = plan.get("main")
    regions = plan.get("regions", [])

    xs = [p[0] for p in room]
    zs = [p[1] for p in room]
    pad = 0.5
    minx, maxx = min(xs) - pad, max(xs) + pad
    minz, maxz = min(zs) - pad, max(zs) + pad
    width_m = maxx - minx
    height_m = maxz - minz
    aspect = width_m / max(height_m, 1e-6)

    fig_w = 11.0
    fig_h = max(8.5, fig_w / aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    # RCP convention: mirror X so the plan reads with floor-plan handedness.
    ax.set_xlim(maxx, minx)
    ax.set_ylim(minz, maxz)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _draw_poly(poly_pts, *, face, edge, label, rel_text, notes, alpha=0.45):
        if not poly_pts:
            return
        pts = [(p[0], p[1]) for p in poly_pts]
        patch = MplPolygon(pts, closed=True, facecolor=face, edgecolor=edge,
                           linewidth=1.5, alpha=alpha)
        ax.add_patch(patch)
        cx = sum(p[0] for p in pts) / len(pts)
        cz = sum(p[1] for p in pts) / len(pts)
        text = f"{label}\n{rel_text}"
        if notes:
            text += f"\n{notes}"
        ax.text(cx, cz, text,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=edge, linewidth=0.6))

    if main:
        _draw_poly(main["polygon"],
                   face=MAIN_TINT, edge="#00897b",
                   label=main.get("label", "Main Ceiling (1)"),
                   rel_text="0 mm",
                   notes=main.get("notes", ""))

    for r in regions:
        rel = r.get("relative_y")
        if rel is None:
            rel_text = "—"
        else:
            sign = "+" if rel >= 0 else "−"
            rel_text = f"{sign}{abs(rel) * 1000:.0f} mm"
        _draw_poly(r["polygon"],
                   face=r.get("tint", "#ff7043"),
                   edge="#444",
                   label=r.get("label", f"region {r['id']}"),
                   rel_text=rel_text,
                   notes=r.get("notes", ""))

    # Room outline + dimension lines
    rxs = [p[0] for p in room] + [room[0][0]]
    rzs = [p[1] for p in room] + [room[0][1]]
    ax.plot(rxs, rzs, color="#222", linewidth=2.0, linestyle="--")

    n = len(room)
    for i in range(n):
        ax_, az_ = room[i]
        bx, bz = room[(i + 1) % n]
        midx, midz = 0.5 * (ax_ + bx), 0.5 * (az_ + bz)
        dx, dz = bx - ax_, bz - az_
        L = (dx ** 2 + dz ** 2) ** 0.5
        if L < 1e-3:
            continue
        # Offset the label slightly outside the polygon along the edge normal.
        nx, nz = -dz / L, dx / L
        offset = 0.18
        # Rough "outside" sense by checking against the polygon centroid
        cx = sum(p[0] for p in room) / n
        cz = sum(p[1] for p in room) / n
        if (midx - cx) * nx + (midz - cz) * nz < 0:
            nx, nz = -nx, -nz
        tx, tz = midx + nx * offset, midz + nz * offset
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
        ax.text(tx, tz, f"{L * 1000:.0f} mm",
                ha="center", va="center", rotation=angle,
                fontsize=7, color="#222",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="#aaa", linewidth=0.4))

    title = f"Reflected Ceiling Plan — session {session_id}"
    ax.set_title(title, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="rcp_{session_id}.pdf"'},
    )


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
