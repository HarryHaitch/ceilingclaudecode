"""Offline / debug entrypoint.

Runs the full pipeline against a scan folder and writes:

  out_<scan_name>/
    ceiling.jpg              raw textured top-down render (downward faces)
    ceiling_planes.jpg       same render with per-cluster region overlay + labels
    footprint.png            binary footprint mask
    plan.json                heights, regions (as polygons), composites, footprint
    report.txt               folder validation summary
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2

from .mesh import inspect_folder, load_mesh
from .planes import detect_periodic_composite, segment_ceiling
from .polygons import footprint_polygon, region_polygon
from .raster import annotate_heights, overlay_regions, render_textured_topdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ceiling-rcp")
    p.add_argument("scan_dir", type=Path,
                   help="Polycam scan folder (or any folder containing .obj + .mtl + textures + mesh_info.json)")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Override output directory (default: ./out_<scan_name>/)")
    p.add_argument("--ppm", type=int, default=200, help="pixels per metre")
    p.add_argument("--max_tilt_deg", type=float, default=30.0)
    p.add_argument("--band_tol_m", type=float, default=0.05)
    p.add_argument("--bin_size_m", type=float, default=0.005)
    p.add_argument("--min_band_area_m2", type=float, default=0.5)
    p.add_argument("--min_region_area_m2", type=float, default=0.25)
    p.add_argument("--no_render", action="store_true",
                   help="Skip the textured render (much faster)")
    args = p.parse_args(argv)

    scan_dir = args.scan_dir.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else Path.cwd() / f"out_{scan_dir.name.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = inspect_folder(scan_dir)

    report_lines = [f"scan_dir: {scan_dir}", f"ok: {rep.ok}",
                    f"obj: {rep.obj}", f"mtl: {rep.mtl}",
                    f"mesh_info: {rep.mesh_info}",
                    f"textures_found: {len(rep.textures_found)}",
                    f"textures_missing: {rep.textures_missing}",
                    "warnings:"]
    report_lines += [f"  - {w}" for w in rep.warnings]
    report_lines.append("errors:")
    report_lines += [f"  - {e}" for e in rep.errors]
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))

    if not rep.ok:
        return 2

    mesh = load_mesh(rep)
    print(f"\nMesh: {mesh.n_verts} verts, {mesh.n_tris} tris")

    seg = segment_ceiling(
        mesh,
        max_tilt_deg=args.max_tilt_deg,
        pixels_per_metre=args.ppm,
        band_tol_m=args.band_tol_m,
        bin_size_m=args.bin_size_m,
        min_band_area_m2=args.min_band_area_m2,
        min_region_area_m2=args.min_region_area_m2,
    )
    print(f"Clusters: {len(seg.clusters)}")
    for ci, c in enumerate(seg.clusters):
        print(f"  [{ci}] y={c.y:.3f}  area={c.area:.2f} m^2")
    print(f"Regions: {len(seg.regions)}")

    # Periodic-composite scan: pair regions whose XZ bboxes overlap.
    composites = []
    for a_idx, b_idx in combinations(range(len(seg.regions)), 2):
        a = seg.regions[a_idx]
        b = seg.regions[b_idx]
        if a.cluster_idx == b.cluster_idx:
            continue
        # Always pass lower-Y first as `lower` for naming consistency.
        lower, upper = (a, b) if a.y < b.y else (b, a)
        meta = detect_periodic_composite(lower, upper, seg.grid)
        if meta:
            composites.append({
                "lower_region": a_idx if lower is a else b_idx,
                "upper_region": b_idx if lower is a else a_idx,
                **meta,
            })
    print(f"Periodic composites detected: {len(composites)}")
    for c in composites:
        print(f"  {c}")

    # Render
    if not args.no_render:
        print("Rendering textured top-down… (slow)")
        canvas, _zbuf = render_textured_topdown(
            mesh, seg.down_face_indices, seg.grid,
        )
        cv2.imwrite(str(out_dir / "ceiling.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        overlay = overlay_regions(canvas, seg.regions, seg.grid)
        overlay = annotate_heights(overlay, seg.regions)
        cv2.imwrite(str(out_dir / "ceiling_planes.jpg"), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    cv2.imwrite(str(out_dir / "footprint.png"), seg.footprint_mask)

    # Plan JSON
    plan = {
        "scan_dir": str(scan_dir),
        "grid": {
            "min_x": seg.grid.min_x, "max_x": seg.grid.max_x,
            "min_z": seg.grid.min_z, "max_z": seg.grid.max_z,
            "pixels_per_metre": seg.grid.pixels_per_metre,
            "width": seg.grid.width, "height": seg.grid.height,
        },
        "clusters": [
            {"y": c.y, "y_lo": c.y_lo, "y_hi": c.y_hi, "area_m2": c.area}
            for c in seg.clusters
        ],
        "regions": [
            {
                "id": i,
                "cluster_idx": r.cluster_idx,
                "y": r.y,
                "area_m2": r.area_m2,
                "polygon": region_polygon(r, seg.grid),
            }
            for i, r in enumerate(seg.regions)
        ],
        "composites": composites,
        "footprint": footprint_polygon(seg.footprint_mask, seg.grid),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
