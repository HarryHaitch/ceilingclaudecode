"""Ceiling-services symbol model — class definitions, classification rules,
and (de)serialisation for the per-session ``symbols.json`` document.

The same module is the source of truth for the upstream segmentation
experiment (``experiments/segmentation_projection``) and the main RCP
viewer. Both load a symbols.json with this shape:

    {
      "version": 1,
      "session_id": "...",
      "ceiling_image": "out/ceiling.jpg" (relative path, optional),
      "grid": { width, height, pixels_per_metre, min_x, max_x, min_z, max_z },
      "symbol_classes": { ...SYMBOL_CLASSES... },
      "thresholds": { downlight_max_diameter_cm, sprinkler_max_diameter_cm },
      "symbols": [
        { id, class, index, centroid_px[u,v], length_px, width_px,
          angle_deg, source: { cluster_id, concept, ... } },
        ...
      ]
    }

The viewer treats this file as opaque metadata: it doesn't need a
``ceiling_image`` reference (the session has its own), and it tolerates
new ``symbol_classes`` keys it doesn't know about (rendered with the
class's declared shape + colour, falling back to a generic dot).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .planes import PlanGrid


# ─── CLASS DEFINITIONS ────────────────────────────────────────────────────

# Default symbol classes. Persisted into every symbols.json so the viewer
# (and a future hand-edit) can read shape/colour without depending on
# this code being available.
SYMBOL_CLASSES: dict[str, dict] = {
    "diffuser": {
        "label": "Diffuser",
        "shape": "rect_with_cross",
        "color_hex": "#40c8ff",
        "fixed_diameter_cm": None,
        "default_size_cm": [30.0, 30.0],
        "source_concept": "vent",
    },
    "downlight": {
        "label": "Downlight",
        "shape": "circle",
        "color_hex": "#ffdc40",
        "fixed_diameter_cm": 20.0,
        "default_size_cm": [20.0, 20.0],
        "source_concept": "light",
    },
    "led_strip_panel": {
        "label": "LED Strip/Panel Light",
        "shape": "rect",
        "color_hex": "#ff9a30",
        "fixed_diameter_cm": None,
        "default_size_cm": [60.0, 60.0],
        "source_concept": "light",
    },
    "sprinkler": {
        "label": "Sprinkler",
        "shape": "dot_in_circle",
        "color_hex": "#ff4040",
        "fixed_diameter_cm": 12.0,
        "fixed_inner_dot_diameter_cm": 5.0,
        "default_size_cm": [12.0, 12.0],
        "source_concept": "ceiling item",
    },
    "miscellaneous": {
        "label": "Miscellaneous",
        "shape": "rect_with_M",
        "color_hex": "#c060ff",
        "fixed_diameter_cm": None,
        "default_size_cm": [30.0, 30.0],
        "source_concept": "ceiling item",
    },
}

DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM = 20.0
DEFAULT_SPRINKLER_MAX_DIAMETER_CM = 5.0
SCHEMA_VERSION = 1


# ─── SYMBOL DATACLASS ─────────────────────────────────────────────────────

@dataclass
class Symbol:
    id: int
    cls: str                                # key in SYMBOL_CLASSES
    index: int                              # per-class numbering (Downlight 1..N)
    centroid_px: tuple[float, float]        # ortho pixel (u, v)
    length_px: float
    width_px: float
    angle_deg: float
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "class": str(self.cls),
            "index": int(self.index),
            "centroid_px": [float(self.centroid_px[0]), float(self.centroid_px[1])],
            "length_px": float(self.length_px),
            "width_px": float(self.width_px),
            "angle_deg": float(self.angle_deg),
            "source": dict(self.source),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Symbol":
        return cls(
            id=int(d["id"]),
            cls=str(d["class"]),
            index=int(d.get("index", 0)),
            centroid_px=(float(d["centroid_px"][0]), float(d["centroid_px"][1])),
            length_px=float(d["length_px"]),
            width_px=float(d["width_px"]),
            angle_deg=float(d.get("angle_deg", 0.0)),
            source=dict(d.get("source", {})),
        )


# ─── CLUSTER → SYMBOL ─────────────────────────────────────────────────────

def classify_cluster(
    cluster: dict, concept: str,
    *,
    downlight_max_d_cm: float = DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM,
    sprinkler_max_d_cm: float = DEFAULT_SPRINKLER_MAX_DIAMETER_CM,
) -> str:
    """Map a (cluster, concept) pair onto a symbol-class key. The size
    metric is the median per-frame mask diameter (real-world cm), with a
    fallback to the cluster footprint mean if the per-frame metric is
    absent (older summaries)."""
    mean_mask_d_cm = float(cluster.get("mean_mask_diameter_cm", 0.0))
    if mean_mask_d_cm <= 0:
        width_m = float(cluster.get("width_m", 0.0))
        length_m = float(cluster.get("length_m", 0.0))
        mean_mask_d_cm = (width_m + length_m) * 50.0

    if concept == "vent":
        return "diffuser"
    if concept == "light":
        return "downlight" if mean_mask_d_cm < downlight_max_d_cm else "led_strip_panel"
    if concept == "ceiling item":
        return "sprinkler" if mean_mask_d_cm < sprinkler_max_d_cm else "miscellaneous"
    return "miscellaneous"


def symbols_from_clusters(
    cluster_summary: dict, *,
    downlight_max_d_cm: float = DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM,
    sprinkler_max_d_cm: float = DEFAULT_SPRINKLER_MAX_DIAMETER_CM,
) -> list[Symbol]:
    """Walk every cluster instance and produce the default-classified
    symbol list with per-class indices."""
    out: list[Symbol] = []
    next_id = 1
    class_idx: dict[str, int] = {k: 0 for k in SYMBOL_CLASSES}

    concepts = cluster_summary.get("concepts", {})
    for concept, info in concepts.items():
        for inst in info.get("instances", []):
            cls = classify_cluster(
                inst, concept,
                downlight_max_d_cm=downlight_max_d_cm,
                sprinkler_max_d_cm=sprinkler_max_d_cm,
            )
            class_idx[cls] += 1
            rr = inst.get("rotated_rect", {})
            out.append(Symbol(
                id=next_id,
                cls=cls,
                index=class_idx[cls],
                centroid_px=(
                    float(inst["centroid_px"][0]),
                    float(inst["centroid_px"][1]),
                ),
                length_px=float(rr.get("h_px", 0.0)),
                width_px=float(rr.get("w_px", 0.0)),
                angle_deg=float(rr.get("angle_deg", 0.0)),
                source=dict(
                    cluster_id=int(inst.get("id", 0)),
                    concept=concept,
                    n_views=int(inst.get("n_views", 0)),
                    median_score=float(inst.get("median_score", 0.0)),
                    width_m=float(inst.get("width_m", 0.0)),
                    length_m=float(inst.get("length_m", 0.0)),
                    mean_mask_diameter_cm=float(inst.get("mean_mask_diameter_cm", 0.0)),
                    # Per-frame contributing polygons in the rotated
                    # 768 × 1024 SAM input coords (= same coords as the
                    # saved per_frame/<id>/input.jpg). The viewer's
                    # thumbnail endpoint reads these to outline the
                    # detection on its source photo. Empty list when
                    # the cluster summary doesn't carry this field
                    # (older runs predating the depth-projection rev).
                    per_frame_polygons=list(inst.get("per_frame_polygons") or []),
                    centroid_xz_m=[float(v) for v in (inst.get("centroid_xz_m") or [0.0, 0.0])],
                ),
            ))
            next_id += 1
    return out


# ─── (DE)SERIALISE ────────────────────────────────────────────────────────

def serialise(
    symbols: list[Symbol], *,
    grid: PlanGrid,
    session_id: str,
    ceiling_image_relpath: str = "",
) -> dict:
    """Build the canonical symbols.json document."""
    return {
        "version": SCHEMA_VERSION,
        "session_id": str(session_id),
        "ceiling_image": str(ceiling_image_relpath),
        "grid": {
            "width": int(grid.width),
            "height": int(grid.height),
            "pixels_per_metre": float(grid.pixels_per_metre),
            "min_x": float(grid.min_x), "max_x": float(grid.max_x),
            "min_z": float(grid.min_z), "max_z": float(grid.max_z),
        },
        "symbol_classes": SYMBOL_CLASSES,
        "thresholds": {
            "downlight_max_diameter_cm": DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM,
            "sprinkler_max_diameter_cm": DEFAULT_SPRINKLER_MAX_DIAMETER_CM,
        },
        "symbols": [s.to_dict() for s in symbols],
    }


def empty_doc(*, grid: PlanGrid, session_id: str) -> dict:
    """Empty symbols document — used as the template the segmentation
    pipeline fills in, and as a fallback for sessions that haven't run
    detection yet."""
    return serialise([], grid=grid, session_id=session_id)


def load_symbols_doc(path: Path) -> dict:
    """Read and lightly validate a symbols.json file."""
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or "symbols" not in doc:
        raise ValueError(f"{path}: not a symbols document (no 'symbols' key)")
    doc.setdefault("version", SCHEMA_VERSION)
    doc.setdefault("symbol_classes", SYMBOL_CLASSES)
    doc.setdefault("thresholds", {
        "downlight_max_diameter_cm": DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM,
        "sprinkler_max_diameter_cm": DEFAULT_SPRINKLER_MAX_DIAMETER_CM,
    })
    return doc


def load_symbols(path: Path) -> list[Symbol]:
    return [Symbol.from_dict(d) for d in load_symbols_doc(path).get("symbols", [])]


def class_counts(symbols: list[Symbol]) -> dict[str, int]:
    out: dict[str, int] = {k: 0 for k in SYMBOL_CLASSES}
    for s in symbols:
        if s.cls in out:
            out[s.cls] += 1
    return out


def doc_class_counts(doc: dict) -> dict[str, int]:
    classes = list((doc.get("symbol_classes") or SYMBOL_CLASSES).keys())
    out = {k: 0 for k in classes}
    for s in doc.get("symbols", []):
        k = s.get("class")
        if k in out:
            out[k] += 1
        else:
            out[k] = 1  # tolerate unknown classes
    return out


def grid_from_doc(doc: dict) -> PlanGrid | None:
    g = doc.get("grid")
    if not g:
        return None
    return PlanGrid(
        min_x=float(g["min_x"]), max_x=float(g["max_x"]),
        min_z=float(g["min_z"]), max_z=float(g["max_z"]),
        pixels_per_metre=float(g["pixels_per_metre"]),
    )


def grid_matches_plan(doc: dict, plan: dict[str, Any]) -> bool:
    """True when the symbols.json grid matches the session plan grid
    closely enough that pixel coordinates can be used as-is. The
    ortho is re-rendered when scan settings change, so the grid in
    plan.json is authoritative — a mismatch means symbols would
    appear shifted/wrong-scale and the user needs to regenerate."""
    g1 = doc.get("grid") or {}
    g2 = plan.get("grid") or {}
    if not g1 or not g2:
        return False
    keys = ("min_x", "max_x", "min_z", "max_z", "pixels_per_metre",
            "width", "height")
    for k in keys:
        if k not in g1 or k not in g2:
            return False
        if abs(float(g1[k]) - float(g2[k])) > 1e-6:
            return False
    return True


__all__ = [
    "SYMBOL_CLASSES",
    "DEFAULT_DOWNLIGHT_MAX_DIAMETER_CM",
    "DEFAULT_SPRINKLER_MAX_DIAMETER_CM",
    "SCHEMA_VERSION",
    "Symbol",
    "classify_cluster",
    "symbols_from_clusters",
    "serialise",
    "empty_doc",
    "load_symbols",
    "load_symbols_doc",
    "class_counts",
    "doc_class_counts",
    "grid_from_doc",
    "grid_matches_plan",
]
