"""OBJ/MTL parsing, Polycam alignment, and folder validation.

A Polycam Space export is a folder that contains at least:
  - one .obj mesh
  - one .mtl material library referenced by the .obj
  - one or more textures (PNG/JPG) referenced by the .mtl
  - mesh_info.json with `alignmentTransform` (column-major 4x4)

The OBJ lives in axis-aligned mesh space; the inverse of
`alignmentTransform` brings vertices into ARKit world space (Y up,
floor near 0). See docs/polycam_coordinate_system.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ─── DATA ─────────────────────────────────────────────────────────────────────

@dataclass
class Mesh:
    """A textured triangle mesh in ARKit-style world space (Y up)."""

    V: np.ndarray            # (n_verts, 3) world-space vertex positions
    VT: np.ndarray           # (n_uvs, 2)   per-corner UV coords
    FV: np.ndarray           # (n_tris, 3)  vertex indices per triangle
    FT: np.ndarray           # (n_tris, 3)  UV indices per triangle
    FM: np.ndarray           # (n_tris,)    material index per triangle
    mat_names: list[str]
    mat_textures: list[Path | None]   # absolute path to each material's diffuse map
    source_obj: Path

    @property
    def n_tris(self) -> int:
        return int(self.FV.shape[0])

    @property
    def n_verts(self) -> int:
        return int(self.V.shape[0])

    def face_normals(self) -> np.ndarray:
        a = self.V[self.FV[:, 0]]
        b = self.V[self.FV[:, 1]]
        c = self.V[self.FV[:, 2]]
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n, axis=1, keepdims=True)
        L[L == 0] = 1.0
        return n / L

    def face_centroids(self) -> np.ndarray:
        return self.V[self.FV].mean(axis=1)

    def face_areas(self) -> np.ndarray:
        a = self.V[self.FV[:, 0]]
        b = self.V[self.FV[:, 1]]
        c = self.V[self.FV[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


@dataclass
class FolderReport:
    """Result of inspecting a user-supplied scan folder."""

    folder: Path
    obj: Path | None = None
    mtl: Path | None = None
    mesh_info: Path | None = None
    textures_found: list[Path] = field(default_factory=list)
    textures_missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ─── FOLDER VALIDATION ────────────────────────────────────────────────────────

def inspect_folder(folder: Path) -> FolderReport:
    """Scan a user upload and report what's present, missing, or ambiguous.

    Accepts either a full Polycam Space folder (with ``mesh_info.json``)
    or a bare mesh folder (just .obj + .mtl + textures). Missing pieces
    are reported, never silently filled in.
    """
    folder = folder.resolve()
    rep = FolderReport(folder=folder)

    if not folder.exists() or not folder.is_dir():
        rep.errors.append(f"Folder does not exist: {folder}")
        return rep

    objs = sorted(folder.rglob("*.obj"))
    if not objs:
        rep.errors.append("No .obj mesh found in upload.")
    else:
        # Pick the largest OBJ (Polycam can ship a low-poly preview alongside).
        rep.obj = max(objs, key=lambda p: p.stat().st_size)
        if len(objs) > 1:
            others = [p.name for p in objs if p != rep.obj]
            rep.warnings.append(
                f"Multiple .obj files; using largest ({rep.obj.name}). Others: {others}"
            )

    if rep.obj is not None:
        # Resolve the mtllib referenced by the OBJ rather than guessing by extension.
        mtllib = _read_mtllib(rep.obj)
        if mtllib:
            cand = (rep.obj.parent / mtllib).resolve()
            if cand.exists():
                rep.mtl = cand
            else:
                rep.errors.append(
                    f".obj references mtllib '{mtllib}' but {cand} does not exist."
                )
        else:
            mtls = sorted(folder.rglob("*.mtl"))
            if mtls:
                rep.mtl = mtls[0]
                rep.warnings.append(
                    f".obj has no `mtllib` directive; falling back to {rep.mtl.name}."
                )
            else:
                rep.errors.append("No .mtl found and .obj has no `mtllib` directive.")

    if rep.mtl is not None:
        for mat_name, tex_rel in _parse_mtl(rep.mtl).items():
            cand = (rep.mtl.parent / tex_rel).resolve()
            if cand.exists():
                rep.textures_found.append(cand)
                continue
            # Browser uploads sometimes flatten directory structure, so also
            # search the whole folder for a file with the same basename.
            base = Path(tex_rel).name
            fallbacks = list(folder.rglob(base))
            if fallbacks:
                rep.textures_found.append(fallbacks[0])
                if fallbacks[0] != cand:
                    rep.warnings.append(
                        f"texture '{tex_rel}' not at expected path; using {fallbacks[0].relative_to(folder)}"
                    )
            else:
                rep.textures_missing.append(f"{mat_name} → {tex_rel}")

        if rep.textures_missing:
            rep.warnings.append(
                f"{len(rep.textures_missing)} texture file(s) referenced by .mtl are missing — "
                "those triangles will render black."
            )

    info_candidates = sorted(folder.rglob("mesh_info.json"))
    if info_candidates:
        rep.mesh_info = info_candidates[0]
    else:
        rep.warnings.append(
            "No mesh_info.json (alignmentTransform). Mesh will be assumed already "
            "Y-up; vertical filtering may be wrong if the scan needs alignment."
        )

    return rep


def _read_mtllib(obj_path: Path) -> str | None:
    with obj_path.open() as fh:
        for line in fh:
            if line.startswith("mtllib"):
                return line.split(maxsplit=1)[1].strip()
    return None


def _parse_mtl(mtl_path: Path) -> dict[str, str]:
    """Return ``{material_name: relative_texture_path}`` from a .mtl."""
    mats: dict[str, str] = {}
    current: str | None = None
    for line in mtl_path.read_text().splitlines():
        s = line.strip().split()
        if not s:
            continue
        if s[0] == "newmtl":
            current = s[1]
        elif s[0] == "map_Kd" and current is not None:
            mats[current] = " ".join(s[1:])
    return mats


# ─── ALIGNMENT ────────────────────────────────────────────────────────────────

def load_alignment(mesh_info_path: Path) -> np.ndarray:
    """Load Polycam ``alignmentTransform`` as a 4×4 matrix (column-major in JSON)."""
    info = json.loads(mesh_info_path.read_text())
    return np.array(info["alignmentTransform"], dtype=np.float64).reshape(4, 4, order="F")


def apply_alignment_inv(V: np.ndarray, M_align: np.ndarray) -> np.ndarray:
    """Apply ``inv(alignmentTransform)`` to bring mesh-space verts into ARKit world."""
    M_inv = np.linalg.inv(M_align)
    Vh = np.hstack([V, np.ones((V.shape[0], 1))])
    return (M_inv @ Vh.T).T[:, :3]


# ─── OBJ PARSING ──────────────────────────────────────────────────────────────

def _parse_obj_raw(obj_path: Path) -> dict:
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces_v: list[tuple[int, int, int]] = []
    faces_vt: list[tuple[int, int, int]] = []
    face_mat: list[int] = []
    mat_names: list[str] = []
    mat_index: dict[str, int] = {}
    mtllib = ""
    current_mat = -1

    with obj_path.open() as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            head = line[:2]
            if head == "v ":
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif head == "vt":
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
            elif head == "f ":
                p = line.split()[1:]
                v_idx: list[int] = []
                vt_idx: list[int] = []
                for tok in p:
                    parts = tok.split("/")
                    v_idx.append(int(parts[0]) - 1)
                    if len(parts) > 1 and parts[1]:
                        vt_idx.append(int(parts[1]) - 1)
                    else:
                        vt_idx.append(v_idx[-1])
                # Triangulate fan for n-gons.
                for i in range(1, len(v_idx) - 1):
                    faces_v.append((v_idx[0], v_idx[i], v_idx[i + 1]))
                    faces_vt.append((vt_idx[0], vt_idx[i], vt_idx[i + 1]))
                    face_mat.append(current_mat)
            elif line.startswith("usemtl"):
                name = line.split()[1]
                if name not in mat_index:
                    mat_index[name] = len(mat_names)
                    mat_names.append(name)
                current_mat = mat_index[name]
            elif line.startswith("mtllib"):
                mtllib = line.split(maxsplit=1)[1].strip()

    return {
        "V": np.asarray(verts, dtype=np.float64),
        "VT": np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((0, 2)),
        "FV": np.asarray(faces_v, dtype=np.int64),
        "FT": np.asarray(faces_vt, dtype=np.int64),
        "FM": np.asarray(face_mat, dtype=np.int32),
        "mat_names": mat_names,
        "mtllib": mtllib,
    }


# ─── HIGH-LEVEL LOADER ────────────────────────────────────────────────────────

def load_mesh(report: FolderReport, *, align: bool = True) -> Mesh:
    """Load the mesh described by an already-validated :class:`FolderReport`.

    If ``align`` is true and a ``mesh_info.json`` is present, vertices are
    transformed into ARKit world space. Otherwise vertices are returned
    as stored in the OBJ.
    """
    if not report.ok:
        raise ValueError(f"Cannot load mesh: {report.errors}")
    assert report.obj is not None  # narrowing for type checkers

    raw = _parse_obj_raw(report.obj)
    V = raw["V"]

    if align and report.mesh_info is not None:
        M = load_alignment(report.mesh_info)
        V = apply_alignment_inv(V, M)

    # Resolve material → texture path in the same order as raw["mat_names"].
    mat_texs: list[Path | None] = []
    folder = report.folder
    if report.mtl is not None:
        tex_map = _parse_mtl(report.mtl)
        for name in raw["mat_names"]:
            rel = tex_map.get(name)
            if rel is None:
                mat_texs.append(None)
                continue
            cand = (report.mtl.parent / rel).resolve()
            if cand.exists():
                mat_texs.append(cand)
                continue
            base = Path(rel).name
            fallbacks = list(folder.rglob(base))
            mat_texs.append(fallbacks[0] if fallbacks else None)
    else:
        mat_texs = [None] * len(raw["mat_names"])

    return Mesh(
        V=V,
        VT=raw["VT"],
        FV=raw["FV"],
        FT=raw["FT"],
        FM=raw["FM"],
        mat_names=raw["mat_names"],
        mat_textures=mat_texs,
        source_obj=report.obj,
    )


def world_y_range(V: np.ndarray) -> tuple[float, float]:
    return float(V[:, 1].min()), float(V[:, 1].max())


def downward_face_mask(normals: np.ndarray, max_tilt_deg: float = 30.0) -> np.ndarray:
    """Faces whose normal points downward in world Y, within ``max_tilt_deg`` of -Y.

    A perfectly horizontal ceiling has ``normal.y == -1``. We accept any
    triangle with ``normal.y < -cos(max_tilt_deg)`` so battens, bulkhead
    sides at slight angles, and noisy LiDAR triangles still survive.
    """
    threshold = -np.cos(np.deg2rad(max_tilt_deg))
    return normals[:, 1] < threshold


__all__ = [
    "Mesh",
    "FolderReport",
    "inspect_folder",
    "load_mesh",
    "load_alignment",
    "apply_alignment_inv",
    "world_y_range",
    "downward_face_mask",
]
