"""Pipeline glue for generating ceiling-services symbols from a Polycam scan.

The pipeline has three stages:

  1. **prepare**   — filter Polycam keyframes to ceiling-facing ones,
                     rotate so world-up is on top, save as a working set.
                     Implemented in ``experiments/segmentation_sam3/prepare_inputs.py``.

  2. **segment**   — run SAM 3 (text prompts: "ceiling item", "light",
                     "vent") over the working set. Two backends:
                       - ``local``  : ``experiments/segmentation_sam3/run_sam3.py``
                                      (Apple Silicon MPS or CUDA, slow on MPS).
                       - ``runpod`` : ``experiments/segmentation_sam3_runpod/run_runpod.py``
                                      (RunPod GPU pod running roboflow/inference-server).

  3. **project**   — back-project per-frame masks onto the ceiling ortho,
                     cluster cross-frame instances into 3D fixtures, classify
                     into Diffuser / Downlight / LED panel / Sprinkler /
                     Misc, and emit the canonical ``symbols.json`` consumed
                     by the viewer.
                     Implemented in ``experiments/segmentation_projection/``.

This module is intentionally thin — it orchestrates the experimental
scripts rather than reimplementing them. The viewer-side endpoint
(``POST /api/sessions/<id>/symbols/generate``) calls
:func:`generate_symbols_for_session` here, which in turn invokes the
relevant subprocess(es) and copies the resulting ``symbols.json`` into
``sessions/<id>/out/symbols.json``.

For now the heavy backends are stubs: the experiments live in this repo
but haven't been wired into a one-button pipeline. The functions below
raise :class:`PipelineNotImplemented` with a helpful message that the
viewer surfaces back to the user. Preloaded example sessions (see
``preload_assets/``) ship a finished ``symbols.json`` so the UI can be
exercised end-to-end without invoking the pipeline.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .sam3_symbols import load_symbols_doc


# ─── PATHS ────────────────────────────────────────────────────────────────

PACKAGE_DIR = Path(__file__).parent
PRELOAD_DIR = PACKAGE_DIR / "preload_assets"


def session_symbols_path(session_dir: Path) -> Path:
    """The canonical location of ``symbols.json`` inside a session."""
    return session_dir / "out" / "symbols.json"


def preload_symbols_path(session_id: str) -> Path:
    """Optional canned ``symbols.json`` shipped with the package, keyed by
    session id. Used to demo the viewer on the example Polycam scan
    without having to run the full segmentation pipeline."""
    return PRELOAD_DIR / f"{session_id}_symbols.json"


# ─── PUBLIC API ───────────────────────────────────────────────────────────

class PipelineNotImplemented(RuntimeError):
    """Raised by :func:`generate_symbols_for_session` when the requested
    backend isn't wired up yet. The viewer turns this into a banner."""


@dataclass
class GenerationResult:
    session_id: str
    backend: str
    symbols_path: Path
    n_symbols: int


def maybe_preload_symbols(session_id: str, session_dir: Path) -> Path | None:
    """If a canned symbols.json for ``session_id`` is bundled with the
    package and the session doesn't already have one, copy the bundled
    file into the session's out/ directory. Returns the destination path
    when a copy happened, else None.

    Called once on session creation / process so the viewer's GET
    endpoint can serve the example without any extra plumbing."""
    src = preload_symbols_path(session_id)
    if not src.exists():
        return None
    dst = session_symbols_path(session_dir)
    if dst.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def read_session_symbols(session_dir: Path) -> dict | None:
    """Return the parsed ``symbols.json`` for a session, or None if no
    symbols have been generated/loaded yet."""
    p = session_symbols_path(session_dir)
    if not p.exists():
        return None
    return load_symbols_doc(p)


def write_session_symbols(session_dir: Path, doc: dict) -> Path:
    """Persist an edited ``symbols.json`` back to the session."""
    p = session_symbols_path(session_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2))
    return p


def generate_symbols_for_session(
    session_id: str,
    session_dir: Path,
    *,
    backend: Literal["local", "runpod"] = "local",
) -> GenerationResult:
    """End-to-end: keyframe filter → SAM 3 → projection → symbols.json.

    NOT IMPLEMENTED for either backend in the viewer right now — the
    individual stages are runnable as standalone experiments but the
    one-button pipeline still has rough edges (Polycam keyframe
    discovery, working-set placement, projection-stage CLI). Wiring is
    the next milestone; for now we surface a clear error and point at
    the manual scripts.
    """
    if backend == "local":
        raise PipelineNotImplemented(
            "Local SAM 3 pipeline not wired up yet — for now run "
            "the three experiment scripts manually:\n"
            "  1. python -m experiments.segmentation_sam3.prepare_inputs\n"
            "  2. python -m experiments.segmentation_sam3.run_sam3 --n 0\n"
            "  3. python -m experiments.segmentation_projection.build_symbols\n"
            "then drop the resulting symbols.json into "
            f"sessions/{session_id}/out/symbols.json."
        )
    if backend == "runpod":
        raise PipelineNotImplemented(
            "RunPod SAM 3 pipeline not wired up yet — see "
            "experiments/segmentation_sam3_runpod/HANDOFF.md for the "
            "manual flow. The pod-lifecycle script "
            "(run_runpod.py) is ready; the projection stage still "
            "needs to be invoked separately."
        )
    raise PipelineNotImplemented(f"unknown backend: {backend}")


__all__ = [
    "PRELOAD_DIR",
    "PipelineNotImplemented",
    "GenerationResult",
    "session_symbols_path",
    "preload_symbols_path",
    "maybe_preload_symbols",
    "read_session_symbols",
    "write_session_symbols",
    "generate_symbols_for_session",
]
