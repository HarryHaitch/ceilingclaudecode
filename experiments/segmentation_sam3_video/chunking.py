"""Chunking strategies for the SAM 3 video-tracking experiment.

The 268 ceiling-facing keyframes are split into chunks (or overlapping
windows) before being fed to SAM 3 video mode. Within each chunk, an
anchor frame must be picked — that's the frame where the text prompt
is attached and propagation starts.

Two chunking patterns:

- ``split_chunks``: disjoint chronological N-way split. The user
  experiment sweeps N ∈ {1, 4, 8, 16, 32, 64}. Last chunk absorbs the
  remainder.
- ``sliding_windows``: overlapping windows of fixed length and stride.
  The last window is pinned to the end of the list so every frame is
  covered. Used to recover identity across chunk boundaries (post-hoc
  IoU stitching — see ``stitch_windows.py``).

One anchor strategy: ``pick_densest`` reads the existing image-mode
``detections.json`` per frame and picks the chunk-local frame with the
most detections of the target prompt. Falls back to chunk midpoint if
the image-mode results are missing — that way the script keeps working
even on a fresh checkout.

Run ``python chunking.py`` directly to execute the self-tests.
"""
from __future__ import annotations

import json
from pathlib import Path


def split_chunks(n_frames: int, n_chunks: int) -> list[list[int]]:
    """Split [0, n_frames) into ``n_chunks`` disjoint contiguous lists.

    The first ``n_frames % n_chunks`` chunks get one extra frame so the
    sizes differ by at most 1. Returns global frame indices, not the
    frame ids.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    if n_frames < n_chunks:
        raise ValueError(
            f"can't split {n_frames} frames into {n_chunks} chunks"
        )
    base, rem = divmod(n_frames, n_chunks)
    out: list[list[int]] = []
    cursor = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        out.append(list(range(cursor, cursor + size)))
        cursor += size
    assert cursor == n_frames
    return out


def sliding_windows(
    n_frames: int, window: int, stride: int,
) -> list[list[int]]:
    """Generate overlapping windows of size ``window`` with step ``stride``.

    The last window is pinned so that ``window[-1] == n_frames - 1``,
    guaranteeing every frame is covered by at least one window. If
    ``window >= n_frames`` returns a single window covering everything.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if window >= n_frames:
        return [list(range(n_frames))]

    out: list[list[int]] = []
    start = 0
    while start + window <= n_frames:
        out.append(list(range(start, start + window)))
        start += stride
    last_start = n_frames - window
    if not out or out[-1][0] != last_start:
        out.append(list(range(last_start, n_frames)))
    return out


def pick_densest(
    chunk_indices: list[int],
    frame_ids: list[str],
    image_mode_root: Path | None,
    prompt: str,
) -> int:
    """Return the chunk-local index with the most image-mode detections.

    Reads ``<image_mode_root>/<frame_id>/detections.json`` and counts
    detections for ``prompt``. Returns the chunk-local index (i.e. an
    integer in ``range(len(chunk_indices))``) of the frame with the
    highest count. Ties broken by earliest chunk-local index.

    Falls back to ``len(chunk_indices) // 2`` (midpoint) if
    ``image_mode_root`` is None, missing, or none of the frames have a
    detections.json.
    """
    if image_mode_root is None or not image_mode_root.exists():
        return len(chunk_indices) // 2

    best_local = len(chunk_indices) // 2
    best_count = -1
    found_any = False
    for local_idx, global_idx in enumerate(chunk_indices):
        det_path = image_mode_root / frame_ids[global_idx] / "detections.json"
        if not det_path.exists():
            continue
        found_any = True
        try:
            doc = json.loads(det_path.read_text())
        except Exception:
            continue
        count = 0
        for pp in doc.get("prompts", []) or []:
            if pp.get("prompt") == prompt:
                count = len(pp.get("detections", []) or [])
                break
        if count > best_count:
            best_count = count
            best_local = local_idx

    if not found_any:
        return len(chunk_indices) // 2
    return best_local


# ─── Self-tests ───────────────────────────────────────────────────────────

def _test_split_chunks() -> None:
    # Plan-mandated sweep: 268 frames at N ∈ {1, 4, 8, 16, 32, 64}.
    for n in (1, 4, 8, 16, 32, 64):
        chunks = split_chunks(268, n)
        assert len(chunks) == n, f"expected {n} chunks, got {len(chunks)}"
        flat = [i for c in chunks for i in c]
        assert flat == list(range(268)), f"split_chunks lost frames at N={n}"
        sizes = sorted({len(c) for c in chunks})
        assert len(sizes) <= 2, f"chunks differ by >1 at N={n}: {sizes}"
    # Edge: n_chunks == n_frames.
    assert split_chunks(5, 5) == [[0], [1], [2], [3], [4]]
    # Edge: n_chunks == 1.
    assert split_chunks(5, 1) == [[0, 1, 2, 3, 4]]
    print("[chunking] split_chunks OK")


def _test_sliding_windows() -> None:
    # Plan default: window=32, stride=16 over 268 frames.
    wins = sliding_windows(268, 32, 16)
    assert all(len(w) == 32 for w in wins), "all windows should be size 32"
    assert wins[0][0] == 0
    assert wins[-1][-1] == 267
    covered = set()
    for w in wins:
        covered.update(w)
    assert covered == set(range(268)), "every frame must be covered"
    # Window > n_frames degenerates to single full window.
    assert sliding_windows(10, 32, 16) == [list(range(10))]
    # Stride == window: disjoint windows.
    wins = sliding_windows(20, 5, 5)
    assert wins[0] == [0, 1, 2, 3, 4]
    assert wins[-1][-1] == 19
    print("[chunking] sliding_windows OK")


def _test_pick_densest() -> None:
    import tempfile

    # Build a fake image-mode results tree with known counts.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame_ids = [f"f{i:03d}" for i in range(10)]
        # f000 → 1 light, f005 → 5 lights, others → 0.
        for fid in frame_ids:
            (root / fid).mkdir()
            counts = {"f000": 1, "f005": 5}.get(fid, 0)
            doc = {
                "image_id": fid,
                "prompts": [
                    {"prompt": "light",
                     "detections": [{"score": 0.9} for _ in range(counts)]},
                    {"prompt": "vent", "detections": []},
                ],
            }
            (root / fid / "detections.json").write_text(json.dumps(doc))

        # Whole sequence → densest light is f005 (idx 5).
        chunk = list(range(10))
        assert pick_densest(chunk, frame_ids, root, "light") == 5
        # Sub-chunk [0..4] → densest light is f000 (local idx 0; only nonzero).
        assert pick_densest([0, 1, 2, 3, 4], frame_ids, root, "light") == 0
        # All-zero prompt → ties → falls back to chunk-local idx 0.
        assert pick_densest([2, 3, 4], frame_ids, root, "vent") == 0
        # Missing root → midpoint fallback.
        assert pick_densest([0, 1, 2, 3, 4], frame_ids, None, "light") == 2
        # Existing root but no detections.json files → midpoint fallback.
        empty_root = root / "empty"
        empty_root.mkdir()
        assert pick_densest(
            [0, 1, 2, 3, 4],
            ["g0", "g1", "g2", "g3", "g4"],
            empty_root,
            "light",
        ) == 2
    print("[chunking] pick_densest OK")


def _run_self_tests() -> None:
    _test_split_chunks()
    _test_sliding_windows()
    _test_pick_densest()
    print("[chunking] all tests passed")


if __name__ == "__main__":
    _run_self_tests()
