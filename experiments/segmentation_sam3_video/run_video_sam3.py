"""Local driver for the SAM 3 video-tracking experiment (HTTP transport).

Spins up a RunPod GPU pod running our custom ``ghcr.io/<user>/sam3-video``
image (built by the GitHub Actions workflow at
``.github/workflows/sam3-video-pod.yml``). The pod exposes a FastAPI
handler on port 9001 with two endpoints:

  POST /upload_frames     multipart upload of all 268 working-set JPEGs
                          → returns ``upload_id``
  POST /sam3/video_segment one chunk of frame indices + prompt + anchor
                          → returns per-track polygons

The driver:
  1. creates the pod (GPU fallback H100 → 4090 → … from the existing
     image-mode runner),
  2. waits for the handler's ``/info`` to respond,
  3. uploads all frames once,
  4. for each (chunking_mode, chunk, prompt) POSTs the chunk and
     stitches the response into a tracks list,
  5. writes ``tracks_<mode>_<prompt>.json`` + ``summary.json`` locally,
  6. terminates the pod.

Reuses pod-lifecycle helpers from
:mod:`ceiling_rcp.sam3_runpod_runner` (atexit + SIGINT/SIGTERM cleanup,
GPU fallback chain).

Default chunking sweep: ``1,4,8,16,32,64`` disjoint chunks plus one
sliding-window mode (``window=32, stride=16``). All three concept
prompts (``"ceiling item"``, ``"light"``, ``"vent"``) run per chunking
mode against the same warm pod.

Usage:

    python experiments/segmentation_sam3_video/run_video_sam3.py \\
        --docker-image ghcr.io/harryhaitch/sam3-video:latest

If ``--working-set`` / ``--image-mode-results`` are omitted, the driver
defaults to the sibling-worktree paths recorded at the time of the
experiment.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ceiling_rcp.sam3_runpod_runner import (  # noqa: E402
    _arm_cleanup_hooks,
    _create_pod_with_fallback,
    _read_key_file,
    set_pod_to_cleanup,
    DEFAULT_RUNPOD_KEY,
)
from chunking import pick_densest, sliding_windows, split_chunks  # noqa: E402


# ─── Defaults ─────────────────────────────────────────────────────────────

WORKTREES_ROOT = REPO_ROOT.parent
DEFAULT_WORKING_SET = (
    WORKTREES_ROOT
    / "trusting-lovelace-911506"
    / "experiments"
    / "segmentation_sam3"
    / "working_set"
)
DEFAULT_IMAGE_MODE_RESULTS = (
    WORKTREES_ROOT
    / "trusting-lovelace-911506"
    / "experiments"
    / "segmentation_sam3_runpod"
    / "results"
)

# Default to the GHA-built image at ghcr.io. Override with --docker-image.
# The owner segment is filled in from the git remote at runtime if absent.
DEFAULT_DOCKER_IMAGE = "ghcr.io/harryhaitch/sam3-video:latest"
HTTP_PORT = 9001


# ─── HTTP transport ───────────────────────────────────────────────────────

def _wait_for_runtime_port(
    pod_id: str, *, port: int, timeout_s: int = 600,
    interval_s: float = 5.0,
) -> None:
    """Poll runpod.get_pod() until the requested private port appears
    in the runtime ports list. RunPod's HTTPS proxy
    (``https://<pod-id>-<port>.proxy.runpod.net``) routes through the
    private-IP mapping just fine, so we don't need ``isIpPublic`` —
    that flag is only set when the pod was created with TCP exposure
    on a directly-routable IP, which community-cloud pods don't get.
    """
    import runpod
    start = time.time()
    last_obs = "no observation yet"
    while time.time() - start < timeout_s:
        info = runpod.get_pod(pod_id)
        if info is not None:
            runtime = info.get("runtime") or {}
            ports = runtime.get("ports") or []
            for p in ports:
                if p.get("privatePort") == port:
                    return
            last_obs = (
                f"status={info.get('desiredStatus')} runtime="
                f"{'present' if runtime else 'null'} "
                f"ports={[(p.get('privatePort'), p.get('publicPort')) for p in ports]}"
            )
        time.sleep(interval_s)
    raise TimeoutError(
        f"pod {pod_id} never exposed port {port} in {timeout_s}s "
        f"(last: {last_obs})"
    )


def _wait_for_http_health(
    base_url: str, *, timeout_s: int = 600, interval_s: float = 5.0,
) -> dict:
    """Poll GET /info until it returns 200."""
    import requests
    start = time.time()
    last_obs = "no attempt yet"
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{base_url}/info", timeout=10)
            if r.status_code == 200:
                print(f"[http] /info OK after {time.time() - start:.0f}s",
                      flush=True)
                return r.json()
            last_obs = f"HTTP {r.status_code}"
        except Exception as e:
            last_obs = f"{type(e).__name__}: {e}"
        time.sleep(interval_s)
    raise TimeoutError(
        f"handler never came up in {timeout_s}s (last: {last_obs})"
    )


def _upload_all_frames(
    base_url: str, working_set_dir: Path, frame_ids: list[str],
) -> str:
    """POST /upload_frames with all working-set JPEGs as multipart.

    Returns the upload_id. ~40 MB total payload at 268 × ~150 KB.
    """
    import requests
    print(f"[http] uploading {len(frame_ids)} frames to /upload_frames…",
          flush=True)
    files = []
    for fid in frame_ids:
        path = working_set_dir / f"{fid}.jpg"
        files.append(("files", (f"{fid}.jpg", path.read_bytes(), "image/jpeg")))
    data = {"frame_ids_json": json.dumps(frame_ids)}
    t0 = time.time()
    r = requests.post(
        f"{base_url}/upload_frames", data=data, files=files, timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    print(f"[http] uploaded {body['n_frames']} frames "
          f"in {time.time() - t0:.1f}s; upload_id={body['upload_id']}",
          flush=True)
    return body["upload_id"]


def _post_chunk(
    base_url: str, *,
    upload_id: str, frame_indices: list[int],
    prompt: str, anchor_local: int, threshold: float,
) -> dict:
    import requests
    r = requests.post(
        f"{base_url}/sam3/video_segment",
        json={
            "upload_id": upload_id,
            "frame_indices": frame_indices,
            "prompt": prompt,
            "anchor_idx_local": anchor_local,
            "threshold": threshold,
        },
        timeout=900,
    )
    if r.status_code >= 400:
        # Surface the handler's traceback when the call fails — without
        # this the driver just sees "500 Internal Server Error" with no
        # idea why.
        body = r.text[:2000]
        raise RuntimeError(
            f"POST /sam3/video_segment {r.status_code}:\n{body}"
        )
    return r.json()


# ─── Sweep helpers (mirror in-pod run_sweep.py) ───────────────────────────

def _build_frame_list(working_set_dir: Path) -> list[str]:
    if (working_set_dir / "_manifest.json").exists():
        manifest = json.loads((working_set_dir / "_manifest.json").read_text())
        return [item["id"] for item in manifest.get("items", [])]
    return sorted(
        p.stem for p in working_set_dir.glob("*.jpg")
        if not p.name.startswith("_")
    )


def _parse_modes(modes_csv: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for raw in modes_csv.split(","):
        m = raw.strip()
        if not m:
            continue
        if m.startswith("disjoint_"):
            n = int(m.removeprefix("disjoint_"))
            out.append((m, {"kind": "disjoint", "n_chunks": n}))
        elif m.startswith("sliding_"):
            # sliding_w32_s16 → split on '_' → ['sliding','w32','s16'].
            # Match the W and S tokens by their full prefix+digits form
            # so "sliding" doesn't accidentally match the "s<N>" filter.
            parts = m.split("_")
            w = int(next(
                p for p in parts
                if p.startswith("w") and p[1:].isdigit()
            ).removeprefix("w"))
            s = int(next(
                p for p in parts
                if p.startswith("s") and p[1:].isdigit()
            ).removeprefix("s"))
            out.append((m, {"kind": "sliding", "window": w, "stride": s}))
        else:
            raise ValueError(f"unrecognised mode: {m!r}")
    return out


def _build_chunks(n_frames: int, mode_cfg: dict) -> list[list[int]]:
    if mode_cfg["kind"] == "disjoint":
        return split_chunks(n_frames, mode_cfg["n_chunks"])
    if mode_cfg["kind"] == "sliding":
        return sliding_windows(
            n_frames, mode_cfg["window"], mode_cfg["stride"],
        )
    raise ValueError(f"unknown mode kind: {mode_cfg['kind']}")


def _aggregate_tracks(
    chunk_idx: int,
    chunk_indices: list[int],
    frame_ids: list[str],
    per_frame_results: list[dict],
    *,
    mode_name: str,
    prompt: str,
) -> list[dict]:
    by_track: dict[int, list[dict]] = {}
    for rec in per_frame_results:
        fidx_local = rec["frame_idx_local"]
        fidx_global = chunk_indices[fidx_local]
        for inst in rec["instances"]:
            by_track.setdefault(inst["obj_id"], []).append({
                "frame_idx_global": fidx_global,
                "frame_idx_local": fidx_local,
                "image_id": frame_ids[fidx_global],
                "bbox": inst["bbox"],
                "polygon": inst["polygon"],
                "score": inst["score"],
                "area_px": inst.get("area_px", 0),
            })

    import statistics as st
    tracks: list[dict] = []
    slug = prompt.replace(" ", "_")
    for obj_id, frames in by_track.items():
        frames.sort(key=lambda f: f["frame_idx_global"])
        scores = [f["score"] for f in frames]
        tracks.append({
            "track_id": f"{mode_name}_c{chunk_idx}_{slug}_{obj_id}",
            "prompt": prompt,
            "chunk_idx": chunk_idx,
            "obj_id": obj_id,
            "lifespan": len(frames),
            "mean_score": float(st.mean(scores)) if scores else 0.0,
            "frames": frames,
        })
    return tracks


def _build_default_modes(chunks_csv: str, sliding: bool,
                         window: int, stride: int) -> str:
    parts = [f"disjoint_{n.strip()}" for n in chunks_csv.split(",") if n.strip()]
    if sliding:
        parts.append(f"sliding_w{window}_s{stride}")
    return ",".join(parts)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--working-set", type=Path, default=DEFAULT_WORKING_SET)
    ap.add_argument("--image-mode-results", type=Path,
                    default=DEFAULT_IMAGE_MODE_RESULTS)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--chunks", default="1,4,8,16,32,64")
    ap.add_argument("--sliding", action="store_true", default=True)
    ap.add_argument("--no-sliding", dest="sliding", action="store_false")
    ap.add_argument("--sliding-window", type=int, default=32)
    ap.add_argument("--sliding-stride", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--prompts", default="ceiling item,light,vent")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="dev-only: cap n_frames for smoke testing")
    ap.add_argument("--cloud", default="COMMUNITY",
                    choices=["COMMUNITY", "SECURE", "ALL"])
    ap.add_argument("--container-disk-gb", type=int, default=40)
    ap.add_argument("--keep-pod", action="store_true",
                    help="don't terminate pod after run")
    ap.add_argument("--reuse-pod-id", default=None,
                    help="attach to an already-running pod instead of "
                         "creating one (still needs ports=9001/http)")
    ap.add_argument("--runpod-key-file", type=Path, default=DEFAULT_RUNPOD_KEY)
    ap.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    args = ap.parse_args()

    if not args.working_set.exists():
        raise SystemExit(
            f"working set not found: {args.working_set}\n"
            "Pass --working-set <path> to override."
        )

    frame_ids = _build_frame_list(args.working_set)
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise SystemExit(f"no frames in {args.working_set}")
    print(f"[driver] {len(frame_ids)} frames in {args.working_set}",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes_csv = _build_default_modes(
        args.chunks, args.sliding, args.sliding_window, args.sliding_stride,
    )
    modes = _parse_modes(modes_csv)
    print(f"[driver] modes: {[m[0] for m in modes]}", flush=True)

    runpod_key = _read_key_file(args.runpod_key_file, "RunPod API")

    # ─── Pod lifecycle ─────────────────────────────────────────────────
    import runpod
    runpod.api_key = runpod_key
    set_pod_to_cleanup(None, terminate=not args.keep_pod)
    _arm_cleanup_hooks()

    if args.reuse_pod_id:
        pod_id = args.reuse_pod_id
        info = runpod.get_pod(pod_id)
        print(f"[driver] reusing pod {pod_id} "
              f"(status={info.get('desiredStatus')})", flush=True)
        # Don't auto-terminate a pod the user wired up themselves. If
        # the sweep crashes mid-way, we still want the pod alive so
        # they can debug or re-run with --reuse-pod-id.
        args.keep_pod = True
    else:
        print(f"[driver] using image {args.docker_image}", flush=True)
        pod = _create_pod_with_fallback(
            runpod_api_key=runpod_key,
            cloud_type=args.cloud,
            container_disk_in_gb=args.container_disk_gb,
            image_name=args.docker_image,
            ports=f"{HTTP_PORT}/http",
            env={},
            name="sam3-video-sweep",
        )
        pod_id = pod["id"]

    set_pod_to_cleanup(pod_id, terminate=not args.keep_pod)

    print(f"[driver] waiting for HTTP port {HTTP_PORT} on pod {pod_id}…",
          flush=True)
    _wait_for_runtime_port(pod_id, port=HTTP_PORT, timeout_s=900)
    base_url = f"https://{pod_id}-{HTTP_PORT}.proxy.runpod.net"
    print(f"[driver] pod proxy URL: {base_url}", flush=True)
    info = _wait_for_http_health(base_url, timeout_s=900)
    print(f"[driver] pod /info: {info}", flush=True)

    # ─── Upload all frames once ────────────────────────────────────────
    upload_id = _upload_all_frames(base_url, args.working_set, frame_ids)

    # ─── Sweep ─────────────────────────────────────────────────────────
    summary: dict = {
        "n_frames": len(frame_ids),
        "image_mode_results": (
            str(args.image_mode_results)
            if args.image_mode_results.exists() else None
        ),
        "threshold": args.threshold,
        "modes": {},
    }
    total_t0 = time.time()
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    for mode_name, mode_cfg in modes:
        chunks = _build_chunks(len(frame_ids), mode_cfg)
        print(f"\n[driver] === mode={mode_name}  n_chunks={len(chunks)} ===",
              flush=True)

        for prompt in prompts:
            mode_t0 = time.time()
            all_tracks: list[dict] = []
            for ci, chunk in enumerate(chunks):
                anchor_local = pick_densest(
                    chunk, frame_ids,
                    args.image_mode_results
                    if args.image_mode_results.exists() else None,
                    prompt,
                )
                print(
                    f"[driver] {mode_name} {prompt!r} "
                    f"chunk {ci+1}/{len(chunks)} "
                    f"len={len(chunk)} "
                    f"anchor_local={anchor_local} "
                    f"({frame_ids[chunk[anchor_local]]})",
                    flush=True,
                )
                resp = _post_chunk(
                    base_url,
                    upload_id=upload_id,
                    frame_indices=chunk,
                    prompt=prompt,
                    anchor_local=anchor_local,
                    threshold=args.threshold,
                )
                tracks = _aggregate_tracks(
                    chunk_idx=ci,
                    chunk_indices=chunk,
                    frame_ids=frame_ids,
                    per_frame_results=resp.get("per_frame", []),
                    mode_name=mode_name,
                    prompt=prompt,
                )
                all_tracks.extend(tracks)
                print(f"          → {len(tracks)} tracks "
                      f"({resp.get('elapsed_s', 0):.1f}s)", flush=True)

            mode_elapsed = time.time() - mode_t0
            slug = prompt.replace(" ", "_")
            out_path = args.out_dir / f"tracks_{mode_name}_{slug}.json"
            out_path.write_text(json.dumps({
                "mode": mode_name,
                "prompt": prompt,
                "n_chunks": len(chunks),
                "n_tracks": len(all_tracks),
                "elapsed_s": mode_elapsed,
                "tracks": all_tracks,
            }, indent=2))

            mode_summary = summary["modes"].setdefault(mode_name, {})
            lifespans = [t["lifespan"] for t in all_tracks] or [0]
            import statistics as st
            mode_summary[prompt] = {
                "n_tracks": len(all_tracks),
                "mean_track_len": float(st.mean(lifespans)),
                "median_track_len": float(st.median(lifespans)),
                "elapsed_s": mode_elapsed,
                "tracks_path": str(out_path.relative_to(args.out_dir)),
            }
            print(f"[driver] {mode_name} {prompt!r}: "
                  f"{len(all_tracks)} tracks in {mode_elapsed:.1f}s",
                  flush=True)

    summary["total_elapsed_s"] = time.time() - total_t0
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[driver] DONE in {summary['total_elapsed_s']:.0f}s — results "
          f"at {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
