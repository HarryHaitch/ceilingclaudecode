# Segmentation experiment — SAM 3 via self-hosted Roboflow inference (RunPod)

Same pipeline as `experiments/segmentation_sam3/` (same 268-image
working set, same three concept prompts: `ceiling item`, `light`,
`vent`) — but the SAM 3 inference is run through a Roboflow
inference server we spin up on a RunPod GPU and tear back down.

Running this script end-to-end (default args) is a single command:

```bash
python3 experiments/segmentation_sam3_runpod/run_runpod.py
```

What that does:

1. Reads RunPod and Roboflow API keys from `~/.runpod_api_key` and
   `~/.roboflow_api_key`.
2. Creates a GPU pod on RunPod community cloud running the
   `roboflow/roboflow-inference-server-gpu:latest` Docker image (RTX
   4090, falls back through 3090 / A5000 / A4000 / L4 if 4090 is
   unavailable). This image ships `CORE_MODEL_SAM3_ENABLED=True`
   baked in — the older `roboflow/inference-server:latest` is
   deprecated and does **not** expose `/sam3/concept_segment`.
3. Waits for the pod's inference server to come up at
   `https://<pod-id>-9001.proxy.runpod.net`.
4. Loops over every image in `experiments/segmentation_sam3/working_set/`
   and calls `sam3_concept_segment` for the three prompts.
5. Writes per-image results to `results/<image_id>/` in the same
   layout as the local SAM 3 run.
6. **Always tears the pod down before exit**, including on Ctrl+C,
   exception, or `kill`. See the cleanup section below.

## Why a separate pipeline from `experiments/segmentation_sam3/`

`experiments/segmentation_sam3/` runs SAM 3 locally via
`transformers.Sam3Model` on Apple Silicon (MPS). That works but is
slow (~110 s per prompt per image) and produces subtly different
results from Roboflow's hosted endpoint — see "Roboflow findings"
below. This pipeline calls SAM 3 the same way Roboflow's hosted API
does, just on a GPU we control.

## Roboflow findings (what's different about their pipeline)

We compared the local-MPS run and Roboflow's hosted SAM 3 on the same
image (`43179113618`, the Coles ceiling). Both at `output_prob_thresh
= 0.5`. Same model weights. Same image. Same three prompts.

**For `light` and `vent`** the two systems are essentially identical
— same number of detections (±1), same top-score distribution, same
maximum mask sizes (within ~5%).

**For `ceiling item`** they diverge in exactly one detection:

| | Roboflow hosted | Local (transformers + MPS) |
|---|---|---|
| count | 20 | 21 |
| largest mask area | **1.4 %** of image | **51.7 %** of image |
| score of that largest mask | (not returned, < 0.5) | 0.528 |

That single 51.7 % "the ceiling itself = a ceiling item" detection is
what made the local overlay look much worse than Roboflow's. It's the
same model semantically interpreting the prompt the same way; the
proposal exists in both systems at borderline confidence (~0.47 –
0.53). It just happens to land on opposite sides of the 0.5 gate.

**Why?** Reading `roboflow/inference/inference/models/sam3/segment_anything3.py`
shows that Roboflow's hosted server loads Meta's reference SAM 3
implementation (`from sam3 import build_sam3_image_model`,
`load_from_HF=False`), not the HuggingFace `transformers.Sam3Model`
port we use locally. Same paper, same weights, two different Python
implementations. The post-processing (`PostProcessImage`) in Meta's
reference yields scores that are systematically ~0.05–0.07 lower than
HuggingFace's `Sam3Processor.post_process_instance_segmentation`. We
verified this offset is constant across every detection that exists
in both systems:

| our score | RF score | diff |
|---|---|---|
| 0.895 | 0.832 | 0.063 |
| 0.887 | 0.820 | 0.067 |
| 0.881 | 0.820 | 0.061 |
| 0.867 | 0.816 | 0.051 |

A constant offset like that doesn't come from semantic understanding —
it comes from the final score-mapping step (sigmoid temperature /
confidence rescale / mask-quality blending). The model's actual
embedding similarity is the same in both pipelines.

**Practical implication:** if we want the cleaner Roboflow result on
our own hardware, we don't need a different model or prompt. We just
need to run Meta's reference implementation, which is what
`roboflow/inference-server` (the open-source library and Docker image)
does. That's exactly what this RunPod pipeline gives us.

A `--max-area-frac` argument (default 0.50) is also wired up as a
safety belt: any single mask covering more than that fraction of the
image is dropped before we save it. For the Roboflow-style pipeline
this should never trigger — but it's there as a sanity gate against
the same borderline case if the model behaviour ever drifts.

## How RunPod is wired up

`run_runpod.py` is one self-contained script that:

- Uses the `runpod` Python SDK to manage the pod.
- Uses the `inference-sdk` HTTP client (the same one Roboflow's hosted
  API uses) to call `sam3_concept_segment` against the pod.
- Hardcodes the proxy URL pattern `https://<pod-id>-9001.proxy.runpod.net`
  (RunPod auto-publishes any port declared as `ports="9001/http"`).

### Pod lifecycle and cleanup

The script registers an `atexit` handler **and** signal handlers for
SIGINT/SIGTERM, both of which terminate the pod. The main body runs
inside `try/finally` so the pod is destroyed even on uncaught
exceptions.

If you Ctrl+C in the middle of a run, you should still see a
`[cleanup] terminating pod ...` line. If you don't (e.g. the script
was force-killed with `kill -9`, or the network dropped during the
terminate call), the pod will keep running and billing — manually
verify on https://console.runpod.io/pods. The script logs a warning
and the pod id if cleanup fails.

By default the pod is **terminated** (destroys disk, costs go to
zero). Pass `--no-terminate` to *stop* the pod instead — that
preserves disk (and the cached SAM 3 weights), so a subsequent run
restarts in ~10 s instead of pulling 3.4 GB again. Stopped pods cost
about $0.005/hr for disk; useful only if you'll be running multiple
batches the same day.

### GPU fallback

`run_runpod.py` tries GPUs in this order, advancing on each "no
capacity" failure:

```
NVIDIA GeForce RTX 4090
NVIDIA GeForce RTX 3090
NVIDIA RTX A5000
NVIDIA RTX A4000
NVIDIA L4
```

All five comfortably fit SAM 3 (~7 GB VRAM at peak). 4090 is the
cheapest GPU on community cloud; the rest are the next-best fallbacks.

### Cost model

For 268 images × 3 prompts = 804 inference calls @ ~2 s each:

- runtime: ~30 minutes (incl. ~60 s pod startup + ~30 s first-call
  weight download)
- $/hr on RTX 4090 community cloud: ~$0.34
- expected cost per full run: ~$0.17

Compare to Roboflow hosted: same 2 s/call, but you pay per call into
their credit ledger and the per-call rate isn't published. Forum
reports of "10× credits when server overloaded" make the per-run cost
unpredictable. RunPod is metered per second of pod uptime — flat,
predictable, and stops the moment we tear the pod down.

## Output layout

```
experiments/segmentation_sam3_runpod/
  run_runpod.py
  README.md
  results/
    <image_id>/
      input.jpg                rotated input, 768×1024
      ceiling_item_mask.png    binary union mask  (0/255)
      ceiling_item_instances.png   uint16 label map (0=bg, 1..N=instance)
      ceiling_item_overlay.jpg     coloured overlay + boxes + score labels
      light_mask.png ...
      vent_mask.png ...
      detections.json          per-prompt list of {score, polygon, box, area_px}
    _summary.json              global record (pod id, GPU, totals)
```

Overlay colours match the local pipeline:
`ceiling item` = red, `light` = yellow, `vent` = blue.

## Reproduce

```bash
# one-time setup
echo "rpa_..." > ~/.runpod_api_key   ; chmod 600 ~/.runpod_api_key
echo "biVZ..." > ~/.roboflow_api_key ; chmod 600 ~/.roboflow_api_key
python3 -m pip install --user runpod inference-sdk

# build the working set (idempotent)
python3 experiments/segmentation_sam3/prepare_inputs.py

# run the full batch (creates pod, processes 268, terminates pod)
python3 experiments/segmentation_sam3_runpod/run_runpod.py

# or just a few images by id
python3 experiments/segmentation_sam3_runpod/run_runpod.py \
    --ids 43179113618,43128777443

# or stop instead of terminate, so disk stays warm for a follow-up run
python3 experiments/segmentation_sam3_runpod/run_runpod.py --no-terminate
```

## Integrating into the main app later

The minimum viable integration is:

1. Reuse `experiments/segmentation_sam3_runpod/run_runpod.py` as a
   library: import `_create_pod_with_fallback`, `_wait_for_health`,
   `_process_image`, `_cleanup`. Drop the CLI wrapper.
2. Decide on a pod-lifecycle policy. Two reasonable ones:
   - **on-demand**: spin up a pod per session import, tear down when
     the session ends (matches today's behaviour). Roughly $0.17 per
     session run.
   - **shared warm pod**: leave one pod running with `--no-terminate`
     and a network volume; have the web server hit it for as long as
     it's needed, then `stop_pod` it from a cron at the end of the
     day. Cheaper if multiple sessions hit it; needs a pod-id stored
     somewhere (e.g. `sessions/_global/runpod_pod_id`).
3. Replace `serverless.roboflow.com` with the pod's proxy URL in the
   inference-sdk client. Same call signature, same response shape.

The constraint nothing in this folder addresses: the pod must finish
booting (~60 s) before a request can be served. For a web app that
expects sub-second response, the pre-warmed shared-pod approach is
the only viable shape.
