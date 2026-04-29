# Segmentation lab

Rapid iteration on auto-region-detection algorithms. Loads a cached
session's height map, room mask and (lazily-built) wall-edge mask, runs
one or many candidate algorithms against it, and either dumps an overlay
PNG or scores against a ground truth and prints a leaderboard.

No browser, no FastAPI — every iteration is a single Python invocation.

## Setup once per scan

```bash
# 1. start the server
ceiling-rcp-server --port 8765

# 2. stage the scan into a session
ceiling-rcp-init "Scan data/<your-polycam-folder>"
# → http://127.0.0.1:8765/?session=<id>

# 3. open that URL, trace the room, run auto-detect, refine each
#    polygon until it matches what you actually want as ground truth
#    (insert / delete vertices, drag, Snap), then:
python -m debug.snapshot_truth <id>
# → debug/ground_truth/<id>.json
```

`ground_truth/<id>.json` is small and committed; you only redo it when
the truth changes.

## Daily loop

```bash
# preview a single algorithm
python -m debug.segment_lab <id> --algo wall_constrained

# side-by-side comparison
python -m debug.segment_lab <id> --compare histogram,felzenszwalb,wall_constrained

# parameter sweep + leaderboard against your truth
python -m debug.experiments <id>

# narrow the sweep while tuning one knob
python -m debug.experiments <id> --filter wall_constrained

# rebuild the wall-edge cache after editing the filter in debug/data.py
python -m debug.segment_lab <id> --algo wall_constrained --refresh-walls
```

Output:

- `debug_out/<id>_<algo>.png` — overlay (textured render + cluster tints + outlines + white wall-edge trace).
- `debug_out/<id>_top<k>_<algo>.png` — side-by-side **truth | top-K prediction**.
- `debug/results/<id>.csv` — sortable leaderboard, easy to diff between runs.
- `debug/results/<id>.json` — full per-experiment record (every match pair, unmatched labels, kwargs).

## What each module does

| File | Purpose |
| --- | --- |
| `data.py` | `load_session(id)` returns a `SegInput` (height map, room mask, wall edges, canvas, grid). `compute_wall_edges` rasterises near-horizontal mesh triangles and caches the result as `wall_edges.npy`. |
| `output.py` | `render_overlay(data, labels, ...)` produces the BGR overlay PNG; `stack_horizontal` builds comparison grids. |
| `algos.py` | Registry of segmentation algorithms (see below). Each takes a `SegInput`, returns an `int32 H×W` label image. |
| `segment_lab.py` | CLI for running one or many algorithms and dumping PNGs (no scoring). |
| `scoring.py` | Confusion matrix, IoU matrix, Hungarian-matched mean / weighted / min IoU. |
| `snapshot_truth.py` | Captures polygons drawn in the browser to `ground_truth/<id>.json`. |
| `experiments.py` | Parameter sweep + scoring + leaderboard. Edit `EXPERIMENTS` to add configurations. |

## Built-in algorithms

| Name | Idea | Tunables |
| --- | --- | --- |
| `histogram` | Current production: peaks of the area-weighted Y histogram, nearest-peak assignment, median filter on labels. | (none — uses production defaults) |
| `felzenszwalb` | `skimage.segmentation.felzenszwalb` on the heightmap. Merges adjacent pixels whose height delta is small — the "gradient between adjacent triangles" signal. | `scale` (region-size sensitivity), `sigma` (pre-blur). |
| `region_growing` | Seed each cluster with the largest CC of pixels within 1.5 cm of a histogram peak, then iteratively dilate, accepting new pixels only where `|height − cluster_mean| < threshold_m`. Stops at bulkheads. | `threshold_m` (default 2 cm), `max_iters`. |
| `wall_constrained` | Connected components of `room_mask & ~wall_edges` are the cells; each cell is labelled by the cluster containing its median height. Boundaries land exactly where the geometry says they should. | `min_cell_px_m2`, `max_clusters`. |

## Adding a new algorithm

1. Write a function in `debug/algos.py`:
   ```python
   def my_algo(data: SegInput, *, my_param: float = 1.0) -> np.ndarray:
       """Returns int32 H×W: -1 outside, 0..N for clusters."""
       ...
   ```
2. Register it:
   ```python
   ALGOS["my_algo"] = my_algo
   ```
3. Add experiment configurations in `debug/experiments.py`:
   ```python
   EXPERIMENTS.append(
       {"name": "my_algo_default", "algo": "my_algo", "kwargs": {}},
   )
   EXPERIMENTS.append(
       {"name": "my_algo_p2", "algo": "my_algo", "kwargs": {"my_param": 2.0}},
   )
   ```
4. Run `python -m debug.segment_lab <id> --algo my_algo` to eyeball the
   PNG, then `python -m debug.experiments <id> --filter my_algo` to see
   how it scores against truth.

## Wall-edge mask

The `wall_constrained` algorithm and the white overlay on every output
PNG come from `compute_wall_edges` in `data.py`:

1. Take every mesh triangle whose normal lies near the XZ plane
   (`|normal.y| < sin(15°)`, i.e. the triangle is nearly vertical).
2. Drop walls — keep only triangles whose Y extent is small
   (`< max_vertical_span_m`, default 60 cm) and at least one vertex is
   within `ceiling_band_m` (80 cm) of the median ceiling height.
3. Fill the kept triangles' XZ projections into a binary raster, dilate
   1 px so thin slivers stay connected.

Tune those thresholds in `data.compute_wall_edges` and pass
`--refresh-walls` to rebuild the cache. The point of the filter is to
keep only **bulkhead / coffer / recess transitions** — vertical faces
that mark where ceiling planes step — and discard floor-to-ceiling
walls.

## Scoring details

`scoring.score(predicted, truth)` returns:

- `mean_iou` — average matched IoU across all truth regions; unmatched
  truth regions count as 0 (penalises under-segmentation).
- `weighted_iou` — same but weighted by truth-polygon area (so getting
  the big main ceiling right matters more than a tiny bulkhead).
- `min_iou` — worst matched pair (catches "everything is OK except this
  one region is totally wrong").
- `n_pred`, `n_truth`, `matches`, `unmatched_pred`, `unmatched_truth`
  — full match record for diagnostics.

Hungarian matching means the predicted label numbering doesn't have to
agree with the truth's — only the spatial overlap matters.
