"""Score a predicted segmentation against a ground-truth polygon set.

The truth is a list of polygons (each = a list of ``(x, z)`` world-metre
tuples). We rasterise it to a label image on the same grid as the
prediction, compute the confusion matrix, and then run a Hungarian
match between predicted and truth labels to find the best one-to-one
assignment maximising IoU.

Why Hungarian: the algorithms emit labels in arbitrary order, so a
predicted "label 3" might correspond to truth "label 0". A naive
per-pixel comparison would give terrible scores even for a perfect
segmentation; Hungarian finds the assignment that best aligns them.

We report mean / min / per-pair IoU and how many predicted regions
were matched / orphaned. Higher is better; perfect match = 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ceiling_rcp.analyse import polygon_to_mask
from ceiling_rcp.planes import PlanGrid


@dataclass
class ScoreResult:
    mean_iou: float                # average IoU across matched pairs
    min_iou: float                 # worst matched pair
    weighted_iou: float            # truth-area-weighted mean IoU
    n_pred: int
    n_truth: int
    matches: list[tuple[int, int, float]]  # (pred_label, truth_label, iou)
    unmatched_pred: list[int]      # predicted labels with no truth partner
    unmatched_truth: list[int]     # truth labels with no predicted partner

    def as_dict(self) -> dict:
        return {
            "mean_iou": float(self.mean_iou),
            "min_iou": float(self.min_iou),
            "weighted_iou": float(self.weighted_iou),
            "n_pred": int(self.n_pred),
            "n_truth": int(self.n_truth),
            "matches": [[int(p), int(t), float(i)] for p, t, i in self.matches],
            "unmatched_pred": [int(x) for x in self.unmatched_pred],
            "unmatched_truth": [int(x) for x in self.unmatched_truth],
        }


def polygons_to_label_image(
    polygons: list[list[list[float]]], grid: PlanGrid,
) -> np.ndarray:
    """Rasterise an ordered list of world-XZ polygons into an int32 label
    image at ``grid``'s resolution. ``-1`` outside any polygon. Later
    polygons in the list overwrite earlier ones, so pass them in the
    order you want overlap-resolution to favour."""
    H, W = grid.height, grid.width
    img = np.full((H, W), -1, dtype=np.int32)
    for li, poly in enumerate(polygons):
        m = polygon_to_mask([tuple(p) for p in poly], grid) > 0
        img[m] = li
    return img


def confusion_matrix(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``cm[i, j]`` = number of pixels labelled ``i`` in prediction AND
    ``j`` in truth. Pixels labelled ``-1`` in either are ignored."""
    valid = (pred >= 0) & (truth >= 0)
    if not valid.any():
        return np.zeros((0, 0), dtype=np.int64)
    p = pred[valid].astype(np.int64)
    t = truth[valid].astype(np.int64)
    n_p = int(p.max() + 1)
    n_t = int(t.max() + 1)
    cm = np.zeros((n_p, n_t), dtype=np.int64)
    np.add.at(cm, (p, t), 1)
    return cm


def iou_matrix(cm: np.ndarray) -> np.ndarray:
    """Per-(pred, truth) IoU computed from a confusion matrix."""
    if cm.size == 0:
        return np.zeros((0, 0))
    row = cm.sum(axis=1, keepdims=True)
    col = cm.sum(axis=0, keepdims=True)
    union = row + col - cm
    return np.where(union > 0, cm / union, 0.0)


def score(pred: np.ndarray, truth: np.ndarray) -> ScoreResult:
    from scipy.optimize import linear_sum_assignment
    cm = confusion_matrix(pred, truth)
    iou = iou_matrix(cm)
    n_pred = iou.shape[0]
    n_truth = iou.shape[1]
    if n_pred == 0 or n_truth == 0:
        return ScoreResult(0.0, 0.0, 0.0, n_pred, n_truth, [], [], [])

    # Hungarian on negative-IoU to maximise.
    row_ind, col_ind = linear_sum_assignment(-iou)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] > 0:
            matches.append((int(r), int(c), float(iou[r, c])))

    matched_pred = {p for p, _, _ in matches}
    matched_truth = {t for _, t, _ in matches}
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_truth = [j for j in range(n_truth) if j not in matched_truth]

    if not matches:
        return ScoreResult(0.0, 0.0, 0.0, n_pred, n_truth, [],
                           unmatched_pred, unmatched_truth)

    iou_vals = np.array([m[2] for m in matches])
    truth_areas = np.array([cm[:, m[1]].sum() for m in matches], dtype=np.float64)
    weighted_iou = float((iou_vals * truth_areas).sum()
                         / max(1.0, truth_areas.sum()))

    # Penalise unmatched truth regions: they count as 0-IoU in the mean.
    full_iou = list(iou_vals) + [0.0] * len(unmatched_truth)
    mean_iou = float(np.mean(full_iou))
    min_iou = float(np.min(full_iou))

    return ScoreResult(
        mean_iou=mean_iou,
        min_iou=min_iou,
        weighted_iou=weighted_iou,
        n_pred=n_pred,
        n_truth=n_truth,
        matches=matches,
        unmatched_pred=unmatched_pred,
        unmatched_truth=unmatched_truth,
    )


__all__ = [
    "ScoreResult",
    "polygons_to_label_image",
    "confusion_matrix",
    "iou_matrix",
    "score",
]
