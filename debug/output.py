"""Render labelled segmentations as side-by-side PNGs."""

from __future__ import annotations

import cv2
import numpy as np

from .data import SegInput


PALETTE_BGR = [
    ( 80, 200, 255),   # amber
    (255, 102,  60),   # orange
    (200, 255, 102),   # lime
    (255, 102, 200),   # magenta
    (102, 200, 255),   # sky
    ( 60,  80, 255),   # red
    (200,  80, 255),   # purple
    (255, 200,  60),   # gold
    (102, 255, 200),   # mint
]


def render_overlay(
    data: SegInput,
    labels: np.ndarray,
    *,
    title: str,
    show_walls: bool = True,
    alpha: float = 0.45,
) -> np.ndarray:
    """Returns BGR uint8 image: textured render + tinted clusters + outlines."""
    img = data.canvas.copy()
    H, W = img.shape[:2]
    overlay = np.zeros_like(img)
    n_labels = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0

    for li in range(n_labels):
        m = labels == li
        if not m.any():
            continue
        col = PALETTE_BGR[li % len(PALETTE_BGR)]
        overlay[m] = col

    blended = cv2.addWeighted(img, 1.0, overlay, alpha, 0)

    for li in range(n_labels):
        m = (labels == li).astype(np.uint8) * 255
        if not m.any():
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = PALETTE_BGR[li % len(PALETTE_BGR)]
        cv2.drawContours(blended, contours, -1, col, 2)

    if show_walls and data.wall_edges.any():
        wall_overlay = np.zeros_like(img)
        wall_overlay[data.wall_edges] = (255, 255, 255)
        blended = cv2.addWeighted(blended, 1.0, wall_overlay, 0.35, 0)

    px_count = int(np.sum(labels >= 0))
    room_count = int(data.room_mask.sum())
    coverage = (100 * px_count / room_count) if room_count else 0
    label_lines = [
        title,
        f"{n_labels} clusters · {coverage:.0f}% coverage",
    ]
    _draw_text(blended, label_lines, (20, 40))
    return blended


def _draw_text(img: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        scale = 0.9 if i == 0 else 0.55
        thickness = 2 if i == 0 else 1
        ypos = y + int(i * (32 * scale + 6))
        cv2.putText(img, line, (x, ypos), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, ypos), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thickness, cv2.LINE_AA)


def stack_horizontal(images: list[np.ndarray]) -> np.ndarray:
    """Pad every image to the tallest height, then hstack."""
    if not images:
        raise ValueError("no images to stack")
    if len(images) == 1:
        return images[0]
    h = max(im.shape[0] for im in images)
    cols = []
    for im in images:
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
            im = np.vstack([im, pad])
        cols.append(im)
    return np.hstack(cols)


__all__ = ["render_overlay", "stack_horizontal", "PALETTE_BGR"]
