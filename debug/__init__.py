"""Segmentation lab: rapid iteration on auto-detect algorithms.

Loads cached session artefacts (height.npy, ceiling.jpg, plan.json) and
exposes a registry of segmentation algorithms producing per-pixel cluster
labels. Visualises the result as a PNG so you can compare techniques
side-by-side without touching the web UI.
"""
