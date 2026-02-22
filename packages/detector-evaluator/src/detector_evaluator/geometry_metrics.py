from __future__ import annotations

import numpy as np

from .io import load_meta
from .types import GeometrySummary, ParsedLabel, SampleRecord


def _apply_h(H: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ones = np.ones((corners.shape[0], 1), dtype=np.float64)
    pts = np.concatenate([corners.astype(np.float64), ones], axis=1)
    proj = (H @ pts.T).T
    out = proj[:, :2] / proj[:, 2:3]
    return out.astype(np.float32)


def _label_px(label: ParsedLabel, w: int, h: int) -> np.ndarray:
    return (label.corners_norm * np.array([[w, h]], dtype=np.float32)).astype(np.float32)


def compute_geometry_metrics(
    records: list[SampleRecord],
    gt_lookup: dict[tuple[str, str], tuple[list[ParsedLabel], int, int]],
) -> GeometrySummary:
    err_label_meta: list[float] = []
    err_h_meta: list[float] = []
    eval_label_meta = 0
    eval_h_meta = 0

    for rec in records:
        key = (rec.split, rec.stem)
        gt_entry = gt_lookup.get(key)
        if gt_entry is None:
            continue

        gt_labels, h, w = gt_entry
        if not gt_labels:
            continue

        meta = load_meta(rec.meta_path)
        if meta is None:
            continue

        projected = np.array(meta.get("projected_corners_px", []), dtype=np.float32)
        if projected.shape == (4, 2):
            gt_px = _label_px(gt_labels[0], w, h)
            err = np.linalg.norm(gt_px - projected, axis=1)
            err_label_meta.append(float(np.mean(err)))
            eval_label_meta += 1

        H = np.array(meta.get("H", []), dtype=np.float64)
        canonical = np.array(meta.get("canonical_corners_px", []), dtype=np.float32)
        if H.shape == (3, 3) and canonical.shape == (4, 2) and projected.shape == (4, 2):
            projected_est = _apply_h(H, canonical)
            err_h = np.linalg.norm(projected_est - projected, axis=1)
            err_h_meta.append(float(np.mean(err_h)))
            eval_h_meta += 1

    return GeometrySummary(
        num_samples=len(records),
        num_evaluable_label_vs_meta=eval_label_meta,
        mean_corner_error_label_vs_meta_px=float(np.mean(err_label_meta)) if err_label_meta else None,
        p95_corner_error_label_vs_meta_px=float(np.percentile(np.array(err_label_meta), 95)) if err_label_meta else None,
        num_evaluable_h_vs_meta=eval_h_meta,
        mean_corner_error_h_vs_meta_px=float(np.mean(err_h_meta)) if err_h_meta else None,
        p95_corner_error_h_vs_meta_px=float(np.percentile(np.array(err_h_meta), 95)) if err_h_meta else None,
    )
