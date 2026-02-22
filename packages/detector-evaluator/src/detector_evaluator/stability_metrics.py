from __future__ import annotations

import numpy as np

from .detection_metrics import polygon_area, polygon_center_drift_px
from .types import ParsedLabel, StabilitySummary


def _to_poly(label: ParsedLabel, w: int, h: int) -> np.ndarray:
    return label.corners_norm * np.array([[w, h]], dtype=np.float32)


def compute_stability_metrics(
    sample_items: list[tuple[str, str, list[ParsedLabel], list[ParsedLabel], int, int]],
    match_lookup: dict[tuple[str, str], dict],
) -> StabilitySummary:
    by_split: dict[str, list[tuple[str, str]]] = {"train": [], "val": []}
    for split, stem, *_ in sample_items:
        by_split.setdefault(split, []).append((split, stem))

    drifts: list[float] = []
    area_vars: list[float] = []

    for split, keys in by_split.items():
        keys.sort(key=lambda x: x[1])
        prev_poly: np.ndarray | None = None

        for key in keys:
            entry = match_lookup.get(key)
            if not entry:
                continue
            matches = entry["matches"]
            preds = entry["pred_labels"]
            h = entry["h"]
            w = entry["w"]
            if not matches or not preds:
                continue

            best = max(matches, key=lambda m: m.iou)
            poly = _to_poly(preds[best.pred_idx], w, h)

            if prev_poly is not None:
                drift = polygon_center_drift_px(prev_poly, poly)
                area_prev = polygon_area(prev_poly)
                area_cur = polygon_area(poly)
                area_var = abs(area_cur - area_prev) / max(area_prev, 1e-9)
                drifts.append(drift)
                area_vars.append(area_var)

            prev_poly = poly

    return StabilitySummary(
        num_pairs=len(drifts),
        mean_center_drift_px=float(np.mean(drifts)) if drifts else None,
        std_center_drift_px=float(np.std(drifts)) if drifts else None,
        mean_area_variation=float(np.mean(area_vars)) if area_vars else None,
        std_area_variation=float(np.std(area_vars)) if area_vars else None,
    )
