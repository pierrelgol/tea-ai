from __future__ import annotations

import numpy as np

from .detection_metrics import area_xyxy, center_drift_px
from .types import ParsedLabel, StabilitySummary


def _to_xyxy(label: ParsedLabel, w: int, h: int) -> tuple[float, float, float, float]:
    corners = label.corners_norm * np.array([[w, h]], dtype=np.float32)
    return (
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    )


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
        prev_box = None

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
            box = _to_xyxy(preds[best.pred_idx], w, h)

            if prev_box is not None:
                drift = center_drift_px(prev_box, box)
                area_prev = area_xyxy(prev_box)
                area_cur = area_xyxy(box)
                area_var = abs(area_cur - area_prev) / max(area_prev, 1e-9)
                drifts.append(drift)
                area_vars.append(area_var)

            prev_box = box

    return StabilitySummary(
        num_pairs=len(drifts),
        mean_center_drift_px=float(np.mean(drifts)) if drifts else None,
        std_center_drift_px=float(np.std(drifts)) if drifts else None,
        mean_area_variation=float(np.mean(area_vars)) if area_vars else None,
        std_area_variation=float(np.std(area_vars)) if area_vars else None,
    )
