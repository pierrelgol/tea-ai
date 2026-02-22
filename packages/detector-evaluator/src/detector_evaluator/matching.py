from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Match:
    gt_idx: int
    pred_idx: int
    iou: float


def _poly_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)

    area_a = _poly_area(a)
    area_b = _poly_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match_one_to_one(
    gt_polys: list[np.ndarray],
    pred_polys: list[np.ndarray],
    iou_threshold: float,
) -> tuple[list[Match], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_polys):
        for pi, p in enumerate(pred_polys):
            iou = polygon_iou(g, p)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(Match(gt_idx=gi, pred_idx=pi, iou=iou))

    unmatched_gt = [i for i in range(len(gt_polys)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_polys)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred
