from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Match:
    gt_idx: int
    pred_idx: int
    iou: float


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match_one_to_one(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[list[Match], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = iou_xyxy(g, p)
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

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred
