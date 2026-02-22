from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import greedy_match_one_to_one, polygon_iou
from .types import DetectionSummary, ParsedLabel


@dataclass(slots=True)
class PerSampleDetection:
    split: str
    stem: str
    tp: int
    fp: int
    fn: int
    matched_ious: list[float]


def _precision(tp: int, fp: int) -> float:
    d = tp + fp
    return tp / d if d > 0 else 0.0


def _recall(tp: int, fn: int) -> float:
    d = tp + fn
    return tp / d if d > 0 else 0.0


def _ap_at_iou_threshold(gt_total: int, candidates: list[tuple[float, int]]) -> float | None:
    if gt_total <= 0:
        return None
    if not candidates:
        return 0.0

    candidates.sort(key=lambda x: -x[0])
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for _conf, is_tp in candidates:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(_precision(tp, fp))
        recalls.append(tp / gt_total)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = 0.0
        for prec, rec in zip(precisions, recalls):
            if rec >= t:
                p = max(p, prec)
        ap += p
    return float(ap / 11.0)


def compute_detection_metrics(
    sample_items: list[tuple[str, str, list[ParsedLabel], list[ParsedLabel], int, int]],
    iou_threshold: float,
) -> tuple[DetectionSummary, list[PerSampleDetection], dict[tuple[str, str], dict]]:
    tp = fp = fn = 0
    all_ious: list[float] = []
    gt_total = 0
    ap_candidates: list[tuple[float, int]] = []
    per_sample: list[PerSampleDetection] = []
    match_lookup: dict[tuple[str, str], dict] = {}

    for split, stem, gt_labels, pred_labels, h, w in sample_items:
        gt_total += len(gt_labels)
        gt_polys = [g.corners_norm * np.array([[w, h]], dtype=np.float32) for g in gt_labels]
        pred_polys = [p.corners_norm * np.array([[w, h]], dtype=np.float32) for p in pred_labels]

        matches, unmatched_gt, unmatched_pred = greedy_match_one_to_one(gt_polys, pred_polys, iou_threshold)
        sample_tp = len(matches)
        sample_fp = len(unmatched_pred)
        sample_fn = len(unmatched_gt)

        tp += sample_tp
        fp += sample_fp
        fn += sample_fn

        ious = [m.iou for m in matches]
        all_ious.extend(ious)

        matched_pred_idx = {m.pred_idx for m in matches}
        match_lookup[(split, stem)] = {
            "matches": matches,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "h": h,
            "w": w,
        }

        for m in matches:
            ap_candidates.append((pred_labels[m.pred_idx].confidence, 1))
        for idx in range(len(pred_labels)):
            if idx not in matched_pred_idx:
                ap_candidates.append((pred_labels[idx].confidence, 0))

        per_sample.append(
            PerSampleDetection(split=split, stem=stem, tp=sample_tp, fp=sample_fp, fn=sample_fn, matched_ious=ious)
        )

    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    summary = DetectionSummary(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        miss_rate=miss_rate,
        mean_iou=float(np.mean(all_ious)) if all_ious else None,
        median_iou=float(np.median(all_ious)) if all_ious else None,
        p90_iou=float(np.percentile(np.array(all_ious), 90)) if all_ious else None,
        p95_iou=float(np.percentile(np.array(all_ious), 95)) if all_ious else None,
        ap_at_iou=_ap_at_iou_threshold(gt_total, ap_candidates),
    )

    return summary, per_sample, match_lookup


def polygon_centroid(poly: np.ndarray) -> tuple[float, float]:
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_center_drift_px(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = polygon_centroid(a)
    bcx, bcy = polygon_centroid(b)
    dx = acx - bcx
    dy = acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)


def iou_for_polygons(a: np.ndarray, b: np.ndarray) -> float:
    return polygon_iou(a, b)
