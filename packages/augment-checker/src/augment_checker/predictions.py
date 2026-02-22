from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import ModelMetrics, ModelSampleMetric, SampleRecord
from .yolo import center_drift_px, iou_xyxy, label_to_xyxy, load_yolo_label


def _prediction_models(predictions_root: Path) -> list[Path]:
    if not predictions_root.exists():
        return []
    return sorted([p for p in predictions_root.iterdir() if p.is_dir()])


def run_prediction_checks(records: list[SampleRecord], predictions_root: Path | None) -> list[ModelMetrics]:
    if predictions_root is None:
        return []

    models = _prediction_models(predictions_root)
    reports: list[ModelMetrics] = []

    for model_dir in models:
        model_name = model_dir.name
        sample_metrics: list[ModelSampleMetric] = []
        ious: list[float] = []
        drifts: list[float] = []
        misses = 0

        for rec in records:
            if rec.image_path is None or rec.label_path is None:
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue

            pred_label = model_dir / "labels" / rec.split / f"{rec.stem}.txt"
            if not pred_label.exists():
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue

            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue
            h, w = img.shape[:2]

            gt = load_yolo_label(rec.label_path)
            pred = load_yolo_label(pred_label)
            gt_xyxy = label_to_xyxy(gt, w, h)
            pred_xyxy = label_to_xyxy(pred, w, h)

            iou = iou_xyxy(gt_xyxy, pred_xyxy)
            drift = center_drift_px(gt_xyxy, pred_xyxy)
            sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, iou, drift, False))
            ious.append(iou)
            drifts.append(drift)

        total = len(records) if records else 1
        report = ModelMetrics(
            model_name=model_name,
            num_scored=len(ious),
            mean_iou=float(np.mean(ious)) if ious else None,
            median_iou=float(np.median(ious)) if ious else None,
            mean_center_drift_px=float(np.mean(drifts)) if drifts else None,
            median_center_drift_px=float(np.median(drifts)) if drifts else None,
            miss_rate=misses / total,
            samples=sample_metrics,
        )
        reports.append(report)

    return reports
