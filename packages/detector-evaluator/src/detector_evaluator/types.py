from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SampleRecord:
    split: str
    stem: str
    image_path: Path | None
    gt_label_path: Path | None
    meta_path: Path | None


@dataclass(slots=True)
class ParsedLabel:
    class_id: int
    corners_norm: np.ndarray  # (4, 2)
    format_name: str  # bbox|obb
    confidence: float


@dataclass(slots=True)
class DetectionPair:
    sample_key: tuple[str, str]
    gt_idx: int
    pred_idx: int
    iou: float
    confidence: float


@dataclass(slots=True)
class DetectionSummary:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    miss_rate: float
    mean_iou: float | None
    median_iou: float | None
    p90_iou: float | None
    p95_iou: float | None
    ap_at_iou: float | None


@dataclass(slots=True)
class StabilitySummary:
    num_pairs: int
    mean_center_drift_px: float | None
    std_center_drift_px: float | None
    mean_area_variation: float | None
    std_area_variation: float | None


@dataclass(slots=True)
class GeometrySummary:
    num_samples: int
    num_evaluable_label_vs_meta: int
    mean_corner_error_label_vs_meta_px: float | None
    p95_corner_error_label_vs_meta_px: float | None
    num_evaluable_h_vs_meta: int
    mean_corner_error_h_vs_meta_px: float | None
    p95_corner_error_h_vs_meta_px: float | None
