from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detector_grader.data import (
    index_ground_truth,
    load_labels,
    load_prediction_labels,
)
from detector_grader.pipeline import resolve_model_source


@dataclass(slots=True)
class Label:
    class_id: int
    corners_norm: np.ndarray
    confidence: float


@dataclass(slots=True)
class Sample:
    split: str
    stem: str
    image_path: Path | None
    gt_label_path: Path | None



def resolve_model_key(
    *,
    model: str,
    artifacts_root: Path,
    predictions_root: Path,
) -> str:
    _weights, model_key, _existing = resolve_model_source(
        model_arg=model,
        weights_arg=None,
        artifacts_root=artifacts_root,
        predictions_root=predictions_root,
    )
    return model_key



def index_samples(dataset_root: Path, splits: list[str]) -> list[Sample]:
    split_set = set(splits)
    records = index_ground_truth(dataset_root)
    out: list[Sample] = []
    for rec in records:
        if rec.split not in split_set:
            continue
        out.append(
            Sample(
                split=rec.split,
                stem=rec.stem,
                image_path=rec.image_path,
                gt_label_path=rec.gt_label_path,
            )
        )
    return out



def load_gt_labels(path: Path | None) -> list[Label]:
    if path is None:
        return []
    labels = load_labels(path, is_prediction=False, conf_threshold=0.0)
    return [Label(class_id=l.class_id, corners_norm=l.corners_norm, confidence=l.confidence) for l in labels]



def load_pred_labels(
    *,
    predictions_root: Path,
    model_name: str,
    split: str,
    stem: str,
    conf_threshold: float,
) -> list[Label]:
    labels = load_prediction_labels(
        predictions_root=predictions_root,
        model_name=model_name,
        split=split,
        stem=stem,
        conf_threshold=conf_threshold,
    )
    return [Label(class_id=l.class_id, corners_norm=l.corners_norm, confidence=l.confidence) for l in labels]



def corners_to_px(corners_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    out = corners_norm.astype(np.float64).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    return out.astype(np.float32)
