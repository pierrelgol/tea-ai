from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import ParsedLabel, SampleRecord


def parse_gt_label_line(line: str) -> ParsedLabel:
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError("GT label line must have 9 OBB fields: class x1 y1 x2 y2 x3 y3 x4 y4")

    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]
    corners = np.array(vals, dtype=np.float32).reshape(4, 2)
    return ParsedLabel(class_id=class_id, corners_norm=corners, format_name="obb", confidence=1.0)


def parse_pred_label_line(line: str) -> ParsedLabel:
    parts = line.strip().split()
    if len(parts) != 10:
        raise ValueError("Prediction line must have 10 OBB fields: class x1 y1 x2 y2 x3 y3 x4 y4 conf")

    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]
    coords = vals[:8]
    conf = vals[8]
    corners = np.array(coords, dtype=np.float32).reshape(4, 2)
    return ParsedLabel(class_id=class_id, corners_norm=corners, format_name="obb", confidence=float(conf))


def load_labels(path: Path, *, is_prediction: bool, conf_threshold: float = 0.0) -> list[ParsedLabel]:
    if not path.exists():
        return []

    parsed: list[ParsedLabel] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = parse_pred_label_line(line) if is_prediction else parse_gt_label_line(line)
        except Exception as exc:
            raise ValueError(f"invalid OBB label at {path}:{line_no}: {exc}") from exc
        if is_prediction and label.confidence < conf_threshold:
            continue
        parsed.append(label)
    return parsed


def index_ground_truth(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        lab_dir = dataset_root / "labels" / split
        meta_dir = dataset_root / "meta" / split

        stems: set[str] = set()
        if img_dir.exists():
            stems |= {p.stem for p in img_dir.iterdir() if p.is_file()}
        if lab_dir.exists():
            stems |= {p.stem for p in lab_dir.iterdir() if p.is_file() and p.suffix == ".txt"}
        if meta_dir.exists():
            stems |= {p.stem for p in meta_dir.iterdir() if p.is_file() and p.suffix == ".json"}

        for stem in sorted(stems):
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    image_path = p
                    break

            label_path = lab_dir / f"{stem}.txt"
            meta_path = meta_dir / f"{stem}.json"

            records.append(
                SampleRecord(
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    gt_label_path=label_path if label_path.exists() else None,
                    meta_path=meta_path if meta_path.exists() else None,
                )
            )
    return records


def load_predictions_for_model(
    predictions_root: Path,
    model_name: str,
    split: str,
    stem: str,
    conf_threshold: float,
) -> list[ParsedLabel]:
    path = predictions_root / model_name / "labels" / split / f"{stem}.txt"
    return load_labels(path, is_prediction=True, conf_threshold=conf_threshold)


def corners_to_pixel(corners_norm: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    out = corners_norm.astype(np.float64).copy()
    out[:, 0] *= image_w
    out[:, 1] *= image_h
    return out.astype(np.float32)


def load_meta(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def image_shape(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return h, w
