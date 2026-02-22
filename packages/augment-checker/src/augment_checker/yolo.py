from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class YoloLabel:
    class_id: int
    corners_norm: np.ndarray  # (4, 2) normalized points
    format_name: str  # "obb"


def parse_yolo_line(line: str, *, is_prediction: bool = False) -> YoloLabel:
    parts = line.strip().split()
    expected = 10 if is_prediction else 9
    if len(parts) != expected:
        if is_prediction:
            raise ValueError("YOLO prediction line must have 10 OBB fields: class x1 y1 x2 y2 x3 y3 x4 y4 conf")
        raise ValueError("YOLO label line must have 9 OBB fields: class x1 y1 x2 y2 x3 y3 x4 y4")

    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]
    coords = vals[:8]
    corners = np.array(coords, dtype=np.float32).reshape(4, 2)
    return YoloLabel(class_id=class_id, corners_norm=corners, format_name="obb")


def load_yolo_labels(path, *, is_prediction: bool = False, conf_threshold: float = 0.0) -> list[YoloLabel]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        if is_prediction:
            raise ValueError(f"Empty label file: {path}")
        return []
    out: list[YoloLabel] = []
    try:
        if not is_prediction:
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                out.append(parse_yolo_line(line, is_prediction=False))
            return out

        for line_no, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 10:
                continue
            conf = float(parts[9])
            if conf < conf_threshold:
                continue
            out.append(parse_yolo_line(line, is_prediction=True))
        if not out:
            raise ValueError("No prediction above confidence threshold")
        return out
    except Exception as exc:
        raise ValueError(f"invalid OBB label at {path}: {exc}") from exc


def load_yolo_label(path, *, is_prediction: bool = False, conf_threshold: float = 0.0) -> YoloLabel:
    labels = load_yolo_labels(path, is_prediction=is_prediction, conf_threshold=conf_threshold)
    return labels[0]


def _polygon_area_norm(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def validate_yolo_label(label: YoloLabel, eps: float = 1e-6) -> list[str]:
    issues: list[str] = []
    for idx, (x, y) in enumerate(label.corners_norm):
        if x < 0.0 or x > 1.0:
            issues.append(f"x{idx + 1} outside [0,1]: {x}")
        if y < 0.0 or y > 1.0:
            issues.append(f"y{idx + 1} outside [0,1]: {y}")

    width = float(np.max(label.corners_norm[:, 0]) - np.min(label.corners_norm[:, 0]))
    height = float(np.max(label.corners_norm[:, 1]) - np.min(label.corners_norm[:, 1]))
    if width <= eps:
        issues.append(f"width degenerate: {width}")
    if height <= eps:
        issues.append(f"height degenerate: {height}")

    area = _polygon_area_norm(label.corners_norm)
    if area <= eps:
        issues.append(f"polygon area degenerate: {area}")

    return issues


def label_to_pixel_corners(label: YoloLabel, image_w: int, image_h: int) -> np.ndarray:
    out = label.corners_norm.astype(np.float64).copy()
    out[:, 0] *= image_w
    out[:, 1] *= image_h
    return out.astype(np.float32)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)
    area_a = _polygon_area_norm(a)
    area_b = _polygon_area_norm(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def polygon_centroid_px(poly: np.ndarray) -> tuple[float, float]:
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def center_drift_px(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = polygon_centroid_px(a)
    bcx, bcy = polygon_centroid_px(b)
    dx = acx - bcx
    dy = acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)
