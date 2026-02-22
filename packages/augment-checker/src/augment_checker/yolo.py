from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class YoloLabel:
    class_id: int
    corners_norm: np.ndarray  # (4, 2) normalized points
    format_name: str  # "bbox" or "obb"


def _bbox_to_corners_norm(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def parse_yolo_line(line: str) -> YoloLabel:
    parts = line.strip().split()
    if len(parts) not in (5, 9):
        raise ValueError("YOLO label line must have 5 (bbox) or 9 (obb) fields")

    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]

    if len(parts) == 5:
        corners = _bbox_to_corners_norm(vals[0], vals[1], vals[2], vals[3])
        return YoloLabel(class_id=class_id, corners_norm=corners, format_name="bbox")

    corners = np.array(vals, dtype=np.float32).reshape(4, 2)
    return YoloLabel(class_id=class_id, corners_norm=corners, format_name="obb")


def load_yolo_label(path) -> YoloLabel:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty label file: {path}")
    return parse_yolo_line(lines[0])


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


def label_to_xyxy(label: YoloLabel, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    corners = label_to_pixel_corners(label, image_w, image_h)
    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    return x1, y1, x2, y2


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


def center_drift_px(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)
