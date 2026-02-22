from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_yolo_line(line: str) -> YoloBox:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("YOLO line must have 5 fields")
    return YoloBox(
        class_id=int(parts[0]),
        x_center=float(parts[1]),
        y_center=float(parts[2]),
        width=float(parts[3]),
        height=float(parts[4]),
    )


def load_yolo_box(path) -> YoloBox:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty label file: {path}")
    return parse_yolo_line(lines[0])


def validate_yolo_box(box: YoloBox, eps: float = 1e-6) -> list[str]:
    issues: list[str] = []
    for name, value in (
        ("x_center", box.x_center),
        ("y_center", box.y_center),
        ("width", box.width),
        ("height", box.height),
    ):
        if value < 0.0 or value > 1.0:
            issues.append(f"{name} outside [0,1]: {value}")

    if box.width <= eps:
        issues.append(f"width degenerate: {box.width}")
    if box.height <= eps:
        issues.append(f"height degenerate: {box.height}")

    return issues


def yolo_to_xyxy(box: YoloBox, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    cx = box.x_center * image_w
    cy = box.y_center * image_h
    bw = box.width * image_w
    bh = box.height * image_h
    return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)


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
