from __future__ import annotations

import numpy as np


def yolo_bbox_to_corners_px(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    cx = x_center * image_w
    cy = y_center * image_h
    bw = width * image_w
    bh = height * image_h

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    return corners


def apply_homography_to_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    projected = (H @ pts_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    return projected_xy.astype(np.float32)


def is_convex_quad(quad: np.ndarray) -> bool:
    if quad.shape != (4, 2):
        return False

    cross_signs: list[float] = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        p2 = quad[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) > 1e-7:
            cross_signs.append(cross)

    if not cross_signs:
        return False

    first_positive = cross_signs[0] > 0
    return all((c > 0) == first_positive for c in cross_signs)


def polygon_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def quad_inside_bounds(quad: np.ndarray, image_w: int, image_h: int) -> bool:
    return (
        np.all(quad[:, 0] >= 0)
        and np.all(quad[:, 0] < image_w)
        and np.all(quad[:, 1] >= 0)
        and np.all(quad[:, 1] < image_h)
    )


def corners_to_xyxy(quad: np.ndarray) -> tuple[float, float, float, float]:
    x1 = float(np.min(quad[:, 0]))
    y1 = float(np.min(quad[:, 1]))
    x2 = float(np.max(quad[:, 0]))
    y2 = float(np.max(quad[:, 1]))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    x_center = cx / image_w
    y_center = cy / image_h
    width = bw / image_w
    height = bh / image_h

    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, width)),
        max(0.0, min(1.0, height)),
    )
