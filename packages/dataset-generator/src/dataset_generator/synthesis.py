from __future__ import annotations

import cv2
import numpy as np


def warp_and_composite(
    background: np.ndarray,
    target: np.ndarray,
    canonical_corners_px: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    bg_h, bg_w = background.shape[:2]

    src_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    polygon = canonical_corners_px.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillConvexPoly(src_mask, polygon, 255)

    warped_target = cv2.warpPerspective(target, H, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(src_mask, H, (bg_w, bg_h), flags=cv2.INTER_NEAREST)

    out = background.copy()
    out[warped_mask > 0] = warped_target[warped_mask > 0]
    return out
