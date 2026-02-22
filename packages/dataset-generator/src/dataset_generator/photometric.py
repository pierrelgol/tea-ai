from __future__ import annotations

import cv2
import numpy as np

from .config import GeneratorConfig


def _odd_kernel(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def _apply_color_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = float(rng.uniform(-8.0, 8.0))
    s_gain = float(rng.uniform(0.75, 1.25))
    v_gain = float(rng.uniform(0.75, 1.25))
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_gain, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = _odd_kernel(int(rng.integers(5, 15)))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(-30.0, 30.0))
    rot = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(float(np.sum(kernel)), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _apply_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(3.0, 14.0))
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_jpeg_artifact(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(40, 85))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_photometric_stack(
    img: np.ndarray,
    *,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> tuple[np.ndarray, dict]:
    out = img
    applied: dict[str, bool] = {
        "color_jitter": False,
        "blur": False,
        "motion_blur": False,
        "noise": False,
        "jpeg_artifact": False,
    }

    if rng.random() < config.color_jitter_prob:
        out = _apply_color_jitter(out, rng)
        applied["color_jitter"] = True

    if rng.random() < config.blur_prob:
        k = _odd_kernel(int(rng.integers(3, 9)))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.0)
        applied["blur"] = True

    if rng.random() < config.motion_blur_prob:
        out = _apply_motion_blur(out, rng)
        applied["motion_blur"] = True

    if rng.random() < config.noise_prob:
        out = _apply_noise(out, rng)
        applied["noise"] = True

    if rng.random() < config.jpeg_artifact_prob:
        out = _apply_jpeg_artifact(out, rng)
        applied["jpeg_artifact"] = True

    return out, applied

