from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import corners_to_pixel, load_meta
from .types import ParsedLabel, SampleRecord


def _draw_poly(img: np.ndarray, corners_px: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    poly = corners_px.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [poly], True, color, 2)
    x0, y0 = int(poly[0][0][0]), int(poly[0][0][1])
    cv2.putText(img, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def write_visual_artifacts(
    records: list[SampleRecord],
    gt_lookup: dict[tuple[str, str], tuple[list[ParsedLabel], int, int]],
    pred_lookup: dict[tuple[str, str], tuple[list[ParsedLabel], int, int]],
    reports_dir: Path,
    model_name: str,
    sample_count_per_split: int,
    seed: int,
) -> list[Path]:
    if sample_count_per_split <= 0:
        return []

    rng = np.random.default_rng(seed)
    written: list[Path] = []

    for split in ("train", "val"):
        candidates = [r for r in records if r.split == split and r.image_path is not None]
        if not candidates:
            continue

        n = min(sample_count_per_split, len(candidates))
        idxs = rng.choice(len(candidates), size=n, replace=False)

        out_dir = reports_dir / "overlays" / model_name / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in idxs:
            rec = candidates[int(idx)]
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]
            key = (rec.split, rec.stem)
            gt = gt_lookup.get(key, ([], h, w))[0]
            pred = pred_lookup.get(key, ([], h, w))[0]

            if gt:
                _draw_poly(img, corners_to_pixel(gt[0].corners_norm, w, h), (0, 255, 0), "GT")
            if pred:
                _draw_poly(img, corners_to_pixel(pred[0].corners_norm, w, h), (255, 255, 0), "PRED")

            meta = load_meta(rec.meta_path)
            if meta is not None:
                corners = np.array(meta.get("projected_corners_px", []), dtype=np.float32)
                if corners.shape == (4, 2):
                    _draw_poly(img, corners, (0, 0, 255), "META")

            out_path = out_dir / f"{rec.stem}.jpg"
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

    return written
