from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import SampleRecord
from .yolo import label_to_pixel_corners, load_yolo_label


def _draw_polygon(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    poly = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)
    x0, y0 = int(poly[0][0][0]), int(poly[0][0][1])
    cv2.putText(image, text, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def export_debug_overlays(records: list[SampleRecord], reports_dir: Path, n_per_split: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    written: list[Path] = []

    for split in ("train", "val"):
        split_records = [r for r in records if r.split == split and r.image_path and r.label_path and r.meta_path]
        if not split_records:
            continue
        count = min(n_per_split, len(split_records))
        idxs = rng.choice(len(split_records), size=count, replace=False)

        out_dir = reports_dir / "overlays" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in idxs:
            rec = split_records[int(idx)]
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            gt = load_yolo_label(rec.label_path)
            h, w = img.shape[:2]
            gt_corners = label_to_pixel_corners(gt, w, h)
            _draw_polygon(img, gt_corners, (0, 255, 0), "GT")

            meta = json.loads(rec.meta_path.read_text(encoding="utf-8"))
            corners = np.array(meta["projected_corners_px"], dtype=np.float32)
            poly = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

            out_path = out_dir / f"{rec.stem}.jpg"
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

    return written
