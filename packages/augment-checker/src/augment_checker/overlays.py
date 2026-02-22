from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import SampleRecord
from .yolo import load_yolo_box, yolo_to_xyxy


def _draw_xyxy(image: np.ndarray, xyxy: tuple[float, float, float, float], color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = xyxy
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, text, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


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

            gt = load_yolo_box(rec.label_path)
            h, w = img.shape[:2]
            gt_xyxy = yolo_to_xyxy(gt, w, h)
            _draw_xyxy(img, gt_xyxy, (0, 255, 0), "GT")

            meta = json.loads(rec.meta_path.read_text(encoding="utf-8"))
            corners = np.array(meta["projected_corners_px"], dtype=np.float32)
            poly = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

            out_path = out_dir / f"{rec.stem}.jpg"
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

    return written
