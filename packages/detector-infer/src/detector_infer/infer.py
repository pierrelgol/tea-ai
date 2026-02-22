from __future__ import annotations

import random
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .config import InferConfig
from .dataset import list_split_images
from .writer import _format_obb_line, write_prediction_file


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _result_to_lines(res: Any) -> list[str]:
    lines: list[str] = []

    if getattr(res, "obb", None) is not None:
        obb = res.obb
        if hasattr(obb, "xyxyxyxyn") and obb.xyxyxyxyn is not None:
            coords = obb.xyxyxyxyn.cpu().numpy()
        elif hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
            px = obb.xyxyxyxy.cpu().numpy()
            h, w = res.orig_shape[:2]
            coords = px.astype(np.float64)
            coords[:, :, 0] /= w
            coords[:, :, 1] /= h
        else:
            coords = np.zeros((0, 4, 2), dtype=np.float32)

        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.ones((coords.shape[0],), dtype=np.float32)
        classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else np.zeros((coords.shape[0],), dtype=int)

        coords = np.clip(coords, 0.0, 1.0)
        for i in range(coords.shape[0]):
            lines.append(_format_obb_line(int(classes[i]), coords[i], float(confs[i])))
        return lines

    raise RuntimeError("Model prediction does not expose OBB output; OBB model/weights are required")


def run_inference(config: InferConfig) -> dict:
    config.validate()
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    model_labels_root = config.output_root / config.model_name / "labels"
    if model_labels_root.exists():
        shutil.rmtree(model_labels_root)

    from ultralytics import YOLO

    model = YOLO(str(config.weights))
    written = 0
    total_images = 0

    for split in config.splits:
        images = list_split_images(config.dataset_root, split)
        for image_path in images:
            total_images += 1
            results = model.predict(
                source=str(image_path),
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                imgsz=config.imgsz,
                device=device,
                verbose=False,
            )
            res = results[0]
            lines = _result_to_lines(res)

            out_path = config.output_root / config.model_name / "labels" / split / f"{image_path.stem}.txt"
            write_prediction_file(out_path, lines, save_empty=config.save_empty)
            written += 1

    return {
        "status": "ok",
        "weights": str(config.weights),
        "model_name": config.model_name,
        "output_root": str(config.output_root),
        "resolved_device": device,
        "images_processed": total_images,
        "label_files_written": written,
    }
