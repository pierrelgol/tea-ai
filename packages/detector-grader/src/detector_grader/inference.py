from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np


def resolve_device(requested: str) -> str:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _format_obb_line(class_id: int, corners_norm: np.ndarray, conf: float) -> str:
    flat = corners_norm.reshape(-1)
    coords = " ".join(f"{float(v):.10f}" for v in flat)
    return f"{class_id} {coords} {float(conf):.10f}"


def _result_to_lines(res: Any) -> list[str]:
    if getattr(res, "obb", None) is None:
        raise RuntimeError("Model prediction does not expose OBB output; OBB model/weights are required")

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

    out: list[str] = []
    for i in range(coords.shape[0]):
        out.append(_format_obb_line(int(classes[i]), coords[i], float(confs[i])))
    return out


def _write_prediction_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _list_split_images(dataset_root: Path, split: str) -> list[Path]:
    split_dir = dataset_root / "images" / split
    if not split_dir.exists():
        return []
    images = [
        p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    return sorted(images)


def run_inference_for_grading(
    *,
    weights: Path,
    dataset_root: Path,
    predictions_root: Path,
    model_name: str,
    splits: list[str],
    imgsz: int,
    device: str,
    conf_threshold: float,
    infer_iou_threshold: float,
    seed: int,
) -> dict:
    set_seed(seed)
    resolved_device = resolve_device(device)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    written = 0
    total_images = 0
    out_root = predictions_root / model_name / "labels"

    for split in splits:
        images = _list_split_images(dataset_root, split)
        for image_path in images:
            total_images += 1
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=infer_iou_threshold,
                imgsz=imgsz,
                device=resolved_device,
                verbose=False,
            )
            lines = _result_to_lines(results[0])
            _write_prediction_file(out_root / split / f"{image_path.stem}.txt", lines)
            written += 1

    return {
        "status": "ok",
        "weights": str(weights),
        "model_name": model_name,
        "resolved_device": resolved_device,
        "images_processed": total_images,
        "label_files_written": written,
        "predictions_root": str(predictions_root),
    }
