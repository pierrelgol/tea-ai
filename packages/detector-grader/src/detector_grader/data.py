from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(slots=True)
class Label:
    class_id: int
    corners_norm: np.ndarray  # (4, 2)
    confidence: float


@dataclass(slots=True)
class SampleRecord:
    split: str
    stem: str
    image_path: Path | None
    gt_label_path: Path | None


def sanitize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def parse_gt_line(line: str) -> Label:
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError("GT label must have 9 OBB fields")
    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]
    corners = np.array(vals, dtype=np.float32).reshape(4, 2)
    return Label(class_id=class_id, corners_norm=corners, confidence=1.0)


def parse_pred_line(line: str) -> Label:
    parts = line.strip().split()
    if len(parts) not in (9, 10):
        raise ValueError("prediction label must have 9 (no conf) or 10 (with conf) OBB fields")
    class_id = int(parts[0])
    corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
    conf = float(parts[9]) if len(parts) == 10 else 1.0
    return Label(class_id=class_id, corners_norm=corners, confidence=conf)


def load_labels(path: Path, *, is_prediction: bool, conf_threshold: float) -> list[Label]:
    if not path.exists():
        return []
    out: list[Label] = []
    parser = parse_pred_line if is_prediction else parse_gt_line
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = parser(line)
        except Exception as exc:
            raise ValueError(f"invalid OBB label at {path}:{line_no}: {exc}") from exc
        if is_prediction and label.confidence < conf_threshold:
            continue
        out.append(label)
    return out


def index_ground_truth(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        lab_dir = dataset_root / "labels" / split

        stems: set[str] = set()
        if img_dir.exists():
            stems |= {p.stem for p in img_dir.iterdir() if p.is_file()}
        if lab_dir.exists():
            stems |= {p.stem for p in lab_dir.iterdir() if p.is_file() and p.suffix == ".txt"}

        for stem in sorted(stems):
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    image_path = p
                    break
            gt = lab_dir / f"{stem}.txt"
            records.append(
                SampleRecord(
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    gt_label_path=gt if gt.exists() else None,
                )
            )
    return records


def load_prediction_labels(
    predictions_root: Path,
    model_name: str,
    split: str,
    stem: str,
    conf_threshold: float,
) -> list[Label]:
    p = predictions_root / model_name / "labels" / split / f"{stem}.txt"
    return load_labels(p, is_prediction=True, conf_threshold=conf_threshold)


def image_shape(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return h, w


def resolve_latest_weights(artifacts_root: Path) -> Path:
    latest_file = artifacts_root / "latest_run.json"
    if latest_file.exists():
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        best = payload.get("weights_best")
        if isinstance(best, str) and best:
            p = Path(best)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                return p

    candidates = sorted(
        (artifacts_root / "runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("could not resolve latest best weights from artifacts root")


def infer_model_name_from_weights(weights: Path) -> str:
    if weights.parent.name == "weights" and weights.parent.parent.name:
        run_name = weights.parent.parent.name
        suffix = weights.stem
        return sanitize_model_name(f"{run_name}_{suffix}")
    return sanitize_model_name(weights.stem)
