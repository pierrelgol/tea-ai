from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(slots=True)
class Label:
    class_id: int
    corners_norm: np.ndarray  # shape (4, 2)
    confidence: float


@dataclass(slots=True)
class Sample:
    split: str
    stem: str
    image_path: Path | None
    label_path: Path | None


def index_samples(dataset_root: Path, splits: list[str]) -> list[Sample]:
    out: list[Sample] = []
    for split in splits:
        img_dir = dataset_root / "images" / split
        lab_dir = dataset_root / "labels" / split
        stems: set[str] = set()

        if img_dir.exists():
            stems |= {p.stem for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        if lab_dir.exists():
            stems |= {p.stem for p in lab_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        for stem in sorted(stems):
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            label_path = lab_dir / f"{stem}.txt"
            out.append(
                Sample(
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                )
            )
    return out


def bbox_to_corners_norm(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def parse_label_line(line: str, is_prediction: bool) -> Label:
    parts = line.strip().split()
    if len(parts) not in (5, 6, 9, 10):
        raise ValueError("expected YOLO bbox/obb line")

    class_id = int(parts[0])
    vals = [float(x) for x in parts[1:]]

    if len(parts) in (5, 6):
        if len(parts) == 6:
            xc, yc, w, h, conf = vals
        else:
            xc, yc, w, h = vals
            conf = 1.0
        return Label(class_id=class_id, corners_norm=bbox_to_corners_norm(xc, yc, w, h), confidence=float(conf))

    if len(parts) == 10:
        coords = vals[:8]
        conf = vals[8]
    else:
        coords = vals
        conf = 1.0
    corners = np.array(coords, dtype=np.float32).reshape(4, 2)
    if not is_prediction:
        conf = 1.0
    return Label(class_id=class_id, corners_norm=corners, confidence=float(conf))


def load_labels(path: Path | None, is_prediction: bool, conf_threshold: float = 0.0) -> list[Label]:
    if path is None or not path.exists():
        return []
    out: list[Label] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = parse_label_line(line, is_prediction=is_prediction)
        if is_prediction and label.confidence < conf_threshold:
            continue
        out.append(label)
    return out


def corners_to_px(corners_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    out = corners_norm.astype(np.float64).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    return out.astype(np.float32)


def resolve_latest_weights(
    artifacts_root: Path,
    explicit_weights: Path | None,
) -> Path:
    if explicit_weights is not None:
        if explicit_weights.exists():
            return explicit_weights
        raise FileNotFoundError(f"weights not found: {explicit_weights}")

    latest_file = artifacts_root / "latest_run.json"
    if latest_file.exists():
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        best = payload.get("weights_best")
        if isinstance(best, str) and best:
            candidate = Path(best)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                return candidate

    # Fallback to newest best.pt under artifacts tree.
    candidates = sorted(
        (artifacts_root / "runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError("could not resolve best weights from latest_run.json or artifacts runs")
