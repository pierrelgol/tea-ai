from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import yolo_bbox_to_corners_px


@dataclass(slots=True)
class CanonicalTarget:
    image_path: Path
    label_path: Path
    class_id_local: int
    class_name: str
    canonical_corners_px: np.ndarray


def load_target_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"Target classes file not found: {classes_file}")
    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_file}")
    return classes


def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 values in YOLO line, got {len(parts)}")
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    return class_id, x_center, y_center, width, height


def load_canonical_targets(
    target_images_dir: Path,
    target_labels_dir: Path,
    target_classes_file: Path,
) -> list[CanonicalTarget]:
    classes = load_target_classes(target_classes_file)
    image_paths = sorted(
        [
            p
            for p in target_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No target images found in {target_images_dir}")

    targets: list[CanonicalTarget] = []
    for image_path in image_paths:
        label_path = target_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        h, w = image.shape[:2]
        line = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not line:
            continue

        class_id, xc, yc, bw, bh = _parse_yolo_line(line[0])
        if class_id < 0 or class_id >= len(classes):
            raise ValueError(f"Class id {class_id} in {label_path} outside classes range")

        corners = yolo_bbox_to_corners_px(xc, yc, bw, bh, w, h)
        targets.append(
            CanonicalTarget(
                image_path=image_path,
                label_path=label_path,
                class_id_local=class_id,
                class_name=classes[class_id],
                canonical_corners_px=corners,
            )
        )

    if not targets:
        raise ValueError("No canonical targets could be loaded")
    return targets


def load_backgrounds_by_split(background_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        split_dir = background_root / "images" / split
        if not split_dir.exists():
            out[split] = []
            continue
        out[split] = sorted(
            [
                p
                for p in split_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
    return out


def write_yolo_label(path: Path, class_id: int, x_center: float, y_center: float, width: float, height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )


def write_yolo_obb_label(path: Path, class_id: int, obb_norm: np.ndarray) -> None:
    """Write YOLO OBB line: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = obb_norm.reshape(-1)
    coords = " ".join(f"{float(v):.10f}" for v in flat)
    path.write_text(f"{class_id} {coords}\n", encoding="utf-8")


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_augmented_classes(output_root: Path, target_classes: list[str], class_offset_base: int) -> None:
    classes_path = output_root / "classes.txt"
    classes_path.parent.mkdir(parents=True, exist_ok=True)

    reserved = [f"__coco_reserved_{i}__" for i in range(class_offset_base)]
    merged = reserved + target_classes
    classes_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    mapping_payload = {
        "class_offset_base": class_offset_base,
        "target_classes": [
            {
                "name": name,
                "local_id": i,
                "exported_id": class_offset_base + i,
            }
            for i, name in enumerate(target_classes)
        ],
    }
    write_metadata(output_root / "classes_map.json", mapping_payload)
