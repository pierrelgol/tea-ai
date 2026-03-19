from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset_generator.config import GeneratorConfig
from dataset_generator.generator import generate_dataset
from dataset_generator.io import load_canonical_targets


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    assert ok


def _base_generator_config(tmp_path: Path) -> GeneratorConfig:
    return GeneratorConfig(
        background_splits={
            "train": tmp_path / "backgrounds" / "train",
            "val": tmp_path / "backgrounds" / "val",
        },
        background_dataset_name="coco256",
        target_images_dir=tmp_path / "targets" / "images",
        target_labels_dir=tmp_path / "targets" / "labels",
        target_classes_file=tmp_path / "targets" / "classes.txt",
        output_root=tmp_path / "augmented",
        curriculum_enabled=False,
        samples_per_background=1,
        seed=42,
        targets_per_image_min=1,
        targets_per_image_max=1,
        empty_sample_prob=0.0,
        scale_min=0.35,
        scale_max=0.35,
        translate_frac=0.05,
        perspective_jitter=0.0,
        min_quad_area_frac=0.001,
        max_attempts=20,
        edge_bias_prob=0.0,
    )


def _write_canonical_target(
    tmp_path: Path,
    *,
    stem: str,
    class_id: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    target_image = np.zeros((64, 64, 3), dtype=np.uint8)
    target_image[:, :] = color
    _write_image(tmp_path / "targets" / "images" / f"{stem}.jpg", target_image)
    (tmp_path / "targets" / "labels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "targets" / "labels" / f"{stem}.txt").write_text(
        f"{class_id} 0 0 1 0 1 1 0 1\n",
        encoding="utf-8",
    )


def _write_backgrounds(tmp_path: Path) -> None:
    bg_train = np.full((256, 256, 3), 80, dtype=np.uint8)
    bg_val = np.full((256, 256, 3), 120, dtype=np.uint8)
    _write_image(tmp_path / "backgrounds" / "train" / "bg_train.jpg", bg_train)
    _write_image(tmp_path / "backgrounds" / "val" / "bg_val.jpg", bg_val)


def test_generate_dataset_uses_threaded_workers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEA_AI_GENERATOR_THREADS", "2")

    _write_canonical_target(tmp_path, stem="target0", class_id=0)
    (tmp_path / "targets" / "classes.txt").write_text("target\n", encoding="utf-8")
    _write_backgrounds(tmp_path)

    config = _base_generator_config(tmp_path)

    results = generate_dataset(config)

    assert len(results) == 2
    summary = json.loads((tmp_path / "augmented" / "generation_summary.json").read_text(encoding="utf-8"))
    assert summary["worker_count"] == 2
    assert (tmp_path / "augmented" / "images" / "train").exists()
    assert (tmp_path / "augmented" / "images" / "val").exists()


def test_generate_dataset_writes_multi_class_targets_and_class_map(tmp_path) -> None:
    _write_canonical_target(tmp_path, stem="can_a0", class_id=0, color=(255, 255, 255))
    _write_canonical_target(tmp_path, stem="can_a1", class_id=0, color=(200, 200, 200))
    _write_canonical_target(tmp_path, stem="can_b0", class_id=1, color=(255, 255, 0))
    (tmp_path / "targets" / "classes.txt").write_text("can\nbottle\n", encoding="utf-8")
    _write_backgrounds(tmp_path)

    config = _base_generator_config(tmp_path)
    config.samples_per_background = 2
    config.class_offset_base = 80

    results = generate_dataset(config)

    assert len(results) == 4

    all_exported_ids: set[int] = set()
    for label_path in (tmp_path / "augmented" / "labels").rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_exported_ids.add(int(line.split()[0]))
    assert all_exported_ids
    assert all_exported_ids <= {80, 81}

    classes = (tmp_path / "augmented" / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert len(classes) == 82
    assert classes[-2:] == ["can", "bottle"]

    class_map = json.loads((tmp_path / "augmented" / "classes_map.json").read_text(encoding="utf-8"))
    assert class_map["class_offset_base"] == 80
    assert class_map["target_classes"] == [
        {"name": "can", "local_id": 0, "exported_id": 80},
        {"name": "bottle", "local_id": 1, "exported_id": 81},
    ]


def test_load_canonical_targets_rejects_multiline_labels(tmp_path) -> None:
    _write_canonical_target(tmp_path, stem="target0", class_id=0)
    (tmp_path / "targets" / "labels" / "target0.txt").write_text(
        "0 0 0 1 0 1 1 0 1\n0 0 0 1 0 1 1 0 1\n",
        encoding="utf-8",
    )
    (tmp_path / "targets" / "classes.txt").write_text("target\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one OBB row"):
        load_canonical_targets(
            tmp_path / "targets" / "images",
            tmp_path / "targets" / "labels",
            tmp_path / "targets" / "classes.txt",
        )


def test_load_canonical_targets_rejects_unknown_class_id(tmp_path) -> None:
    _write_canonical_target(tmp_path, stem="target0", class_id=2)
    (tmp_path / "targets" / "classes.txt").write_text("target\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside classes range"):
        load_canonical_targets(
            tmp_path / "targets" / "images",
            tmp_path / "targets" / "labels",
            tmp_path / "targets" / "classes.txt",
        )
