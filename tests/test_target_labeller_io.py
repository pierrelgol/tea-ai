from __future__ import annotations

from pathlib import Path

import pytest

from target_labeller.io import (
    YoloBox,
    ensure_class_id,
    load_classes,
    load_yolo_label,
    save_yolo_label,
)


def test_ensure_class_id_reuses_existing_entries(tmp_path: Path) -> None:
    classes_file = tmp_path / "targets" / "classes.txt"

    can_id = ensure_class_id(classes_file, "can")
    bottle_id = ensure_class_id(classes_file, "bottle")
    can_id_again = ensure_class_id(classes_file, "can")

    assert can_id == 0
    assert bottle_id == 1
    assert can_id_again == 0
    assert load_classes(classes_file) == ["can", "bottle"]


def test_save_and_load_yolo_label_round_trip_class_id(tmp_path: Path) -> None:
    label_file = tmp_path / "targets" / "labels" / "sample.txt"
    box = YoloBox(x_center=0.5, y_center=0.5, width=0.25, height=0.4)

    save_yolo_label(label_file, 1, box)
    loaded = load_yolo_label(label_file)

    assert loaded is not None
    class_id, loaded_box = loaded
    assert class_id == 1
    assert loaded_box.x_center == pytest.approx(box.x_center)
    assert loaded_box.y_center == pytest.approx(box.y_center)
    assert loaded_box.width == pytest.approx(box.width)
    assert loaded_box.height == pytest.approx(box.height)
