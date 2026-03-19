from __future__ import annotations

from pathlib import Path

import yaml

from detector_train.data_yaml import write_data_yaml


def test_write_data_yaml_uses_all_class_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "classes.txt").write_text("can\nbottle\ncup\n", encoding="utf-8")

    output_path = tmp_path / "data.yaml"
    written_path, names = write_data_yaml(dataset_root, output_path)

    payload = yaml.safe_load(written_path.read_text(encoding="utf-8"))
    assert names == ["can", "bottle", "cup"]
    assert payload["nc"] == 3
    assert payload["names"] == ["can", "bottle", "cup"]
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
