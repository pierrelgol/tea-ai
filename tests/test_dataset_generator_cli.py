from __future__ import annotations

from dataset_generator.cli import load_dataset_config


def test_load_dataset_config_falls_back_for_generated_coco_subsets(tmp_path) -> None:
    config = load_dataset_config(tmp_path, "coco256")
    assert config["name"] == "coco256"
    assert config["splits"]["train_images_rel"] == "images/train2017"
    assert config["splits"]["val_images_rel"] == "images/val2017"
