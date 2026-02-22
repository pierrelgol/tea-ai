from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import GeneratorConfig
from .generator import generate_dataset


def load_dataset_config(dataset_name: str) -> dict:
    config_path = Path("configs/datasets") / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"dataset config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented dataset")
    parser.add_argument(
        "--dataset", default="coco8", help="Dataset directory name under dataset/"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset_root = Path("dataset")
    dataset_name = args.dataset

    dataset_config = load_dataset_config(dataset_name)
    splits = dataset_config.get("splits", {})
    train_rel = splits.get("train_images_rel", "images/train")
    val_rel = splits.get("val_images_rel", "images/val")

    background_splits = {
        "train": dataset_root / dataset_name / train_rel,
        "val": dataset_root / dataset_name / val_rel,
    }
    output_root = dataset_root / "augmented" / dataset_name

    config = GeneratorConfig(
        background_splits=background_splits,
        background_dataset_name=dataset_name,
        target_images_dir=dataset_root / "targets" / "images",
        target_labels_dir=dataset_root / "targets" / "labels",
        target_classes_file=dataset_root / "targets" / "classes.txt",
        output_root=output_root,
        seed=args.seed,
    )

    results = generate_dataset(config)
    print(f"generated {len(results)} synthetic samples in {config.output_root}")
