from __future__ import annotations

import argparse
from pathlib import Path

from .config import GeneratorConfig
from .generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented dataset")
    parser.add_argument("--dataset", default="coco8", help="Dataset directory name under --dataset-root")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Top-level datasets directory")
    parser.add_argument("--output-root", type=Path, default=None, help="Augmented output root")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    dataset_name = args.dataset
    background_splits = {
        "train": dataset_root / dataset_name / "images" / "train",
        "val": dataset_root / dataset_name / "images" / "val",
    }
    output_root = args.output_root if args.output_root else dataset_root / "augmented" / dataset_name

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
