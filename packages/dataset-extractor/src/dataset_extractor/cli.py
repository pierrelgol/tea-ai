from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import load_pipeline_config

from .fetch import extract_coco17_family


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract train2017.zip into pipeline dataset layouts")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    train_zip_path = shared.dataset.get("train_zip")
    if not isinstance(train_zip_path, Path):
        raise RuntimeError("config.dataset.train_zip must resolve to a filesystem path")

    coco17_dir, subsets = extract_coco17_family(
        train_zip_path=train_zip_path,
        dataset_root=shared.paths["dataset_root"],
        configs_root=shared.paths["configs_root"],
    )
    print(f"coco17 ready at: {coco17_dir}")
    print(f"train_zip: {train_zip_path}")
    print(f"train_images: {coco17_dir / 'images/train2017'}")
    print(f"val_images: {coco17_dir / 'images/val2017'}")
    for subset in subsets:
        print(
            f"subset: {subset['name']} "
            f"(train={subset['train_images']}, val={subset['val_images']})"
        )


if __name__ == "__main__":
    main()
