from __future__ import annotations

import argparse
from pathlib import Path

from .app import launch_gui
from .data import index_samples, resolve_model_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual checker for detector predictions vs ground truth")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under dataset/augmented/")
    parser.add_argument("--model", default="latest", help="Model key/source used under predictions")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()

    dataset_root = Path("dataset/augmented") / args.dataset

    samples = index_samples(dataset_root=dataset_root, splits=["val"])
    if not samples:
        raise RuntimeError(f"no samples found in dataset: {dataset_root}")

    model_key = resolve_model_key(
        model=args.model,
        artifacts_root=Path("artifacts/detector-train"),
        predictions_root=Path("predictions"),
    )

    print(f"dataset_root: {dataset_root}")
    print("predictions_root: predictions")
    print(f"model_key: {model_key}")
    print(f"samples: {len(samples)}")

    launch_gui(
        samples=samples,
        predictions_root=Path("predictions"),
        model_name=model_key,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
