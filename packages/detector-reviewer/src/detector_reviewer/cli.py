from __future__ import annotations

import argparse
from pathlib import Path

from .app import launch_gui
from .data import index_samples, resolve_model_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual checker for detector predictions vs ground truth")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--model", default="latest", help="Model key/source used under predictions")
    parser.add_argument("--splits", default="val", help="Comma-separated splits to review (default: val)")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    samples = index_samples(dataset_root=dataset_root, splits=splits)
    if not samples:
        raise RuntimeError(f"no samples found in dataset: {dataset_root}")

    model_key = resolve_model_key(
        model=args.model,
        artifacts_root=args.artifacts_root,
        predictions_root=args.predictions_root,
    )

    print(f"dataset_root: {dataset_root}")
    print(f"predictions_root: {args.predictions_root}")
    print(f"model_key: {model_key}")
    print(f"samples: {len(samples)}")

    launch_gui(
        samples=samples,
        predictions_root=args.predictions_root,
        model_name=model_key,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
