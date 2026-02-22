from __future__ import annotations

import argparse
from pathlib import Path

from .config import InferConfig
from .infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference and export YOLO predictions")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="coco8", help="Dataset name under dataset/augmented/")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path("dataset/augmented") / args.dataset

    cfg = InferConfig(
        weights=args.weights,
        dataset_root=dataset_root,
        output_root=Path("predictions"),
        model_name=args.model_name,
        imgsz=640,
        device="auto",
        conf_threshold=args.conf_threshold,
        iou_threshold=0.7,
        seed=args.seed,
        splits=["train", "val"],
        save_empty=True,
    )

    summary = run_inference(cfg)
    print(f"status: {summary['status']}")
    print(f"model: {summary['model_name']}")
    print(f"images_processed: {summary['images_processed']}")
    print(f"label_files_written: {summary['label_files_written']}")
    print(f"output_root: {summary['output_root']}")
