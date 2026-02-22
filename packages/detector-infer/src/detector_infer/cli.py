from __future__ import annotations

import argparse
from pathlib import Path

from .config import InferConfig
from .infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference and export YOLO predictions")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--output-root", type=Path, default=Path("predictions"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--save-empty", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset

    cfg = InferConfig(
        weights=args.weights,
        dataset_root=dataset_root,
        output_root=args.output_root,
        model_name=args.model_name,
        imgsz=args.imgsz,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        seed=args.seed,
        splits=splits,
        save_empty=args.save_empty,
    )

    summary = run_inference(cfg)
    print(f"status: {summary['status']}")
    print(f"model: {summary['model_name']}")
    print(f"images_processed: {summary['images_processed']}")
    print(f"label_files_written: {summary['label_files_written']}")
    print(f"output_root: {summary['output_root']}")
