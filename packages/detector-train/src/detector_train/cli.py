from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import TrainConfig
from .trainer import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model (lean pipeline)")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset
    run_name = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")

    config = TrainConfig(
        dataset_root=dataset_root,
        artifacts_root=args.artifacts_root,
        model=args.model,
        name=run_name,
        seed=args.seed,
    )

    summary = train_detector(config)
    print(f"status: {summary['status']}")
    if summary["status"] == "ok":
        print(f"run_dir: {summary['artifacts']['save_dir']}")
        print(f"best_weights: {summary['artifacts']['weights_best']}")
        print(f"wandb_mode: {summary['wandb']['mode_used']}")
        if summary["wandb"].get("error"):
            print(f"wandb_note: {summary['wandb']['error']}")
