from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

from .config import TrainConfig
from .trainer import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model (OBB) and track run artifacts")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("artifacts/detector-train/runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="tea-ai-detector")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "auto"], default="auto")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval-epochs", type=int, default=5)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--eval-conf-threshold", type=float, default=0.25)
    parser.add_argument("--eval-viz-samples", type=int, default=0)

    args = parser.parse_args()

    run_name = args.name or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    config = TrainConfig(
        dataset_root=args.dataset_root,
        artifacts_root=args.artifacts_root,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        save_json=args.save_json,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        eval_enabled=args.eval,
        eval_interval_epochs=args.eval_interval_epochs,
        eval_iou_threshold=args.eval_iou_threshold,
        eval_conf_threshold=args.eval_conf_threshold,
        eval_viz_samples=args.eval_viz_samples,
    )

    summary = train_detector(config)
    print(f"status: {summary['status']}")
    if summary["status"] == "ok":
        print(f"run_dir: {summary['artifacts']['save_dir']}")
        print(f"best_weights: {summary['artifacts']['weights_best']}")
        print(f"wandb_mode: {summary['wandb']['mode_used']}")
        if summary["wandb"].get("error"):
            print(f"wandb_note: {summary['wandb']['error']}")
