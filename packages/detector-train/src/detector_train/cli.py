from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

from .config import TrainConfig
from .trainer import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model (OBB) and track run artifacts")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
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
    parser.add_argument("--train-profile", choices=["default", "obb_precision_v1"], default="obb_precision_v1")

    parser.add_argument("--optimizer", choices=["SGD", "AdamW", "auto"], default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--shear", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--hsv-h", type=float, default=None)
    parser.add_argument("--hsv-s", type=float, default=None)
    parser.add_argument("--hsv-v", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=None)
    parser.add_argument("--copy-paste", type=float, default=None)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="tea-ai-detector")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "auto"], default="auto")
    parser.add_argument("--wandb-log-profile", choices=["core", "core+diag"], default="core+diag")
    parser.add_argument("--wandb-log-system-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-log-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval-epochs", type=int, default=5)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--eval-conf-threshold", type=float, default=0.25)
    parser.add_argument("--eval-viz-samples", type=int, default=0)

    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset
    run_name = args.name or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    config = TrainConfig(
        dataset_root=dataset_root,
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
        train_profile=args.train_profile,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        flipud=args.flipud,
        copy_paste=args.copy_paste,
        multi_scale=args.multi_scale,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_log_profile=args.wandb_log_profile,
        wandb_log_system_metrics=args.wandb_log_system_metrics,
        wandb_log_every_epoch=args.wandb_log_every_epoch,
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
