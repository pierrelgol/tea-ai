from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config

from .config import TrainConfig
from .trainer import train_detector

HF_DEFAULT_ALIAS = "hf-openvision-yolo26-n-obb"


def _resolve_model_arg(model_arg: str) -> str:
    if model_arg != HF_DEFAULT_ALIAS:
        return model_arg
    try:
        from huggingface_hub import hf_hub_download

        return str(
            hf_hub_download(
                repo_id="openvision/yolo26-n-obb",
                filename="model.pt",
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to resolve default HF OBB model; set run.model to a local OBB .pt path"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    run_id = str(shared.run["run_id"])
    model_key = str(shared.run["model_key"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=run_id,
    )
    model_path = _resolve_model_arg(str(shared.run["model"]))

    tc = shared.train
    config = TrainConfig(
        dataset_root=dataset_root,
        artifacts_root=layout.run_root,
        project=layout.train_root / "ultralytics",
        model=model_path,
        name=run_id,
        seed=int(shared.run["seed"]),
        device=str(tc.get("device", "auto")),
        epochs=int(tc.get("epochs", 128)),
        imgsz=int(tc.get("imgsz", 512)),
        batch=int(tc.get("batch", 16)),
        workers=int(tc.get("workers", 16)),
        patience=int(tc.get("patience", 30)),
        cache=str(tc.get("cache", "auto")),
        amp=bool(tc.get("amp", True)),
        plots=bool(tc.get("plots", True)),
        tf32=bool(tc.get("tf32", True)),
        cudnn_benchmark=bool(tc.get("cudnn_benchmark", True)),
        optimizer=str(tc.get("optimizer", "AdamW")),
        lr0=float(tc.get("lr0", 0.0012)),
        lrf=float(tc.get("lrf", 0.01)),
        weight_decay=float(tc.get("weight_decay", 0.0006)),
        warmup_epochs=float(tc.get("warmup_epochs", 6.0)),
        cos_lr=bool(tc.get("cos_lr", True)),
        close_mosaic=int(tc.get("close_mosaic", 12)),
        mosaic=float(tc.get("mosaic", 0.5)),
        mixup=float(tc.get("mixup", 0.03)),
        degrees=float(tc.get("degrees", 1.0)),
        translate=float(tc.get("translate", 0.035)),
        scale=float(tc.get("scale", 0.35)),
        shear=float(tc.get("shear", 0.0)),
        perspective=float(tc.get("perspective", 0.0)),
        hsv_h=float(tc.get("hsv_h", 0.01)),
        hsv_s=float(tc.get("hsv_s", 0.30)),
        hsv_v=float(tc.get("hsv_v", 0.22)),
        fliplr=float(tc.get("fliplr", 0.5)),
        flipud=float(tc.get("flipud", 0.0)),
        copy_paste=float(tc.get("copy_paste", 0.0)),
        multi_scale=bool(tc.get("multi_scale", False)),
        freeze=None if tc.get("freeze") is None else int(tc.get("freeze")),
        dino_root=Path(str(tc.get("dino_root", "dinov3"))),
        dino_distill_warmup_epochs=int(tc.get("dino_distill_warmup_epochs", 5)),
        dino_distill_layers=tuple(int(v) for v in tc.get("dino_distill_layers", [19])),
        dino_distill_channels=int(tc.get("dino_distill_channels", 32)),
        dino_distill_object_weight=float(tc.get("dino_distill_object_weight", 1.15)),
        dino_distill_background_weight=float(tc.get("dino_distill_background_weight", 0.15)),
        stage_a_ratio=float(tc.get("stage_a_ratio", 0.30)),
        stage_a_freeze=int(tc.get("stage_a_freeze", 10)),
        stage_a_distill_weight=float(tc.get("stage_a_distill_weight", 0.25)),
        stage_b_distill_weight=float(tc.get("stage_b_distill_weight", 0.08)),
        dino_viz_enabled=bool(tc.get("dino_viz_enabled", True)),
        dino_viz_every_n_epochs=int(tc.get("dino_viz_every_n_epochs", 5)),
        dino_viz_max_samples=int(tc.get("dino_viz_max_samples", 4)),
        wandb_enabled=bool(tc.get("wandb_enabled", True)),
        wandb_project=str(tc.get("wandb_project", "tea-ai-detector")),
        wandb_entity=tc.get("wandb_entity"),
        wandb_run_name=tc.get("wandb_run_name") or run_id,
        wandb_tags=list(tc.get("wandb_tags", [])),
        wandb_notes=tc.get("wandb_notes"),
        wandb_mode=str(tc.get("wandb_mode", "auto")),
        wandb_log_system_metrics=bool(tc.get("wandb_log_system_metrics", False)),
        wandb_log_every_epoch=bool(tc.get("wandb_log_every_epoch", True)),
        eval_enabled=bool(tc.get("eval_enabled", True)),
        eval_interval_epochs=int(tc.get("eval_interval_epochs", 2)),
        eval_iou_threshold=float(tc.get("eval_iou_threshold", 0.75)),
        eval_conf_threshold=float(tc.get("eval_conf_threshold", 0.90)),
        eval_viz_samples=int(tc.get("eval_viz_samples", 8)),
        eval_viz_split=str(tc.get("eval_viz_split", "val")),
    )

    summary = train_detector(config)
    print(f"status: {summary['status']}")
    if summary["status"] == "ok":
        print(f"run_root: {layout.run_root}")
        print(f"run_dir: {summary['artifacts']['save_dir']}")
        print(f"best_weights: {summary['artifacts']['weights_best']}")
        print(f"wandb_mode: {summary['wandb']['mode_used']}")
        if summary["wandb"].get("error"):
            print(f"wandb_note: {summary['wandb']['error']}")


if __name__ == "__main__":
    main()
