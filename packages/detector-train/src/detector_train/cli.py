from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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
            "failed to resolve default HF OBB model; pass --model with a local OBB .pt path"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model (lean pipeline)")
    parser.add_argument("--dataset", default="coco1024", help="Dataset name under dataset/augmented/")
    parser.add_argument("--model", default=HF_DEFAULT_ALIAS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path("dataset/augmented") / args.dataset
    wandb_run_name = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    model_path = _resolve_model_arg(args.model)

    config = TrainConfig(
        dataset_root=dataset_root,
        artifacts_root=Path("artifacts/detector-train"),
        model=model_path,
        name="current",
        seed=args.seed,
        wandb_run_name=wandb_run_name,
    )

    summary = train_detector(config)
    print(f"status: {summary['status']}")
    if summary["status"] == "ok":
        print(f"run_dir: {summary['artifacts']['save_dir']}")
        print(f"best_weights: {summary['artifacts']['weights_best']}")
        print(f"wandb_mode: {summary['wandb']['mode_used']}")
        if summary["wandb"].get("error"):
            print(f"wandb_note: {summary['wandb']['error']}")
