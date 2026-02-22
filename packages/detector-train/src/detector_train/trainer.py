from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import csv
import json

from .config import TrainConfig
from .data_yaml import write_data_yaml
from .wandb_logger import finish_wandb, init_wandb, log_wandb


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _extract_last_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}

    out: dict[str, float] = {}
    for k, v in rows[-1].items():
        if v is None:
            continue
        vv = v.strip()
        if vv == "":
            continue
        try:
            out[k] = float(vv)
        except Exception:
            continue
    return out


def _json_safe(obj):
    """Convert objects (e.g. Path) into JSON-serializable primitives."""
    return json.loads(json.dumps(obj, default=str))


def _run_periodic_eval(
    *,
    config: TrainConfig,
    save_dir: Path,
    epoch: int,
    device: str,
    run_name: str,
    wandb_run,
) -> dict[str, Any]:
    from detector_evaluator.evaluator import evaluate_models
    from detector_infer.config import InferConfig
    from detector_infer.infer import run_inference

    weights_last = save_dir / "weights" / "last.pt"
    if not weights_last.exists():
        return {"status": "skipped", "reason": f"missing checkpoint: {weights_last}"}

    pred_root = config.artifacts_root / "eval_predictions" / run_name / f"epoch_{epoch:03d}"
    reports_dir = config.artifacts_root / "eval_reports" / run_name / f"epoch_{epoch:03d}"
    model_key = run_name

    infer_cfg = InferConfig(
        weights=weights_last,
        dataset_root=config.dataset_root,
        output_root=pred_root,
        model_name=model_key,
        imgsz=config.imgsz,
        device=device,
        conf_threshold=config.eval_conf_threshold,
        iou_threshold=config.eval_iou_threshold,
        seed=config.seed,
        splits=["train", "val"],
        save_empty=True,
    )
    infer_summary = run_inference(infer_cfg)

    eval_summary = evaluate_models(
        dataset_root=config.dataset_root,
        predictions_root=pred_root,
        reports_dir=reports_dir,
        iou_threshold=config.eval_iou_threshold,
        conf_threshold=config.eval_conf_threshold,
        seed=config.seed,
        viz_samples=config.eval_viz_samples,
        models_filter=[model_key],
    )

    row = eval_summary["models"][0] if eval_summary["models"] else None
    if row is not None:
        log_wandb(
            wandb_run,
            {
                "eval/precision": row.get("precision"),
                "eval/recall": row.get("recall"),
                "eval/miss_rate": row.get("miss_rate"),
                "eval/mean_iou": row.get("mean_iou"),
                "eval/ap_at_iou": row.get("ap_at_iou"),
                "eval/mean_center_drift_px": row.get("mean_center_drift_px"),
                "eval/geometry_mean_corner_error_label_vs_meta_px": row.get(
                    "geometry_mean_corner_error_label_vs_meta_px"
                ),
            },
            step=epoch,
        )

    return {
        "status": "ok",
        "epoch": epoch,
        "infer": infer_summary,
        "eval": eval_summary,
    }


def train_detector(config: TrainConfig) -> dict[str, Any]:
    config.validate()
    device = _resolve_device(config.device)

    data_yaml_path, names = write_data_yaml(
        dataset_root=config.dataset_root,
        output_path=config.artifacts_root / "data.yaml",
    )

    wandb_cfg = {
        "dataset_root": str(config.dataset_root),
        "data_yaml": str(data_yaml_path),
        "model": config.model,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": device,
        "seed": config.seed,
        "workers": config.workers,
        "patience": config.patience,
        "classes_count": len(names),
    }

    run_name = config.wandb_run_name or config.name
    wandb_run, wandb_state = init_wandb(
        enabled=config.wandb_enabled,
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=run_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=wandb_cfg,
    )

    try:
        from ultralytics import YOLO

        model = YOLO(config.model)
        periodic_eval: list[dict[str, Any]] = []
        last_eval_epoch = -1

        def _on_fit_epoch_end(trainer) -> None:
            nonlocal last_eval_epoch
            if not config.eval_enabled:
                return

            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            if current_epoch <= 0:
                return
            if current_epoch == last_eval_epoch:
                return

            should_eval = (current_epoch % config.eval_interval_epochs == 0) or (current_epoch == config.epochs)
            if not should_eval:
                return

            save_dir_cb = Path(getattr(trainer, "save_dir", config.project / config.name))
            try:
                eval_result = _run_periodic_eval(
                    config=config,
                    save_dir=save_dir_cb,
                    epoch=current_epoch,
                    device=device,
                    run_name=config.name,
                    wandb_run=wandb_run,
                )
                periodic_eval.append(eval_result)
            except Exception as exc:
                periodic_eval.append(
                    {
                        "status": "error",
                        "epoch": current_epoch,
                        "error": str(exc),
                    }
                )
                log_wandb(wandb_run, {"eval/error": 1, "eval/error_message": str(exc)}, step=current_epoch)

            last_eval_epoch = current_epoch

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        train_result = model.train(
            data=str(data_yaml_path),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=device,
            project=str(config.project),
            name=config.name,
            seed=config.seed,
            workers=config.workers,
            patience=config.patience,
            exist_ok=True,
        )

        save_dir = Path(getattr(train_result, "save_dir", config.project / config.name))
        weights_dir = save_dir / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        results_csv = save_dir / "results.csv"

        metrics = _extract_last_metrics(results_csv)
        log_wandb(wandb_run, metrics)

        summary = {
            "status": "ok",
            "config": _json_safe(asdict(config)),
            "wandb": asdict(wandb_state),
            "artifacts": {
                "data_yaml": str(data_yaml_path),
                "save_dir": str(save_dir),
                "weights_best": str(best_weights),
                "weights_last": str(last_weights),
                "results_csv": str(results_csv),
            },
            "metrics": metrics,
            "resolved_device": device,
            "periodic_eval": periodic_eval,
        }
    except Exception as exc:
        summary = {
            "status": "error",
            "config": _json_safe(asdict(config)),
            "wandb": asdict(wandb_state),
            "resolved_device": device,
            "error": str(exc),
        }
        raise
    finally:
        finish_wandb(wandb_run)

    if config.save_json:
        save_dir = Path(summary["artifacts"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        run_summary_path = save_dir / "train_summary.json"
        run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        latest_path = config.artifacts_root / "latest_run.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_payload = {
            "run_name": config.name,
            "save_dir": summary["artifacts"]["save_dir"],
            "weights_best": summary["artifacts"]["weights_best"],
            "weights_last": summary["artifacts"]["weights_last"],
            "data_yaml": summary["artifacts"]["data_yaml"],
            "wandb": summary["wandb"],
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return summary
