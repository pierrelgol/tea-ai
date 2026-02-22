from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import csv
import json
import re

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


def _json_safe(obj):
    """Convert objects (e.g. Path) into JSON-serializable primitives."""
    return json.loads(json.dumps(obj, default=str))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return out


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace("(", "").replace(")", "")
    key = re.sub(r"\s+", "", key)
    return key


def _canonical_key(raw_key: str) -> str | None:
    k = _normalize_key(raw_key)

    if "metrics/precision" in k:
        return "val/precision"
    if "metrics/recall" in k:
        return "val/recall"
    if "metrics/map50-95" in k or "metrics/map50_95" in k:
        return "val/map50_95"
    if "metrics/map50" in k:
        return "val/map50"

    if "train/box_loss" in k or k.endswith("box_loss"):
        return "train/loss_box"
    if "train/cls_loss" in k or k.endswith("cls_loss"):
        return "train/loss_cls"
    if "train/dfl_loss" in k or k.endswith("dfl_loss"):
        return "train/loss_dfl"

    if k in {"lr", "train/lr"} or "lr/pg0" in k:
        return "train/lr"

    if "instances" in k:
        return "train/num_instances"

    return None


def _drop_non_profile_metrics(payload: dict[str, float], profile: str) -> dict[str, float]:
    if profile == "core+diag":
        return payload
    diag_keys = {"train/speed_ms_per_img", "train/num_instances"}
    return {k: v for k, v in payload.items() if k not in diag_keys}


def _extract_epoch_metrics(trainer, profile: str) -> dict[str, float]:
    out: dict[str, float] = {}

    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        speed_vals: list[float] = []
        for key, value in metrics.items():
            fv = _to_float(value)
            if fv is None:
                continue
            canonical = _canonical_key(key)
            if canonical is not None:
                out[canonical] = fv
            nk = _normalize_key(str(key))
            if nk.startswith("speed/"):
                speed_vals.append(fv)
        if speed_vals:
            out["train/speed_ms_per_img"] = float(sum(speed_vals))

    tloss = getattr(trainer, "tloss", None)
    if tloss is not None:
        try:
            arr = tloss.detach().cpu().numpy().reshape(-1)
            if arr.size > 0:
                box = _to_float(arr[0])
                if box is not None:
                    out["train/loss_box"] = box
            if arr.size > 1:
                cls = _to_float(arr[1])
                if cls is not None:
                    out["train/loss_cls"] = cls
            if arr.size > 2:
                dfl = _to_float(arr[2])
                if dfl is not None:
                    out["train/loss_dfl"] = dfl
        except Exception:
            pass

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        try:
            lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
            if lrs:
                out["train/lr"] = lrs[0]
        except Exception:
            pass

    loss_keys = ("train/loss_box", "train/loss_cls", "train/loss_dfl")
    losses = [out[k] for k in loss_keys if k in out]
    if losses:
        out["train/loss_total"] = float(sum(losses))

    return _drop_non_profile_metrics(out, profile)


def _extract_last_metrics(results_csv: Path, profile: str) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}

    raw_row = rows[-1]
    out: dict[str, float] = {}
    speed_vals: list[float] = []

    for key, value in raw_row.items():
        if value is None:
            continue
        fv = _to_float(str(value).strip())
        if fv is None:
            continue

        canonical = _canonical_key(key)
        if canonical is not None:
            out[canonical] = fv

        nk = _normalize_key(key)
        if nk.startswith("speed/"):
            speed_vals.append(fv)

    if speed_vals:
        out["train/speed_ms_per_img"] = float(sum(speed_vals))

    loss_keys = ("train/loss_box", "train/loss_cls", "train/loss_dfl")
    losses = [out[k] for k in loss_keys if k in out]
    if losses:
        out["train/loss_total"] = float(sum(losses))

    return _drop_non_profile_metrics(out, profile)


def _apply_train_profile_defaults(config: TrainConfig) -> dict[str, Any]:
    if config.train_profile == "default":
        return {}
    if config.train_profile == "obb_precision_v1":
        # OBB profile tuned to preserve geometry and reduce localization distortion.
        return {
            "close_mosaic": 10,
            "mosaic": 0.60,
            "mixup": 0.0,
            "degrees": 2.0,
            "translate": 0.08,
            "scale": 0.40,
            "shear": 0.0,
            "perspective": 0.0,
            "hsv_h": 0.010,
            "hsv_s": 0.40,
            "hsv_v": 0.30,
        }
    if config.train_profile == "obb_precision_v2":
        # Balanced profile for complex synthetic scenes: stronger optimizer + gentler geometry jitter.
        return {
            "optimizer": "AdamW",
            "lr0": 0.002,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "cos_lr": True,
            "close_mosaic": 20,
            "mosaic": 0.45,
            "mixup": 0.0,
            "degrees": 1.5,
            "translate": 0.06,
            "scale": 0.35,
            "shear": 0.0,
            "perspective": 0.0,
            "hsv_h": 0.010,
            "hsv_s": 0.35,
            "hsv_v": 0.25,
        }
    raise ValueError(f"unsupported train profile: {config.train_profile}")


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
        log_wandb(wandb_run, {"eval/status": 0.0}, step=epoch)
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
                "eval/status": 1.0,
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
    else:
        log_wandb(wandb_run, {"eval/status": 0.0}, step=epoch)

    return {
        "status": "ok",
        "epoch": epoch,
        "infer": infer_summary,
        "eval": eval_summary,
    }


def train_detector(config: TrainConfig) -> dict[str, Any]:
    config.validate()
    device = _resolve_device(config.device)
    project_dir = config.project if config.project.is_absolute() else (Path.cwd() / config.project)
    project_dir = project_dir.resolve()

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
        "train_profile": config.train_profile,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "cos_lr": config.cos_lr,
        "close_mosaic": config.close_mosaic,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "fliplr": config.fliplr,
        "flipud": config.flipud,
        "copy_paste": config.copy_paste,
        "multi_scale": config.multi_scale,
        "freeze": config.freeze,
        "classes_count": len(names),
        "wandb_log_profile": config.wandb_log_profile,
        "wandb_log_every_epoch": config.wandb_log_every_epoch,
        "wandb_log_system_metrics": config.wandb_log_system_metrics,
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
        log_system_metrics=config.wandb_log_system_metrics,
    )

    try:
        from ultralytics import YOLO

        model = YOLO(config.model)
        periodic_eval: list[dict[str, Any]] = []
        last_eval_epoch = -1
        logged_keys_by_step: dict[int, set[str]] = {}

        def _log_step_payload(step: int, payload: dict[str, float]) -> None:
            if step <= 0 or not payload:
                return
            sent = logged_keys_by_step.setdefault(step, set())
            unique_payload = {k: v for k, v in payload.items() if k not in sent}
            if not unique_payload:
                return
            log_wandb(wandb_run, unique_payload, step=step)
            sent.update(unique_payload.keys())

        def _on_fit_epoch_end(trainer) -> None:
            nonlocal last_eval_epoch

            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            if current_epoch <= 0:
                return

            if config.wandb_log_every_epoch:
                epoch_metrics = _extract_epoch_metrics(trainer, config.wandb_log_profile)
                _log_step_payload(current_epoch, epoch_metrics)

            if not config.eval_enabled:
                return
            if current_epoch == last_eval_epoch:
                return

            should_eval = (current_epoch % config.eval_interval_epochs == 0) or (current_epoch == config.epochs)
            if not should_eval:
                return

            save_dir_cb = Path(getattr(trainer, "save_dir", project_dir / config.name))
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
                _log_step_payload(current_epoch, {"eval/status": 0.0, "eval/error_flag": 1.0})

            last_eval_epoch = current_epoch

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        train_kwargs: dict[str, Any] = {
            "data": str(data_yaml_path),
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "device": device,
            "project": str(project_dir),
            "name": config.name,
            "seed": config.seed,
            "workers": config.workers,
            "patience": config.patience,
            "exist_ok": True,
            "optimizer": config.optimizer,
            "cos_lr": config.cos_lr,
            "multi_scale": config.multi_scale,
        }
        optional_train_kwargs = {
            "lr0": config.lr0,
            "lrf": config.lrf,
            "weight_decay": config.weight_decay,
            "warmup_epochs": config.warmup_epochs,
            "close_mosaic": config.close_mosaic,
            "mosaic": config.mosaic,
            "mixup": config.mixup,
            "degrees": config.degrees,
            "translate": config.translate,
            "scale": config.scale,
            "shear": config.shear,
            "perspective": config.perspective,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "fliplr": config.fliplr,
            "flipud": config.flipud,
            "copy_paste": config.copy_paste,
            "freeze": config.freeze,
        }
        profile_defaults = _apply_train_profile_defaults(config)
        for key, value in profile_defaults.items():
            train_kwargs[key] = value
        for key, value in optional_train_kwargs.items():
            if value is not None:
                train_kwargs[key] = value

        train_result = model.train(
            **train_kwargs,
        )

        save_dir = Path(getattr(train_result, "save_dir", project_dir / config.name))
        weights_dir = save_dir / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        results_csv = save_dir / "results.csv"

        metrics = _extract_last_metrics(results_csv, config.wandb_log_profile)
        if metrics:
            final_payload = {f"final/{k.replace('/', '_')}": v for k, v in metrics.items()}
            _log_step_payload(config.epochs, final_payload)

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
