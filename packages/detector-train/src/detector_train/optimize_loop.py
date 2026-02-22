from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .config import TrainConfig
from .trainer import train_detector


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "run"


def _sanitize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def _infer_model_key_from_weights(weights_best: Path) -> str:
    if weights_best.parent.name == "weights" and weights_best.parent.parent.name:
        run_name = weights_best.parent.parent.name
        return _sanitize_model_name(f"{run_name}_{weights_best.stem}")
    return _sanitize_model_name(weights_best.stem)


def _parse_baseline_grade(baseline_file: Path) -> float:
    if not baseline_file.exists():
        raise FileNotFoundError(f"baseline file not found: {baseline_file}")
    text = baseline_file.read_text(encoding="utf-8")
    match = re.search(r"run_grade_0_100:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise ValueError(f"could not parse run_grade_0_100 from {baseline_file}")
    return float(match.group(1))


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _update_best_config(
    *,
    best_config_path: Path,
    iteration: int,
    technique: str,
    train_config: TrainConfig,
    grade_report: dict[str, Any],
    baseline_grade: float,
) -> None:
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = grade_report["aggregate"]
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "technique": technique,
        "baseline_grade_0_100": baseline_grade,
        "run_grade_0_100": float(aggregate["run_grade_0_100"]),
        "config": json.loads(json.dumps(asdict(train_config), default=str)),
        "run_detection": aggregate.get("run_detection", {}),
    }
    best_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_commit_best(best_config_path: Path, grade: float, iteration: int) -> str:
    subprocess.run(["git", "add", str(best_config_path)], check=True)
    staged_check = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    if staged_check.returncode == 0:
        return "skipped_no_changes"
    message = f"improve detector grade to {grade:.4f} at iter {iteration:03d}"
    subprocess.run(["git", "commit", "-m", message], check=True)
    return message


def _run_grader(
    *,
    dataset: str,
    datasets_base_root: Path,
    predictions_root: Path,
    artifacts_root: Path,
    model: str,
    device: str,
    conf_threshold: float,
    infer_iou_threshold: float,
    match_iou_threshold: float,
) -> None:
    cmd = [
        "uv",
        "run",
        "detector-grader",
        "--dataset",
        dataset,
        "--datasets-base-root",
        str(datasets_base_root),
        "--predictions-root",
        str(predictions_root),
        "--artifacts-root",
        str(artifacts_root),
        "--model",
        model,
        "--run-inference",
        "--device",
        device,
        "--conf-threshold",
        f"{conf_threshold}",
        "--infer-iou-threshold",
        f"{infer_iou_threshold}",
        "--match-iou-threshold",
        f"{match_iou_threshold}",
        "--splits",
        "train,val",
    ]
    subprocess.run(cmd, check=True)


def _technique_queue() -> list[dict[str, Any]]:
    return [
        {
            "name": "adamw-cosine-lr",
            "overrides": {
                "optimizer": "AdamW",
                "lr0": 0.002,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "cos_lr": True,
            },
        },
        {
            "name": "obb-geometry-aug-balance",
            "overrides": {
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
            },
        },
        {
            "name": "high-resolution-pass",
            "overrides": {
                "imgsz": 960,
                "batch": 8,
            },
        },
        {
            "name": "warmup-regularization-retune",
            "overrides": {
                "warmup_epochs": 5.0,
                "weight_decay": 0.0007,
                "patience": 75,
            },
        },
        {
            "name": "mixup-copy-paste-enabled",
            "overrides": {
                "mixup": 0.20,
                "copy_paste": 0.15,
            },
        },
        {
            "name": "mixup-copy-paste-disabled",
            "overrides": {
                "mixup": 0.0,
                "copy_paste": 0.0,
            },
        },
        {
            "name": "multi-scale-training",
            "overrides": {
                "multi_scale": True,
            },
        },
        {
            "name": "sgd-cosine-strong-baseline",
            "overrides": {
                "optimizer": "SGD",
                "lr0": 0.01,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "cos_lr": True,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative detector optimization loop (train + grade + compare)")
    parser.add_argument("--dataset", default="coco128")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions"))
    parser.add_argument("--baseline-file", type=Path, default=Path("baseline.txt"))
    parser.add_argument("--best-config-path", type=Path, default=Path("configs/models/detector_opt_best.json"))
    parser.add_argument("--history-path", type=Path, default=Path("artifacts/detector-train/optimization/history.jsonl"))

    parser.add_argument("--target-grade", type=float, default=90.57)
    parser.add_argument("--min-win-delta", type=float, default=0.50)
    parser.add_argument("--min-precision-proxy", type=float, default=1.0)
    parser.add_argument("--min-recall-proxy", type=float, default=1.0)
    parser.add_argument("--max-no-win", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--eval-interval-epochs", type=int, default=5)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--eval-conf-threshold", type=float, default=0.25)
    parser.add_argument("--eval-viz-samples", type=int, default=0)

    parser.add_argument("--grade-conf-threshold", type=float, default=0.25)
    parser.add_argument("--grade-infer-iou-threshold", type=float, default=0.7)
    parser.add_argument("--grade-match-iou-threshold", type=float, default=0.5)

    parser.add_argument("--wandb-project", default="tea-ai-detector")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--commit-wins", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    baseline_grade = _parse_baseline_grade(args.baseline_file)
    dataset_root = args.datasets_base_root / args.dataset
    project_root = args.artifacts_root / "runs"

    queue = _technique_queue()
    if args.max_iterations < 1:
        raise ValueError("max-iterations must be >= 1")
    if not (0.0 <= args.min_precision_proxy <= 1.0):
        raise ValueError("min-precision-proxy must be in [0,1]")
    if not (0.0 <= args.min_recall_proxy <= 1.0):
        raise ValueError("min-recall-proxy must be in [0,1]")

    history = _load_history(args.history_path) if args.resume else []
    best_grade = baseline_grade
    consecutive_no_win = 0
    start_iteration = 1
    if history:
        start_iteration = int(history[-1]["iteration"]) + 1
        eligible_rows = [
            row
            for row in history
            if float(row.get("precision_proxy", 0.0) or 0.0) >= args.min_precision_proxy
            and float(row.get("recall_proxy", 0.0) or 0.0) >= args.min_recall_proxy
        ]
        if eligible_rows:
            best_grade = max(best_grade, max(float(row.get("run_grade_0_100", baseline_grade)) for row in eligible_rows))
        consecutive_no_win = int(history[-1].get("consecutive_no_win", 0))
        print(
            f"resuming from iteration {start_iteration} "
            f"(best={best_grade:.4f}, consecutive_no_win={consecutive_no_win})"
        )
    end_iteration = min(start_iteration + args.max_iterations - 1, len(queue))

    for iteration in range(start_iteration, end_iteration + 1):
        if consecutive_no_win >= args.max_no_win:
            print(f"plateau reached after {consecutive_no_win} consecutive no-win iterations")
            break
        if iteration > len(queue):
            print("technique queue exhausted")
            break

        candidate = queue[iteration - 1]
        technique = candidate["name"]
        overrides: dict[str, Any] = candidate["overrides"]
        run_name = f"opt-iter-{iteration:03d}-{_slug(technique)}-{_utc_stamp()}"
        run_imgsz = int(overrides.get("imgsz", args.imgsz))
        run_batch = int(overrides.get("batch", args.batch))
        run_patience = int(overrides.get("patience", args.patience))

        tags = [
            "opt-loop",
            f"iter-{iteration:03d}",
            f"baseline-{baseline_grade:.4f}",
            _slug(technique),
        ]

        cfg = TrainConfig(
            dataset_root=dataset_root,
            artifacts_root=args.artifacts_root,
            model=args.model,
            epochs=args.epochs,
            imgsz=run_imgsz,
            batch=run_batch,
            device=args.device,
            project=project_root,
            name=run_name,
            seed=args.seed,
            workers=args.workers,
            patience=run_patience,
            save_json=True,
            optimizer=str(overrides.get("optimizer", "auto")),
            lr0=overrides.get("lr0"),
            lrf=overrides.get("lrf"),
            weight_decay=overrides.get("weight_decay"),
            warmup_epochs=overrides.get("warmup_epochs"),
            cos_lr=bool(overrides.get("cos_lr", False)),
            close_mosaic=overrides.get("close_mosaic"),
            mosaic=overrides.get("mosaic"),
            mixup=overrides.get("mixup"),
            degrees=overrides.get("degrees"),
            translate=overrides.get("translate"),
            scale=overrides.get("scale"),
            shear=overrides.get("shear"),
            perspective=overrides.get("perspective"),
            hsv_h=overrides.get("hsv_h"),
            hsv_s=overrides.get("hsv_s"),
            hsv_v=overrides.get("hsv_v"),
            fliplr=overrides.get("fliplr"),
            flipud=overrides.get("flipud"),
            copy_paste=overrides.get("copy_paste"),
            multi_scale=bool(overrides.get("multi_scale", False)),
            wandb_enabled=True,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=run_name,
            wandb_tags=tags,
            wandb_notes=f"iterative optimization run ({technique})",
            wandb_mode="online",
            wandb_log_profile="core+diag",
            wandb_log_system_metrics=False,
            wandb_log_every_epoch=True,
            eval_enabled=True,
            eval_interval_epochs=args.eval_interval_epochs,
            eval_iou_threshold=args.eval_iou_threshold,
            eval_conf_threshold=args.eval_conf_threshold,
            eval_viz_samples=args.eval_viz_samples,
        )

        summary = train_detector(cfg)
        if summary["status"] != "ok":
            raise RuntimeError(f"training failed at iteration {iteration}: {summary}")

        weights_best = Path(summary["artifacts"]["weights_best"])
        model_key = _infer_model_key_from_weights(weights_best)
        _run_grader(
            dataset=args.dataset,
            datasets_base_root=args.datasets_base_root,
            predictions_root=args.predictions_root,
            artifacts_root=args.artifacts_root,
            model="latest",
            device=args.device,
            conf_threshold=args.grade_conf_threshold,
            infer_iou_threshold=args.grade_infer_iou_threshold,
            match_iou_threshold=args.grade_match_iou_threshold,
        )

        report_path = dataset_root / "grade_reports" / f"grade_report_{model_key}.json"
        if not report_path.exists():
            raise FileNotFoundError(f"expected grade report was not produced: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        aggregate = report["aggregate"]
        grade = float(aggregate["run_grade_0_100"])
        detection = aggregate.get("run_detection", {})
        precision = float(detection.get("precision_proxy", 0.0) or 0.0)
        recall = float(detection.get("recall_proxy", 0.0) or 0.0)
        delta_vs_best = grade - best_grade
        delta_vs_baseline = grade - baseline_grade
        is_win = (
            delta_vs_best >= args.min_win_delta
            and precision >= args.min_precision_proxy
            and recall >= args.min_recall_proxy
        )

        commit_status = "not_applicable"
        if is_win:
            consecutive_no_win = 0
            best_grade = grade
            _update_best_config(
                best_config_path=args.best_config_path,
                iteration=iteration,
                technique=technique,
                train_config=cfg,
                grade_report=report,
                baseline_grade=baseline_grade,
            )
            if args.commit_wins:
                commit_status = _maybe_commit_best(args.best_config_path, grade, iteration)
            else:
                commit_status = "skipped_disabled"
        else:
            consecutive_no_win += 1
            commit_status = "skipped_not_win"

        history_row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "technique": technique,
            "run_name": run_name,
            "model_key": model_key,
            "baseline_grade_0_100": baseline_grade,
            "run_grade_0_100": grade,
            "best_grade_0_100": best_grade,
            "delta_vs_baseline": delta_vs_baseline,
            "delta_vs_best_before_decision": delta_vs_best,
            "is_win": is_win,
            "precision_proxy": precision,
            "recall_proxy": recall,
            "consecutive_no_win": consecutive_no_win,
            "target_grade_0_100": args.target_grade,
            "min_precision_proxy": args.min_precision_proxy,
            "min_recall_proxy": args.min_recall_proxy,
            "report_path": str(report_path),
            "weights_best": str(weights_best),
            "commit_status": commit_status,
            "hyperparams": {
                "optimizer": cfg.optimizer,
                "lr0": cfg.lr0,
                "lrf": cfg.lrf,
                "weight_decay": cfg.weight_decay,
                "warmup_epochs": cfg.warmup_epochs,
                "cos_lr": cfg.cos_lr,
                "close_mosaic": cfg.close_mosaic,
                "mosaic": cfg.mosaic,
                "mixup": cfg.mixup,
                "degrees": cfg.degrees,
                "translate": cfg.translate,
                "scale": cfg.scale,
                "shear": cfg.shear,
                "perspective": cfg.perspective,
                "hsv_h": cfg.hsv_h,
                "hsv_s": cfg.hsv_s,
                "hsv_v": cfg.hsv_v,
                "fliplr": cfg.fliplr,
                "flipud": cfg.flipud,
                "copy_paste": cfg.copy_paste,
                "multi_scale": cfg.multi_scale,
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "epochs": cfg.epochs,
                "patience": cfg.patience,
            },
        }
        _write_jsonl(args.history_path, history_row)

        print(
            f"iteration={iteration} technique={technique} grade={grade:.4f} "
            f"delta_baseline={delta_vs_baseline:+.4f} delta_best={delta_vs_best:+.4f} "
            f"win={is_win} best={best_grade:.4f}"
        )

        if best_grade >= args.target_grade:
            print(f"target reached: {best_grade:.4f} >= {args.target_grade:.4f}")
            break

    print(
        f"optimization finished: baseline={baseline_grade:.4f} "
        f"best={best_grade:.4f} target={args.target_grade:.4f}"
    )


if __name__ == "__main__":
    main()
