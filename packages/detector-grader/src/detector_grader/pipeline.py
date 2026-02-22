from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from detector_infer.config import InferConfig
from detector_infer.infer import run_inference

from .data import (
    image_shape,
    index_ground_truth,
    infer_model_name_from_weights,
    load_labels,
    load_prediction_labels,
    resolve_latest_weights,
    sanitize_model_name,
)
from .reporting import aggregate_scores, write_reports
from .scoring import ScoreWeights, score_sample


@dataclass(slots=True)
class GradingConfig:
    dataset_root: Path
    predictions_root: Path = Path("predictions")
    artifacts_root: Path = Path("artifacts/detector-train")
    reports_dir: Path | None = None
    model: str = "latest"
    weights: Path | None = None
    run_inference: bool = True
    splits: list[str] | None = None
    imgsz: int = 640
    device: str = "auto"
    conf_threshold: float = 0.25
    infer_iou_threshold: float = 0.7
    match_iou_threshold: float = 0.5
    weights_json: Path | None = None
    strict_obb: bool = True
    max_samples: int | None = None
    seed: int = 42



def load_weights_profile(path: Path | None) -> ScoreWeights:
    if path is None:
        return ScoreWeights().normalized()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoreWeights(
        iou=float(payload.get("iou", 0.50)),
        corner=float(payload.get("corner", 0.28)),
        angle=float(payload.get("angle", 0.08)),
        center=float(payload.get("center", 0.06)),
        shape=float(payload.get("shape", 0.08)),
        fn_penalty=float(payload.get("fn_penalty", 0.35)),
        fp_penalty=float(payload.get("fp_penalty", 0.20)),
        containment_miss_penalty=float(payload.get("containment_miss_penalty", 0.35)),
        containment_outside_penalty=float(payload.get("containment_outside_penalty", 0.20)),
        tau_corner_px=float(payload.get("tau_corner_px", 20.0)),
        tau_center_px=float(payload.get("tau_center_px", 24.0)),
        iou_gamma=float(payload.get("iou_gamma", 1.6)),
    ).normalized()


def resolve_model_source(
    *,
    model_arg: str,
    weights_arg: Path | None,
    artifacts_root: Path,
    predictions_root: Path,
) -> tuple[Path | None, str, bool]:
    if weights_arg is not None:
        if not weights_arg.exists():
            raise FileNotFoundError(f"weights not found: {weights_arg}")
        key = infer_model_name_from_weights(weights_arg)
        existing = (predictions_root / key / "labels").exists()
        return weights_arg, key, existing

    normalized = model_arg.strip()
    latest_aliases = {".", "latest", "latest-best", "best"}
    if normalized in latest_aliases:
        w = resolve_latest_weights(artifacts_root)
        key = infer_model_name_from_weights(w)
        existing = (predictions_root / key / "labels").exists()
        return w, key, existing

    key = sanitize_model_name(normalized)
    existing = (predictions_root / key / "labels").exists()
    return None, key, existing


def _split_geometry_summary(aggregate: dict[str, Any]) -> dict[str, float | None]:
    splits = aggregate.get("splits", [])
    total = 0
    iou_acc = 0.0
    drift_acc = 0.0
    iou_seen = 0
    drift_seen = 0
    for split in splits:
        n = int(split.get("num_samples", 0) or 0)
        if n <= 0:
            continue
        total += n
        geom = split.get("geometry", {})
        iou = geom.get("iou_mean")
        drift = geom.get("center_error_px_mean")
        if iou is not None:
            iou_acc += float(iou) * n
            iou_seen += n
        if drift is not None:
            drift_acc += float(drift) * n
            drift_seen += n
    return {
        "mean_iou": (iou_acc / iou_seen) if iou_seen > 0 else None,
        "mean_center_drift_px": (drift_acc / drift_seen) if drift_seen > 0 else None,
    }


def run_grading(config: GradingConfig) -> dict[str, Any]:
    splits = config.splits if config.splits is not None else ["train", "val"]
    reports_dir = config.reports_dir if config.reports_dir is not None else config.dataset_root / "grade_reports"
    weights_profile = load_weights_profile(config.weights_json)

    weights_path, model_key, has_existing = resolve_model_source(
        model_arg=config.model,
        weights_arg=config.weights,
        artifacts_root=config.artifacts_root,
        predictions_root=config.predictions_root,
    )

    inference_summary: dict[str, Any] | None = None
    if config.run_inference:
        if weights_path is None:
            raise RuntimeError(
                "run-inference requested but no weights resolved. "
                "Pass weights, use model latest/. , or provide a run/weights path."
            )
        infer_cfg = InferConfig(
            weights=weights_path,
            dataset_root=config.dataset_root,
            output_root=config.predictions_root,
            model_name=model_key,
            imgsz=config.imgsz,
            device=config.device,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.infer_iou_threshold,
            seed=config.seed,
            splits=splits,
            save_empty=True,
        )
        inference_summary = run_inference(infer_cfg)
    else:
        if not has_existing:
            raise RuntimeError(
                f"prediction set not found for model key '{model_key}' under {config.predictions_root}. "
                "Enable run-inference or provide a valid predictions model key."
            )

    records = index_ground_truth(config.dataset_root)
    sample_rows: list[dict[str, Any]] = []
    invalid_count = 0
    indexed_count = 0
    missing_image_count = 0
    missing_gt_count = 0
    missing_pred_file_count = 0
    by_split_indexed: dict[str, int] = {s: 0 for s in splits}
    by_split_invalid: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_image: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_gt: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_pred_file: dict[str, int] = {s: 0 for s in splits}

    split_set = set(splits)
    for rec in records:
        if rec.split not in split_set:
            continue
        if config.max_samples is not None and len(sample_rows) >= config.max_samples:
            break
        if rec.image_path is None:
            missing_image_count += 1
            by_split_missing_image[rec.split] = by_split_missing_image.get(rec.split, 0) + 1
            continue
        if rec.gt_label_path is None:
            missing_gt_count += 1
            by_split_missing_gt[rec.split] = by_split_missing_gt.get(rec.split, 0) + 1
            continue
        shape = image_shape(rec.image_path)
        if shape is None:
            missing_image_count += 1
            by_split_missing_image[rec.split] = by_split_missing_image.get(rec.split, 0) + 1
            continue
        h, w = shape
        indexed_count += 1
        by_split_indexed[rec.split] = by_split_indexed.get(rec.split, 0) + 1

        pred_file = config.predictions_root / model_key / "labels" / rec.split / f"{rec.stem}.txt"
        if not pred_file.exists():
            missing_pred_file_count += 1
            by_split_missing_pred_file[rec.split] = by_split_missing_pred_file.get(rec.split, 0) + 1

        try:
            gt_labels = load_labels(rec.gt_label_path, is_prediction=False, conf_threshold=0.0)
            pred_labels = load_prediction_labels(
                predictions_root=config.predictions_root,
                model_name=model_key,
                split=rec.split,
                stem=rec.stem,
                conf_threshold=config.conf_threshold,
            )
        except Exception:
            if config.strict_obb:
                raise
            invalid_count += 1
            by_split_invalid[rec.split] = by_split_invalid.get(rec.split, 0) + 1
            continue

        row = score_sample(
            split=rec.split,
            stem=rec.stem,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            w=w,
            h=h,
            iou_threshold=config.match_iou_threshold,
            weights=weights_profile,
        )
        sample_rows.append(row)

    aggregate = aggregate_scores(sample_rows)
    aggregate["invalid_samples_skipped"] = invalid_count
    aggregate["data_quality"] = {
        "indexed_samples": indexed_count,
        "missing_image_samples": missing_image_count,
        "missing_gt_label_samples": missing_gt_count,
        "missing_prediction_file_samples": missing_pred_file_count,
        "invalid_samples_skipped": invalid_count,
        "by_split": {
            split: {
                "indexed_samples": by_split_indexed.get(split, 0),
                "missing_image_samples": by_split_missing_image.get(split, 0),
                "missing_gt_label_samples": by_split_missing_gt.get(split, 0),
                "missing_prediction_file_samples": by_split_missing_pred_file.get(split, 0),
                "invalid_samples_skipped": by_split_invalid.get(split, 0),
            }
            for split in splits
        },
    }

    run_detection = aggregate.get("run_detection", {})
    eval_like = {
        "precision": run_detection.get("precision_proxy"),
        "recall": run_detection.get("recall_proxy"),
        "miss_rate": run_detection.get("miss_rate_proxy"),
        **_split_geometry_summary(aggregate),
    }

    report_config = {
        "dataset_root": str(config.dataset_root),
        "predictions_root": str(config.predictions_root),
        "model_arg": config.model,
        "resolved_model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "run_inference": config.run_inference,
        "splits": splits,
        "conf_threshold": config.conf_threshold,
        "infer_iou_threshold": config.infer_iou_threshold,
        "match_iou_threshold": config.match_iou_threshold,
        "weights_profile": {
            "iou": weights_profile.iou,
            "corner": weights_profile.corner,
            "angle": weights_profile.angle,
            "center": weights_profile.center,
            "shape": weights_profile.shape,
            "fn_penalty": weights_profile.fn_penalty,
            "fp_penalty": weights_profile.fp_penalty,
            "containment_miss_penalty": weights_profile.containment_miss_penalty,
            "containment_outside_penalty": weights_profile.containment_outside_penalty,
            "tau_corner_px": weights_profile.tau_corner_px,
            "tau_center_px": weights_profile.tau_center_px,
            "iou_gamma": weights_profile.iou_gamma,
        },
    }

    out = write_reports(
        reports_dir=reports_dir,
        model_name=model_key,
        config=report_config,
        sample_rows=sample_rows,
        aggregate=aggregate,
    )

    return {
        "dataset_root": str(config.dataset_root),
        "predictions_root": str(config.predictions_root),
        "reports_dir": str(reports_dir),
        "model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "inference": inference_summary,
        "aggregate": aggregate,
        "eval_like": eval_like,
        "reports": out,
    }
