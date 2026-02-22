from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .detection_metrics import compute_detection_metrics
from .geometry_metrics import compute_geometry_metrics
from .io import (
    image_shape,
    index_ground_truth,
    load_labels,
    load_predictions_for_model,
)
from .reporting import write_model_report, write_summary
from .stability_metrics import compute_stability_metrics
from .visualize import write_visual_artifacts


def _discover_models(predictions_root: Path | None, models_filter: list[str] | None) -> list[str]:
    if predictions_root is None or not predictions_root.exists():
        return []
    models = sorted([p.name for p in predictions_root.iterdir() if p.is_dir()])
    if models_filter:
        wanted = set(models_filter)
        return [m for m in models if m in wanted]
    return models


def evaluate_models(
    dataset_root: Path,
    predictions_root: Path | None,
    reports_dir: Path,
    iou_threshold: float,
    conf_threshold: float,
    seed: int,
    viz_samples: int,
    models_filter: list[str] | None = None,
) -> dict:
    records = index_ground_truth(dataset_root)

    gt_lookup: dict[tuple[str, str], tuple[list, int, int]] = {}
    for rec in records:
        if rec.gt_label_path is None or rec.image_path is None:
            continue
        shape = image_shape(rec.image_path)
        if shape is None:
            continue
        h, w = shape
        gt_labels = load_labels(rec.gt_label_path, is_prediction=False)
        gt_lookup[(rec.split, rec.stem)] = (gt_labels, h, w)

    geometry_summary = compute_geometry_metrics(records, gt_lookup)
    models = _discover_models(predictions_root, models_filter)

    model_rows: list[dict] = []
    model_report_paths: list[str] = []

    for model_name in models:
        pred_lookup: dict[tuple[str, str], tuple[list, int, int]] = {}
        sample_items: list[tuple[str, str, list, list, int, int]] = []

        for rec in records:
            key = (rec.split, rec.stem)
            gt_entry = gt_lookup.get(key)
            if gt_entry is None:
                continue

            gt_labels, h, w = gt_entry
            preds = load_predictions_for_model(
                predictions_root=predictions_root,
                model_name=model_name,
                split=rec.split,
                stem=rec.stem,
                conf_threshold=conf_threshold,
            )
            pred_lookup[key] = (preds, h, w)
            sample_items.append((rec.split, rec.stem, gt_labels, preds, h, w))

        detection_summary, per_sample, match_lookup = compute_detection_metrics(sample_items, iou_threshold)
        stability_summary = compute_stability_metrics(sample_items, match_lookup)

        config = {
            "dataset_root": str(dataset_root),
            "predictions_root": str(predictions_root) if predictions_root else None,
            "iou_threshold": iou_threshold,
            "conf_threshold": conf_threshold,
            "seed": seed,
            "viz_samples": viz_samples,
        }

        per_sample_rows = [
            {
                "split": row.split,
                "stem": row.stem,
                "tp": row.tp,
                "fp": row.fp,
                "fn": row.fn,
                "matched_ious": row.matched_ious,
            }
            for row in per_sample
        ]

        report_path = write_model_report(
            reports_dir=reports_dir,
            model_name=model_name,
            config=config,
            detection=detection_summary,
            stability=stability_summary,
            geometry=geometry_summary,
            per_sample_rows=per_sample_rows,
        )
        model_report_paths.append(str(report_path))

        write_visual_artifacts(
            records=records,
            gt_lookup=gt_lookup,
            pred_lookup=pred_lookup,
            reports_dir=reports_dir,
            model_name=model_name,
            sample_count_per_split=viz_samples,
            seed=seed,
        )

        model_rows.append(
            {
                "model_name": model_name,
                "precision": detection_summary.precision,
                "recall": detection_summary.recall,
                "miss_rate": detection_summary.miss_rate,
                "mean_iou": detection_summary.mean_iou,
                "ap_at_iou": detection_summary.ap_at_iou,
                "mean_center_drift_px": stability_summary.mean_center_drift_px,
                "geometry_mean_corner_error_label_vs_meta_px": geometry_summary.mean_corner_error_label_vs_meta_px,
            }
        )

    summary_json, summary_md = write_summary(reports_dir, model_rows)

    return {
        "num_samples_indexed": len(records),
        "num_samples_with_gt": len(gt_lookup),
        "models": model_rows,
        "model_report_paths": model_report_paths,
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "geometry": asdict(geometry_summary),
    }
