from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import (
    image_shape,
    index_ground_truth,
    infer_model_name_from_weights,
    load_labels,
    load_prediction_labels,
    resolve_latest_weights,
    sanitize_model_name,
)
from .inference import run_inference_for_grading
from .reporting import aggregate_scores, write_reports
from .scoring import ScoreWeights, score_sample


def _load_weights_profile(path: Path | None) -> ScoreWeights:
    if path is None:
        return ScoreWeights()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoreWeights(
        iou=float(payload.get("iou", 0.40)),
        corner=float(payload.get("corner", 0.25)),
        angle=float(payload.get("angle", 0.15)),
        center=float(payload.get("center", 0.10)),
        shape=float(payload.get("shape", 0.10)),
        fn_penalty=float(payload.get("fn_penalty", 0.35)),
        fp_penalty=float(payload.get("fp_penalty", 0.20)),
        containment_miss_penalty=float(payload.get("containment_miss_penalty", 0.25)),
        tau_corner_px=float(payload.get("tau_corner_px", 25.0)),
        tau_center_px=float(payload.get("tau_center_px", 30.0)),
    )


def _split_csv(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _resolve_model_source(
    *,
    model_arg: str,
    weights_arg: Path | None,
    artifacts_root: Path,
    predictions_root: Path,
) -> tuple[Path | None, str, bool]:
    # returns (weights, model_key, has_existing_predictions)
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

    model_path = Path(normalized)
    if model_path.exists():
        if model_path.is_file():
            w = model_path
            key = infer_model_name_from_weights(w)
            existing = (predictions_root / key / "labels").exists()
            return w, key, existing
        # directory case: run folder or predictions folder
        best = model_path / "weights" / "best.pt"
        if best.exists():
            key = infer_model_name_from_weights(best)
            existing = (predictions_root / key / "labels").exists()
            return best, key, existing
        # fallback: treat dir name as model key in predictions root
        key = sanitize_model_name(model_path.name)
        existing = (predictions_root / key / "labels").exists()
        return None, key, existing

    # treat as prediction model key
    key = sanitize_model_name(normalized)
    existing = (predictions_root / key / "labels").exists()
    return None, key, existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade detector runs from strict OBB geometry quality")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions"))
    parser.add_argument("--model", required=True, help="Model source: latest|.|weights path|run dir|prediction model key")
    parser.add_argument("--weights", type=Path, default=None, help="Explicit weights path override")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--run-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for infer + grading")
    parser.add_argument("--infer-iou-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--weights-json", type=Path, default=None, help="Scoring weight profile JSON")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--strict-obb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset
    reports_dir = args.reports_dir if args.reports_dir is not None else dataset_root / "grade_reports"
    weights_profile = _load_weights_profile(args.weights_json).normalized()
    splits = _split_csv(args.splits)

    weights_path, model_key, has_existing = _resolve_model_source(
        model_arg=args.model,
        weights_arg=args.weights,
        artifacts_root=args.artifacts_root,
        predictions_root=args.predictions_root,
    )

    inference_summary: dict[str, Any] | None = None
    if args.run_inference:
        if weights_path is None:
            raise RuntimeError(
                "run-inference requested but no weights resolved. "
                "Pass --weights, use --model latest/. , or provide a run/weights path."
            )
        inference_summary = run_inference_for_grading(
            weights=weights_path,
            dataset_root=dataset_root,
            predictions_root=args.predictions_root,
            model_name=model_key,
            splits=splits,
            imgsz=args.imgsz,
            device=args.device,
            conf_threshold=args.conf_threshold,
            infer_iou_threshold=args.infer_iou_threshold,
            seed=args.seed,
        )
    else:
        if not has_existing:
            raise RuntimeError(
                f"prediction set not found for model key '{model_key}' under {args.predictions_root}. "
                "Enable --run-inference or provide a valid predictions model key."
            )

    records = index_ground_truth(dataset_root)
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

    for rec in records:
        if rec.split not in set(splits):
            continue
        if args.max_samples is not None and len(sample_rows) >= args.max_samples:
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

        pred_file = args.predictions_root / model_key / "labels" / rec.split / f"{rec.stem}.txt"
        if not pred_file.exists():
            missing_pred_file_count += 1
            by_split_missing_pred_file[rec.split] = by_split_missing_pred_file.get(rec.split, 0) + 1

        try:
            gt_labels = load_labels(rec.gt_label_path, is_prediction=False, conf_threshold=0.0)
            pred_labels = load_prediction_labels(
                predictions_root=args.predictions_root,
                model_name=model_key,
                split=rec.split,
                stem=rec.stem,
                conf_threshold=args.conf_threshold,
            )
        except Exception:
            if args.strict_obb:
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
            iou_threshold=args.match_iou_threshold,
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

    config = {
        "dataset_root": str(dataset_root),
        "predictions_root": str(args.predictions_root),
        "model_arg": args.model,
        "resolved_model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "run_inference": args.run_inference,
        "splits": splits,
        "conf_threshold": args.conf_threshold,
        "infer_iou_threshold": args.infer_iou_threshold,
        "match_iou_threshold": args.match_iou_threshold,
        "weights_profile": {
            "iou": weights_profile.iou,
            "corner": weights_profile.corner,
            "angle": weights_profile.angle,
            "center": weights_profile.center,
            "shape": weights_profile.shape,
            "fn_penalty": weights_profile.fn_penalty,
            "fp_penalty": weights_profile.fp_penalty,
            "containment_miss_penalty": weights_profile.containment_miss_penalty,
            "tau_corner_px": weights_profile.tau_corner_px,
            "tau_center_px": weights_profile.tau_center_px,
        },
    }

    out = write_reports(
        reports_dir=reports_dir,
        model_name=model_key,
        config=config,
        sample_rows=sample_rows,
        aggregate=aggregate,
    )

    print("Model Source")
    print(f"- requested: {args.model}")
    print(f"- resolved_model_key: {model_key}")
    print(f"- weights: {weights_path if weights_path is not None else 'N/A (using existing predictions)'}")
    print(f"- predictions_root: {args.predictions_root / model_key / 'labels'}")
    print("")
    print("Inference")
    if inference_summary is None:
        print("- executed: no")
        print("- reason: using existing predictions")
    else:
        print("- executed: yes")
        print(f"- device: {inference_summary['resolved_device']}")
        print(f"- images_processed: {inference_summary['images_processed']}")
        print(f"- label_files_written: {inference_summary['label_files_written']}")
    print("")
    print("Grading")
    print(f"- dataset_root: {dataset_root}")
    print(f"- samples_indexed: {indexed_count}")
    print(f"- samples_scored: {aggregate['num_samples_scored']}")
    print(f"- missing_image_samples: {missing_image_count}")
    print(f"- missing_gt_label_samples: {missing_gt_count}")
    print(f"- missing_prediction_file_samples: {missing_pred_file_count}")
    print(f"- invalid_samples_skipped: {invalid_count}")
    print(f"- run_grade_0_100: {aggregate['run_grade_0_100']:.4f}")
    run_det = aggregate.get("run_detection", {})
    print(f"- run_precision_proxy: {_fmt_float(run_det.get('precision_proxy'))}")
    print(f"- run_recall_proxy: {_fmt_float(run_det.get('recall_proxy'))}")
    print(f"- run_miss_rate_proxy: {_fmt_float(run_det.get('miss_rate_proxy'))}")
    print(f"- run_class_match_rate: {_fmt_float(run_det.get('class_match_rate'))}")
    print("")
    print("Split Metrics")
    for s in aggregate["splits"]:
        score_dist = s.get("score_distribution_0_100", {})
        det = s.get("detection", {})
        geom = s.get("geometry", {})
        counts = s.get("counts", {})
        dq_split = aggregate["data_quality"]["by_split"].get(s["split"], {})
        print(
            f"- {s['split']}: grade={s['grade_0_100']:.4f}, samples={s['num_samples']}, "
            f"score_p50={_fmt_float(score_dist.get('median'))}, score_p90={_fmt_float(score_dist.get('p90'))}"
        )
        print(
            f"  objects gt/pred/matched={counts.get('gt_total', 0)}/{counts.get('pred_total', 0)}/{counts.get('matched_total', 0)}, "
            f"fn={counts.get('fn_total', 0)}, fp={counts.get('fp_total', 0)}, class_mismatch={counts.get('class_mismatch_total', 0)}"
        )
        print(
            f"  precision_proxy={_fmt_float(det.get('precision_proxy'))}, recall_proxy={_fmt_float(det.get('recall_proxy'))}, "
            f"miss_rate={_fmt_float(det.get('miss_rate_proxy'))}, fdr={_fmt_float(det.get('false_discovery_rate_proxy'))}"
        )
        print(
            f"  penalties fn/fp/containment={_fmt_float(s.get('mean_penalty_fn'))}/"
            f"{_fmt_float(s.get('mean_penalty_fp'))}/{_fmt_float(s.get('mean_penalty_containment'))}"
        )
        print(
            f"  mean_iou={_fmt_float(geom.get('iou_mean'))}, iou_p90={_fmt_float(geom.get('iou_p90'))}, "
            f"iou_ge_0.50={_fmt_float(geom.get('iou_ge_50_rate'))}, iou_ge_0.75={_fmt_float(geom.get('iou_ge_75_rate'))}"
        )
        print(
            f"  corner_err_px mean/p90={_fmt_float(geom.get('corner_error_px_mean'))}/{_fmt_float(geom.get('corner_error_px_p90'))}, "
            f"center_err_px mean/p90={_fmt_float(geom.get('center_error_px_mean'))}/{_fmt_float(geom.get('center_error_px_p90'))}"
        )
        print(
            f"  center_err_norm_mean={_fmt_float(geom.get('center_error_norm_mean'))}, "
            f"angle_err_deg mean/p90={_fmt_float(geom.get('angle_error_deg_mean'))}/{_fmt_float(geom.get('angle_error_deg_p90'))}"
        )
        print(
            f"  orientation<=5deg={_fmt_float(geom.get('orientation_within_5deg_rate'))}, "
            f"orientation<=10deg={_fmt_float(geom.get('orientation_within_10deg_rate'))}"
        )
        print(
            f"  area_ratio mean/median={_fmt_float(geom.get('area_ratio_mean'))}/{_fmt_float(geom.get('area_ratio_median'))}, "
            f"abs_log_area_ratio_mean={_fmt_float(geom.get('abs_log_area_ratio_mean'))}, edge_rel_error_mean={_fmt_float(geom.get('edge_rel_error_mean'))}, "
            f"gt_area_missed_mean={_fmt_float(geom.get('gt_area_missed_ratio_mean'))}, gt_area_missed_p90={_fmt_float(geom.get('gt_area_missed_ratio_p90'))}, "
            f"pred_outside_mean={_fmt_float(geom.get('pred_outside_ratio_mean'))}"
        )
        print(
            f"  pred_conf mean/median={_fmt_float(geom.get('pred_confidence_mean'))}/{_fmt_float(geom.get('pred_confidence_median'))}"
        )
        print(
            f"  data_quality indexed/missing_img/missing_gt/missing_pred/invalid="
            f"{dq_split.get('indexed_samples', 0)}/{dq_split.get('missing_image_samples', 0)}/"
            f"{dq_split.get('missing_gt_label_samples', 0)}/{dq_split.get('missing_prediction_file_samples', 0)}/"
            f"{dq_split.get('invalid_samples_skipped', 0)}"
        )
    print("")
    print("Reports")
    print(f"- summary_json: {out['summary_json']}")
    print(f"- samples_jsonl: {out['sample_jsonl']}")
    print(f"- summary_md: {out['summary_md']}")


if __name__ == "__main__":
    main()
