from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector predictions against YOLO GT with stability and geometry metrics")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("dataset/augmented/eval_reports"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz-samples", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names filter")
    args = parser.parse_args()

    models_filter = [x.strip() for x in args.models.split(",")] if args.models else None
    viz_samples = 0 if args.no_viz else args.viz_samples

    result = evaluate_models(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        reports_dir=args.reports_dir,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        seed=args.seed,
        viz_samples=viz_samples,
        models_filter=models_filter,
    )

    print(f"indexed samples: {result['num_samples_indexed']}")
    print(f"samples with GT+image: {result['num_samples_with_gt']}")
    if result["models"]:
        for row in result["models"]:
            print(
                f"{row['model_name']}: P={row['precision']:.4f} R={row['recall']:.4f} "
                f"miss={row['miss_rate']:.4f} meanIoU={row['mean_iou']} AP={row['ap_at_iou']} "
                f"drift={row['mean_center_drift_px']}"
            )
    else:
        print("no models evaluated (predictions root missing/empty or filtered out)")

    print(f"summary json: {result['summary_json']}")
    print(f"summary md: {result['summary_md']}")
