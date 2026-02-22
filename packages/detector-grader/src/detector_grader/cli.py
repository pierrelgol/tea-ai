from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import GradingConfig, run_grading


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade detector runs from strict OBB geometry quality")
    parser.add_argument("--dataset", default="coco1024", help="Dataset name under dataset/augmented/")
    parser.add_argument("--model", default="latest", help="Prediction model key or latest")
    parser.add_argument("--run-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for infer + grading")
    parser.add_argument("--calibrate-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path("dataset/augmented") / args.dataset

    result = run_grading(
        GradingConfig(
            dataset_root=dataset_root,
            predictions_root=Path("predictions"),
            artifacts_root=Path("artifacts/detector-train"),
            reports_dir=None,
            model=args.model,
            weights=None,
            run_inference=args.run_inference,
            splits=["val"],
            imgsz=640,
            device="auto",
            conf_threshold=args.conf_threshold,
            calibrate_confidence=args.calibrate_confidence,
            infer_iou_threshold=0.7,
            match_iou_threshold=0.5,
            weights_json=None,
            strict_obb=True,
            max_samples=None,
            seed=args.seed,
        )
    )

    aggregate = result["aggregate"]
    run_det = aggregate.get("run_detection", {})

    print("Model Source")
    print(f"- resolved_model_key: {result['model_key']}")
    print(f"- weights: {result['weights_path'] or 'N/A (using existing predictions)'}")
    print(f"- predictions_root: {Path('predictions') / result['model_key'] / 'labels'}")
    print("")
    print("Inference")
    if result["inference"] is None:
        print("- executed: no")
        print("- reason: using existing predictions")
    else:
        print("- executed: yes")
        print(f"- device: {result['inference']['resolved_device']}")
        print(f"- images_processed: {result['inference']['images_processed']}")
        print(f"- label_files_written: {result['inference']['label_files_written']}")
    print("")
    print("Grading")
    print(f"- dataset_root: {dataset_root}")
    print(f"- samples_scored: {aggregate['num_samples_scored']}")
    print(f"- run_grade_0_100: {aggregate['run_grade_0_100']:.4f}")
    print(f"- run_precision_proxy: {_fmt_float(run_det.get('precision_proxy'))}")
    print(f"- run_recall_proxy: {_fmt_float(run_det.get('recall_proxy'))}")
    print(f"- run_miss_rate_proxy: {_fmt_float(run_det.get('miss_rate_proxy'))}")
    print(f"- run_class_match_rate: {_fmt_float(run_det.get('class_match_rate'))}")
    print("")
    print("Artifacts")
    print(f"- summary_json: {result['reports']['summary_json']}")
    print(f"- sample_jsonl: {result['reports']['sample_jsonl']}")
    print(f"- summary_md: {result['reports']['summary_md']}")


if __name__ == "__main__":
    main()
