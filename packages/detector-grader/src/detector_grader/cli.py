from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import GradingConfig, run_grading


def _split_csv(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade detector runs from strict OBB geometry quality")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions"))
    parser.add_argument("--model", default="latest", help="Model source: latest|.|weights path|run dir|prediction model key")
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

    result = run_grading(
        GradingConfig(
            dataset_root=dataset_root,
            predictions_root=args.predictions_root,
            artifacts_root=args.artifacts_root,
            reports_dir=args.reports_dir,
            model=args.model,
            weights=args.weights,
            run_inference=args.run_inference,
            splits=_split_csv(args.splits),
            imgsz=args.imgsz,
            device=args.device,
            conf_threshold=args.conf_threshold,
            infer_iou_threshold=args.infer_iou_threshold,
            match_iou_threshold=args.match_iou_threshold,
            weights_json=args.weights_json,
            strict_obb=args.strict_obb,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )

    aggregate = result["aggregate"]
    run_det = aggregate.get("run_detection", {})

    print("Model Source")
    print(f"- resolved_model_key: {result['model_key']}")
    print(f"- weights: {result['weights_path'] or 'N/A (using existing predictions)'}")
    print(f"- predictions_root: {args.predictions_root / result['model_key'] / 'labels'}")
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
