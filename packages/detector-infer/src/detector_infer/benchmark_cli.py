from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config
from pipeline_runtime_utils import resolve_latest_weights_from_artifacts

from .benchmark import BenchmarkConfig, run_latency_benchmark


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detector inference latency on eval splits")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    model_key = str(shared.run["model_key"])
    run_id = str(shared.run["run_id"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=run_id,
    )

    model_value = str(shared.run["model"])
    model_path = Path(model_value)
    if model_path.exists() and model_path.suffix == ".pt":
        weights = model_path
    else:
        weights = resolve_latest_weights_from_artifacts(shared.paths["artifacts_root"])

    infer_cfg = shared.infer
    grade_cfg = shared.grade
    splits = args.splits or [str(s) for s in grade_cfg.get("splits", infer_cfg.get("splits", ["val"]))]
    report_path = args.report_path or (layout.eval_root / "latency_benchmark.json")

    result = run_latency_benchmark(
        BenchmarkConfig(
            weights=weights,
            dataset_root=dataset_root,
            model_name=model_key,
            imgsz=int(infer_cfg.get("imgsz", 640) if args.imgsz is None else args.imgsz),
            device=str(infer_cfg.get("device", "auto") if args.device is None else args.device),
            conf_threshold=float(
                infer_cfg.get("conf_threshold", 0.25) if args.conf_threshold is None else args.conf_threshold
            ),
            iou_threshold=float(
                infer_cfg.get("iou_threshold", 0.7) if args.iou_threshold is None else args.iou_threshold
            ),
            seed=int(shared.run["seed"]),
            splits=splits,
            batch_size=int(infer_cfg.get("batch_size", 16) if args.batch_size is None else args.batch_size),
            warmup_batches=int(args.warmup_batches),
            max_images=None if args.max_images is None else int(args.max_images),
            report_path=report_path,
        )
    )

    print("Latency Benchmark")
    print(f"- model: {result['model_name']}")
    print(f"- weights: {result['weights']}")
    print(f"- device: {result['resolved_device']}")
    print(f"- splits: {', '.join(result['splits'])}")
    print(f"- images_considered: {result['images_considered']}")
    print(f"- warmup_batches_run: {result['warmup_batches_run']}")
    print(f"- timed_batches: {result['timed_batches']}")
    print(f"- timed_images: {result['timed_images']}")
    print(f"- mean_batch_latency_ms: {_fmt_float(result['mean_batch_latency_ms'])}")
    print(f"- p95_batch_latency_ms: {_fmt_float(result['p95_batch_latency_ms'])}")
    print(f"- mean_image_latency_ms: {_fmt_float(result['mean_image_latency_ms'])}")
    print(f"- throughput_images_per_second: {_fmt_float(result['throughput_images_per_second'])}")
    print(f"- report_path: {result['report_path']}")


if __name__ == "__main__":
    main()
