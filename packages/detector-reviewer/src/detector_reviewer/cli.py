from __future__ import annotations

import argparse
from pathlib import Path

from .app import launch_gui
from .data import index_samples, resolve_latest_weights


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual checker for detector predictions vs ground truth")
    parser.add_argument("--dataset", default="coco8", help="Dataset name under --datasets-base-root")
    parser.add_argument("--datasets-base-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Explicit dataset root override")
    parser.add_argument("--weights", type=Path, default=None, help="Model weights path. Defaults to latest best model.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/detector-train"))
    parser.add_argument("--model-name", default="latest-best")
    parser.add_argument("--splits", default="val", help="Comma-separated splits to review (default: val)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root is not None else args.datasets_base_root / args.dataset
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = index_samples(dataset_root=dataset_root, splits=splits)
    if not samples:
        raise RuntimeError(f"no samples found in dataset: {dataset_root}")

    weights = resolve_latest_weights(args.artifacts_root, args.weights)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model_name = args.model_name if args.model_name else weights.stem

    print(f"dataset_root: {dataset_root}")
    print(f"weights: {weights}")
    print(f"samples: {len(samples)}")

    launch_gui(
        samples=samples,
        model=model,
        model_name=model_name,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
        device=_resolve_device(args.device),
    )


if __name__ == "__main__":
    main()
