from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config

from .config import InferConfig
from .infer import run_inference


def _resolve_latest_weights(artifacts_root: Path) -> Path:
    latest_file = artifacts_root / "latest_run.json"
    if latest_file.exists():
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        best = payload.get("weights_best")
        if isinstance(best, str) and best:
            p = Path(best)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                return p
    candidates = sorted(
        artifacts_root.glob("*/runs/*/train/ultralytics/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"could not resolve latest best weights from {artifacts_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference and export YOLO predictions")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
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
        weights = _resolve_latest_weights(shared.paths["artifacts_root"])

    ic = shared.infer
    cfg = InferConfig(
        weights=weights,
        dataset_root=dataset_root,
        output_root=layout.infer_root,
        model_name=model_key,
        imgsz=int(ic.get("imgsz", 640)),
        device=str(ic.get("device", "auto")),
        conf_threshold=float(ic.get("conf_threshold", 0.25)),
        iou_threshold=float(ic.get("iou_threshold", 0.7)),
        seed=int(shared.run["seed"]),
        splits=[str(s) for s in ic.get("splits", ["val"])],
        save_empty=bool(ic.get("save_empty", True)),
    )

    summary = run_inference(cfg)
    print(f"status: {summary['status']}")
    print(f"model: {summary['model_name']}")
    print(f"images_processed: {summary['images_processed']}")
    print(f"label_files_written: {summary['label_files_written']}")
    print(f"output_root: {summary['output_root']}")


if __name__ == "__main__":
    main()
