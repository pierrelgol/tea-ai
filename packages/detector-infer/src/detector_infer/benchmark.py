from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from pipeline_runtime_utils import resolve_device, set_seed

from .config import InferConfig
from .dataset import list_split_images


@dataclass(slots=True)
class BenchmarkConfig:
    weights: Path
    dataset_root: Path
    model_name: str
    imgsz: int
    device: str
    conf_threshold: float
    iou_threshold: float
    seed: int
    splits: list[str]
    batch_size: int = 1
    warmup_batches: int = 0
    max_images: int | None = None
    report_path: Path | None = None

    def validate(self) -> None:
        InferConfig(
            weights=self.weights,
            dataset_root=self.dataset_root,
            output_root=Path("."),
            model_name=self.model_name,
            imgsz=self.imgsz,
            device=self.device,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            seed=self.seed,
            splits=self.splits,
            save_empty=True,
            batch_size=self.batch_size,
        ).validate()
        if self.warmup_batches < 0:
            raise ValueError("warmup_batches must be >= 0")
        if self.max_images is not None and self.max_images < 1:
            raise ValueError("max_images must be >= 1")


def _sync_device(device: str) -> None:
    try:
        import torch

        if (
            device == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "synchronize")
        ):
            torch.mps.synchronize()
            return
        if device not in {"cpu", "mps"} and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _collect_images(
    dataset_root: Path, splits: list[str], max_images: int | None
) -> list[Path]:
    images: list[Path] = []
    for split in splits:
        images.extend(list_split_images(dataset_root, split))
    if max_images is not None:
        return images[:max_images]
    return images


def _summarize_latencies(
    latencies_ms: list[float], batch_sizes: list[int]
) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {
            "timed_batches": 0,
            "timed_images": 0,
            "mean_batch_latency_ms": None,
            "median_batch_latency_ms": None,
            "p95_batch_latency_ms": None,
            "min_batch_latency_ms": None,
            "max_batch_latency_ms": None,
            "mean_image_latency_ms": None,
            "throughput_images_per_second": None,
        }

    total_images = int(sum(batch_sizes))
    total_seconds = float(sum(latencies_ms) / 1000.0)
    mean_batch = float(np.mean(latencies_ms))
    median_batch = float(np.median(latencies_ms))
    p95_batch = float(np.percentile(latencies_ms, 95))
    mean_image = float(
        sum(
            lat / size
            for lat, size in zip(latencies_ms, batch_sizes, strict=True)
        )
        / len(latencies_ms)
    )

    return {
        "timed_batches": len(latencies_ms),
        "timed_images": total_images,
        "mean_batch_latency_ms": mean_batch,
        "median_batch_latency_ms": median_batch,
        "p95_batch_latency_ms": p95_batch,
        "min_batch_latency_ms": float(min(latencies_ms)),
        "max_batch_latency_ms": float(max(latencies_ms)),
        "mean_image_latency_ms": mean_image,
        "throughput_images_per_second": float(total_images / total_seconds)
        if total_seconds > 0
        else None,
    }


def run_latency_benchmark(config: BenchmarkConfig) -> dict:
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)

    images = _collect_images(
        config.dataset_root, config.splits, config.max_images
    )
    if not images:
        raise ValueError(
            f"no images found for splits={config.splits} under {config.dataset_root}"
        )

    from ultralytics import YOLO

    model = YOLO(str(config.weights))
    batch_latencies_ms: list[float] = []
    batch_sizes: list[int] = []
    warmup_batches_run = 0

    for batch_index, start in enumerate(
        range(0, len(images), config.batch_size)
    ):
        batch_paths = images[start : start + config.batch_size]
        _sync_device(device)
        started = perf_counter()
        model.predict(
            source=[str(path) for path in batch_paths],
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            imgsz=config.imgsz,
            device=device,
            verbose=False,
        )
        _sync_device(device)
        elapsed_ms = (perf_counter() - started) * 1000.0

        if batch_index < config.warmup_batches:
            warmup_batches_run += 1
            continue

        batch_latencies_ms.append(elapsed_ms)
        batch_sizes.append(len(batch_paths))

    summary = _summarize_latencies(batch_latencies_ms, batch_sizes)
    result = {
        "status": "ok",
        "weights": str(config.weights),
        "model_name": config.model_name,
        "dataset_root": str(config.dataset_root),
        "resolved_device": device,
        "imgsz": config.imgsz,
        "batch_size": config.batch_size,
        "splits": list(config.splits),
        "warmup_batches_requested": config.warmup_batches,
        "warmup_batches_run": warmup_batches_run,
        "images_considered": len(images),
        "max_images": config.max_images,
        **summary,
    }

    if config.report_path is not None:
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        result["report_path"] = str(config.report_path)

    return result
