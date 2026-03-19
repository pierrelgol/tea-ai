from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from detector_infer.benchmark_cli import main


def test_detector_infer_benchmark_cli_writes_report(tmp_path, monkeypatch, capsys) -> None:
    dataset_root = tmp_path / "dataset" / "augmented" / "coco128"
    image_dir = dataset_root / "images" / "val"
    image_dir.mkdir(parents=True)
    for idx in range(3):
        (image_dir / f"sample_{idx}.jpg").write_bytes(b"x")

    artifacts_root = tmp_path / "artifacts"
    weights_path = artifacts_root / "k" / "runs" / "r" / "train" / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"weights")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "dataset_root": "dataset",
                    "artifacts_root": "artifacts",
                    "configs_root": "configs/datasets",
                    "targets_source_root": "targets",
                },
                "run": {
                    "dataset": "coco128",
                    "model": str(weights_path),
                    "model_key": "k",
                    "run_id": "r",
                    "seed": 42,
                },
                "dataset": {
                    "name": "coco128",
                    "augmented_subdir": "augmented",
                    "splits": ["val"],
                    "train_zip": "train2017.zip",
                },
                "generator": {"seed": 42},
                "tuner": {
                    "enabled": False,
                    "dataset": "coco128",
                    "coarse_epochs": 1,
                    "confirm_epochs": 1,
                    "vram_target_utilization": 0.9,
                    "batch_min": 1,
                    "batch_max_cap": 1,
                    "imgsz_candidates": [64],
                    "workers_candidates": [0],
                    "cache_candidates": ["disk"],
                    "amp_candidates": [False],
                    "tf32_candidates": [False],
                    "cudnn_benchmark_candidates": [False],
                    "max_trials": 1,
                    "artifacts_subdir": "tuner",
                },
                "train": {
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 1,
                    "batch_mode": "fixed",
                    "batch_max": 1,
                    "batch_utilization_target": 0.9,
                    "oom_backoff_factor": 0.85,
                    "workers": 0,
                    "workers_auto": False,
                    "workers_max": 1,
                    "patience": 1,
                    "cache": "disk",
                    "throughput_mode": "balanced",
                    "device": "cpu",
                    "optimizer": "AdamW",
                    "lr0": 0.001,
                    "lrf": 0.01,
                    "weight_decay": 0.0,
                    "warmup_epochs": 0.0,
                    "cos_lr": False,
                    "close_mosaic": 0,
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "degrees": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "hsv_v": 0.0,
                    "fliplr": 0.0,
                    "flipud": 0.0,
                    "copy_paste": 0.0,
                    "multi_scale": False,
                    "freeze": None,
                    "amp": False,
                    "plots": False,
                    "tf32": False,
                    "cudnn_benchmark": False,
                    "dino_root": "dinov3",
                    "dino_distill_warmup_epochs": 0,
                    "dino_distill_layers": [1],
                    "dino_distill_channels": 8,
                    "dino_to_yolo_taps": {"1": 1},
                    "dino_feat_layer_weights": {"1": 1.0},
                    "dino_distill_object_weight": 0.0,
                    "dino_distill_background_weight": 0.0,
                    "stage_a_ratio": 0.5,
                    "stage_a_freeze": 0,
                    "dino_attn_enabled": False,
                    "dino_attn_layers": [1],
                    "dino_attn_weight_stage_a": 0.0,
                    "dino_attn_weight_stage_b": 0.0,
                    "dino_objmap_enabled": False,
                    "dino_objmap_layer": 1,
                    "dino_objmap_weight_stage_a": 0.0,
                    "dino_objmap_weight_stage_b": 0.0,
                    "dino_objmap_apply_inside_gt_only": False,
                    "stage_a_distill_weight": 0.0,
                    "stage_b_distill_weight": 0.0,
                    "dino_viz_enabled": False,
                    "dino_viz_mode": "off",
                    "dino_viz_every_n_epochs": 1,
                    "dino_viz_max_samples": 1,
                    "wandb_enabled": False,
                    "wandb_project": "p",
                    "wandb_entity": None,
                    "wandb_run_name": None,
                    "wandb_tags": [],
                    "wandb_notes": None,
                    "wandb_mode": "offline",
                    "wandb_log_system_metrics": False,
                    "wandb_log_every_epoch": False,
                    "eval_enabled": False,
                    "periodic_eval_mode": "off",
                    "periodic_eval_sparse_epochs": 1,
                    "eval_interval_epochs": 1,
                    "eval_iou_threshold": 0.5,
                    "eval_conf_threshold": 0.5,
                    "eval_viz_samples": 0,
                    "eval_viz_split": "val",
                    "tuned_gpu_signature": None,
                    "tuned_at_utc": None,
                    "tuned_by": None,
                    "tuned_profile_path": None,
                },
                "infer": {
                    "imgsz": 64,
                    "device": "cpu",
                    "conf_threshold": 0.25,
                    "iou_threshold": 0.7,
                    "splits": ["val"],
                    "save_empty": True,
                    "batch_size": 1,
                },
                "grade": {
                    "splits": ["val"],
                    "imgsz": 64,
                    "device": "cpu",
                    "conf_threshold": 0.25,
                    "infer_iou_threshold": 0.7,
                    "match_iou_threshold": 0.5,
                    "strict_obb": True,
                    "max_samples": None,
                    "calibrate_confidence": False,
                    "calibration_candidates": None,
                    "weights_json": None,
                    "run_inference": False,
                },
                "review": {"split": "val", "conf_threshold": 0.25},
                "checks": {
                    "outlier_threshold_px": 2.0,
                    "debug_overlays_per_split": 10,
                    "gui": False,
                    "seed": 42,
                },
                "profile": {
                    "dataset": "coco128",
                    "train_epochs": 1,
                    "enable_gpu_sampling": False,
                    "baseline_run": "baseline",
                    "regression_gate": {
                        "min_run_grade_delta": 0.0,
                        "max_precision_drop": 0.02,
                        "max_recall_drop": 0.02,
                        "max_miss_rate_increase": 0.02,
                        "max_total_duration_increase_pct": 5.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeYOLO:
        calls: list[dict] = []

        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, *, source, conf, iou, imgsz, device, verbose):
            self.calls.append(
                {
                    "source": source,
                    "conf": conf,
                    "iou": iou,
                    "imgsz": imgsz,
                    "device": device,
                    "verbose": verbose,
                }
            )
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    ticks = iter([0.0, 0.005, 0.010, 0.016, 0.020, 0.029])
    monkeypatch.setattr("detector_infer.benchmark.perf_counter", lambda: next(ticks))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["detector-infer-benchmark", "--config", str(config_path), "--warmup-batches", "1"],
    )

    main()

    captured = capsys.readouterr().out
    report_path = artifacts_root / "k" / "runs" / "r" / "eval" / "latency_benchmark.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert "Latency Benchmark" in captured
    assert payload["status"] == "ok"
    assert payload["warmup_batches_run"] == 1
    assert payload["timed_batches"] == 2
    assert payload["timed_images"] == 2
    assert payload["mean_batch_latency_ms"] == pytest.approx(7.5)
    assert payload["p95_batch_latency_ms"] == pytest.approx(8.85)
    assert len(FakeYOLO.calls) == 3
