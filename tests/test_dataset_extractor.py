from __future__ import annotations

import json
from pathlib import Path
import zipfile

from dataset_extractor.cli import main
from dataset_extractor.fetch import extract_coco17_family, materialize_coco_power_of_two_subsets


def _build_train_zip(path: Path, count: int) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("__MACOSX/._ignored", b"x")
        zf.writestr("train2017/readme.txt", b"ignore")
        for idx in range(count):
            zf.writestr(f"train2017/{idx:06d}.jpg", b"sample")


def test_extract_coco17_family_creates_disjoint_train_val_and_configs(tmp_path) -> None:
    train_zip = tmp_path / "train2017.zip"
    dataset_root = tmp_path / "dataset"
    configs_root = tmp_path / "configs" / "datasets"
    _build_train_zip(train_zip, 20)

    coco17_dir, subsets = extract_coco17_family(
        train_zip_path=train_zip,
        dataset_root=dataset_root,
        configs_root=configs_root,
    )

    train_images = sorted((coco17_dir / "images" / "train2017").iterdir())
    val_images = sorted((coco17_dir / "images" / "val2017").iterdir())
    assert len(train_images) == 15
    assert len(val_images) == 5
    assert {path.name for path in train_images}.isdisjoint({path.name for path in val_images})
    assert json.loads((configs_root / "coco17.json").read_text(encoding="utf-8"))["source"]["type"] == "local_dir"
    assert subsets == []


def test_materialize_coco_power_of_two_subsets_writes_configs(tmp_path) -> None:
    coco17_dir = tmp_path / "dataset" / "coco17"
    train_dir = coco17_dir / "images" / "train2017"
    val_dir = coco17_dir / "images" / "val2017"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for idx in range(300):
        (train_dir / f"train-{idx:04d}.jpg").write_bytes(b"x")
    for idx in range(100):
        (val_dir / f"val-{idx:04d}.jpg").write_bytes(b"x")

    subsets = materialize_coco_power_of_two_subsets(
        coco17_dir=coco17_dir,
        dataset_root=tmp_path / "dataset",
        configs_root=tmp_path / "configs" / "datasets",
    )

    names = [item["name"] for item in subsets]
    assert names == ["coco128", "coco256"]
    coco128_train = list((tmp_path / "dataset" / "coco128" / "images" / "train2017").iterdir())
    coco128_val = list((tmp_path / "dataset" / "coco128" / "images" / "val2017").iterdir())
    assert len(coco128_train) == 128
    assert len(coco128_val) == 32
    assert all(path.is_symlink() for path in coco128_train)
    coco128_cfg = json.loads((tmp_path / "configs" / "datasets" / "coco128.json").read_text(encoding="utf-8"))
    assert coco128_cfg["splits"]["train_images_rel"] == "images/train2017"
    assert coco128_cfg["source"]["local_path"].endswith("dataset/coco128")


def test_dataset_extractor_cli_uses_pipeline_config(tmp_path, monkeypatch, capsys) -> None:
    train_zip = tmp_path / "train2017.zip"
    _build_train_zip(train_zip, 512)
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
                    "model": "m",
                    "model_key": "k",
                    "run_id": "r",
                    "seed": 42,
                },
                "dataset": {
                    "name": "coco128",
                    "augmented_subdir": "augmented",
                    "splits": ["train", "val"],
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
                    "dino_attn_enabled": False,
                    "dino_attn_layers": [1],
                    "dino_attn_weight_stage_a": 0.0,
                    "dino_attn_weight_stage_b": 0.0,
                    "dino_objmap_enabled": False,
                    "dino_objmap_layer": 1,
                    "dino_objmap_weight_stage_a": 0.0,
                    "dino_objmap_weight_stage_b": 0.0,
                    "dino_objmap_apply_inside_gt_only": False,
                    "stage_a_ratio": 0.5,
                    "stage_a_freeze": 0,
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
                    "debug_overlays_per_split": 1,
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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["dataset-extractor", "--config", str(config_path)])
    main()

    captured = capsys.readouterr().out
    assert "coco17 ready at:" in captured
    assert (tmp_path / "dataset" / "coco128" / "images" / "train2017").exists()
    assert (tmp_path / "configs" / "datasets" / "coco128.json").exists()
