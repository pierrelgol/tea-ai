from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    dataset_root: Path
    artifacts_root: Path
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    project: Path
    name: str
    seed: int
    workers: int
    patience: int
    save_json: bool
    train_profile: str

    optimizer: str
    lr0: float | None
    lrf: float | None
    weight_decay: float | None
    warmup_epochs: float | None
    cos_lr: bool

    close_mosaic: int | None
    mosaic: float | None
    mixup: float | None
    degrees: float | None
    translate: float | None
    scale: float | None
    shear: float | None
    perspective: float | None
    hsv_h: float | None
    hsv_s: float | None
    hsv_v: float | None
    fliplr: float | None
    flipud: float | None
    copy_paste: float | None
    multi_scale: bool
    freeze: int | None

    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_tags: list[str]
    wandb_notes: str | None
    wandb_mode: str  # online|offline|auto
    wandb_log_profile: str  # core|core+diag
    wandb_log_system_metrics: bool
    wandb_log_every_epoch: bool

    eval_enabled: bool
    eval_interval_epochs: int
    eval_iou_threshold: float
    eval_conf_threshold: float
    eval_viz_samples: int

    def validate(self) -> None:
        project_resolved = self.project if self.project.is_absolute() else (Path.cwd() / self.project)
        project_resolved = project_resolved.resolve()
        cwd_resolved = Path.cwd().resolve()
        if not project_resolved.is_relative_to(cwd_resolved):
            raise ValueError("project path must be inside the repository working directory")

        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.train_profile not in {"default", "obb_precision_v1", "obb_precision_v2"}:
            raise ValueError("train_profile must be one of: default, obb_precision_v1, obb_precision_v2")
        if self.optimizer not in {"SGD", "AdamW", "auto"}:
            raise ValueError("optimizer must be one of: SGD, AdamW, auto")
        if self.lr0 is not None and self.lr0 <= 0:
            raise ValueError("lr0 must be > 0")
        if self.lrf is not None and self.lrf <= 0:
            raise ValueError("lrf must be > 0")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.warmup_epochs is not None and self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.close_mosaic is not None and self.close_mosaic < 0:
            raise ValueError("close_mosaic must be >= 0")
        if self.freeze is not None and self.freeze < 0:
            raise ValueError("freeze must be >= 0")
        if self.wandb_mode not in {"online", "offline", "auto"}:
            raise ValueError("wandb_mode must be one of: online, offline, auto")
        if self.wandb_log_profile not in {"core", "core+diag"}:
            raise ValueError("wandb_log_profile must be one of: core, core+diag")
        if self.eval_interval_epochs < 1:
            raise ValueError("eval_interval_epochs must be >= 1")
        if self.eval_iou_threshold < 0 or self.eval_iou_threshold > 1:
            raise ValueError("eval_iou_threshold must be in [0,1]")
        if self.eval_conf_threshold < 0 or self.eval_conf_threshold > 1:
            raise ValueError("eval_conf_threshold must be in [0,1]")
        if self.eval_viz_samples < 0:
            raise ValueError("eval_viz_samples must be >= 0")
