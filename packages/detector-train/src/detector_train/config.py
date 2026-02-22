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
