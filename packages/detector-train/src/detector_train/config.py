from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    dataset_root: Path
    artifacts_root: Path
    model: str
    name: str
    seed: int

    device: str = "auto"
    project: Path = Path("artifacts/detector-train/runs")

    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    patience: int = 50

    optimizer: str = "AdamW"
    lr0: float = 0.002
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    cos_lr: bool = True

    close_mosaic: int = 20
    mosaic: float = 0.45
    mixup: float = 0.0
    degrees: float = 1.5
    translate: float = 0.06
    scale: float = 0.35
    shear: float = 0.0
    perspective: float = 0.0
    hsv_h: float = 0.010
    hsv_s: float = 0.35
    hsv_v: float = 0.25
    fliplr: float = 0.5
    flipud: float = 0.0
    copy_paste: float = 0.0
    multi_scale: bool = False
    freeze: int | None = None

    wandb_enabled: bool = True
    wandb_project: str = "tea-ai-detector"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_mode: str = "auto"  # online|offline|auto
    wandb_log_system_metrics: bool = False
    wandb_log_every_epoch: bool = True

    eval_enabled: bool = True
    eval_interval_epochs: int = 5
    eval_iou_threshold: float = 0.5
    eval_conf_threshold: float = 0.25
    eval_viz_samples: int = 0

    save_json: bool = True

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
        if self.optimizer not in {"SGD", "AdamW", "auto"}:
            raise ValueError("optimizer must be one of: SGD, AdamW, auto")
        if self.lr0 <= 0:
            raise ValueError("lr0 must be > 0")
        if self.lrf <= 0:
            raise ValueError("lrf must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.close_mosaic < 0:
            raise ValueError("close_mosaic must be >= 0")
        if self.freeze is not None and self.freeze < 0:
            raise ValueError("freeze must be >= 0")
        if self.wandb_mode not in {"online", "offline", "auto"}:
            raise ValueError("wandb_mode must be one of: online, offline, auto")
        if self.eval_interval_epochs < 1:
            raise ValueError("eval_interval_epochs must be >= 1")
        if self.eval_iou_threshold < 0 or self.eval_iou_threshold > 1:
            raise ValueError("eval_iou_threshold must be in [0,1]")
        if self.eval_conf_threshold < 0 or self.eval_conf_threshold > 1:
            raise ValueError("eval_conf_threshold must be in [0,1]")
        if self.eval_viz_samples < 0:
            raise ValueError("eval_viz_samples must be >= 0")
