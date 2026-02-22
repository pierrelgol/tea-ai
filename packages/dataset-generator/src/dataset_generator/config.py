from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class GeneratorConfig:
    background_splits: dict[str, Path] = field(
        default_factory=lambda: {
            "train": Path("backgrounds/train"),
            "val": Path("backgrounds/val"),
        }
    )
    background_dataset_name: str = "default"
    target_images_dir: Path = Path("targets/images")
    target_labels_dir: Path = Path("targets/labels")
    target_classes_file: Path = Path("targets/classes.txt")
    output_root: Path = Path("augmented/default")

    samples_per_background: int = 1
    seed: int | None = None
    generator_version: str = "obb_robust_v1"
    complexity_profile: str = "obb_robust_v1"

    targets_per_image_min: int = 2
    targets_per_image_max: int = 4
    max_occlusion_ratio: float = 0.45
    allow_partial_visibility: bool = True

    scale_min: float = 0.4
    scale_max: float = 1.2
    translate_frac: float = 0.25
    perspective_jitter: float = 0.08
    min_quad_area_frac: float = 0.002
    max_attempts: int = 50

    class_offset_base: int = 80
    blur_prob: float = 0.35
    motion_blur_prob: float = 0.20
    noise_prob: float = 0.25
    jpeg_artifact_prob: float = 0.20
    color_jitter_prob: float = 0.50

    def validate(self) -> None:
        if self.samples_per_background < 1:
            raise ValueError("samples_per_background must be >= 1")
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale bounds must be > 0")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be <= scale_max")
        if self.translate_frac < 0:
            raise ValueError("translate_frac must be >= 0")
        if self.perspective_jitter < 0:
            raise ValueError("perspective_jitter must be >= 0")
        if self.min_quad_area_frac <= 0:
            raise ValueError("min_quad_area_frac must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.class_offset_base < 0:
            raise ValueError("class_offset_base must be >= 0")
        if self.targets_per_image_min < 1:
            raise ValueError("targets_per_image_min must be >= 1")
        if self.targets_per_image_max < self.targets_per_image_min:
            raise ValueError("targets_per_image_max must be >= targets_per_image_min")
        if self.max_occlusion_ratio < 0 or self.max_occlusion_ratio >= 1:
            raise ValueError("max_occlusion_ratio must be in [0,1)")
        for name, value in (
            ("blur_prob", self.blur_prob),
            ("motion_blur_prob", self.motion_blur_prob),
            ("noise_prob", self.noise_prob),
            ("jpeg_artifact_prob", self.jpeg_artifact_prob),
            ("color_jitter_prob", self.color_jitter_prob),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if "train" not in self.background_splits or "val" not in self.background_splits:
            raise ValueError("background_splits must define train and val paths")
        for split, path in self.background_splits.items():
            if not path.exists():
                raise FileNotFoundError(f"background split path does not exist ({split}): {path}")
