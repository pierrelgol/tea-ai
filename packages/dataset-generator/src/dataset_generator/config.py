from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GeneratorConfig:
    background_root: Path = Path("dataset/coco8")
    target_images_dir: Path = Path("dataset/targets/images")
    target_labels_dir: Path = Path("dataset/targets/labels")
    target_classes_file: Path = Path("dataset/targets/classes.txt")
    output_root: Path = Path("dataset/augmented")

    samples_per_background: int = 1
    seed: int | None = None

    scale_min: float = 0.4
    scale_max: float = 1.2
    translate_frac: float = 0.25
    perspective_jitter: float = 0.08
    min_quad_area_frac: float = 0.002
    max_attempts: int = 50

    class_offset_base: int = 80

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
