from __future__ import annotations

import argparse
from pathlib import Path

from .config import GeneratorConfig
from .generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented dataset from coco8 and targets")
    parser.add_argument("--background-root", default="dataset/coco8")
    parser.add_argument("--target-images-dir", default="dataset/targets/images")
    parser.add_argument("--target-labels-dir", default="dataset/targets/labels")
    parser.add_argument("--target-classes-file", default="dataset/targets/classes.txt")
    parser.add_argument("--output-root", default="dataset/augmented")
    parser.add_argument("--samples-per-background", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--translate-frac", type=float, default=0.25)
    parser.add_argument("--perspective-jitter", type=float, default=0.08)
    parser.add_argument("--min-quad-area-frac", type=float, default=0.002)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--class-offset-base", type=int, default=80)

    args = parser.parse_args()

    config = GeneratorConfig(
        background_root=Path(args.background_root),
        target_images_dir=Path(args.target_images_dir),
        target_labels_dir=Path(args.target_labels_dir),
        target_classes_file=Path(args.target_classes_file),
        output_root=Path(args.output_root),
        samples_per_background=args.samples_per_background,
        seed=args.seed,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        translate_frac=args.translate_frac,
        perspective_jitter=args.perspective_jitter,
        min_quad_area_frac=args.min_quad_area_frac,
        max_attempts=args.max_attempts,
        class_offset_base=args.class_offset_base,
    )

    results = generate_dataset(config)
    print(f"generated {len(results)} synthetic samples in {config.output_root}")
