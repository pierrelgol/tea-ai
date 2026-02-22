from __future__ import annotations

import argparse
from pathlib import Path

from .config import GeneratorConfig
from .generator import generate_dataset
from .profiles import resolve_profile, resolve_split_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented dataset from selected background dataset")
    parser.add_argument("--dataset", default="coco8", help="Dataset profile name (default: coco8)")
    parser.add_argument("--dataset-root", default="dataset", help="Top-level datasets directory (default: dataset)")
    parser.add_argument("--profile", type=Path, default=None, help="Explicit dataset profile JSON path")
    parser.add_argument("--background-root", default=None, help="Manual background dataset root override")
    parser.add_argument("--target-images-dir", default=None, help="Defaults to <dataset-root>/targets/images")
    parser.add_argument("--target-labels-dir", default=None, help="Defaults to <dataset-root>/targets/labels")
    parser.add_argument("--target-classes-file", default=None, help="Defaults to <dataset-root>/targets/classes.txt")
    parser.add_argument("--output-root", default=None, help="Augmented output root (default: dataset/augmented/<dataset>)")
    parser.add_argument("--samples-per-background", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--complexity-profile", choices=["legacy", "obb_robust_v1"], default="obb_robust_v1")

    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--translate-frac", type=float, default=0.25)
    parser.add_argument("--perspective-jitter", type=float, default=0.08)
    parser.add_argument("--min-quad-area-frac", type=float, default=0.002)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--class-offset-base", type=int, default=80)
    parser.add_argument("--targets-per-image-min", type=int, default=2)
    parser.add_argument("--targets-per-image-max", type=int, default=4)
    parser.add_argument("--max-occlusion-ratio", type=float, default=0.45)
    parser.add_argument("--allow-partial-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blur-prob", type=float, default=0.35)
    parser.add_argument("--motion-blur-prob", type=float, default=0.20)
    parser.add_argument("--noise-prob", type=float, default=0.25)
    parser.add_argument("--jpeg-artifact-prob", type=float, default=0.20)
    parser.add_argument("--color-jitter-prob", type=float, default=0.50)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.background_root:
        dataset_name = args.dataset
        background_splits = {
            "train": Path(args.background_root) / "images" / "train",
            "val": Path(args.background_root) / "images" / "val",
        }
    else:
        profile, _profile_path = resolve_profile(dataset_name=args.dataset, profile_path=args.profile)
        dataset_name = profile.name
        background_splits = resolve_split_dirs(dataset_root=dataset_root, profile=profile)

    output_root = Path(args.output_root) if args.output_root else dataset_root / "augmented" / dataset_name
    target_images_dir = Path(args.target_images_dir) if args.target_images_dir else dataset_root / "targets" / "images"
    target_labels_dir = Path(args.target_labels_dir) if args.target_labels_dir else dataset_root / "targets" / "labels"
    target_classes_file = (
        Path(args.target_classes_file) if args.target_classes_file else dataset_root / "targets" / "classes.txt"
    )

    if args.complexity_profile == "legacy":
        args.targets_per_image_min = 1
        args.targets_per_image_max = 1
        args.max_occlusion_ratio = 0.0
        args.allow_partial_visibility = False
        args.blur_prob = 0.0
        args.motion_blur_prob = 0.0
        args.noise_prob = 0.0
        args.jpeg_artifact_prob = 0.0
        args.color_jitter_prob = 0.0

    config = GeneratorConfig(
        background_splits=background_splits,
        background_dataset_name=dataset_name,
        target_images_dir=target_images_dir,
        target_labels_dir=target_labels_dir,
        target_classes_file=target_classes_file,
        output_root=output_root,
        samples_per_background=args.samples_per_background,
        seed=args.seed,
        generator_version="obb_robust_v1",
        complexity_profile=args.complexity_profile,
        targets_per_image_min=args.targets_per_image_min,
        targets_per_image_max=args.targets_per_image_max,
        max_occlusion_ratio=args.max_occlusion_ratio,
        allow_partial_visibility=args.allow_partial_visibility,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        translate_frac=args.translate_frac,
        perspective_jitter=args.perspective_jitter,
        min_quad_area_frac=args.min_quad_area_frac,
        max_attempts=args.max_attempts,
        class_offset_base=args.class_offset_base,
        blur_prob=args.blur_prob,
        motion_blur_prob=args.motion_blur_prob,
        noise_prob=args.noise_prob,
        jpeg_artifact_prob=args.jpeg_artifact_prob,
        color_jitter_prob=args.color_jitter_prob,
    )

    results = generate_dataset(config)
    print(f"generated {len(results)} synthetic samples in {config.output_root}")
