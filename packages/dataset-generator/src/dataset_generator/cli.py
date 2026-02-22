from __future__ import annotations

import argparse
from pathlib import Path

from .config import GeneratorConfig
from .generator import generate_dataset
from .profiles import resolve_profile, resolve_split_dirs


def _apply_profile_overrides(args: argparse.Namespace) -> None:
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
        args.edge_bias_prob = 0.0
        return

    if args.complexity_profile == "obb_robust_v2_hard":
        args.targets_per_image_min = 3
        args.targets_per_image_max = 6
        args.max_occlusion_ratio = 0.60
        args.allow_partial_visibility = True
        args.scale_min = 0.25
        args.scale_max = 1.35
        args.translate_frac = 0.35
        args.perspective_jitter = 0.12
        args.min_quad_area_frac = 0.0015
        args.edge_bias_prob = 0.40
        args.edge_band_frac = 0.22
        args.blur_prob = 0.55
        args.motion_blur_prob = 0.35
        args.noise_prob = 0.45
        args.jpeg_artifact_prob = 0.35
        args.color_jitter_prob = 0.75
        args.color_hue_shift_max_deg = 14.0
        args.color_sat_gain_min = 0.60
        args.color_sat_gain_max = 1.45
        args.color_val_gain_min = 0.60
        args.color_val_gain_max = 1.40
        args.gaussian_blur_kernel_min = 5
        args.gaussian_blur_kernel_max = 11
        args.motion_blur_kernel_min = 7
        args.motion_blur_kernel_max = 19
        args.motion_blur_angle_max_deg = 45.0
        args.noise_sigma_min = 5.0
        args.noise_sigma_max = 20.0
        args.jpeg_quality_min = 25
        args.jpeg_quality_max = 75


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
    parser.add_argument(
        "--complexity-profile",
        choices=["legacy", "obb_robust_v1", "obb_robust_v2_hard"],
        default="obb_robust_v1",
    )

    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--translate-frac", type=float, default=0.25)
    parser.add_argument("--perspective-jitter", type=float, default=0.08)
    parser.add_argument("--min-quad-area-frac", type=float, default=0.002)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--edge-bias-prob", type=float, default=0.10)
    parser.add_argument("--edge-band-frac", type=float, default=0.18)
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
    parser.add_argument("--color-hue-shift-max-deg", type=float, default=8.0)
    parser.add_argument("--color-sat-gain-min", type=float, default=0.75)
    parser.add_argument("--color-sat-gain-max", type=float, default=1.25)
    parser.add_argument("--color-val-gain-min", type=float, default=0.75)
    parser.add_argument("--color-val-gain-max", type=float, default=1.25)
    parser.add_argument("--gaussian-blur-kernel-min", type=int, default=3)
    parser.add_argument("--gaussian-blur-kernel-max", type=int, default=9)
    parser.add_argument("--motion-blur-kernel-min", type=int, default=5)
    parser.add_argument("--motion-blur-kernel-max", type=int, default=15)
    parser.add_argument("--motion-blur-angle-max-deg", type=float, default=30.0)
    parser.add_argument("--noise-sigma-min", type=float, default=3.0)
    parser.add_argument("--noise-sigma-max", type=float, default=14.0)
    parser.add_argument("--jpeg-quality-min", type=int, default=40)
    parser.add_argument("--jpeg-quality-max", type=int, default=85)

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

    _apply_profile_overrides(args)

    config = GeneratorConfig(
        background_splits=background_splits,
        background_dataset_name=dataset_name,
        target_images_dir=target_images_dir,
        target_labels_dir=target_labels_dir,
        target_classes_file=target_classes_file,
        output_root=output_root,
        samples_per_background=args.samples_per_background,
        seed=args.seed,
        generator_version="obb_robust_v2" if args.complexity_profile == "obb_robust_v2_hard" else "obb_robust_v1",
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
        edge_bias_prob=args.edge_bias_prob,
        edge_band_frac=args.edge_band_frac,
        class_offset_base=args.class_offset_base,
        blur_prob=args.blur_prob,
        motion_blur_prob=args.motion_blur_prob,
        noise_prob=args.noise_prob,
        jpeg_artifact_prob=args.jpeg_artifact_prob,
        color_jitter_prob=args.color_jitter_prob,
        color_hue_shift_max_deg=args.color_hue_shift_max_deg,
        color_sat_gain_min=args.color_sat_gain_min,
        color_sat_gain_max=args.color_sat_gain_max,
        color_val_gain_min=args.color_val_gain_min,
        color_val_gain_max=args.color_val_gain_max,
        gaussian_blur_kernel_min=args.gaussian_blur_kernel_min,
        gaussian_blur_kernel_max=args.gaussian_blur_kernel_max,
        motion_blur_kernel_min=args.motion_blur_kernel_min,
        motion_blur_kernel_max=args.motion_blur_kernel_max,
        motion_blur_angle_max_deg=args.motion_blur_angle_max_deg,
        noise_sigma_min=args.noise_sigma_min,
        noise_sigma_max=args.noise_sigma_max,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
    )

    results = generate_dataset(config)
    print(f"generated {len(results)} synthetic samples in {config.output_root}")
