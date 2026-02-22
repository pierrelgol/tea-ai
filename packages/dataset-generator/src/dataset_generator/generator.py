from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import GeneratorConfig
from .geometry import apply_homography_to_points, corners_px_to_yolo_obb, corners_to_xyxy, xyxy_to_yolo
from .homography import HomographyParams, sample_valid_homography
from .io import (
    CanonicalTarget,
    load_backgrounds_by_split,
    load_canonical_targets,
    load_target_classes,
    write_augmented_classes,
    write_metadata,
    write_yolo_obb_label,
)
from .synthesis import warp_and_composite


@dataclass(slots=True)
class SampleResult:
    image_out_path: Path
    label_out_path: Path
    metadata_out_path: Path


def _output_stem(split: str, background_stem: str, target_stem: str, sample_idx: int) -> str:
    return f"{split}_{background_stem}_t{target_stem}_s{sample_idx:03d}"


def _ensure_output_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "meta" / split).mkdir(parents=True, exist_ok=True)


def generate_dataset(config: GeneratorConfig) -> list[SampleResult]:
    config.validate()

    targets = load_canonical_targets(
        target_images_dir=config.target_images_dir,
        target_labels_dir=config.target_labels_dir,
        target_classes_file=config.target_classes_file,
    )
    target_classes = load_target_classes(config.target_classes_file)
    backgrounds_by_split = load_backgrounds_by_split(config.background_splits)

    _ensure_output_layout(config.output_root)
    write_augmented_classes(config.output_root, target_classes, config.class_offset_base)

    homography_params = HomographyParams(
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        translate_frac=config.translate_frac,
        perspective_jitter=config.perspective_jitter,
        min_quad_area_frac=config.min_quad_area_frac,
        max_attempts=config.max_attempts,
    )

    rng = np.random.default_rng(config.seed)

    results: list[SampleResult] = []

    for split in ("train", "val"):
        backgrounds = backgrounds_by_split.get(split, [])
        for bg_path in backgrounds:
            background = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            if background is None:
                continue

            bg_h, bg_w = background.shape[:2]

            for sample_idx in range(config.samples_per_background):
                target: CanonicalTarget = targets[int(rng.integers(0, len(targets)))]
                target_image = cv2.imread(str(target.image_path), cv2.IMREAD_COLOR)
                if target_image is None:
                    continue

                try:
                    hs = sample_valid_homography(
                        canonical_corners_px=target.canonical_corners_px,
                        background_w=bg_w,
                        background_h=bg_h,
                        rng=rng,
                        params=homography_params,
                    )
                except RuntimeError:
                    # Skip unsatisfiable target/background placements instead of aborting the full generation run.
                    continue

                projected_corners = apply_homography_to_points(hs.H, target.canonical_corners_px)
                projected_corners_norm = corners_px_to_yolo_obb(projected_corners, bg_w, bg_h)
                x1, y1, x2, y2 = corners_to_xyxy(projected_corners)
                x_center, y_center, width, height = xyxy_to_yolo(x1, y1, x2, y2, bg_w, bg_h)

                composited = warp_and_composite(
                    background=background,
                    target=target_image,
                    canonical_corners_px=target.canonical_corners_px,
                    H=hs.H,
                )

                stem = _output_stem(split, bg_path.stem, target.image_path.stem, sample_idx)
                image_out_path = config.output_root / "images" / split / f"{stem}{bg_path.suffix}"
                label_out_path = config.output_root / "labels" / split / f"{stem}.txt"
                meta_out_path = config.output_root / "meta" / split / f"{stem}.json"

                image_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_out_path), composited)

                class_id_exported = config.class_offset_base + target.class_id_local
                write_yolo_obb_label(label_out_path, class_id_exported, projected_corners_norm)

                metadata = {
                    "seed": config.seed,
                    "background_dataset_name": config.background_dataset_name,
                    "background_image": str(bg_path),
                    "target_image": str(target.image_path),
                    "target_class_name": target.class_name,
                    "target_class_id_local": target.class_id_local,
                    "target_class_id_exported": class_id_exported,
                    "H": hs.H.tolist(),
                    "canonical_corners_px": target.canonical_corners_px.tolist(),
                    "projected_corners_px": projected_corners.tolist(),
                    "projected_corners_yolo_obb": projected_corners_norm.tolist(),
                    "bbox_xyxy_px": [x1, y1, x2, y2],
                    "bbox_yolo": [x_center, y_center, width, height],
                }
                write_metadata(meta_out_path, metadata)

                results.append(
                    SampleResult(
                        image_out_path=image_out_path,
                        label_out_path=label_out_path,
                        metadata_out_path=meta_out_path,
                    )
                )

    return results
