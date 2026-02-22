from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import GeneratorConfig
from .geometry import apply_homography_to_points, corners_px_to_yolo_obb
from .homography import HomographyParams, sample_valid_homography
from .io import (
    CanonicalTarget,
    load_backgrounds_by_split,
    load_canonical_targets,
    load_target_classes,
    write_augmented_classes,
    write_metadata,
    write_yolo_obb_labels,
)
from .photometric import apply_photometric_stack
from .synthesis import blend_layer, visible_ratio, warp_target_and_mask


@dataclass(slots=True)
class SampleResult:
    image_out_path: Path
    label_out_path: Path
    metadata_out_path: Path


@dataclass(slots=True)
class PlacedTarget:
    target: CanonicalTarget
    projected_corners_px: np.ndarray
    projected_corners_norm: np.ndarray
    warped_mask: np.ndarray
    class_id_exported: int
    placement: dict[str, Any]


def _output_stem(split: str, background_stem: str, sample_idx: int) -> str:
    return f"{split}_{background_stem}_s{sample_idx:03d}"


def _ensure_output_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "meta" / split).mkdir(parents=True, exist_ok=True)


def _try_place_target(
    *,
    background: np.ndarray,
    target: CanonicalTarget,
    target_image: np.ndarray,
    occupancy_mask: np.ndarray,
    homography_params: HomographyParams,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> PlacedTarget | None:
    bg_h, bg_w = background.shape[:2]
    hs = sample_valid_homography(
        canonical_corners_px=target.canonical_corners_px,
        background_w=bg_w,
        background_h=bg_h,
        rng=rng,
        params=homography_params,
    )
    projected_corners = apply_homography_to_points(hs.H, target.canonical_corners_px)
    projected_corners_norm = corners_px_to_yolo_obb(projected_corners, bg_w, bg_h)

    warped_target, warped_mask = warp_target_and_mask(
        target=target_image,
        canonical_corners_px=target.canonical_corners_px,
        H=hs.H,
        out_w=bg_w,
        out_h=bg_h,
    )
    ratio_visible = visible_ratio(warped_mask=warped_mask, occupancy_mask=occupancy_mask)
    occlusion_ratio = 1.0 - ratio_visible
    if (not config.allow_partial_visibility and ratio_visible < 0.999) or occlusion_ratio > config.max_occlusion_ratio:
        return None

    class_id_exported = config.class_offset_base + target.class_id_local
    placement = {
        "target_image": str(target.image_path),
        "target_class_name": target.class_name,
        "target_class_id_local": target.class_id_local,
        "target_class_id_exported": class_id_exported,
        "H": hs.H.tolist(),
        "canonical_corners_px": target.canonical_corners_px.tolist(),
        "projected_corners_px": projected_corners.tolist(),
        "projected_corners_yolo_obb": projected_corners_norm.tolist(),
        "visible_ratio": ratio_visible,
        "occlusion_ratio": occlusion_ratio,
    }
    return PlacedTarget(
        target=target,
        projected_corners_px=projected_corners,
        projected_corners_norm=projected_corners_norm,
        warped_mask=warped_mask,
        class_id_exported=class_id_exported,
        placement=placement,
    )


def generate_dataset(config: GeneratorConfig) -> list[SampleResult]:
    config.validate()

    targets = load_canonical_targets(
        target_images_dir=config.target_images_dir,
        target_labels_dir=config.target_labels_dir,
        target_classes_file=config.target_classes_file,
    )
    target_classes = load_target_classes(config.target_classes_file)
    backgrounds_by_split = load_backgrounds_by_split(config.background_splits)
    target_images_cache: dict[str, np.ndarray] = {}

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
                n_targets = int(rng.integers(config.targets_per_image_min, config.targets_per_image_max + 1))
                composited = background.copy()
                occupancy_mask = np.zeros((bg_h, bg_w), dtype=bool)
                placed: list[PlacedTarget] = []

                for _ in range(n_targets):
                    placed_target: PlacedTarget | None = None
                    for _attempt in range(config.max_attempts):
                        target = targets[int(rng.integers(0, len(targets)))]
                        key = str(target.image_path)
                        if key not in target_images_cache:
                            image = cv2.imread(str(target.image_path), cv2.IMREAD_COLOR)
                            if image is None:
                                break
                            target_images_cache[key] = image
                        target_image = target_images_cache[key]
                        placed_target = _try_place_target(
                            background=composited,
                            target=target,
                            target_image=target_image,
                            occupancy_mask=occupancy_mask,
                            homography_params=homography_params,
                            rng=rng,
                            config=config,
                        )
                        if placed_target is not None:
                            break
                    if placed_target is None:
                        continue

                    warped_target, warped_mask = warp_target_and_mask(
                        target=target_images_cache[str(placed_target.target.image_path)],
                        canonical_corners_px=placed_target.target.canonical_corners_px,
                        H=np.array(placed_target.placement["H"], dtype=np.float64),
                        out_w=bg_w,
                        out_h=bg_h,
                    )
                    composited = blend_layer(
                        background=composited,
                        warped_target=warped_target,
                        warped_mask=warped_mask,
                        feather_px=5,
                    )
                    occupancy_mask = occupancy_mask | (warped_mask > 0)
                    placed.append(placed_target)

                if not placed:
                    continue

                composited, photometric_applied = apply_photometric_stack(composited, rng=rng, config=config)
                labels_out = [(p.class_id_exported, p.projected_corners_norm) for p in placed]

                stem = _output_stem(split, bg_path.stem, sample_idx)
                image_out_path = config.output_root / "images" / split / f"{stem}{bg_path.suffix}"
                label_out_path = config.output_root / "labels" / split / f"{stem}.txt"
                meta_out_path = config.output_root / "meta" / split / f"{stem}.json"

                image_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_out_path), composited)
                write_yolo_obb_labels(label_out_path, labels_out)

                metadata = {
                    "seed": config.seed,
                    "generator_version": config.generator_version,
                    "complexity_profile": config.complexity_profile,
                    "background_dataset_name": config.background_dataset_name,
                    "background_image": str(bg_path),
                    "num_targets": len(placed),
                    "photometric_applied": photometric_applied,
                    "targets": [p.placement for p in placed],
                }
                # Backward-compatible top-level fields from first target.
                first = placed[0].placement
                metadata["target_image"] = first["target_image"]
                metadata["target_class_name"] = first["target_class_name"]
                metadata["target_class_id_local"] = first["target_class_id_local"]
                metadata["target_class_id_exported"] = first["target_class_id_exported"]
                metadata["H"] = first["H"]
                metadata["canonical_corners_px"] = first["canonical_corners_px"]
                metadata["projected_corners_px"] = first["projected_corners_px"]
                metadata["projected_corners_yolo_obb"] = first["projected_corners_yolo_obb"]
                write_metadata(meta_out_path, metadata)

                results.append(
                    SampleResult(
                        image_out_path=image_out_path,
                        label_out_path=label_out_path,
                        metadata_out_path=meta_out_path,
                    )
                )

    return results

