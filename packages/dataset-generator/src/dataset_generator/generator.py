from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from .config import GeneratorConfig
from .geometry import (
    apply_homography_to_points,
    corners_px_to_yolo_obb,
    is_convex_quad,
    polygon_area,
    quad_inside_bounds,
)
from .homography import HomographyParams, sample_valid_homography
from .io import (
    CanonicalTarget,
    audit_background_split_overlap,
    enforce_disjoint_background_splits,
    load_backgrounds_by_split,
    load_canonical_targets,
    load_target_classes,
    write_augmented_classes,
    write_metadata,
    write_yolo_obb_labels,
)
from .photometric import apply_photometric_stack
from .synthesis import blend_layer, visible_ratio, warp_target_and_mask

MIN_RAW_RECT_IOU = 0.72


@dataclass(slots=True)
class SampleResult:
    image_out_path: Path
    label_out_path: Path
    metadata_out_path: Path


@dataclass(slots=True)
class PlacedTarget:
    target: CanonicalTarget
    projected_corners_px: np.ndarray
    projected_corners_px_raw: np.ndarray
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


def _scaled_homography_params_for_target_count(
    *,
    base: HomographyParams,
    n_targets: int,
    config: GeneratorConfig,
) -> HomographyParams:
    """Shrink target scale bounds when placing more targets on the same image."""
    min_targets = config.targets_per_image_min
    max_targets = config.targets_per_image_max
    if max_targets <= min_targets:
        return base

    crowd_ratio = (n_targets - min_targets) / float(max_targets - min_targets)
    crowd_ratio = float(np.clip(crowd_ratio, 0.0, 1.0))

    # At max crowd, targets are scaled down to configured floor.
    min_scale_factor = float(config.crowd_scale_floor)
    scale_factor = 1.0 - (1.0 - min_scale_factor) * crowd_ratio

    scaled_min = max(0.05, base.scale_min * scale_factor)
    scaled_max = max(scaled_min, base.scale_max * scale_factor)
    return replace(base, scale_min=scaled_min, scale_max=scaled_max)


def _is_valid_projected_obb(projected_corners: np.ndarray, image_w: int, image_h: int, config: GeneratorConfig) -> bool:
    if projected_corners.shape != (4, 2):
        return False
    if not is_convex_quad(projected_corners):
        return False
    if not quad_inside_bounds(projected_corners, image_w, image_h):
        return False
    if polygon_area(projected_corners) < config.min_target_area_px:
        return False
    edge_lengths: list[float] = []
    for i in range(4):
        p0 = projected_corners[i]
        p1 = projected_corners[(i + 1) % 4]
        edge_len = float(np.linalg.norm(p1 - p0))
        edge_lengths.append(edge_len)
        if edge_len < config.min_edge_length_px:
            return False
    longest = max(edge_lengths)
    shortest = max(1e-9, min(edge_lengths))
    if (longest / shortest) > config.max_edge_aspect_ratio:
        return False
    for i in range(4):
        v_prev = projected_corners[(i - 1) % 4] - projected_corners[i]
        v_next = projected_corners[(i + 1) % 4] - projected_corners[i]
        n_prev = float(np.linalg.norm(v_prev))
        n_next = float(np.linalg.norm(v_next))
        if n_prev <= 1e-9 or n_next <= 1e-9:
            return False
        c = float(np.clip(np.dot(v_prev, v_next) / (n_prev * n_next), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(c)))
        if angle < config.min_corner_angle_deg or angle > config.max_corner_angle_deg:
            return False
    return True


def _principal_angle_deg(quad: np.ndarray) -> float:
    edges: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        v = quad[(i + 1) % 4] - quad[i]
        edges.append((float(np.linalg.norm(v)), v))
    edges.sort(key=lambda x: x[0], reverse=True)
    vec = edges[0][1]
    return float(np.degrees(np.arctan2(float(vec[1]), float(vec[0]))) % 180.0)


def _angle_bin(angle_deg: float, n_bins: int = 12) -> int:
    step = 180.0 / float(n_bins)
    idx = int(np.floor(angle_deg / step))
    return max(0, min(n_bins - 1, idx))


def _canonicalize_quad_cw_start_tl(quad: np.ndarray) -> np.ndarray:
    center = np.mean(quad, axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    order = np.argsort(angles)
    ccw = quad[order]
    cw = ccw[::-1].copy()
    start = int(np.argmin(cw[:, 1] * 100000.0 + cw[:, 0]))
    return np.roll(cw, -start, axis=0).astype(np.float32)


def _polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _fit_rectangular_obb(projected_quad: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(projected_quad.astype(np.float32))
    corners = cv2.boxPoints(rect).astype(np.float32)
    return _canonicalize_quad_cw_start_tl(corners)


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
    projected_corners_raw = apply_homography_to_points(hs.H, target.canonical_corners_px)
    if not _is_valid_projected_obb(projected_corners_raw, bg_w, bg_h, config):
        return None
    projected_corners_rect = _fit_rectangular_obb(projected_corners_raw)
    if not _is_valid_projected_obb(projected_corners_rect, bg_w, bg_h, config):
        return None
    raw_rect_iou = _polygon_iou(projected_corners_raw, projected_corners_rect)
    if raw_rect_iou < MIN_RAW_RECT_IOU:
        return None
    projected_corners_norm = corners_px_to_yolo_obb(projected_corners_rect, bg_w, bg_h)

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
        "projected_corners_px_raw": projected_corners_raw.tolist(),
        "projected_corners_px_rect_obb": projected_corners_rect.tolist(),
        "projected_corners_px": projected_corners_rect.tolist(),
        "projected_corners_yolo_obb": projected_corners_norm.tolist(),
        "rect_fit_iou_raw_vs_rect": raw_rect_iou,
        "visible_ratio": ratio_visible,
        "occlusion_ratio": occlusion_ratio,
    }
    return PlacedTarget(
        target=target,
        projected_corners_px=projected_corners_rect,
        projected_corners_px_raw=projected_corners_raw,
        projected_corners_norm=projected_corners_norm,
        warped_mask=warped_mask,
        class_id_exported=class_id_exported,
        placement=placement,
    )


def generate_dataset(config: GeneratorConfig) -> list[SampleResult]:
    config.validate()
    if config.output_root.exists():
        shutil.rmtree(config.output_root)

    targets = load_canonical_targets(
        target_images_dir=config.target_images_dir,
        target_labels_dir=config.target_labels_dir,
        target_classes_file=config.target_classes_file,
    )
    target_classes = load_target_classes(config.target_classes_file)
    backgrounds_by_split = load_backgrounds_by_split(config.background_splits)
    original_audit = audit_background_split_overlap(backgrounds_by_split)
    if original_audit.get("overlap_count", 0) > 0:
        backgrounds_by_split, enforced_audit = enforce_disjoint_background_splits(backgrounds_by_split)
        split_audit = {
            "enforced": True,
            "original": original_audit,
            "post_enforcement": enforced_audit,
            "overlap_count": int(enforced_audit.get("overlap_count", 0)),
        }
    else:
        split_audit = {
            "enforced": False,
            "original": original_audit,
            "post_enforcement": {
                "original_train_count": len(backgrounds_by_split.get("train", [])),
                "original_val_count": len(backgrounds_by_split.get("val", [])),
                "original_overlap_count": 0,
                "reassigned_overlap_to_train_count": 0,
                "reassigned_overlap_to_val_count": 0,
                "final_train_count": len(backgrounds_by_split.get("train", [])),
                "final_val_count": len(backgrounds_by_split.get("val", [])),
                "policy": "none_required",
                "overlap_count": 0,
            },
            "overlap_count": 0,
        }
    write_metadata(config.output_root / "split_audit.json", split_audit)
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
        edge_bias_prob=config.edge_bias_prob,
        edge_band_frac=config.edge_band_frac,
    )
    rng = np.random.default_rng(config.seed)
    angle_bin_counts = np.zeros(12, dtype=np.int32)
    results: list[SampleResult] = []

    for split in ("train", "val"):
        backgrounds = backgrounds_by_split.get(split, [])
        for bg_path in backgrounds:
            background = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            if background is None:
                continue
            bg_h, bg_w = background.shape[:2]

            for sample_idx in range(config.samples_per_background):
                planned_empty = bool(rng.random() < config.empty_sample_prob)
                if planned_empty:
                    n_targets = 0
                else:
                    n_targets = int(rng.integers(config.targets_per_image_min, config.targets_per_image_max + 1))
                sample_homography_params = _scaled_homography_params_for_target_count(
                    base=homography_params,
                    n_targets=n_targets,
                    config=config,
                )
                composited = background.copy()
                occupancy_mask = np.zeros((bg_h, bg_w), dtype=bool)
                placed: list[PlacedTarget] = []

                for _ in range(n_targets):
                    placed_target: PlacedTarget | None = None
                    placed_score = -1.0
                    for _attempt in range(config.max_attempts):
                        target = targets[int(rng.integers(0, len(targets)))]
                        key = str(target.image_path)
                        if key not in target_images_cache:
                            image = cv2.imread(str(target.image_path), cv2.IMREAD_COLOR)
                            if image is None:
                                break
                            target_images_cache[key] = image
                        target_image = target_images_cache[key]
                        candidate = _try_place_target(
                            background=composited,
                            target=target,
                            target_image=target_image,
                            occupancy_mask=occupancy_mask,
                            homography_params=sample_homography_params,
                            rng=rng,
                            config=config,
                        )
                        if candidate is not None:
                            angle_deg = _principal_angle_deg(candidate.projected_corners_px)
                            angle_idx = _angle_bin(angle_deg)
                            rarity_bonus = 1.0 / float(1 + angle_bin_counts[angle_idx])
                            area_px = polygon_area(candidate.projected_corners_px)
                            area_ratio = float(np.clip(area_px / max(1.0, float(bg_w * bg_h)), 0.0, 1.0))
                            size_penalty = 1.0 - area_ratio
                            score = 0.75 * rarity_bonus + 0.25 * size_penalty
                            if score > placed_score:
                                placed_score = score
                                candidate.placement["principal_angle_deg"] = angle_deg
                                placed_target = candidate
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
                    angle_deg = float(
                        placed_target.placement.get(
                            "principal_angle_deg",
                            _principal_angle_deg(placed_target.projected_corners_px),
                        )
                    )
                    angle_bin_counts[_angle_bin(angle_deg)] += 1

                if not placed and n_targets > 0:
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
                    "background_dataset_name": config.background_dataset_name,
                    "background_image": str(bg_path),
                    "num_targets": len(placed),
                    "planned_empty": planned_empty,
                    "photometric_applied": photometric_applied,
                    "targets": [p.placement for p in placed],
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
