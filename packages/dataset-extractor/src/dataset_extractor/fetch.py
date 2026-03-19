from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


COCO_DATASET_NAME = "coco17"
COCO_SUBSET_MIN_SIZE = 128
COCO_SUBSET_SEED = 42
COCO_VAL_RATIO = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _replace_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _deterministic_image_subset(
    images: list[Path],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[Path]:
    if len(images) <= limit:
        return images

    ranked = sorted(
        images,
        key=lambda path: hashlib.sha1(f"{seed}:{split}:{path.stem}".encode("utf-8")).hexdigest(),
    )
    return ranked[:limit]


def _max_power_of_two_at_or_below(value: int) -> int:
    if value < 1:
        return 0
    return 1 << (value.bit_length() - 1)


def _symlink_images(images: list[Path], *, images_dst: Path) -> None:
    _replace_dir(images_dst)
    images_dst.mkdir(parents=True, exist_ok=True)
    for image_src in images:
        link_path = images_dst / image_src.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(image_src.resolve())


def _select_train_and_val_images(
    extracted_images: list[Path],
    *,
    seed: int,
    val_ratio: int,
) -> tuple[list[Path], list[Path]]:
    if not extracted_images:
        raise RuntimeError("archive did not contain any supported image files")

    val_count = max(1, len(extracted_images) // max(1, val_ratio))
    val_selected = _deterministic_image_subset(
        extracted_images,
        split="val",
        limit=val_count,
        seed=seed,
    )
    val_names = {path.name for path in val_selected}
    train_selected = [path for path in extracted_images if path.name not in val_names]
    if not train_selected:
        raise RuntimeError("archive is too small to create a disjoint train/val split")
    return train_selected, val_selected


def _extract_train_archive(train_zip_path: Path, dataset_root: Path, dataset_dir_name: str) -> Path:
    if not train_zip_path.exists():
        raise FileNotFoundError(f"train zip not found: {train_zip_path}")
    if not zipfile.is_zipfile(train_zip_path):
        raise RuntimeError(f"train zip is not a valid zip archive: {train_zip_path}")

    target_dir = dataset_root / dataset_dir_name
    _replace_dir(target_dir)

    with tempfile.TemporaryDirectory(prefix=f"{dataset_dir_name}-extract-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(train_zip_path) as archive:
            archive.extractall(tmp_root)

        extracted_root = tmp_root / "train2017"
        if not extracted_root.exists() or not extracted_root.is_dir():
            raise RuntimeError("train zip must contain a top-level train2017/ directory")

        extracted_images = [
            path for path in sorted(extracted_root.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        train_selected, val_selected = _select_train_and_val_images(
            extracted_images,
            seed=COCO_SUBSET_SEED,
            val_ratio=COCO_VAL_RATIO,
        )

        train_dir = target_dir / "images" / "train2017"
        val_dir = target_dir / "images" / "val2017"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        for image_path in train_selected:
            shutil.move(str(image_path), str(train_dir / image_path.name))
        for image_path in val_selected:
            shutil.move(str(image_path), str(val_dir / image_path.name))

    return target_dir


def write_dataset_config(
    *,
    configs_root: Path,
    dataset_name: str,
    dataset_dir_name: str,
    dataset_local_path: Path,
) -> Path:
    configs_root.mkdir(parents=True, exist_ok=True)
    config_path = configs_root / f"{dataset_name}.json"
    payload = {
        "version": 1,
        "name": dataset_name,
        "dataset_dir_name": dataset_dir_name,
        "source": {
            "type": "local_dir",
            "local_path": str(dataset_local_path),
        },
        "splits": {
            "train_images_rel": "images/train2017",
            "val_images_rel": "images/val2017",
        },
        "validation": {
            "required_paths_rel": [
                "images/train2017",
                "images/val2017",
            ]
        },
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def materialize_coco_power_of_two_subsets(
    *,
    coco17_dir: Path,
    dataset_root: Path,
    configs_root: Path,
    min_size: int = COCO_SUBSET_MIN_SIZE,
    seed: int = COCO_SUBSET_SEED,
) -> list[dict[str, int | str]]:
    train_images_dir = coco17_dir / "images" / "train2017"
    val_images_dir = coco17_dir / "images" / "val2017"
    train_images = _list_images(train_images_dir)
    val_images = _list_images(val_images_dir)
    if not train_images:
        raise RuntimeError(f"no train images found under {train_images_dir}")
    if not val_images:
        raise RuntimeError(f"no val images found under {val_images_dir}")

    max_train_subset = _max_power_of_two_at_or_below(len(train_images))
    subset_size = max(1, int(min_size))
    created: list[dict[str, int | str]] = []

    while subset_size <= max_train_subset:
        subset_name = f"coco{subset_size}"
        subset_root = dataset_root / subset_name
        train_selected = _deterministic_image_subset(
            train_images,
            split="train",
            limit=subset_size,
            seed=seed,
        )
        val_selected = _deterministic_image_subset(
            val_images,
            split="val",
            limit=min(len(val_images), max(1, subset_size // COCO_VAL_RATIO)),
            seed=seed,
        )
        _symlink_images(train_selected, images_dst=subset_root / "images" / "train2017")
        _symlink_images(val_selected, images_dst=subset_root / "images" / "val2017")
        write_dataset_config(
            configs_root=configs_root,
            dataset_name=subset_name,
            dataset_dir_name=subset_name,
            dataset_local_path=subset_root,
        )
        created.append(
            {
                "name": subset_name,
                "train_images": len(train_selected),
                "val_images": len(val_selected),
            }
        )
        subset_size *= 2

    return created


def extract_coco17_family(
    *,
    train_zip_path: Path,
    dataset_root: Path,
    configs_root: Path,
) -> tuple[Path, list[dict[str, int | str]]]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    coco17_dir = _extract_train_archive(train_zip_path, dataset_root, COCO_DATASET_NAME)
    write_dataset_config(
        configs_root=configs_root,
        dataset_name=COCO_DATASET_NAME,
        dataset_dir_name=COCO_DATASET_NAME,
        dataset_local_path=coco17_dir,
    )
    subsets = materialize_coco_power_of_two_subsets(
        coco17_dir=coco17_dir,
        dataset_root=dataset_root,
        configs_root=configs_root,
    )
    return coco17_dir, subsets
