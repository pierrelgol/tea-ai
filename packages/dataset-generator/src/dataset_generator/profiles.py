from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class DatasetProfile:
    version: int
    name: str
    dataset_dir_name: str
    train_images_rel: str
    val_images_rel: str


class ProfileError(ValueError):
    pass


def profiles_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "configs" / "datasets"
        if candidate.exists():
            return candidate
    raise ProfileError("could not locate configs/datasets from current package path")


def resolve_profile(dataset_name: str, profile_path: Path | None) -> tuple[DatasetProfile, Path]:
    path = profile_path if profile_path is not None else profiles_root() / f"{dataset_name}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"dataset profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid dataset profile JSON: {path}") from exc

    try:
        splits = payload["splits"]
        profile = DatasetProfile(
            version=int(payload.get("version", 1)),
            name=str(payload["name"]),
            dataset_dir_name=str(payload["dataset_dir_name"]),
            train_images_rel=str(splits["train_images_rel"]),
            val_images_rel=str(splits["val_images_rel"]),
        )
    except Exception as exc:
        raise ProfileError(f"invalid dataset profile schema: {path}") from exc

    return profile, path


def resolve_split_dirs(dataset_root: Path, profile: DatasetProfile) -> dict[str, Path]:
    base = dataset_root / profile.dataset_dir_name
    return {
        "train": base / profile.train_images_rel,
        "val": base / profile.val_images_rel,
    }
