from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class DatasetProfile:
    version: int
    name: str
    dataset_dir_name: str
    source_type: str
    urls: list[str]
    local_path: str | None
    train_images_rel: str
    val_images_rel: str
    required_paths_rel: list[str]


class ProfileError(ValueError):
    pass


def profiles_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "configs" / "datasets"
        if candidate.exists():
            return candidate
    raise ProfileError("could not locate configs/datasets from current package path")


def load_profile(profile_path: Path) -> DatasetProfile:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"dataset profile not found: {profile_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid JSON in dataset profile: {profile_path}") from exc

    try:
        source = payload["source"]
        splits = payload["splits"]
        validation = payload.get("validation", {})
        profile = DatasetProfile(
            version=int(payload.get("version", 1)),
            name=str(payload["name"]),
            dataset_dir_name=str(payload["dataset_dir_name"]),
            source_type=str(source["type"]),
            urls=[str(u) for u in source.get("urls", [])],
            local_path=(str(source["local_path"]) if "local_path" in source else None),
            train_images_rel=str(splits["train_images_rel"]),
            val_images_rel=str(splits["val_images_rel"]),
            required_paths_rel=[str(p) for p in validation.get("required_paths_rel", [])],
        )
    except Exception as exc:  # pragma: no cover - schema errors
        raise ProfileError(f"invalid dataset profile schema: {profile_path}") from exc

    if profile.source_type not in {"ultralytics_zip", "remote_zip", "local_dir"}:
        raise ProfileError(f"unsupported source.type in {profile_path}: {profile.source_type}")

    if profile.source_type in {"ultralytics_zip", "remote_zip"} and not profile.urls:
        raise ProfileError(f"profile {profile.name} requires non-empty source.urls")

    if profile.source_type == "local_dir" and not profile.local_path:
        raise ProfileError(f"profile {profile.name} requires source.local_path")

    return profile


def resolve_profile(dataset_name: str, profile_path: Path | None) -> tuple[DatasetProfile, Path]:
    path = profile_path if profile_path is not None else profiles_root() / f"{dataset_name}.json"
    return load_profile(path), path
