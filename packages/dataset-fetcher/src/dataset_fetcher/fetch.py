from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from .profiles import DatasetProfile
from .validate import validate_dataset


def _download_archive(urls: list[str], archive_path: Path) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, archive_path)
            return url
        except Exception as exc:  # pragma: no cover - network-specific
            last_error = exc
    raise RuntimeError("failed to download dataset from provided URLs") from last_error


def _fetch_zip_dataset(dataset_root: Path, dataset_dir_name: str, urls: list[str]) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / dataset_dir_name

    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    with tempfile.TemporaryDirectory(prefix=f"{dataset_dir_name}-") as tmp_dir:
        archive_path = Path(tmp_dir) / f"{dataset_dir_name}.zip"
        _download_archive(urls, archive_path)

        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(dataset_root)

    return target_dir


def _fetch_local_dataset(local_path: str, dataset_root: Path, dataset_dir_name: str) -> Path:
    candidate = Path(local_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    expected = (dataset_root / dataset_dir_name).resolve()
    if candidate == expected:
        return expected

    # Keep operation cheap and non-destructive by using a symlink into dataset_root.
    dataset_root.mkdir(parents=True, exist_ok=True)
    link_path = dataset_root / dataset_dir_name
    if not link_path.exists():
        link_path.symlink_to(candidate, target_is_directory=True)
    return link_path.resolve()


def fetch_dataset(
    profile: DatasetProfile,
    dataset_root: Path,
    source_url_override: str | None = None,
    dataset_dir_name_override: str | None = None,
) -> Path:
    dataset_dir_name = dataset_dir_name_override or profile.dataset_dir_name

    if profile.source_type == "local_dir":
        dataset_dir = _fetch_local_dataset(profile.local_path or "", dataset_root, dataset_dir_name)
    else:
        urls = [source_url_override] if source_url_override else profile.urls
        dataset_dir = _fetch_zip_dataset(dataset_root, dataset_dir_name, urls)

    validate_dataset(profile, dataset_dir)
    return dataset_dir
