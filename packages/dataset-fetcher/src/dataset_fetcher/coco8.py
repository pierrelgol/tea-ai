from pathlib import Path
import tempfile
import urllib.request
import zipfile

COCO8_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
    "https://ultralytics.com/assets/coco8.zip",
]


def fetch_coco8(dataset_root: Path = Path("dataset")) -> Path:
    """Download and extract COCO8 into dataset_root/coco8.

    Returns the extracted dataset path.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / "coco8"

    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    with tempfile.TemporaryDirectory(prefix="coco8-") as tmp_dir:
        archive_path = Path(tmp_dir) / "coco8.zip"

        last_error: Exception | None = None
        for url in COCO8_URLS:
            try:
                urllib.request.urlretrieve(url, archive_path)
                break
            except Exception as exc:  # pragma: no cover - network-specific
                last_error = exc
        else:
            raise RuntimeError("Failed to download COCO8 from known Ultralytics URLs") from last_error

        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(dataset_root)

        extracted = dataset_root / "coco8"
        if not extracted.exists():
            raise RuntimeError(
                f"COCO8 archive extracted, but expected '{dataset_root / 'coco8'}' was not found"
            )

        return extracted
