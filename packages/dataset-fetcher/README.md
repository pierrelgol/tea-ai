# dataset-fetcher

> Declarative fetching and subsetting of remote and local datasets.

The `dataset-fetcher` package handles the ingestion of ground-truth datasets into the AI pipeline. It operates heavily on the concept of a `DatasetProfile`—a declarative JSON schema that dictates where a dataset comes from, how it should be subsetted, and what directory structure it must expose.

## Architecture

Fetching is abstracted into "source types" to support a variety of dataset origins (e.g., zip files from remote URLs, local directories, or dynamically subsetted local COCO datasets). 

```text
+-----------------------+      +-----------------------+      +-----------------------+
|                       |      |                       |      |                       |
|   configs/datasets/   |      |   fetch.py            |      |   validate.py         |
|   profile.json        +------>   (Downloader &       +------>   (Schema & Layout    |
|                       |      |   Subsetter)          |      |   Assertion)          |
+-----------------------+      +-----------+-----------+      +-----------+-----------+
                                           |                              |
                                           v                              v
                               +------------------------------------------------------+
                               |                                                      |
                               |                   dataset/<name>/                    |
                               |                   (Normalized Filesystem)            |
                               |                                                      |
                               +------------------------------------------------------+
```

## Features

*   **Declarative Profiles**: Datasets are defined entirely via JSON files (e.g., `coco128.json`), decoupling code from dataset-specific URLs or layout quirks.
*   **Multiple Ingestion Strategies**:
    *   `remote_zip` / `ultralytics_zip`: Downloads and extracts zip archives.
    *   `local_dir`: Fast, non-destructive symlinking of an existing local dataset.
    *   `coco_subset_local`: Deterministically samples a subset of an existing local COCO dataset (useful for rapid prototyping on smaller scales).
    *   `coco_ids_local`: Constructs a dataset split by explicitly reading a list of image IDs.
*   **Deterministic Sampling**: When subsetting (e.g., taking 100 images from a 10k image dataset), the package uses seeded cryptographic hashing on filenames to ensure the exact same images are selected across different runs and machines.
*   **Strict Validation**: Post-fetch, the dataset directory is validated against `required_paths_rel` to guarantee the expected structure (e.g., `images/train`, `labels/train`) exists before downstream tasks execute.

## Usage

```python
from pathlib import Path
from dataset_fetcher.profiles import resolve_profile
from dataset_fetcher.fetch import fetch_dataset

# 1. Resolve the profile from the configurations folder
profile, _ = resolve_profile("coco128", profile_path=None)

# 2. Fetch, subset (if configured), and validate the dataset
dataset_path = fetch_dataset(
    profile=profile,
    dataset_root=Path("./dataset"),
)

print(f"Dataset ready at: {dataset_path}")
```
