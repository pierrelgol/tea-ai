import argparse
from pathlib import Path

from .coco8 import fetch_coco8


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch COCO8 into top-level dataset directory")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Directory where datasets are stored (default: ./dataset)",
    )
    args = parser.parse_args()

    dataset_path = fetch_coco8(args.dataset_root)
    print(f"COCO8 ready at: {dataset_path}")
