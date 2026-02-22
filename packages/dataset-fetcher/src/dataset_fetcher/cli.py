import argparse
from pathlib import Path

from .fetch import fetch_dataset
from .profiles import resolve_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a selected dataset into top-level dataset directory")
    parser.add_argument(
        "--dataset",
        default="coco8",
        help="Dataset profile name to use (default: coco8)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Explicit profile path (JSON). Overrides --dataset lookup.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Directory where datasets are stored (default: ./dataset)",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional remote ZIP URL override for built-in/custom profiles.",
    )
    parser.add_argument(
        "--dataset-dir-name",
        default=None,
        help="Optional extracted dataset directory name override.",
    )
    args = parser.parse_args()

    profile, profile_path = resolve_profile(dataset_name=args.dataset, profile_path=args.profile)
    dataset_path = fetch_dataset(
        profile=profile,
        dataset_root=args.dataset_root,
        source_url_override=args.source_url,
        dataset_dir_name_override=args.dataset_dir_name,
    )
    print(f"{profile.name} ready at: {dataset_path}")
    print(f"profile: {profile_path}")
    print(f"train_images: {dataset_path / profile.train_images_rel}")
    print(f"val_images: {dataset_path / profile.val_images_rel}")
