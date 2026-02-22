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
    args = parser.parse_args()

    profile, profile_path = resolve_profile(dataset_name=args.dataset, profile_path=None)
    dataset_path = fetch_dataset(
        profile=profile,
        dataset_root=Path("dataset"),
        source_url_override=None,
        dataset_dir_name_override=None,
    )
    print(f"{profile.name} ready at: {dataset_path}")
    print(f"profile: {profile_path}")
    print(f"train_images: {dataset_path / profile.train_images_rel}")
    print(f"val_images: {dataset_path / profile.val_images_rel}")
