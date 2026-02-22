from __future__ import annotations

import argparse
from pathlib import Path

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI tool for labeling target images in YOLO format")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--targets-subdir", default="targets")
    parser.add_argument("--images-dir", type=Path, default=Path("targets"))
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="File extension filter. Can be repeated. Defaults to jpg,jpeg,png,bmp",
    )
    args = parser.parse_args()

    targets_root = args.dataset_root / args.targets_subdir
    labels_dir = args.labels_dir if args.labels_dir is not None else targets_root / "labels"
    classes_file = args.classes_file if args.classes_file is not None else targets_root / "classes.txt"
    exts = args.ext if args.ext else ["jpg", "jpeg", "png", "bmp"]
    run_app(
        images_dir=args.images_dir,
        labels_dir=labels_dir,
        classes_file=classes_file,
        export_root=targets_root,
        exts=exts,
    )
