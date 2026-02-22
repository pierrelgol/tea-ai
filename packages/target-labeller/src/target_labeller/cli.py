from __future__ import annotations

import argparse
from pathlib import Path

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI tool for labeling target images in YOLO format")
    parser.add_argument("--images-dir", type=Path, default=Path("targets"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/targets/labels"))
    parser.add_argument("--classes-file", type=Path, default=Path("dataset/targets/classes.txt"))
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="File extension filter. Can be repeated. Defaults to jpg,jpeg,png,bmp",
    )
    args = parser.parse_args()

    exts = args.ext if args.ext else ["jpg", "jpeg", "png", "bmp"]
    run_app(images_dir=args.images_dir, labels_dir=args.labels_dir, classes_file=args.classes_file, exts=exts)
