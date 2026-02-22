from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI tool for labeling target images in YOLO format")
    parser.parse_args()

    targets_root = Path("dataset") / "targets"
    for rel in ("images", "labels"):
        p = targets_root / rel
        if p.exists():
            shutil.rmtree(p)
    run_app(
        images_dir=Path("targets"),
        labels_dir=targets_root / "labels",
        classes_file=targets_root / "classes.txt",
        export_root=targets_root,
        exts=["jpg", "jpeg", "png", "bmp"],
    )
