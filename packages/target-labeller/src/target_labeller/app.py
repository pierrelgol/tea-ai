from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .io import discover_images, ensure_class_id, load_classes, load_yolo_label, save_yolo_label
from .widgets import ImageCanvas


class LabelerWindow(QMainWindow):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        classes_file: Path,
        export_root: Path,
        exts: list[str],
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file
        self.export_root = export_root
        self.images = discover_images(images_dir, exts)
        self.index = 0

        self.setWindowTitle("Target Labeller")
        self.resize(1200, 800)

        self.canvas = ImageCanvas()
        self.image_info = QLabel()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name")
        self.class_picker = QComboBox()
        self.class_picker.currentTextChanged.connect(self._on_class_selected)

        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.finish_btn = QPushButton("Finish")

        self.save_btn.clicked.connect(self.save_current)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.finish_btn.clicked.connect(self.finish_and_close)

        self._setup_layout()
        self._setup_shortcuts()
        self._reload_class_picker()

        if not self.images:
            QMessageBox.critical(self, "No Images", f"No images found in: {images_dir}")
            raise RuntimeError(f"No images found in {images_dir}")

        self.load_current_image()

    def _setup_layout(self) -> None:
        side = QVBoxLayout()
        side.addWidget(self.image_info)
        side.addWidget(QLabel("Class"))
        side.addWidget(self.class_input)
        side.addWidget(QLabel("Existing classes"))
        side.addWidget(self.class_picker)
        side.addWidget(self.save_btn)
        side.addWidget(self.prev_btn)
        side.addWidget(self.next_btn)
        side.addStretch()
        side.addWidget(self.finish_btn)

        root = QHBoxLayout()
        root.addWidget(self.canvas, 1)

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(280)
        root.addWidget(side_widget)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.prev_image)
        QShortcut(QKeySequence(Qt.Key_A), self, activated=self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_image)
        QShortcut(QKeySequence(Qt.Key_D), self, activated=self.next_image)
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_current)

        clear_action = QAction(self)
        clear_action.setShortcut(QKeySequence(Qt.Key_Delete))
        clear_action.triggered.connect(self.canvas.clear_box)
        self.addAction(clear_action)

    def _reload_class_picker(self) -> None:
        classes = load_classes(self.classes_file)
        self.class_picker.blockSignals(True)
        self.class_picker.clear()
        self.class_picker.addItems(classes)
        self.class_picker.blockSignals(False)

    def _on_class_selected(self, text: str) -> None:
        if text:
            self.class_input.setText(text)

    def _current_image(self) -> Path:
        return self.images[self.index]

    def _label_file_for(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def load_current_image(self) -> None:
        image_path = self._current_image()
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Error", f"Failed to load image: {image_path}")
            return

        self.canvas.set_image(pixmap)
        self.image_info.setText(f"{self.index + 1}/{len(self.images)} - {image_path.name}")

        label_data = load_yolo_label(self._label_file_for(image_path))
        if label_data is None:
            self.canvas.clear_box()
            return

        class_id, yolo_box = label_data
        pixel_box = self.canvas.yolo_to_pixel_box(yolo_box)
        self.canvas.set_box(pixel_box)

        classes = load_classes(self.classes_file)
        if 0 <= class_id < len(classes):
            class_name = classes[class_id]
            self.class_input.setText(class_name)
            self.class_picker.setCurrentText(class_name)

    def save_current(self) -> None:
        class_name = self.class_input.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Missing Class", "Please set a class name before saving.")
            return

        yolo_box = self.canvas.pixel_box_to_yolo()
        if yolo_box is None or yolo_box.width <= 0 or yolo_box.height <= 0:
            QMessageBox.warning(self, "Missing Box", "Please draw a bounding box before saving.")
            return

        class_id = ensure_class_id(self.classes_file, class_name)
        save_yolo_label(self._label_file_for(self._current_image()), class_id, yolo_box)
        self._reload_class_picker()

    def prev_image(self) -> None:
        if self.index <= 0:
            return
        self.index -= 1
        self.load_current_image()

    def next_image(self) -> None:
        if self.index >= len(self.images) - 1:
            return
        self.index += 1
        self.load_current_image()

    def finish_and_close(self) -> None:
        try:
            exported_count = self._export_targets_structure()
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to export targets structure:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Finished",
            (
                f"Exported {exported_count} labeled image(s) to:\n"
                f"{self.export_root / 'images'}\n"
                f"{self.export_root / 'labels'}"
            ),
        )
        self.close()

    def _export_targets_structure(self) -> int:
        export_images_dir = self.export_root / "images"
        export_labels_dir = self.export_root / "labels"
        legacy_labels_dir = self.export_root / "lables"
        export_images_dir.mkdir(parents=True, exist_ok=True)
        export_labels_dir.mkdir(parents=True, exist_ok=True)
        if legacy_labels_dir.exists():
            shutil.rmtree(legacy_labels_dir)

        classes = load_classes(self.classes_file)
        used_stems: set[str] = set()
        class_counts: dict[str, int] = {}
        cleaned_classes: set[str] = set()
        exported_count = 0

        for image_path in self.images:
            source_label = self._resolve_label_file_for_export(image_path)
            if source_label is None:
                continue
            label_data = load_yolo_label(source_label)
            if label_data is None:
                continue

            class_id, _ = label_data
            class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
            class_slug = self._slugify(class_name)
            base_stem = class_slug
            if class_slug not in cleaned_classes:
                self._remove_old_image_variants(export_images_dir, class_slug, None)
                self._remove_old_label_variants(export_labels_dir, class_slug, None)
                cleaned_classes.add(class_slug)

            class_counts[class_slug] = class_counts.get(class_slug, 0) + 1
            if class_counts[class_slug] > 1:
                base_stem = f"{class_slug}_{class_counts[class_slug]}"
            target_stem = self._unique_stem(base_stem, used_stems, export_images_dir, export_labels_dir)
            used_stems.add(target_stem)

            target_image = export_images_dir / f"{target_stem}{image_path.suffix}"
            target_label = export_labels_dir / f"{target_stem}.txt"

            self._remove_old_image_variants(export_images_dir, target_stem, target_image)
            self._remove_old_label_variants(export_labels_dir, target_stem, target_label)

            shutil.copy2(image_path, target_image)
            target_label.write_text(source_label.read_text(encoding="utf-8"), encoding="utf-8")

            if source_label.resolve() != target_label.resolve():
                source_label.unlink(missing_ok=True)
            exported_count += 1

        return exported_count

    def _resolve_label_file_for_export(self, image_path: Path) -> Path | None:
        primary = self._label_file_for(image_path)
        if primary.exists():
            return primary

        variants = sorted(self.labels_dir.glob(f"{image_path.stem}_*.txt"))
        if variants:
            return variants[0]
        return None

    @staticmethod
    def _remove_old_image_variants(export_images_dir: Path, image_stem: str, keep_file: Path | None) -> None:
        for candidate in export_images_dir.glob(f"{image_stem}_*.*"):
            if not keep_file or candidate.resolve() != keep_file.resolve():
                candidate.unlink(missing_ok=True)
        for candidate in export_images_dir.glob(f"{image_stem}.*"):
            if not keep_file or candidate.resolve() != keep_file.resolve():
                candidate.unlink(missing_ok=True)

    @staticmethod
    def _remove_old_label_variants(export_labels_dir: Path, image_stem: str, keep_file: Path | None) -> None:
        for candidate in export_labels_dir.glob(f"{image_stem}_*.txt"):
            if not keep_file or candidate.resolve() != keep_file.resolve():
                candidate.unlink(missing_ok=True)
        legacy_primary = export_labels_dir / f"{image_stem}.txt"
        if not keep_file or legacy_primary.resolve() != keep_file.resolve():
            legacy_primary.unlink(missing_ok=True)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
        return slug or "class"

    @staticmethod
    def _unique_stem(
        base_stem: str,
        used_stems: set[str],
        export_images_dir: Path,
        export_labels_dir: Path,
    ) -> str:
        stem = base_stem
        index = 2
        while (
            stem in used_stems
            or any(export_images_dir.glob(f"{stem}.*"))
            or (export_labels_dir / f"{stem}.txt").exists()
        ):
            stem = f"{base_stem}_{index}"
            index += 1
        return stem


def run_app(images_dir: Path, labels_dir: Path, classes_file: Path, export_root: Path, exts: list[str]) -> None:
    app = QApplication(sys.argv)
    window = LabelerWindow(
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes_file=classes_file,
        export_root=export_root,
        exts=exts,
    )
    window.show()
    app.exec()
