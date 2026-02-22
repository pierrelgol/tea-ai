from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data import Label, Sample, corners_to_px, load_labels


@dataclass(slots=True)
class ReviewConfig:
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    device: str


class ReviewWindow(QMainWindow):
    def __init__(
        self,
        samples: list[Sample],
        model,
        model_name: str,
        cfg: ReviewConfig,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.model = model
        self.model_name = model_name
        self.cfg = cfg
        self.index = 0
        self.pred_cache: dict[tuple[str, str], list[Label]] = {}

        self.setWindowTitle(f"Detector Reviewer - {model_name}")
        self.resize(1400, 900)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(900, 700)

        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self.on_row_changed)

        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_sample)
        self.next_btn.clicked.connect(self.next_sample)

        self.show_gt = QCheckBox("Show GT (green)")
        self.show_gt.setChecked(True)
        self.show_pred = QCheckBox("Show Pred (red)")
        self.show_pred.setChecked(True)
        self.show_gt.stateChanged.connect(lambda _s: self.render_current())
        self.show_pred.stateChanged.connect(lambda _s: self.render_current())

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(round(cfg.conf_threshold * 100)))
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.conf_label = QLabel(f"Conf: {self._conf_value():.2f}")

        self.info = QTextEdit()
        self.info.setReadOnly(True)

        self.build_layout()
        for sample in self.samples:
            self.sample_list.addItem(f"{sample.split}/{sample.stem}")
        if self.samples:
            self.sample_list.setCurrentRow(0)

    def _conf_value(self) -> float:
        return float(self.conf_slider.value()) / 100.0

    def build_layout(self) -> None:
        side = QVBoxLayout()
        side.addWidget(self.prev_btn)
        side.addWidget(self.next_btn)
        side.addWidget(self.show_gt)
        side.addWidget(self.show_pred)
        side.addWidget(self.conf_label)
        side.addWidget(self.conf_slider)
        side.addWidget(QLabel("Samples"))
        side.addWidget(self.sample_list, 1)
        side.addWidget(QLabel("Info"))
        side.addWidget(self.info, 1)

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(420)

        root = QHBoxLayout()
        root.addWidget(self.image_label, 1)
        root.addWidget(side_widget)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def on_conf_changed(self) -> None:
        self.conf_label.setText(f"Conf: {self._conf_value():.2f}")
        self.render_current()

    def prev_sample(self) -> None:
        if not self.samples:
            return
        self.sample_list.setCurrentRow(max(0, self.sample_list.currentRow() - 1))

    def next_sample(self) -> None:
        if not self.samples:
            return
        self.sample_list.setCurrentRow(min(len(self.samples) - 1, self.sample_list.currentRow() + 1))

    def on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.samples):
            return
        self.index = row
        self.render_current()

    def _predict(self, sample: Sample) -> list[Label]:
        key = (sample.split, sample.stem)
        if key in self.pred_cache:
            return self.pred_cache[key]
        if sample.image_path is None:
            self.pred_cache[key] = []
            return []

        results = self.model.predict(
            source=str(sample.image_path),
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        res = results[0]
        preds: list[Label] = []

        if getattr(res, "obb", None) is not None:
            obb = res.obb
            if hasattr(obb, "xyxyxyxyn") and obb.xyxyxyxyn is not None:
                coords = obb.xyxyxyxyn.cpu().numpy()
            elif hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
                px = obb.xyxyxyxy.cpu().numpy()
                h, w = res.orig_shape[:2]
                coords = px.astype(np.float64)
                coords[:, :, 0] /= w
                coords[:, :, 1] /= h
            else:
                coords = np.zeros((0, 4, 2), dtype=np.float32)
            confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.ones((coords.shape[0],), dtype=np.float32)
            classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else np.zeros((coords.shape[0],), dtype=int)
            coords = np.clip(coords, 0.0, 1.0)
            for i in range(coords.shape[0]):
                preds.append(Label(class_id=int(classes[i]), corners_norm=coords[i], confidence=float(confs[i])))

        self.pred_cache[key] = preds
        return preds

    @staticmethod
    def _draw_poly(img: np.ndarray, corners_px: np.ndarray, color: tuple[int, int, int], text: str) -> None:
        poly = corners_px.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [poly], True, color, 2)
        x = int(np.min(corners_px[:, 0]))
        y = int(np.min(corners_px[:, 1])) - 6
        cv2.putText(img, text, (max(2, x), max(16, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def render_current(self) -> None:
        if not self.samples:
            return
        sample = self.samples[self.index]
        if sample.image_path is None:
            self.info.setPlainText(f"{sample.split}/{sample.stem}\nmissing image")
            return

        img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if img is None:
            self.info.setPlainText(f"{sample.split}/{sample.stem}\nfailed to read image")
            return
        h, w = img.shape[:2]

        gt = load_labels(sample.label_path, is_prediction=False)
        preds = self._predict(sample)
        conf_threshold = self._conf_value()
        preds = [p for p in preds if p.confidence >= conf_threshold]

        if self.show_gt.isChecked():
            for idx, label in enumerate(gt):
                px = corners_to_px(label.corners_norm, w, h)
                self._draw_poly(img, px, (0, 255, 0), f"GT c{label.class_id} #{idx}")

        if self.show_pred.isChecked():
            for idx, label in enumerate(preds):
                px = corners_to_px(label.corners_norm, w, h)
                self._draw_poly(img, px, (0, 0, 255), f"PR c{label.class_id} {label.confidence:.2f} #{idx}")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        lines = [
            f"sample: {sample.split}/{sample.stem}",
            f"image: {sample.image_path}",
            f"gt_boxes: {len(gt)}",
            f"pred_boxes@{conf_threshold:.2f}: {len(preds)}",
            f"model: {self.model_name}",
        ]
        self.info.setPlainText("\n".join(lines))


def launch_gui(
    *,
    samples: list[Sample],
    model,
    model_name: str,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
) -> None:
    app = QApplication([])
    window = ReviewWindow(
        samples=samples,
        model=model,
        model_name=model_name,
        cfg=ReviewConfig(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            device=device,
        ),
    )
    window.show()
    app.exec()
