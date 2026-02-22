from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils.torch_utils import unwrap_model

from dinov3_bridge import DinoV3Teacher, resolve_local_dinov3_root


class DinoOBBTrainer(OBBTrainer):
    """OBB trainer with mandatory DINOv3 feature distillation."""
    _configured_dino_root: Path = Path("dinov3")
    _configured_distill_weight: float = 0.2
    _configured_warmup_epochs: int = 5
    _configured_student_embed_layer: int = -2

    @classmethod
    def configure(
        cls,
        *,
        dino_root: Path,
        distill_weight: float,
        warmup_epochs: int,
        student_embed_layer: int,
    ) -> None:
        cls._configured_dino_root = dino_root.resolve()
        cls._configured_distill_weight = float(distill_weight)
        cls._configured_warmup_epochs = int(warmup_epochs)
        cls._configured_student_embed_layer = int(student_embed_layer)

    def _setup_train(self) -> None:
        super()._setup_train()

        model = unwrap_model(self.model)
        self._dino_teacher = DinoV3Teacher(
            resolve_local_dinov3_root(self._configured_dino_root),
            device=self.device,
        )
        self._dino_base_weight = max(0.0, float(self._configured_distill_weight))
        self._dino_warmup_epochs = max(0, int(self._configured_warmup_epochs))
        self._dino_active_weight = 0.0
        self._dino_last_loss = 0.0
        self._dino_last_weight = 0.0
        self._dino_loss_sum = 0.0
        self._dino_loss_count = 0
        self._dino_epoch_loss = 0.0

        embed_idx = int(self._configured_student_embed_layer)
        layer_count = len(getattr(model, "model", []))
        if layer_count <= 0:
            raise RuntimeError("unable to resolve YOLO model layers for DINOv3 distillation")
        if embed_idx < 0:
            embed_idx = layer_count + embed_idx
        if embed_idx < 0 or embed_idx >= layer_count:
            raise ValueError(
                f"dino_student_embed_layer resolved to invalid index {embed_idx} for layer_count={layer_count}"
            )
        self._dino_embed_idx = embed_idx

        original_loss = model.loss
        trainer = self

        def _loss_with_distill(model_self, batch, preds=None):
            det_out = original_loss(batch, preds)
            if not isinstance(det_out, (tuple, list)) or len(det_out) < 2:
                return det_out

            det_loss, loss_items = det_out[0], det_out[1]
            if trainer._dino_active_weight <= 0.0:
                trainer._dino_last_loss = 0.0
                return det_loss, loss_items

            imgs = batch["img"]
            student_embed_raw = model_self.predict(imgs, embed=[trainer._dino_embed_idx])
            if isinstance(student_embed_raw, torch.Tensor):
                student_embed = student_embed_raw
            elif isinstance(student_embed_raw, (tuple, list)):
                student_embed = torch.stack([
                    v if v.ndim == 1 else v.reshape(-1) for v in student_embed_raw
                ], dim=0)
            else:
                raise RuntimeError("unexpected student embedding output type for DINOv3 distillation")

            if student_embed.ndim != 2:
                student_embed = student_embed.reshape(student_embed.shape[0], -1)

            teacher_embed = trainer._dino_teacher.extract_features(imgs)
            teacher_embed = teacher_embed.to(device=student_embed.device, dtype=student_embed.dtype)
            sdim = int(student_embed.shape[1])
            tdim = int(teacher_embed.shape[1])
            if sdim == tdim:
                projected = student_embed
            elif sdim > tdim:
                projected = student_embed[:, :tdim]
            else:
                projected = F.pad(student_embed, (0, tdim - sdim))
            distill_raw = 1.0 - F.cosine_similarity(
                F.normalize(projected, dim=1),
                F.normalize(teacher_embed, dim=1),
                dim=1,
            ).mean()
            distill_loss = distill_raw * trainer._dino_active_weight

            trainer._dino_last_loss = float(distill_loss.detach().item())
            trainer._dino_last_weight = float(trainer._dino_active_weight)
            trainer._dino_loss_sum += trainer._dino_last_loss
            trainer._dino_loss_count += 1
            trainer._dino_epoch_loss = trainer._dino_loss_sum / max(1, trainer._dino_loss_count)

            return det_loss + distill_loss, loss_items

        model.loss = types.MethodType(_loss_with_distill, model)

        def _on_train_epoch_start(cb_trainer) -> None:
            cb_trainer._dino_loss_sum = 0.0
            cb_trainer._dino_loss_count = 0
            cb_trainer._dino_epoch_loss = 0.0

            warmup = cb_trainer._dino_warmup_epochs
            if warmup <= 0:
                weight_scale = 1.0
            else:
                epoch_idx = int(getattr(cb_trainer, "epoch", 0)) + 1
                weight_scale = min(1.0, float(epoch_idx) / float(warmup))
            cb_trainer._dino_active_weight = cb_trainer._dino_base_weight * weight_scale

        self.add_callback("on_train_epoch_start", _on_train_epoch_start)
