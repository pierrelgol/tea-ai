from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils.torch_utils import unwrap_model

from dinov3_bridge import DinoV3Teacher, resolve_local_dinov3_root


@dataclass(frozen=True, slots=True)
class DinoDistillConfig:
    dino_root: Path
    weight: float
    warmup_epochs: int
    student_layers: tuple[int, ...]
    channels: int
    object_weight: float
    background_weight: float


class DinoOBBTrainer(OBBTrainer):
    """OBB trainer with mandatory spatial DINOv3 feature distillation."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None, *, dino_cfg: DinoDistillConfig | None = None):
        if cfg is None:
            super().__init__(overrides=overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if dino_cfg is None:
            raise ValueError("dino_cfg is required for DinoOBBTrainer")
        self._dino_cfg = dino_cfg

    def _setup_train(self) -> None:
        super()._setup_train()

        model = unwrap_model(self.model)
        self._dino_teacher = DinoV3Teacher(
            resolve_local_dinov3_root(self._dino_cfg.dino_root),
            device=self.device,
        )
        self._dino_base_weight = max(0.0, float(self._dino_cfg.weight))
        self._dino_warmup_epochs = max(0, int(self._dino_cfg.warmup_epochs))
        self._dino_active_weight = 0.0
        self._dino_last_weight = 0.0
        self._dino_loss_sum = 0.0
        self._dino_loss_count = 0
        self._dino_epoch_loss = 0.0
        self._dino_epoch_obj_loss = 0.0
        self._dino_epoch_bg_loss = 0.0

        layer_count = len(getattr(model, "model", []))
        if layer_count <= 0:
            raise RuntimeError("unable to resolve YOLO model layers for DINOv3 distillation")
        resolved_layers: list[int] = []
        for idx in self._dino_cfg.student_layers:
            ridx = layer_count + idx if idx < 0 else idx
            if ridx < 0 or ridx >= layer_count:
                raise ValueError(f"invalid dino student layer index {idx} for layer_count={layer_count}")
            resolved_layers.append(int(ridx))
        if not resolved_layers:
            raise ValueError("at least one dino student layer is required")
        self._dino_layers = tuple(dict.fromkeys(resolved_layers))

        self._dino_feature_cache: dict[int, torch.Tensor] = {}
        self._dino_hooks = []

        def _capture_layer(layer_idx: int):
            def _hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output
                elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output[0]
            return _hook

        for layer_idx in self._dino_layers:
            handle = model.model[layer_idx].register_forward_hook(_capture_layer(layer_idx))
            self._dino_hooks.append(handle)

        original_loss = model.loss
        trainer = self

        def _build_object_mask(batch: dict[str, torch.Tensor], out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
            h, w = out_hw
            mask = torch.zeros((int(batch["img"].shape[0]), 1, h, w), device=device)
            batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
            boxes = batch["bboxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                return mask

            cx = boxes[:, 0].clamp(0.0, 1.0)
            cy = boxes[:, 1].clamp(0.0, 1.0)
            bw = boxes[:, 2].clamp(0.0, 1.0)
            bh = boxes[:, 3].clamp(0.0, 1.0)
            x1 = ((cx - bw * 0.5) * w).floor().to(torch.long).clamp(0, max(0, w - 1))
            x2 = ((cx + bw * 0.5) * w).ceil().to(torch.long).clamp(1, w)
            y1 = ((cy - bh * 0.5) * h).floor().to(torch.long).clamp(0, max(0, h - 1))
            y2 = ((cy + bh * 0.5) * h).ceil().to(torch.long).clamp(1, h)

            for i in range(int(boxes.shape[0])):
                bi = int(batch_idx[i].item())
                xa, xb = int(x1[i].item()), int(x2[i].item())
                ya, yb = int(y1[i].item()), int(y2[i].item())
                if xa < xb and ya < yb and 0 <= bi < mask.shape[0]:
                    mask[bi, 0, ya:yb, xa:xb] = 1.0
            return mask

        def _compute_distill_loss(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if trainer._dino_active_weight <= 0.0:
                z = batch["img"].sum() * 0.0
                return z, z, z

            teacher_map = trainer._dino_teacher.extract_feature_map(batch["img"])
            obj_weight = max(0.0, float(trainer._dino_cfg.object_weight))
            bg_weight = max(0.0, float(trainer._dino_cfg.background_weight))

            layer_losses: list[torch.Tensor] = []
            layer_obj_losses: list[torch.Tensor] = []
            layer_bg_losses: list[torch.Tensor] = []

            for layer_idx in trainer._dino_layers:
                if layer_idx not in trainer._dino_feature_cache:
                    raise RuntimeError(f"missing captured student feature for layer {layer_idx}")
                student_map = trainer._dino_feature_cache[layer_idx]
                if student_map.ndim != 4:
                    raise RuntimeError(f"student feature at layer {layer_idx} must be BCHW, got {student_map.shape}")

                target_c = min(
                    max(8, int(trainer._dino_cfg.channels)),
                    int(student_map.shape[1]),
                    int(teacher_map.shape[1]),
                )
                smap = student_map[:, :target_c, :, :]
                tbase = teacher_map[:, :target_c, :, :].to(device=smap.device, dtype=smap.dtype)
                tmap = F.interpolate(
                    tbase,
                    size=smap.shape[-2:],
                    mode="nearest",
                )

                sproj = F.normalize(smap.flatten(2), dim=1)
                tproj = F.normalize(tmap.flatten(2), dim=1)
                cos_dist = 1.0 - (sproj * tproj).sum(dim=1, keepdim=True)  # Bx1xHW

                obj_mask = _build_object_mask(batch, smap.shape[-2:], smap.device)
                bg_mask = 1.0 - obj_mask

                obj_den = obj_mask.sum().clamp_min(1.0)
                bg_den = bg_mask.sum().clamp_min(1.0)
                obj_loss = (cos_dist * obj_mask.flatten(2)).sum() / obj_den
                bg_loss = (cos_dist * bg_mask.flatten(2)).sum() / bg_den
                weighted = obj_weight * obj_loss + bg_weight * bg_loss

                layer_losses.append(weighted)
                layer_obj_losses.append(obj_loss)
                layer_bg_losses.append(bg_loss)

            mean_loss = torch.stack(layer_losses).mean() * trainer._dino_active_weight
            mean_obj = torch.stack(layer_obj_losses).mean() * trainer._dino_active_weight
            mean_bg = torch.stack(layer_bg_losses).mean() * trainer._dino_active_weight
            return mean_loss, mean_obj, mean_bg

        def _loss_with_distill(model_self, batch, preds=None):
            trainer._dino_feature_cache.clear()
            if preds is None:
                preds = model_self.predict(batch["img"])
            det_out = original_loss(batch, preds)
            if not isinstance(det_out, (tuple, list)) or len(det_out) < 2:
                return det_out

            det_loss, loss_items = det_out[0], det_out[1]
            distill_loss, obj_loss, bg_loss = _compute_distill_loss(batch)

            trainer._dino_last_weight = float(trainer._dino_active_weight)
            trainer._dino_loss_sum += float(distill_loss.detach().item())
            trainer._dino_loss_count += 1
            trainer._dino_epoch_loss = trainer._dino_loss_sum / max(1, trainer._dino_loss_count)
            trainer._dino_epoch_obj_loss = float(obj_loss.detach().item())
            trainer._dino_epoch_bg_loss = float(bg_loss.detach().item())

            return det_loss + distill_loss, loss_items

        model.loss = types.MethodType(_loss_with_distill, model)

        def _on_train_epoch_start(cb_trainer) -> None:
            cb_trainer._dino_loss_sum = 0.0
            cb_trainer._dino_loss_count = 0
            cb_trainer._dino_epoch_loss = 0.0
            cb_trainer._dino_epoch_obj_loss = 0.0
            cb_trainer._dino_epoch_bg_loss = 0.0

            warmup = cb_trainer._dino_warmup_epochs
            if warmup <= 0:
                weight_scale = 1.0
            else:
                epoch_idx = int(getattr(cb_trainer, "epoch", 0)) + 1
                weight_scale = min(1.0, float(epoch_idx) / float(warmup))
            cb_trainer._dino_active_weight = cb_trainer._dino_base_weight * weight_scale

        self.add_callback("on_train_epoch_start", _on_train_epoch_start)

    def __del__(self):
        hooks = getattr(self, "_dino_hooks", None)
        if not hooks:
            return
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
