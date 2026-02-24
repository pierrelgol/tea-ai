from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import unwrap_model

from .distill import DinoTeacherAdapter, DistillRegistry, DistillRegistryConfig


@dataclass(frozen=True, slots=True)
class DinoDistillConfig:
    dino_root: Path
    stage_a_epochs: int
    stage_a_freeze: int
    stage_a_weight: float
    stage_b_weight: float
    warmup_epochs: int
    dino_layers: tuple[int, ...]
    dino_to_yolo_taps: dict[int, int]
    channels: int
    object_weight: float
    background_weight: float
    feat_layer_weights: dict[int, float]
    attn_enabled: bool
    attn_layers: tuple[int, ...]
    attn_weight_stage_a: float
    attn_weight_stage_b: float
    objmap_enabled: bool
    objmap_layer: int
    objmap_weight_stage_a: float
    objmap_weight_stage_b: float
    objmap_apply_inside_gt_only: bool
    viz_enabled: bool
    viz_mode: str
    viz_every_n_epochs: int
    total_epochs: int
    viz_max_samples: int


class DinoOBBTrainer(OBBTrainer):
    """OBB trainer with DINOv3 multi-loss distillation."""

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
        self._dino_layers = tuple(dict.fromkeys(int(v) for v in self._dino_cfg.dino_layers))
        if not self._dino_layers:
            raise ValueError("at least one dino distillation layer is required")

        self._dino_stage_a_epochs = max(0, int(self._dino_cfg.stage_a_epochs))
        self._dino_stage_a_freeze = max(0, int(self._dino_cfg.stage_a_freeze))
        self._dino_stage_a_weight = max(0.0, float(self._dino_cfg.stage_a_weight))
        self._dino_stage_b_weight = max(0.0, float(self._dino_cfg.stage_b_weight))
        self._dino_warmup_epochs = max(0, int(self._dino_cfg.warmup_epochs))
        self._dino_attn_stage_a_weight = max(0.0, float(self._dino_cfg.attn_weight_stage_a))
        self._dino_attn_stage_b_weight = max(0.0, float(self._dino_cfg.attn_weight_stage_b))
        self._dino_obj_stage_a_weight = max(0.0, float(self._dino_cfg.objmap_weight_stage_a))
        self._dino_obj_stage_b_weight = max(0.0, float(self._dino_cfg.objmap_weight_stage_b))
        self._dino_viz_enabled = bool(self._dino_cfg.viz_enabled)
        self._dino_viz_mode = str(self._dino_cfg.viz_mode)
        self._dino_viz_every_n_epochs = max(1, int(self._dino_cfg.viz_every_n_epochs))
        self._dino_total_epochs = max(1, int(self._dino_cfg.total_epochs))
        self._dino_viz_capture_epoch = False
        self._dino_viz_max_samples = max(1, int(self._dino_cfg.viz_max_samples))

        self._dino_active_weight = 0.0
        self._dino_attn_active_weight = 0.0
        self._dino_obj_active_weight = 0.0
        self._dino_last_weight = 0.0
        self._dino_loss_sum = 0.0
        self._dino_loss_count = 0
        self._dino_epoch_loss = 0.0
        self._dino_epoch_obj_loss = 0.0
        self._dino_epoch_bg_loss = 0.0
        self._dino_epoch_terms: dict[str, float] = {}
        self._dino_viz_snapshot: dict[str, object] | None = None
        self._dino_stage_a_active = self._dino_stage_a_epochs > 0 and self._dino_stage_a_freeze > 0
        self._dino_stage_a_prefixes = tuple(f"model.{i}." for i in range(self._dino_stage_a_freeze))

        layer_count = len(getattr(model, "model", []))
        if layer_count <= 0:
            raise RuntimeError("unable to resolve YOLO model layers for DINO distillation")

        resolved_taps: dict[int, int] = {}
        for dlayer, ytap in self._dino_cfg.dino_to_yolo_taps.items():
            ridx = layer_count + ytap if ytap < 0 else ytap
            if ridx < 0 or ridx >= layer_count:
                raise ValueError(f"invalid yolo tap layer index {ytap} for layer_count={layer_count}")
            resolved_taps[int(dlayer)] = int(ridx)
        missing_mapped = [int(v) for v in self._dino_layers if int(v) not in resolved_taps]
        if missing_mapped:
            raise ValueError(f"missing dino_to_yolo_taps entries for dino layers: {missing_mapped}")
        self._dino_to_yolo_taps = resolved_taps

        required_taps = tuple(sorted(set(self._dino_to_yolo_taps.values())))
        self._dino_feature_cache: dict[int, torch.Tensor] = {}
        self._dino_hooks = []

        def _capture_layer(layer_idx: int):
            def _hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output
                elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output[0]

            return _hook

        for layer_idx in required_taps:
            handle = model.model[layer_idx].register_forward_hook(_capture_layer(layer_idx))
            self._dino_hooks.append(handle)

        self._teacher_adapter = DinoTeacherAdapter(
            dino_root=self._dino_cfg.dino_root,
            device=self.device,
            feature_layers=self._dino_layers,
            attn_layers=tuple(int(v) for v in self._dino_cfg.attn_layers),
        )

        def _add_params(params: list[nn.Parameter]) -> None:
            if not params:
                return
            if getattr(self, "optimizer", None) is None or not self.optimizer.param_groups:
                raise RuntimeError("optimizer is unavailable while registering distillation parameters")
            # Keep param group count unchanged; Ultralytics scheduler expects fixed group cardinality.
            group0 = self.optimizer.param_groups[0]
            existing_ids = {id(p) for p in group0.get("params", [])}
            for p in params:
                if id(p) not in existing_ids:
                    group0["params"].append(p)
                    existing_ids.add(id(p))

        self._distill_registry = DistillRegistry(
            DistillRegistryConfig(
                dino_layers=self._dino_layers,
                dino_to_yolo_taps=self._dino_to_yolo_taps,
                channels=int(self._dino_cfg.channels),
                object_weight=float(self._dino_cfg.object_weight),
                background_weight=float(self._dino_cfg.background_weight),
                feat_layer_weights={int(k): float(v) for k, v in self._dino_cfg.feat_layer_weights.items()},
                attn_enabled=bool(self._dino_cfg.attn_enabled),
                attn_layers=tuple(int(v) for v in self._dino_cfg.attn_layers),
                objmap_enabled=bool(self._dino_cfg.objmap_enabled),
                objmap_layer=int(self._dino_cfg.objmap_layer),
                objmap_apply_inside_gt_only=bool(self._dino_cfg.objmap_apply_inside_gt_only),
            ),
            add_params=_add_params,
        ).to(self.device)

        original_loss = model.loss
        trainer = self

        def _build_object_mask(batch: dict[str, torch.Tensor], out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
            h, w = out_hw
            mask = torch.zeros((int(batch["img"].shape[0]), 1, h, w), device=device)
            batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
            boxes = batch["bboxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                return mask
            boxes_px = boxes.clone()
            boxes_px[:, 0] = boxes_px[:, 0].clamp(0.0, 1.0) * w
            boxes_px[:, 1] = boxes_px[:, 1].clamp(0.0, 1.0) * h
            boxes_px[:, 2] = boxes_px[:, 2].clamp(0.0, 1.0) * w
            boxes_px[:, 3] = boxes_px[:, 3].clamp(0.0, 1.0) * h
            polys = ops.xywhr2xyxyxyxy(boxes_px).detach().cpu().numpy()
            for bi in range(mask.shape[0]):
                idxs = (batch_idx == bi).nonzero(as_tuple=False).view(-1).detach().cpu().numpy()
                if idxs.size == 0:
                    continue
                canvas = np.zeros((h, w), dtype=np.uint8)
                pts: list[np.ndarray] = []
                for i in idxs.tolist():
                    poly = np.round(polys[i]).astype(np.int32).reshape(-1, 2)
                    poly[:, 0] = np.clip(poly[:, 0], 0, max(0, w - 1))
                    poly[:, 1] = np.clip(poly[:, 1], 0, max(0, h - 1))
                    if poly.shape[0] == 4:
                        pts.append(poly)
                if pts:
                    cv2.fillPoly(canvas, pts, color=1)
                    mask[bi, 0] = torch.from_numpy(canvas).to(device=device, dtype=mask.dtype)
            return mask

        def _extract_one2many(preds) -> dict[str, torch.Tensor | list[torch.Tensor]] | None:
            payload = preds[1] if isinstance(preds, tuple) and len(preds) > 1 else preds
            if not isinstance(payload, dict):
                return None
            if "one2many" in payload and isinstance(payload["one2many"], dict):
                return payload["one2many"]
            if all(k in payload for k in ("boxes", "scores", "feats")):
                return payload
            return None

        def _compute_distill_loss(batch: dict[str, torch.Tensor], preds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
            if trainer._dino_active_weight <= 0.0:
                z = batch["img"].sum() * 0.0
                return z, z, z, {}

            for tap in required_taps:
                if tap not in trainer._dino_feature_cache:
                    raise RuntimeError(f"missing captured student feature for tap layer {tap}")

            teacher_out = trainer._teacher_adapter.forward(batch["img"])
            student_taps = {tap: trainer._dino_feature_cache[tap] for tap in required_taps}
            mask_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

            for fmap in student_taps.values():
                hw = (int(fmap.shape[-2]), int(fmap.shape[-1]))
                if hw not in mask_cache:
                    obj = _build_object_mask(batch, hw, fmap.device)
                    mask_cache[hw] = (obj, 1.0 - obj)

            feat_loss, feat_logs, signal_map = trainer._distill_registry.feature_loss(
                teacher=teacher_out,
                student_taps=student_taps,
                masks_by_hw=mask_cache,
                active_weight=trainer._dino_active_weight,
            )

            attn_loss, attn_logs = trainer._distill_registry.attention_loss(
                teacher=teacher_out,
                student_taps=student_taps,
                active_weight=trainer._dino_active_weight,
                stage_attn_weight=trainer._dino_attn_active_weight,
            )

            one2many = _extract_one2many(preds)
            if one2many is None:
                obj_loss = feat_loss * 0.0
                obj_logs: dict[str, float] = {}
            else:
                obj_loss, obj_logs = trainer._distill_registry.objectness_map_loss(
                    teacher=teacher_out,
                    preds_one2many=one2many,
                    masks_by_hw=mask_cache,
                    active_weight=trainer._dino_active_weight,
                    stage_obj_weight=trainer._dino_obj_active_weight,
                )

            if trainer._dino_viz_enabled and trainer._dino_viz_capture_epoch:
                sample_count = min(int(batch["img"].shape[0]), trainer._dino_viz_max_samples)
                if sample_count > 0:
                    t_layer = trainer._dino_cfg.objmap_layer if trainer._dino_cfg.objmap_layer in teacher_out.features else trainer._dino_layers[-1]
                    tmap = teacher_out.features[t_layer]
                    thw = (int(tmap.shape[-2]), int(tmap.shape[-1]))
                    obj_mask, _ = mask_cache.get(thw, (_build_object_mask(batch, thw, tmap.device), None))
                    viz_signal = signal_map if signal_map is not None else tmap[:, 0]
                    trainer._dino_viz_snapshot = {
                        "images": batch["img"][:sample_count].detach().float().cpu().numpy(),
                        "teacher": tmap[:sample_count].detach().float().cpu().numpy(),
                        "obj_mask": obj_mask[:sample_count, 0].detach().float().cpu().numpy(),
                        "signal_map": viz_signal[:sample_count].detach().float().cpu().numpy(),
                    }

            total = feat_loss + attn_loss + obj_loss
            logs = {**feat_logs, **attn_logs, **obj_logs}
            obj_proxy = torch.tensor(float(obj_logs.get(f"obj_l{trainer._dino_cfg.objmap_layer}", 0.0)), device=total.device, dtype=total.dtype)
            bg_proxy = torch.tensor(float(sum(attn_logs.values())), device=total.device, dtype=total.dtype)
            return total, obj_proxy, bg_proxy, logs

        def _loss_with_distill(model_self, batch, preds=None):
            trainer._dino_feature_cache.clear()
            if preds is None:
                preds = model_self.predict(batch["img"])
            det_out = original_loss(batch, preds)
            if not isinstance(det_out, (tuple, list)) or len(det_out) < 2:
                return det_out

            det_loss, loss_items = det_out[0], det_out[1]
            distill_loss, obj_loss, bg_loss, term_logs = _compute_distill_loss(batch, preds)

            trainer._dino_last_weight = float(trainer._dino_active_weight)
            trainer._dino_loss_sum += float(distill_loss.detach().item())
            trainer._dino_loss_count += 1
            trainer._dino_epoch_loss = trainer._dino_loss_sum / max(1, trainer._dino_loss_count)
            trainer._dino_epoch_obj_loss = float(obj_loss.detach().item())
            trainer._dino_epoch_bg_loss = float(bg_loss.detach().item())
            trainer._dino_epoch_terms = term_logs

            return det_loss + distill_loss, loss_items

        model.loss = types.MethodType(_loss_with_distill, model)

        def _on_train_epoch_start(cb_trainer) -> None:
            cb_trainer._dino_loss_sum = 0.0
            cb_trainer._dino_loss_count = 0
            cb_trainer._dino_epoch_loss = 0.0
            cb_trainer._dino_epoch_obj_loss = 0.0
            cb_trainer._dino_epoch_bg_loss = 0.0
            cb_trainer._dino_epoch_terms = {}
            cb_trainer._dino_viz_snapshot = None

            epoch_idx = int(getattr(cb_trainer, "epoch", 0))
            in_stage_a = epoch_idx < cb_trainer._dino_stage_a_epochs
            cb_trainer._dino_stage_a_active = (
                in_stage_a and cb_trainer._dino_stage_a_epochs > 0 and cb_trainer._dino_stage_a_freeze > 0
            )
            warmup = cb_trainer._dino_warmup_epochs
            if warmup <= 0:
                weight_scale = 1.0
            else:
                weight_scale = min(1.0, float(epoch_idx + 1) / float(warmup))
            base_weight = cb_trainer._dino_stage_a_weight if in_stage_a else cb_trainer._dino_stage_b_weight
            cb_trainer._dino_active_weight = base_weight * weight_scale
            cb_trainer._dino_attn_active_weight = cb_trainer._dino_attn_stage_a_weight if in_stage_a else cb_trainer._dino_attn_stage_b_weight
            cb_trainer._dino_obj_active_weight = cb_trainer._dino_obj_stage_a_weight if in_stage_a else cb_trainer._dino_obj_stage_b_weight

            if cb_trainer._dino_viz_enabled:
                if cb_trainer._dino_viz_mode == "off":
                    cb_trainer._dino_viz_capture_epoch = False
                elif cb_trainer._dino_viz_mode == "final_only":
                    cb_trainer._dino_viz_capture_epoch = (epoch_idx + 1) == cb_trainer._dino_total_epochs
                else:
                    cb_trainer._dino_viz_capture_epoch = (
                        (epoch_idx + 1) == cb_trainer._dino_total_epochs
                        or ((epoch_idx + 1) % cb_trainer._dino_viz_every_n_epochs == 0)
                    )
            else:
                cb_trainer._dino_viz_capture_epoch = False

            if cb_trainer._dino_stage_a_active and cb_trainer._dino_stage_a_prefixes:
                base_model = unwrap_model(cb_trainer.model)
                for name, module in base_model.named_modules():
                    if any(name.startswith(pref) for pref in cb_trainer._dino_stage_a_prefixes) and isinstance(
                        module, nn.BatchNorm2d
                    ):
                        module.eval()

        self.add_callback("on_train_epoch_start", _on_train_epoch_start)

    def optimizer_step(self):
        if getattr(self, "_dino_stage_a_active", False):
            base_model = unwrap_model(self.model)
            freeze_prefixes = tuple(getattr(self, "_dino_stage_a_prefixes", ()))
            for name, param in base_model.named_parameters():
                if any(name.startswith(pref) for pref in freeze_prefixes):
                    param.grad = None
        return super().optimizer_step()

    def __del__(self):
        hooks = getattr(self, "_dino_hooks", None)
        if hooks:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        teacher_adapter = getattr(self, "_teacher_adapter", None)
        if teacher_adapter is not None:
            try:
                teacher_adapter.close()
            except Exception:
                pass
