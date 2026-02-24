from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dino_teacher import TeacherOutputs


@dataclass(frozen=True, slots=True)
class DistillRegistryConfig:
    dino_layers: tuple[int, ...]
    dino_to_yolo_taps: dict[int, int]
    channels: int
    object_weight: float
    background_weight: float
    feat_layer_weights: dict[int, float]
    attn_enabled: bool
    attn_layers: tuple[int, ...]
    objmap_enabled: bool
    objmap_layer: int
    objmap_apply_inside_gt_only: bool


class DistillRegistry(nn.Module):
    def __init__(self, cfg: DistillRegistryConfig, add_params: Callable[[list[nn.Parameter]], None]) -> None:
        super().__init__()
        self.cfg = cfg
        self._add_params = add_params
        self._proj_yolo = nn.ModuleDict()
        self._proj_dino = nn.ModuleDict()

    def _make_conv(self, key: str, in_ch: int, out_ch: int, bank: nn.ModuleDict, ref: Tensor) -> nn.Conv2d:
        module = bank[key] if key in bank else None
        if module is None:
            # Keep projection heads in FP32 so GradScaler can safely unscale under AMP.
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False).to(device=ref.device, dtype=torch.float32)
            bank[key] = conv
            self._add_params(list(conv.parameters()))
            return conv
        if not isinstance(module, nn.Conv2d):
            raise RuntimeError(f"projection module for {key} is not Conv2d")
        return module.to(device=ref.device)

    def _get_proj_pair(self, layer: int, y: Tensor, d: Tensor, out_ch: int) -> tuple[Tensor, Tensor]:
        ky = f"y_{layer}"
        kd = f"d_{layer}"
        py = self._make_conv(ky, int(y.shape[1]), out_ch, self._proj_yolo, y)
        pd = self._make_conv(kd, int(d.shape[1]), out_ch, self._proj_dino, d)
        y_proj = py(y.to(dtype=py.weight.dtype))
        d_proj = pd(d.to(dtype=pd.weight.dtype))
        # Return in student dtype for downstream loss math consistency.
        return y_proj.to(dtype=y.dtype), d_proj.to(dtype=y.dtype)

    def feature_loss(
        self,
        *,
        teacher: TeacherOutputs,
        student_taps: dict[int, Tensor],
        masks_by_hw: dict[tuple[int, int], tuple[Tensor, Tensor]],
        active_weight: float,
    ) -> tuple[Tensor, dict[str, float], Tensor | None]:
        if active_weight <= 0.0:
            z = next(iter(student_taps.values())).sum() * 0.0
            return z, {}, None

        losses: list[Tensor] = []
        logs: dict[str, float] = {}
        signal_map: Tensor | None = None

        for layer in self.cfg.dino_layers:
            tap = int(self.cfg.dino_to_yolo_taps[layer])
            if tap not in student_taps:
                raise RuntimeError(f"missing student tap feature for layer {tap} (mapped from dino {layer})")
            y = student_taps[tap]
            if y.ndim != 4:
                raise RuntimeError(f"student tap {tap} must be BCHW, got {y.shape}")
            d = teacher.features[layer].to(device=y.device, dtype=y.dtype)
            d = F.interpolate(d, size=y.shape[-2:], mode="bilinear", align_corners=False)

            proj_ch = max(8, int(self.cfg.channels))
            y_proj, d_proj = self._get_proj_pair(layer, y, d, proj_ch)
            y_proj = F.normalize(y_proj, p=2.0, dim=1)
            d_proj = F.normalize(d_proj, p=2.0, dim=1)

            cos_dist = 1.0 - F.cosine_similarity(y_proj, d_proj, dim=1).unsqueeze(1)
            hw = (int(y.shape[-2]), int(y.shape[-1]))
            if hw not in masks_by_hw:
                raise RuntimeError(f"missing masks for resolution {hw}")
            obj_mask, bg_mask = masks_by_hw[hw]
            pix_w = float(self.cfg.object_weight) * obj_mask + float(self.cfg.background_weight) * bg_mask
            den = pix_w.sum().clamp_min(1.0)
            l = (cos_dist * pix_w).sum() / den
            lw = float(self.cfg.feat_layer_weights.get(layer, 1.0))
            losses.append(l * lw)

            logs[f"feat_l{layer}"] = float(l.detach().item())
            if signal_map is None:
                signal_map = (cos_dist[:, 0] * pix_w[:, 0]).detach()

        if not losses:
            z = next(iter(student_taps.values())).sum() * 0.0
            return z, logs, signal_map

        feat = torch.stack(losses).sum() * float(active_weight)
        return feat, logs, signal_map

    def attention_loss(
        self,
        *,
        teacher: TeacherOutputs,
        student_taps: dict[int, Tensor],
        active_weight: float,
        stage_attn_weight: float,
    ) -> tuple[Tensor, dict[str, float]]:
        if not self.cfg.attn_enabled or active_weight <= 0.0 or stage_attn_weight <= 0.0:
            z = next(iter(student_taps.values())).sum() * 0.0
            return z, {}
        if not teacher.attn:
            z = next(iter(student_taps.values())).sum() * 0.0
            return z, {}

        losses: list[Tensor] = []
        logs: dict[str, float] = {}
        hp, wp = teacher.patch_hw
        patch_n = hp * wp

        for layer in self.cfg.attn_layers:
            att = teacher.attn.get(layer)
            if att is None:
                continue
            if att.ndim != 4:
                raise RuntimeError(f"teacher attention for layer {layer} must be BHQQ, got {att.shape}")
            tap = int(self.cfg.dino_to_yolo_taps[layer])
            if tap not in student_taps:
                continue
            y = student_taps[tap]
            patch_start = int(att.shape[-1]) - patch_n
            if patch_start <= 0:
                continue
            cls_to_patch = att[:, :, 0, patch_start:]  # B,H,P
            dmap = cls_to_patch.mean(dim=1).reshape(att.shape[0], 1, hp, wp)
            dmap = F.interpolate(dmap, size=y.shape[-2:], mode="bilinear", align_corners=False)

            ymap = torch.sqrt(torch.mean(y.square(), dim=1, keepdim=True) + 1e-8)
            dprob = torch.softmax(dmap.flatten(1), dim=1)
            ylog = torch.log_softmax(ymap.flatten(1), dim=1)
            l = F.kl_div(ylog, dprob, reduction="batchmean")
            losses.append(l)
            logs[f"attn_l{layer}"] = float(l.detach().item())

        if not losses:
            z = next(iter(student_taps.values())).sum() * 0.0
            return z, logs

        loss = torch.stack(losses).mean() * float(stage_attn_weight) * float(active_weight)
        return loss, logs

    def objectness_map_loss(
        self,
        *,
        teacher: TeacherOutputs,
        preds_one2many: dict[str, Tensor | list[Tensor]],
        masks_by_hw: dict[tuple[int, int], tuple[Tensor, Tensor]],
        active_weight: float,
        stage_obj_weight: float,
    ) -> tuple[Tensor, dict[str, float]]:
        if not self.cfg.objmap_enabled or active_weight <= 0.0 or stage_obj_weight <= 0.0:
            feats = preds_one2many.get("feats")
            if isinstance(feats, list) and feats:
                z = feats[0].sum() * 0.0
                return z, {}
            raise RuntimeError("missing feats in one2many predictions")

        tokens = teacher.tokens
        if tokens is None or self.cfg.objmap_layer not in tokens:
            feats = preds_one2many.get("feats")
            if isinstance(feats, list) and feats:
                z = feats[0].sum() * 0.0
                return z, {}
            raise RuntimeError("missing teacher tokens")

        token_map = tokens[self.cfg.objmap_layer]
        hp, wp = teacher.patch_hw
        patch_n = hp * wp
        patch_start = int(token_map.shape[1]) - patch_n
        cls_tok = token_map[:, 0:1, :]
        patch_tok = token_map[:, patch_start:, :]
        cls_tok = F.normalize(cls_tok, p=2.0, dim=-1)
        patch_tok = F.normalize(patch_tok, p=2.0, dim=-1)
        sim = (patch_tok * cls_tok).sum(dim=-1)  # B,P
        sim = sim.reshape(sim.shape[0], 1, hp, wp)

        smin = sim.amin(dim=(2, 3), keepdim=True)
        smax = sim.amax(dim=(2, 3), keepdim=True)
        target = (sim - smin) / (smax - smin + 1e-6)
        target = target.pow(2.0)

        scores = preds_one2many.get("scores")
        feats = preds_one2many.get("feats")
        if not isinstance(scores, torch.Tensor) or not isinstance(feats, list) or not feats:
            raise RuntimeError("missing one2many scores/feats for objmap distillation")

        # Distill on highest-resolution head (P3 / stride-8): level 0
        level = 0
        counts = [int(f.shape[-2] * f.shape[-1]) for f in feats]
        start = 0
        selected: Tensor | None = None
        for li, c in enumerate(counts):
            end = start + c
            part = scores[:, :, start:end]
            if li == level:
                h, w = int(feats[li].shape[-2]), int(feats[li].shape[-1])
                selected = part.reshape(part.shape[0], part.shape[1], h, w)
                break
            start = end
        if selected is None:
            raise RuntimeError("unable to slice level-0 scores for objmap")

        logits = selected.max(dim=1).values.unsqueeze(1)
        target = F.interpolate(target.to(device=logits.device, dtype=logits.dtype), size=logits.shape[-2:], mode="bilinear", align_corners=False)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.cfg.objmap_apply_inside_gt_only:
            hw = (int(logits.shape[-2]), int(logits.shape[-1]))
            obj_mask, _ = masks_by_hw[hw]
            den = obj_mask.sum().clamp_min(1.0)
            l = (bce * obj_mask).sum() / den
        else:
            l = bce.mean()

        loss = l * float(stage_obj_weight) * float(active_weight)
        return loss, {f"obj_l{self.cfg.objmap_layer}": float(l.detach().item())}
