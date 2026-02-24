from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb

from dinov3_bridge import DinoV3Teacher, resolve_local_dinov3_root


@dataclass(slots=True)
class TeacherOutputs:
    features: dict[int, Tensor]  # BCHW
    attn: dict[int, Tensor] | None  # B,H,T,T
    tokens: dict[int, Tensor] | None  # B,T,C
    patch_hw: tuple[int, int]


class DinoTeacherAdapter:
    def __init__(
        self,
        *,
        dino_root: Path,
        device: torch.device,
        feature_layers: tuple[int, ...],
        attn_layers: tuple[int, ...],
    ) -> None:
        self._teacher = DinoV3Teacher(resolve_local_dinov3_root(dino_root), device=device)
        self._teacher.model.eval()
        self._teacher.model.requires_grad_(False)
        self._device = device
        self._feature_layers = tuple(dict.fromkeys(int(v) for v in feature_layers))
        self._attn_layers = tuple(dict.fromkeys(int(v) for v in attn_layers))
        self._attn_cache: dict[int, Tensor] = {}
        self._handles = []
        self._register_attn_hooks()

    @property
    def teacher(self) -> DinoV3Teacher:
        return self._teacher

    @property
    def patch_skip_tokens(self) -> int:
        return 1 + max(0, int(self._teacher.stats.num_register_tokens))

    def _register_attn_hooks(self) -> None:
        if not self._attn_layers:
            return

        layer_count = len(getattr(self._teacher.model, "layer", []))
        for layer_idx in self._attn_layers:
            if layer_idx < 0 or layer_idx >= layer_count:
                raise ValueError(f"invalid DINO attention layer {layer_idx} for layer_count={layer_count}")
            module = self._teacher.model.layer[layer_idx].attention

            def _make_hook(idx: int):
                def _hook(attn_module, inputs, kwargs, _output):
                    if not inputs:
                        return
                    hidden_states = inputs[0]
                    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                        return
                    b, t, _ = hidden_states.shape
                    q = attn_module.q_proj(hidden_states)
                    k = attn_module.k_proj(hidden_states)
                    q = q.view(b, t, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
                    k = k.view(b, t, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
                    pos = kwargs.get("position_embeddings")
                    if isinstance(pos, tuple) and len(pos) == 2:
                        q, k = apply_rotary_pos_emb(q, k, pos[0], pos[1])
                    logits = torch.matmul(q, k.transpose(-2, -1)) * float(attn_module.scaling)
                    self._attn_cache[idx] = torch.softmax(logits, dim=-1)

                return _hook

            self._handles.append(module.register_forward_hook(_make_hook(layer_idx), with_kwargs=True))

    @torch.no_grad()
    def forward(self, images: Tensor) -> TeacherOutputs:
        self._teacher.model.eval()
        self._attn_cache.clear()

        x = self._teacher._prepare(images)
        out = self._teacher.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("DINO teacher did not return hidden_states")

        all_layers = tuple(dict.fromkeys(self._feature_layers + self._attn_layers))
        features: dict[int, Tensor] = {}
        tokens: dict[int, Tensor] = {}

        patch_hw: tuple[int, int] | None = None
        skip = self.patch_skip_tokens

        for layer_idx in all_layers:
            hs_idx = int(layer_idx) + 1  # hidden_states[0] is embedding output
            if hs_idx < 0 or hs_idx >= len(hidden_states):
                raise ValueError(f"invalid hidden state index for layer {layer_idx}; count={len(hidden_states)-1}")
            layer_tokens = hidden_states[hs_idx]
            if layer_tokens.ndim != 3:
                raise RuntimeError(f"unexpected DINO hidden state shape at layer {layer_idx}: {layer_tokens.shape}")
            tokens[layer_idx] = layer_tokens

            if layer_tokens.shape[1] <= skip:
                raise RuntimeError("DINO hidden states do not contain patch tokens")
            patch_tokens = layer_tokens[:, skip:, :]
            n_patches = int(patch_tokens.shape[1])
            side = int(round(n_patches ** 0.5))
            if side * side != n_patches:
                raise RuntimeError(f"DINO patch token count is not square: {n_patches}")
            fmap = patch_tokens.transpose(1, 2).reshape(layer_tokens.shape[0], layer_tokens.shape[2], side, side)
            features[layer_idx] = fmap
            if patch_hw is None:
                patch_hw = (side, side)

        if patch_hw is None:
            raise RuntimeError("failed to resolve teacher patch shape")

        attn_out: dict[int, Tensor] | None = None
        if self._attn_layers:
            attn_out = {k: v for k, v in self._attn_cache.items() if k in self._attn_layers}

        return TeacherOutputs(
            features=features,
            attn=attn_out,
            tokens=tokens,
            patch_hw=patch_hw,
        )

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:
        self.close()
