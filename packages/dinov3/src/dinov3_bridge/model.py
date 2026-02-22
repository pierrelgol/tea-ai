from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel

from .config import DinoV3Config


@dataclass(slots=True)
class DinoStats:
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class DinoV3Teacher:
    def __init__(self, cfg: DinoV3Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = AutoModel.from_pretrained(
            str(cfg.root),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.stats = self._load_stats(cfg.preprocessor_file)

    @staticmethod
    def _load_stats(path: Path) -> DinoStats:
        payload = json.loads(path.read_text(encoding="utf-8"))
        size_obj = payload.get("size") or {}
        image_size = int(size_obj.get("height", 224))
        mean = tuple(float(v) for v in payload.get("image_mean", [0.485, 0.456, 0.406]))
        std = tuple(float(v) for v in payload.get("image_std", [0.229, 0.224, 0.225]))
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("DINOv3 preprocessor_config.json must provide RGB mean/std length 3")
        return DinoStats(image_size=image_size, mean=mean, std=std)

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 1024))

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Tensor:
        x = images
        if x.dtype != torch.float32:
            x = x.float()
        x = F.interpolate(
            x,
            size=(self.stats.image_size, self.stats.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self.stats.mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.stats.std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model(pixel_values=x.to(self.device))
        last_hidden = out.last_hidden_state
        if last_hidden.ndim != 3 or last_hidden.shape[1] < 1:
            raise RuntimeError("DINOv3 output last_hidden_state has unexpected shape")
        cls_token = last_hidden[:, 0, :]
        return cls_token
