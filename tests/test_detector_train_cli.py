from __future__ import annotations

import sys
import types

from detector_train.cli import _resolve_model_arg


def test_resolve_model_arg_downloads_openvision_repo_id(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_hf_hub_download(*, repo_id: str, filename: str) -> str:
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return "/tmp/model.pt"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    resolved = _resolve_model_arg("openvision/yolo26-m-obb")

    assert resolved == "/tmp/model.pt"
    assert captured == {"repo_id": "openvision/yolo26-m-obb", "filename": "model.pt"}
