from __future__ import annotations

from detector_train.cli import describe_tuner_state
from gpu_auto_tuner.cli import _estimate_seed_batch, _find_max_feasible_batch
from gpu_auto_tuner.search import binary_search_max_feasible
from gpu_auto_tuner.system import signature_from_parts


def test_signature_from_parts_is_stable() -> None:
    a = signature_from_parts(
        name="NVIDIA RTX 4090",
        total_vram_mb=24564,
        compute_capability="8.9",
        driver_version="550.54.14",
    )
    b = signature_from_parts(
        name="NVIDIA RTX 4090",
        total_vram_mb=24564,
        compute_capability="8.9",
        driver_version="550.54.14",
    )
    assert a == b
    assert len(a) == 16


def test_binary_search_max_feasible_finds_boundary() -> None:
    res = binary_search_max_feasible(
        low=1, high=32, is_feasible=lambda x: x <= 19
    )
    assert res.best_value == 19
    assert res.attempts


def test_binary_search_returns_none_when_no_value_feasible() -> None:
    res = binary_search_max_feasible(low=3, high=7, is_feasible=lambda _: False)
    assert res.best_value is None


def test_seed_batch_scales_for_smaller_images_and_large_cards() -> None:
    seed = _estimate_seed_batch(
        combo={"imgsz": 512},
        train={"batch": 32, "imgsz": 1024},
        batch_min=1,
        batch_cap=256,
    )
    assert seed == 128


def test_seeded_batch_search_reaches_large_boundary() -> None:
    best, attempts = _find_max_feasible_batch(
        batch_min=1,
        batch_cap=1024,
        seed_batch=128,
        remaining_trials=16,
        is_feasible=lambda x: x <= 300,
    )
    assert best == 300
    assert attempts[0] == 128
    assert len(attempts) <= 16


def test_describe_tuner_state_allows_training_without_tuning(monkeypatch) -> None:
    monkeypatch.setattr("detector_train.cli.resolve_device", lambda _device: "0")
    shared = type(
        "Shared",
        (),
        {
            "tuner": {"enabled": True},
            "train": {"device": "auto", "tuned_gpu_signature": None},
        },
    )()
    note = describe_tuner_state(shared)
    assert note is not None
    assert "proceeding without tuned gpu profile" in note.lower()
