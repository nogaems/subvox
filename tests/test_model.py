"""Владение моделью: освобождение VRAM перед переводчиком.

Whisper large-v3 (~3 ГБ VRAM) и NLLB (~1.5 ГБ) в 8 ГБ не уживаются:
перевод падает с OOM и уходит на CPU. После транскрипции сама модель
больше не нужна, но её устройство и причина фолбэка ещё нужны отчёту —
поэтому release() обнуляет только модель.
"""

from __future__ import annotations

from subvox.model import Loaded


def test_release_frees_model_but_keeps_report_fields() -> None:
    loaded = Loaded(model=object(), device="cuda", fallback_reason=None)

    loaded.release()

    assert loaded.model is None
    assert loaded.device == "cuda"
    assert loaded.fallback_reason is None


def test_release_keeps_fallback_reason() -> None:
    loaded = Loaded(model=object(), device="cpu", fallback_reason="cuda: boom")

    loaded.release()

    assert loaded.model is None
    assert loaded.device == "cpu"
    assert loaded.fallback_reason == "cuda: boom"


def test_release_is_idempotent() -> None:
    loaded = Loaded(model=object(), device="cuda")

    loaded.release()
    loaded.release()

    assert loaded.model is None
