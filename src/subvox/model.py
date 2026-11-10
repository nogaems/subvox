"""Загрузка моделей и выбор устройства.

Две ловушки, обе проверены на практике:

1. CTranslate2 на GPU требует CUDA 12 и cuDNN 9. Система может иметь CUDA 13,
   и тогда модель *загрузится*, но упадёт на первом вычислении с
   ``libcublas.so.12 is not found``. Поэтому проверка устройства — реальный
   прогон, а не факт создания объекта.
2. Нужные библиотеки можно положить в venv питоновскими колёсами
   (``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12``) и не трогать систему.
   Но ``LD_LIBRARY_PATH`` читается загрузчиком при старте процесса, поэтому
   библиотеки подгружаются заранее через ``ctypes`` с ``RTLD_GLOBAL`` —
   дальше ``dlopen`` по имени находит уже загруженные.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from subvox import paths
from subvox.cuda import preload as _preload_cuda

DEFAULT_WHISPER = "large-v3"

WARMUP_SECONDS = 30.0
"""Столько Whisper обрабатывает за один проход; этого хватает, чтобы
задеть свёрточные слои и выяснить, работает ли cuDNN."""


class ModelError(RuntimeError):
    """Модель не загрузилась ни на одном устройстве."""


@dataclass(frozen=True, slots=True)
class Loaded:
    model: Any
    device: str
    fallback_reason: str | None = None


def _warmup(model: Any, compute_type: str) -> None:
    """Настоящий прогон: загрузка модели ещё ничего не доказывает."""
    silence = np.zeros(int(WARMUP_SECONDS * 16_000), dtype=np.float32)
    model.detect_language(audio=silence)


def load_whisper(
    size: str = DEFAULT_WHISPER,
    device: str = "auto",
    compute_type: str | None = None,
) -> Loaded:
    """Загрузить модель Whisper и проверить, что она действительно считает."""
    _preload_cuda()

    from faster_whisper import WhisperModel

    candidates: list[tuple[str, str]] = []
    if device in ("auto", "cuda"):
        candidates.append(("cuda", compute_type or "float16"))
    if device in ("auto", "cpu"):
        candidates.append(("cpu", compute_type or "int8"))
    if not candidates:
        raise ValueError(f"неизвестное устройство: {device!r}")

    errors: list[str] = []
    for dev, ctype in candidates:
        try:
            model = WhisperModel(
                size,
                device=dev,
                compute_type=ctype,
                download_root=str(paths.models_dir()),
            )
            _warmup(model, ctype)
        except Exception as exc:  # noqa: BLE001 — причин может быть много
            errors.append(f"{dev}/{ctype}: {exc}")
            continue
        return Loaded(model=model, device=dev, fallback_reason=errors[-1] if errors else None)

    raise ModelError("не удалось загрузить модель — " + "; ".join(errors))
