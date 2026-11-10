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

Скачивание тоже здесь: faster-whisper отключает прогресс-бары
(``disabled_tqdm``), и первый запуск выглядел бы как зависание
посреди гигабайтов тишины. Поэтому модель префетчится до загрузки —
с анонсом через ``note`` («что, куда, сколько») и живыми барами
``snapshot_download``; ``WhisperModel`` после этого находит её в кеше
и повторно не качает. Библиотечные модули ничего не печатают сами:
``note`` — колбэк вызывающего слоя.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from subvox import paths
from subvox.cuda import preload as _preload_cuda

DEFAULT_WHISPER = "large-v3"

WARMUP_SECONDS = 30.0
"""Столько Whisper обрабатывает за один проход; этого хватает, чтобы
задеть свёрточные слои и выяснить, работает ли cuDNN."""

APPROX_GBYTES: dict[str, float] = {
    # Приблизительные размеры для анонса скачивания; точность не критична.
    "tiny": 0.1,
    "base": 0.15,
    "small": 0.5,
    "medium": 1.5,
    "large-v1": 3.1,
    "large-v2": 3.1,
    "large-v3": 3.1,
    "large": 3.1,
    "large-v3-turbo": 1.6,
    "turbo": 1.6,
}


class ModelError(RuntimeError):
    """Модель не загрузилась ни на одном устройстве."""


@dataclass(frozen=True, slots=True)
class Loaded:
    model: Any
    device: str
    fallback_reason: str | None = None

    def release(self) -> None:
        """Отпустить модель, сохранив сведения о ней.

        Нужно перед загрузкой NLLB: Whisper large-v3 (~3 ГБ VRAM) и
        переводчик (~1.5 ГБ) в 8 ГБ не уживаются, и перевод падает с
        OOM, уходя на CPU. После транскрипции сама модель больше не
        нужна, а ``device``/``fallback_reason`` ещё нужны отчёту —
        поэтому обнуляется только ``model``.
        """
        object.__setattr__(self, "model", None)
        gc.collect()


def _warmup(model: Any, compute_type: str) -> None:
    """Настоящий прогон: загрузка модели ещё ничего не доказывает."""
    silence = np.zeros(int(WARMUP_SECONDS * 16_000), dtype=np.float32)
    model.detect_language(audio=silence)


def _repo_id(size: str) -> str:
    """repo-id Hugging Face для имени модели Whisper."""
    if "/" in size:
        return size
    from faster_whisper.utils import _MODELS

    try:
        return _MODELS[size]
    except KeyError:
        raise ValueError(
            f"неизвестная модель {size!r}; известны: {', '.join(sorted(_MODELS))}"
        ) from None


def _prefetch_whisper(size: str, note: Callable[[str], None] | None) -> None:
    """Скачать модель до загрузки, если её нет в кеше.

    Список файлов дублирует faster-whisper осознанно: если он изменится,
    недостающее молча докачает ``WhisperModel`` — корректность не пострадает,
    пострадает только тишина этого файла.
    """
    if os.path.isdir(size):
        return  # локальный путь — скачивать нечего

    from faster_whisper.utils import download_model
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache = str(paths.models_dir())
    try:
        download_model(size, cache_dir=cache, local_files_only=True)
        return  # уже в кеше
    except LocalEntryNotFoundError:
        pass

    if note is not None:
        approx = APPROX_GBYTES.get(size)
        size_part = f" (~{approx:.1f} ГБ)" if approx else ""
        note(f"модель Whisper {size}{size_part} качаю в {cache}")

    snapshot_download(
        _repo_id(size),
        cache_dir=cache,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ],
    )


def load_whisper(
    size: str = DEFAULT_WHISPER,
    device: str = "auto",
    compute_type: str | None = None,
    note: Callable[[str], None] | None = None,
) -> Loaded:
    """Загрузить модель Whisper и проверить, что она действительно считает."""
    _preload_cuda()

    try:
        _prefetch_whisper(size, note)
    except Exception as exc:
        raise ModelError(f"модель Whisper не скачалась в {paths.models_dir()}: {exc}") from exc

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
