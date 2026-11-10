"""Silero VAD: поиск речи в аудио.

Живёт отдельным модулем, потому что нужен двум стадиям — определению
языка и распознаванию — и не должен принадлежать ни одной из них.

Лестница порогов существует не для красоты. Silero уверенно находит
чистую речь и отказывается находить речь поверх громкой музыки:
типичный мем-ролик с репом на фоне он помечает как тишину. Если на
стандартном пороге речи нет, порог понижается, а факт понижения
возвращается наружу — пользователь должен знать, что аудио шумное.
"""

from __future__ import annotations

import numpy as np

from subvox.audio import SAMPLE_RATE

THRESHOLDS: tuple[float, ...] = (0.5, 0.35, 0.2, 0.1)
"""Порог Silero VAD по умолчанию — 0.5. Дальше — для шумного аудио."""

MIN_SPEECH_SECONDS = 5.0
"""Меньше пяти секунд речи — определение языка превращается в угадайку."""

WINDOW_SECONDS = 30.0
"""Whisper видит не больше 30 секунд за раз — длиннее окно всё равно обрежется."""


class NoSpeech(RuntimeError):
    """Речи нет: тишина, музыка или шум."""


def _find(audio: np.ndarray, threshold: float, sample_rate: int) -> list[dict]:
    """Один прогон Silero VAD. Отдельная функция — для подмены в тестах."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(
        audio, VadOptions(threshold=threshold), sampling_rate=sample_rate
    )


def speech_regions(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[list[dict], float, float]:
    """Найти речь, понижая порог VAD, если не получилось.

    Возвращает регионы речи, их суммарную длительность в секундах
    и порог, на котором речь нашлась.
    """
    enough = MIN_SPEECH_SECONDS * sample_rate
    default = THRESHOLDS[0]
    default_regions: list[dict] = []
    default_total = 0

    for threshold in THRESHOLDS:
        regions = _find(audio, threshold, sample_rate)
        total = sum(region["end"] - region["start"] for region in regions)
        if threshold == default:
            default_regions, default_total = regions, total
        if total >= enough:
            return regions, total / sample_rate, threshold

    # Лестница не помогла, но на стандартном пороге что-то было —
    # редкая, но настоящая речь. Берём её, не маскируя под тишину.
    if default_total > 0:
        return default_regions, default_total / sample_rate, default

    raise NoSpeech(
        "речь не найдена даже на минимальном пороге VAD: тишина, музыка или шум"
    )


def windows(
    audio: np.ndarray,
    regions: list[dict],
    count: int = 6,
    sample_rate: int = SAMPLE_RATE,
) -> list[np.ndarray]:
    """Нарезать до ``count`` окон, размазанных по найденной речи."""
    total = sum(region["end"] - region["start"] for region in regions)
    if total <= 0:
        raise NoSpeech("речи нет")

    max_window = int(WINDOW_SECONDS * sample_rate)
    min_window = int(MIN_SPEECH_SECONDS * sample_rate)

    fits = max(1, int(total // min_window))
    count = max(1, min(count, fits))

    chunks: list[np.ndarray] = []
    for i in range(count):
        target = (i + 0.5) * total / count
        consumed = 0
        start = regions[-1]["start"]
        for region in regions:
            length = region["end"] - region["start"]
            if consumed + length >= target:
                start = region["start"] + int(target - consumed)
                break
            consumed += length

        end = min(start + max_window, len(audio))
        if end - start < min_window:
            start = max(0, end - min_window)
        chunks.append(audio[start:end])

    return chunks
