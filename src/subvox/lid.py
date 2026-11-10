"""Определение языка речи.

Один вызов ``detect_language`` смотрит первые 30 секунд файла. На видео
с заставкой, музыкой или титрами это врёт: модель уверенно называет язык
заставки. Поэтому здесь VAD режет речь, из неё берутся окна по всей
длительности, и вероятности усредняются.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subvox import langs, vad

CONFIDENCE = 0.60
"""Ниже этого top-1 считается неуверенным, и об этом надо сказать вслух."""

MARGIN = 0.15
"""Отрыв от второго места. Без него sv/no/da неразличимы."""


@dataclass(frozen=True, slots=True)
class LanguageScore:
    code: str
    name: str
    probability: float


@dataclass(frozen=True, slots=True)
class Detection:
    language: str
    name: str
    probability: float
    alternatives: tuple[LanguageScore, ...]
    windows: int
    speech_seconds: float
    duration: float
    vad_threshold: float

    @property
    def confident(self) -> bool:
        runner_up = self.alternatives[0].probability if self.alternatives else 0.0
        return self.probability >= CONFIDENCE and self.probability - runner_up >= MARGIN


def _label(code: str) -> str:
    lang = langs.get(code)
    return lang.name if lang else code


def detect(
    model,
    audio: np.ndarray,
    *,
    windows: int = 6,
    sample_rate: int = vad.SAMPLE_RATE,
) -> Detection:
    """Определить язык по нескольким окнам речи."""
    regions, speech_seconds, threshold = vad.speech_regions(audio, sample_rate)
    chunks = vad.windows(audio, regions, windows, sample_rate)

    totals: dict[str, float] = {}
    for chunk in chunks:
        _, _, probs = model.detect_language(audio=chunk)
        for code, probability in probs or []:
            totals[code] = totals.get(code, 0.0) + probability

    if not totals:
        raise vad.NoSpeech("модель не смогла определить язык ни на одном окне")

    # Каждое окно даёт распределение с суммой ~1, поэтому делим на их число.
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    scores = [
        LanguageScore(code=code, name=_label(code), probability=total / len(chunks))
        for code, total in ranked
    ]

    top = scores[0]
    return Detection(
        language=top.code,
        name=top.name,
        probability=top.probability,
        alternatives=tuple(scores[1:4]),
        windows=len(chunks),
        speech_seconds=speech_seconds,
        duration=len(audio) / sample_rate,
        vad_threshold=threshold,
    )
