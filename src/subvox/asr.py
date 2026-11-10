"""Распознавание речи в таймкоды.

Здесь получается сырой материал для всего остального: текст, границы слов
и сегментов. Единицы перевода (предложения) и единицы отображения (реплики)
строятся позже и отдельно — Whisper режет по своим окнам и рвёт фразы
посередине, поэтому его сегменты нельзя переводить напрямую.

Против галлюцинаций — три фильтра, все честно посчитаны в отчёте:

1. Пустой текст — сегмент есть, слов нет.
2. «Тихий» сегмент: высокая вероятность отсутствия речи при низкой
   уверенности в тексте. Так Whisper читает музыку и тишину.
3. Цикл повторов: тот же текст с той же отметкой начала. Настоящее
   повторение не может начинаться дважды в одну и ту же секунду.

``condition_on_previous_text=False`` — сознательный выбор: контекст
предыдущего окна даёт зацикливание на мусоре. ``hallucination_silence_threshold``
просит Whisper пропускать длинные паузы внутри подозрительных сегментов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NO_SPEECH_PROB = 0.5
"""Вероятность «здесь нет речи», выше которой сегмент подозрителен."""

AVG_LOGPROB = -1.0
"""Уверенность модели в тексте; ниже — подозрительно."""

DUPLICATE_START_EPSILON = 0.1
"""Насколько близко должны совпасть начала сегментов, чтобы считать их циклом."""

HALLUCINATION_SILENCE = 2.0
"""Паузы длиннее считаются местом галлюцинации и вырезаются."""

BEAM_SIZE = 5


class TranscriptionError(RuntimeError):
    """Распознавание не дало ни одного живого сегмента."""


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...]
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str
    duration: float
    segments: tuple[Segment, ...]
    dropped_segments: int
    duplicated_segments: int

    @property
    def words(self) -> int:
        return sum(len(segment.words) for segment in self.segments)


def transcribe(
    model: Any,
    audio: np.ndarray,
    *,
    language: str,
    vad_threshold: float = 0.5,
    beam_size: int = BEAM_SIZE,
) -> Transcript:
    """Распознать речь, вернув текст с пословными таймкодами."""
    iterator, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"threshold": vad_threshold},
        condition_on_previous_text=False,
        hallucination_silence_threshold=HALLUCINATION_SILENCE,
        beam_size=beam_size,
    )

    kept: list[Segment] = []
    dropped = 0
    duplicated = 0
    last: tuple[str, float] | None = None

    for raw in iterator:
        text = raw.text.strip()
        if not text:
            dropped += 1
            continue
        if raw.no_speech_prob >= NO_SPEECH_PROB and raw.avg_logprob <= AVG_LOGPROB:
            dropped += 1
            continue
        if (
            last is not None
            and text == last[0]
            and abs(raw.start - last[1]) < DUPLICATE_START_EPSILON
        ):
            duplicated += 1
            continue

        words = tuple(
            Word(
                start=word.start,
                end=word.end,
                text=word.word.strip(),
                probability=word.probability,
            )
            for word in raw.words or []
        )
        kept.append(
            Segment(
                start=max(raw.start, 0.0),
                end=max(raw.end, raw.start + 0.1),
                text=text,
                words=words,
                avg_logprob=raw.avg_logprob,
                no_speech_prob=raw.no_speech_prob,
            )
        )
        last = (text, raw.start)

    if not kept:
        raise TranscriptionError(
            "распознавание не дало ни одного сегмента: "
            f"отброшено {dropped} пустых или подозрительных, {duplicated} повторов"
        )

    return Transcript(
        language=info.language,
        duration=info.duration,
        segments=tuple(kept),
        dropped_segments=dropped,
        duplicated_segments=duplicated,
    )


def to_dict(transcript: Transcript) -> dict:
    """Машинное представление — основа кэша и обмена между стадиями."""
    return {
        "language": transcript.language,
        "duration": round(transcript.duration, 3),
        "dropped_segments": transcript.dropped_segments,
        "duplicated_segments": transcript.duplicated_segments,
        "segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text,
                "avg_logprob": round(segment.avg_logprob, 3),
                "no_speech_prob": round(segment.no_speech_prob, 3),
                "words": [
                    {
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "text": word.text,
                        "probability": round(word.probability, 3),
                    }
                    for word in segment.words
                ],
            }
            for segment in transcript.segments
        ],
    }


def from_dict(data: dict) -> Transcript:
    """Обратная сторона ``to_dict``: восстановление из манифеста.

    Битая запись падает исключением, а не тащит мусор в пайплайн —
    слой кеша ловит это и считает промахом.
    """
    segments = tuple(
        Segment(
            start=raw["start"],
            end=raw["end"],
            text=raw["text"],
            words=tuple(
                Word(
                    start=word["start"],
                    end=word["end"],
                    text=word["text"],
                    probability=word["probability"],
                )
                for word in raw["words"]
            ),
            avg_logprob=raw["avg_logprob"],
            no_speech_prob=raw["no_speech_prob"],
        )
        for raw in data["segments"]
    )
    if not segments:
        raise ValueError("запись транскрипта без сегментов")

    return Transcript(
        language=data["language"],
        duration=data["duration"],
        segments=segments,
        dropped_segments=data["dropped_segments"],
        duplicated_segments=data["duplicated_segments"],
    )
