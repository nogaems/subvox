"""Группировка слов в предложения.

Единица перевода — предложение, а не сегмент Whisper: модели портятся
на обрубках, а Whisper режет по своим окнам и рвёт фразы посередине.
Единица отображения — реплика субтитра — строится позже, из перевода.

Границы предложения ищутся там, где сошлись два признака: пунктуация
(?!… в конце слова) и пауза после него. Пауза без пунктуации тоже граница —
речь небрежная, а Whisper ставит точки неохотно. Без паузы и без знака
предложение продолжается, пока не упрётся в лимит: слишком длинное
предложение переводится хуже и не влезает в реплику.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subvox.asr import Segment, Word

MAX_SENTENCE_CHARS = 300
"""Предел для одного предложения перевода: длиннее — режем по паузе."""

PAUSE_SECONDS = 1.2
"""Пауза после слова, достаточная для границы предложения."""

TERMINATORS = re.compile(r"[.?!…:;]")
"""Знаки, заканчивающие слово."""


class SentenceError(RuntimeError):
    """Из слов не собралось ни одного предложения."""


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: float
    end: float
    words: tuple[Word, ...]

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.1)


def _is_break(word: Word, next_word: Word | None) -> bool:
    """Кончается ли слово на границе предложения."""
    if TERMINATORS.search(word.text[-1]):
        return True
    if next_word is None:
        return True
    return next_word.start - word.end >= PAUSE_SECONDS


def _cut_long(chunk: list[Word]) -> list[list[Word]]:
    """Слишком длинный кусок без знаков — режем по самой большой паузе."""
    if len(" ".join(w.text for w in chunk)) <= MAX_SENTENCE_CHARS:
        return [chunk]

    best_gap, best_at = -1.0, len(chunk) - 1
    for i in range(len(chunk) - 1):
        gap = chunk[i + 1].start - chunk[i].end
        if gap > best_gap:
            best_gap, best_at = gap, i

    left, right = chunk[: best_at + 1], chunk[best_at + 1 :]
    if not left or not right:  # паника: режем пополам
        middle = len(chunk) // 2
        left, right = chunk[:middle], chunk[middle:]

    return _cut_long(left) + _cut_long(right)


OPENING = ("(", "[", "«", '"', "'", "—", "-", "–")
"""Знаки, перед которыми пробел не нужен."""


def _join(chunk: list[Word]) -> str:
    """Собрать текст предложения из слов с учётом открывающих знаков."""
    parts: list[str] = []
    for word in chunk:
        text = word.text
        if not parts or text.startswith(OPENING):
            parts.append(text)
        else:
            parts.append(" " + text)
    return "".join(parts).strip()


def sentences(segments: tuple[Segment, ...]) -> list[Sentence]:
    """Собрать слова всех сегментов в предложения с таймингами."""
    words: list[Word] = []
    for segment in segments:
        words.extend(segment.words)

    if not words:
        # Слов с таймкодами не дали (word_timestamps не сработал) —
        # предложениями становятся сами сегменты.
        words = [
            Word(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                probability=1.0,
            )
            for segment in segments
        ]

    chunks: list[list[Word]] = []
    current: list[Word] = []

    for i, word in enumerate(words):
        current.append(word)
        if _is_break(word, words[i + 1] if i + 1 < len(words) else None):
            if current:
                chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    chunks = [part for chunk in chunks for part in _cut_long(chunk)]

    result: list[Sentence] = []
    for chunk in chunks:
        text = _join(chunk)
        if not text:
            continue
        result.append(
            Sentence(
                text=text,
                start=chunk[0].start,
                end=chunk[-1].end,
                words=tuple(chunk),
            )
        )

    if not result:
        raise SentenceError("из транскрипта не собралось ни одного предложения")
    return result
