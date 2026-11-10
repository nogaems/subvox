"""Нарезка переведённых предложений на реплики субтитров.

Единица отображения — реплика: до двух строк, до 42 символов в строке,
1–7 секунд. Это единственный способ, которым субтитр остаётся читаемым.

Тайминги здесь не подгоняются под длину перевода: длинный перевод
остаётся длинным, а реплика — сколько получилось. Пользователь всегда
может поставить паузу. Зато каждая реплика знает свой запас скорости
чтения (CPS) и максимальную длительность — задел под будущую озвучку.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE_CHARS = 42
"""Классический предел Netflix для латиницы и кириллицы."""

MAX_LINES = 2
CPS_SOFT = 21.0
"""Мягкий предел скорости чтения: символы в секунду. Выше — предупреждение."""

MIN_DURATION = 1.0
MAX_DURATION = 7.0
GAP_MIN = 0.1
"""Меньше — реплики сливаются в кашу на экране."""

SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Cue:
    text: str
    start: float
    end: float
    cps: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cps", self.chars / max(self.duration, 0.1))

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def too_fast(self) -> bool:
        return self.cps > CPS_SOFT


def _clean(text: str) -> str:
    """Убрать артефакты перевода, которые режут глаз в субтитре."""
    text = SPACES.sub(" ", text).strip()
    # Французская типографика NLLB: пробел перед знаком препинания.
    text = re.sub(r"\s+([!?.,;:»])", r"\1", text)
    # "? !" -> "?!"
    text = re.sub(r"([!?])\s+([!?.,])", r"\1\2", text)
    return text


def _split_lines(text: str) -> str:
    """Разбить реплику на строки — в точке, ближайшей к середине."""
    if len(text) <= LINE_CHARS:
        return text

    words = text.split(" ")
    best, best_delta = None, len(text)
    for i in range(1, len(words)):
        length = len(" ".join(words[:i]))
        if length > LINE_CHARS:
            break  # дальше только длиннее
        rest = len(text) - length - 1
        if rest > LINE_CHARS:
            continue  # эта точка не годится, ищем дальше
        delta = abs(length - rest)
        if delta < best_delta:
            best, best_delta = i, delta

    if best is None:
        return text
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def _split_cue(text: str, start: float, end: float) -> list[Cue]:
    """Разрезать реплику на две, если она не влезает в две строки."""
    limit = LINE_CHARS * MAX_LINES
    if len(text) <= limit:
        return [Cue(text=_split_lines(text), start=start, end=end)]

    duration = max(end - start, MIN_DURATION)
    per_char = duration / max(len(text), 1)

    chunks: list[str] = []
    words = text.split(" ")
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > LINE_CHARS and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)

    # Пары строк -> реплики; время делится по числу символов.
    cues: list[Cue] = []
    pairs = [chunks[i : i + MAX_LINES] for i in range(0, len(chunks), MAX_LINES)]
    for pair in pairs:
        part = " ".join(pair)
        span = per_char * len(part)
        cues.append(
            Cue(
                text=_split_lines(part),
                start=start,
                end=min(start + span, end),
            )
        )
        start += span

    if cues:
        last = cues[-1]
        cues[-1] = Cue(text=last.text, start=last.start, end=end)
    return cues


def build(
    entries: list[tuple[str, float, float]],
    next_start: float | None = None,
) -> list[Cue]:
    """Собрать реплики из (текст, начало, конец) с расширением в паузы.

    ``next_start`` — начало следующего предложения: реплика может
    продлиться в паузу после неё, но не дальше.
    """
    cues: list[Cue] = []
    for i, (text, start, end) in enumerate(entries):
        boundary = (
            min(next_start, entries[i + 1][1])
            if i + 1 < len(entries)
            else next_start
        ) if next_start is not None else None

        ceiling = boundary - GAP_MIN if boundary is not None else end
        ceiling = max(ceiling, end)

        text = _clean(text)
        if not text:
            continue
        if end - start < MIN_DURATION:
            end = min(start + MIN_DURATION, max(ceiling, start + MIN_DURATION))

        cues.extend(_split_cue(text, start, max(end, start + 0.1)))

    # Реплики не должны наезжать друг на друга.
    fixed: list[Cue] = []
    for i, cue in enumerate(cues):
        if i + 1 < len(cues):
            limit = cues[i + 1].start - GAP_MIN
            if cue.end > limit:
                cue = Cue(text=cue.text, start=cue.start, end=max(limit, cue.start + 0.1))
        fixed.append(cue)

    return fixed
