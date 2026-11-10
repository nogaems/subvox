"""Нарезка переведённых предложений на реплики субтитров.

Единица отображения — реплика: до двух строк, до 42 символов в строке.
Сверху реплику ограничивает длина текста, а не таймер: перевод длиннее
оригинала остаётся длинным, и резать его по секундам значило бы рвать
фразу посреди мысли. Это единственный способ, которым субтитр остаётся
читаемым.

Короткая реплика при плотной речи — не ошибка: пользователь может
поставить паузу. Поэтому секунда — ориентир, а не запрет.

Один неразрывный токен длиннее двух строк — не субтитр, а бред модели
(артефакт распознавания или перевода). Такая реплика пропускается
целиком: разложить её по строкам нельзя, а резать и заменять текст
здесь нечем. Токены короче (немецкие композиты, URL) остаются: лучше
переполнить строку, чем потерять содержимое.

Тайминги здесь не подгоняются под длину перевода: длинный перевод
остаётся длинным, а реплика — сколько получилось. Пользователь всегда
может поставить паузу. Зато каждая реплика знает свой запас скорости
чтения (CPS) — его видно в отчёте и можно оценить, где текст плотный.
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
"""Ориентир: короче секунды читать неудобно, но плотная речь бывает быстрее."""

GAP_MIN = 0.1
"""Меньше — реплики сливаются в кашу на экране."""

UNRENDERABLE = LINE_CHARS * MAX_LINES
"""Неразрывный токен длиннее этого — артефакт: реплика пропускается."""

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


def _fits(text: str) -> bool:
    """Раскладывается ли текст в две строки по бюджету."""
    return all(len(line) <= LINE_CHARS for line in _split_lines(text).split("\n"))


def _balanced(words: list[str], count: int) -> list[str]:
    """Разделить слова на ``count`` кусков, почти равных по длине."""
    lengths = [0]
    for word in words:
        lengths.append(lengths[-1] + len(word) + 1)  # +1 — пробел
    total = lengths[-1] - 1

    bounds = [0]
    for part in range(1, count):
        target = total * part / count
        low = bounds[-1] + 1
        high = len(words) - (count - part)
        best = min(range(low, high + 1), key=lambda i: abs(lengths[i] - 1 - target))
        bounds.append(best)
    bounds.append(len(words))

    return [" ".join(words[bounds[i] : bounds[i + 1]]) for i in range(count)]


def _greedy(words: list[str]) -> list[str]:
    """Набивать строку словами до бюджета — запасной путь.

    Нужен, когда среди слов есть неразрывный токен длиннее строки:
    идеального деления не существует, но всё, кроме самого токена,
    остаётся в бюджете.
    """
    chunks: list[str] = []
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
    return chunks


def _parts(text: str) -> list[str]:
    """Разложить текст на минимальное число кусков, влезающих в две строки."""
    if _fits(text):
        return [text]

    words = text.split(" ")
    if len(words) > 1:
        # Меньше кусков, чем позволяет суммарная длина, не бывает —
        # начинаем с этой оценки, а не с двух.
        minimum = max(2, -(-len(text) // UNRENDERABLE))
        for count in range(minimum, len(words) + 1):
            parts = _balanced(words, count)
            if all(_fits(part) for part in parts):
                return parts

    return _greedy(words)


def _split_cue(text: str, start: float, end: float) -> list[Cue]:
    """Разложить реплику по строкам, разделив её на реплики, если нужно."""
    parts = _parts(text)
    if len(parts) == 1:
        return [Cue(text=_split_lines(text), start=start, end=end)]

    duration = max(end - start, MIN_DURATION)
    per_char = duration / max(len(text), 1)

    # Время делится по числу символов: куски почти равны, значит и
    # реплики почти равны по длительности.
    cues: list[Cue] = []
    for part in parts:
        span = per_char * len(part)
        cues.append(
            Cue(
                text=_split_lines(part),
                start=start,
                end=min(start + span, end),
            )
        )
        start += span

    last = cues[-1]
    cues[-1] = Cue(text=last.text, start=last.start, end=end)
    return cues


def build(
    entries: list[tuple[str, float, float]],
    next_start: float | None = None,
) -> tuple[list[Cue], int]:
    """Собрать реплики из (текст, начало, конец) с расширением в паузы.

    ``next_start`` — начало следующего предложения: реплика может
    продлиться в паузу после неё, но не дальше.

    Возвращает реплики и число пропущенных неотображаемых: строка
    длиннее двух строк — не субтитр, а бред модели
    (см. ``UNRENDERABLE``).
    """
    cues: list[Cue] = []
    skipped = 0
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

        # Неразрывный токен длиннее двух строк не рендерится ни в одной
        # реплике — такие реплики пропускаются, тайминги соседних не
        # сдвигаются. Токен 43–84 знака остаётся переполнением строки.
        parts = _split_cue(text, start, max(end, start + 0.1))
        renderable = [
            cue
            for cue in parts
            if all(len(line) <= UNRENDERABLE for line in cue.text.split("\n"))
        ]
        skipped += len(parts) - len(renderable)
        cues.extend(renderable)

    # Реплики не должны наезжать друг на друга.
    fixed: list[Cue] = []
    for i, cue in enumerate(cues):
        if i + 1 < len(cues):
            limit = cues[i + 1].start - GAP_MIN
            if cue.end > limit:
                cue = Cue(text=cue.text, start=cue.start, end=max(limit, cue.start + 0.1))
        fixed.append(cue)

    return fixed, skipped
