"""Запись субтитров на диск.

Пишется финальный файл с репликами и черновой — с сегментами Whisper
как есть. Черновик полезен для отладки таймингов: видно, что дала
модель, до нарезки и перевода.

Формат определяется расширением пути — его знает pysubs2. Поддержанное
нами: ``.srt`` (везде) и ``.vtt`` (браузеры, тег ``<track>``). Прочие
расширения pysubs2 не обязана понять, поэтому путь проверяется заранее
(``format_of``), а не после минут транскрипции на шаге записи.
"""

from __future__ import annotations

from pathlib import Path

import pysubs2

from subvox.asr import Segment
from subvox.cues import Cue

FORMATS = ("srt", "vtt")
"""Что мы обещаем писать. Всё прочее — вне контракта."""


def format_of(path: Path) -> str:
    """Формат по расширению пути; неизвестный — ошибка с текстом."""
    ext = path.suffix.lower().lstrip(".")
    if ext not in FORMATS:
        expected = " или ".join(f".{fmt}" for fmt in FORMATS)
        raise ValueError(f"неизвестный формат субтитров: {path.name} (ожидалось {expected})")
    return ext


def _events(rows: list[tuple[str, float, float]]) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    for text, start, end in rows:
        subs.events.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=start),
                end=pysubs2.make_time(s=end),
                text=text,
            )
        )
    return subs


def write(segments: tuple[Segment, ...], path: Path) -> Path:
    """Записать черновой SRT: сегменты Whisper как есть."""
    if not segments:
        raise ValueError("транскрипт пуст")
    rows = [(s.text, s.start, s.end) for s in segments]
    _events(rows).save(str(path))
    return path


def write_cues(cues: list[Cue], path: Path) -> Path:
    """Записать финальные субтитры из реплик."""
    if not cues:
        raise ValueError("реплик нет")
    rows = [(cue.text, cue.start, cue.end) for cue in cues]
    _events(rows).save(str(path))
    return path


def write_empty(path: Path) -> Path:
    """Записать пустой файл: речи нет, реплик нет, но покрытие полное.

    Ошибка «речи не найдено» остаётся видимой в CLI; пустой файл — не
    подстановка текста, а честный результат для видео без речи.
    """
    _events([]).save(str(path))
    return path
