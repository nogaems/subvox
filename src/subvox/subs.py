"""Запись субтитров на диск.

Пишется финальный файл с репликами и черновой — с сегментами Whisper
как есть. Черновик полезен для отладки таймингов: видно, что дала
модель, до нарезки и перевода.
"""

from __future__ import annotations

from pathlib import Path

import pysubs2

from subvox.asr import Segment
from subvox.cues import Cue


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
    """Записать финальный SRT из реплик."""
    if not cues:
        raise ValueError("реплик нет")
    rows = [(cue.text, cue.start, cue.end) for cue in cues]
    _events(rows).save(str(path))
    return path
