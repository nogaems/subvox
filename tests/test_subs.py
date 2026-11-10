"""Запись субтитров: один код, два формата — SRT и VTT.

Формат определяется расширением пути, писатель один. Проверяются
гарантии форматов (заголовок WEBVTT, разделитель миллисекунд,
переносы строк) и раундтрип: записанное читается обратно без потерь.
"""

from __future__ import annotations

from pathlib import Path

import pysubs2
import pytest

from subvox import subs
from subvox.asr import Segment, Word
from subvox.cues import Cue


@pytest.fixture
def cue() -> list[Cue]:
    return [
        Cue(text="Привет,\\Nмир", start=1.5, end=4.0),
        Cue(text="Вторая реплика", start=10.0, end=12.0),
    ]


def _segments() -> tuple[Segment, ...]:
    return (
        Segment(
            start=0.5,
            end=2.0,
            text="первый",
            words=(Word(0.5, 2.0, "первый", 0.9),),
            avg_logprob=-0.3,
            no_speech_prob=0.05,
        ),
    )


def test_srt_writes_comma_milliseconds(tmp_path: Path, cue: list[Cue]) -> None:
    path = tmp_path / "out.srt"
    subs.write_cues(cue, path)

    text = path.read_text(encoding="utf-8")
    assert "00:00:01,500 --> 00:00:04,000" in text
    assert "Привет,\nмир" in text  # \N стал переносом строки


def test_vtt_writes_header_and_dots(tmp_path: Path, cue: list[Cue]) -> None:
    path = tmp_path / "out.vtt"
    subs.write_cues(cue, path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "00:00:01.500 --> 00:00:04.000" in text
    assert "Привет,\nмир" in text


def test_vtt_roundtrip_keeps_timings_and_lines(tmp_path: Path, cue: list[Cue]) -> None:
    path = tmp_path / "out.vtt"
    subs.write_cues(cue, path)

    loaded = pysubs2.SSAFile.load(str(path))
    assert len(loaded.events) == 2
    first = loaded.events[0]
    assert first.start == 1500 and first.end == 4000
    assert first.plaintext == "Привет,\nмир"


def test_empty_subtitle_is_valid(tmp_path: Path) -> None:
    """Видео без речи: файл есть, реплик ноль.

    Пустой SRT — это файл из 0 байт (блоков нет); VTT обязан нести
    заголовок ``WEBVTT``, иначе браузер его не примет.
    """
    srt = tmp_path / "empty.srt"
    vtt = tmp_path / "empty.vtt"

    subs.write_empty(srt)
    subs.write_empty(vtt)

    assert srt.is_file()
    assert srt.read_text(encoding="utf-8") == ""
    assert len(pysubs2.SSAFile.load(str(vtt)).events) == 0
    assert vtt.read_text(encoding="utf-8").startswith("WEBVTT")


def test_draft_segments_write_both_formats(tmp_path: Path) -> None:
    srt = tmp_path / "draft.srt"
    vtt = tmp_path / "draft.vtt"

    subs.write(_segments(), srt)
    subs.write(_segments(), vtt)

    assert srt.read_text(encoding="utf-8").startswith("1")
    assert vtt.read_text(encoding="utf-8").startswith("WEBVTT")


def test_format_of_accepts_supported(tmp_path: Path) -> None:
    assert subs.format_of(tmp_path / "a.srt") == "srt"
    assert subs.format_of(tmp_path / "a.VTT") == "vtt"  # регистр не важен


def test_format_of_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="txt"):
        subs.format_of(tmp_path / "a.txt")

    with pytest.raises(ValueError, match="формат"):
        subs.format_of(tmp_path / "noext")
