"""Извлечение звука: тесты на настоящих файлах через ffmpeg.

ffmpeg — системная зависимость, поэтому эти тесты проверяют не моки,
а реальный контракт: что именно приходит в Whisper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from subvox import audio


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    """Стерео 44.1 кГц — как в типичном видео, а не как в тестовом наборе."""
    path = tmp_path / "tone.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-ac", "2", "-ar", "44100", str(path),
    )
    return path


@pytest.fixture
def silent_video(tmp_path: Path) -> Path:
    path = tmp_path / "silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
        "-an", str(path),
    )
    return path


def test_probe_reads_streams(tone: Path) -> None:
    info = audio.probe(tone)
    assert info.has_audio
    assert info.duration == pytest.approx(1.0, abs=0.1)
    assert len(info.streams) == 1
    assert info.streams[0].channels == 2
    assert info.streams[0].sample_rate == 44100


def test_probe_reports_missing_audio(silent_video: Path) -> None:
    info = audio.probe(silent_video)
    assert not info.has_audio
    with pytest.raises(audio.NoAudioStream):
        info.stream(0)


def test_extract_downmixes_to_whisper_format(tone: Path) -> None:
    samples = audio.extract(tone)

    assert samples.dtype == np.float32
    assert samples.ndim == 1
    # 1 секунда при 16 кГц, допуск на округление кадров.
    assert len(samples) == pytest.approx(audio.SAMPLE_RATE, rel=0.02)
    # Диапазон float32: сигнал не клиппирован и не тишина.
    assert 0.01 < float(np.abs(samples).max()) <= 1.0


def test_extract_selects_track(tmp_path: Path) -> None:
    left = tmp_path / "left.wav"
    right = tmp_path / "right.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=1", str(left))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=1200:duration=1", str(right))

    mixed = tmp_path / "two.mka"
    _ffmpeg("-i", str(left), "-i", str(right), "-map", "0:a", "-map", "1:a", "-c", "copy", str(mixed))

    info = audio.probe(mixed)
    assert len(info.streams) == 2

    first = audio.extract(mixed, info.stream(0))
    second = audio.extract(mixed, info.stream(1))
    assert not np.array_equal(first, second)


def test_missing_track_is_a_clear_error(tone: Path) -> None:
    info = audio.probe(tone)
    with pytest.raises(audio.AudioError, match="запрошена 3"):
        info.stream(3)
