"""Лестница порогов VAD.

Проверяется на подменном ``_find``: сам Silero в тестах не нужен,
важна логика понижения порога и честность отказа.
"""

from __future__ import annotations

import numpy as np
import pytest

from subvox import vad


def _fake_finder(by_threshold: dict[float, list[tuple[float, float]]]):
    """threshold -> регионы речи в секундах."""

    def _find(audio: np.ndarray, threshold: float, sample_rate: int) -> list[dict]:
        return [
            {"start": int(start * sample_rate), "end": int(end * sample_rate)}
            for start, end in by_threshold.get(threshold, [])
        ]

    return _find


def _audio(seconds: float = 15.0) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def test_speech_over_music_lowers_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Мем-ролик: на 0.5 речи нет совсем, на 0.1 — есть."""
    monkeypatch.setattr(
        vad,
        "_find",
        _fake_finder({0.5: [], 0.35: [(1.0, 1.4)], 0.2: [(1.0, 2.2)], 0.1: [(0, 2.0), (5.0, 9.3)]}),
    )
    regions, total, threshold = vad.speech_regions(_audio())

    assert threshold == 0.1
    assert total == pytest.approx(6.3)
    assert len(regions) == 2


def test_clean_speech_keeps_default_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad, "_find", _fake_finder({0.5: [(0, 8.0)]}))
    _regions, total, threshold = vad.speech_regions(_audio())

    assert threshold == 0.5
    assert total == pytest.approx(8.0)


def test_sparse_speech_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Редкая короткая реплика: лестница не помогла, но на 0.5 речь была."""
    monkeypatch.setattr(
        vad,
        "_find",
        _fake_finder({0.5: [(2.0, 5.0)], 0.35: [(2.0, 5.1)], 0.2: [(2.0, 5.2)], 0.1: [(2.0, 5.3)]}),
    )
    _regions, total, threshold = vad.speech_regions(_audio())

    assert threshold == 0.5
    assert total == pytest.approx(3.0)


def test_silence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad, "_find", _fake_finder({}))
    with pytest.raises(vad.NoSpeech):
        vad.speech_regions(_audio())


def test_click_shorter_than_fallback_floor_is_not_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0.9 с шума на стандартном пороге — не «редкая речь» (fail-nospeech)."""
    monkeypatch.setattr(vad, "_find", _fake_finder({0.5: [(10.0, 10.9)]}))

    with pytest.raises(vad.NoSpeech):
        vad.speech_regions(_audio(seconds=30.0))


def test_real_vad_finds_nothing_in_silence() -> None:
    with pytest.raises(vad.NoSpeech):
        vad.speech_regions(np.zeros(3 * 16_000, dtype=np.float32))


def test_windows_are_spread_over_speech() -> None:
    regions = [
        {"start": 0, "end": 16000 * 10},
        {"start": 16000 * 30, "end": 16000 * 40},
    ]
    chunks = vad.windows(np.zeros(16000 * 50, np.float32), regions, count=6)

    # 20 секунд речи при минимуме 5 — не больше четырёх окон.
    assert len(chunks) == 4
    assert all(len(chunk) <= int(vad.WINDOW_SECONDS * 16_000) for chunk in chunks)
    assert all(len(chunk) >= int(vad.MIN_SPEECH_SECONDS * 16_000) for chunk in chunks)


def test_short_speech_yields_one_window() -> None:
    regions = [{"start": 0, "end": 16000 * 3}]
    chunks = vad.windows(np.zeros(16000 * 10, np.float32), regions, count=6)

    assert len(chunks) == 1
