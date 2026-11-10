"""Определение языка.

Логика усреднения по окнам проверяется на подставной модели —
важно усреднение и честность признака неуверенности.
"""

from __future__ import annotations

import numpy as np
import pytest

from subvox import lid, vad


class FakeModel:
    """Модель, которая на каждое окно отдаёт заранее заданное распределение."""

    def __init__(self, distributions: list[list[tuple[str, float]]]) -> None:
        self._distributions = distributions
        self.calls = 0

    def detect_language(self, audio: np.ndarray | None = None, **_: object):
        distribution = self._distributions[self.calls % len(self._distributions)]
        self.calls += 1
        return distribution[0][0], distribution[0][1], distribution


@pytest.fixture
def two_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Два окна речи по 10 секунд, без запуска настоящего VAD."""
    monkeypatch.setattr(
        vad,
        "speech_regions",
        lambda audio, sample_rate: ([{"start": 0, "end": 100}], 20.0, 0.5),
    )
    monkeypatch.setattr(
        vad,
        "windows",
        lambda audio, regions, count, sample_rate: [np.zeros(10, np.float32)] * 2,
    )


def test_averages_probabilities_across_windows(two_windows: None) -> None:
    model = FakeModel(
        [
            [("en", 0.60), ("no", 0.40)],
            [("en", 0.80), ("no", 0.20)],
        ]
    )
    result = lid.detect(model, np.zeros(100, np.float32))

    assert result.language == "en"
    assert result.probability == pytest.approx(0.70)
    assert result.alternatives[0].code == "no"
    assert result.alternatives[0].probability == pytest.approx(0.30)
    assert result.windows == 2
    assert result.speech_seconds == pytest.approx(20.0)
    assert result.vad_threshold == 0.5


def test_close_race_is_not_confident(two_windows: None) -> None:
    """sv/no/da — ровно тот случай, где нельзя делать вид, что всё ясно."""
    model = FakeModel([[("sv", 0.40), ("no", 0.35), ("da", 0.25)]])
    result = lid.detect(model, np.zeros(100, np.float32))

    assert result.language == "sv"
    assert not result.confident


def test_clear_winner_is_confident(two_windows: None) -> None:
    model = FakeModel([[("ru", 0.95), ("uk", 0.03), ("bg", 0.02)]])
    result = lid.detect(model, np.zeros(100, np.float32))

    assert result.confident
