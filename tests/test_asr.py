"""Распознавание: фильтры галлюцинаций и модель данных.

Модель подставная — проверяется, что именно остаётся после фильтров
и что слова доходят до потребителя не искажёнными.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from subvox import asr, subs


def word(text: str, start: float, end: float, probability: float = 0.9):
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


def segment(
    text: str,
    start: float,
    end: float,
    *,
    words: list | None = None,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.05,
):
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=words if words is not None else [word(text, start, end)],
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class FakeModel:
    def __init__(self, segments: list, language: str = "ru", duration: float = 10.0):
        self._segments = segments
        self._language = language
        self._duration = duration
        self.kwargs: dict | None = None

    def transcribe(self, audio: np.ndarray, **kwargs):
        self.kwargs = kwargs
        info = SimpleNamespace(language=self._language, duration=self._duration)
        return iter(self._segments), info


def test_words_are_kept_with_timestamps() -> None:
    raw = segment(
        " привет мир", 1.0, 2.0, words=[word(" привет", 1.0, 1.4), word(" мир", 1.5, 2.0)]
    )
    result = asr.transcribe(FakeModel([raw]), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.text == "привет мир"
    assert [w.text for w in seg.words] == ["привет", "мир"]
    assert seg.words[0].start == pytest.approx(1.0)
    assert result.words == 2


def test_empty_text_is_dropped() -> None:
    raws = [segment(" ", 0.0, 1.0), segment("жизнь", 1.0, 2.0)]
    result = asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 1
    assert result.dropped_segments == 1


def test_silent_segment_is_dropped() -> None:
    """Музыка: модель не слышит речь и не уверена в тексте."""
    raws = [
        segment("какая-то музыка", 0.0, 5.0, avg_logprob=-1.5, no_speech_prob=0.8),
        segment("настоящая речь", 5.0, 7.0),
    ]
    result = asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 1
    assert result.segments[0].text == "настоящая речь"
    assert result.dropped_segments == 1


def test_suspicious_but_confident_text_is_kept() -> None:
    """Высокий no_speech_prob при уверенном тексте — не повод выбрасывать."""
    raw = segment("речь сквозь шум", 0.0, 5.0, avg_logprob=-0.5, no_speech_prob=0.9)
    result = asr.transcribe(FakeModel([raw]), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 1


def test_repetition_loop_is_collapsed() -> None:
    """Цикл: тот же текст с той же отметкой начала."""
    raws = [
        segment("спасибо за просмотр", 10.0, 12.0),
        segment("спасибо за просмотр", 10.0, 12.0),
        segment("спасибо за просмотр", 10.05, 12.0),
    ]
    result = asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 1
    assert result.duplicated_segments == 2


def test_genuine_repetition_is_kept() -> None:
    """Настоящее повторение — в другое время, не тот же старт."""
    raws = [
        segment("и снова привет", 1.0, 2.0),
        segment("и снова привет", 5.0, 6.0),
    ]
    result = asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")

    assert len(result.segments) == 2
    assert result.duplicated_segments == 0


def test_negative_start_is_clamped() -> None:
    raw = segment("речь", -0.5, 1.0)
    result = asr.transcribe(FakeModel([raw]), np.zeros(100, np.float32), language="ru")

    assert result.segments[0].start == 0.0


def test_transcription_passes_vad_threshold_through() -> None:
    model = FakeModel([segment("речь", 0.0, 1.0)])
    asr.transcribe(model, np.zeros(100, np.float32), language="ru", vad_threshold=0.1)

    assert model.kwargs["vad_parameters"]["threshold"] == 0.1
    assert model.kwargs["word_timestamps"] is True
    assert model.kwargs["condition_on_previous_text"] is False


def test_everything_dropped_is_an_error() -> None:
    raws = [segment(" ", 0.0, 1.0)]
    with pytest.raises(asr.TranscriptionError):
        asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")


def test_to_dict_rounds_and_includes_words() -> None:
    raw = segment("привет", 1.0, 2.0, words=[word(" привет", 1.0, 2.0)])
    result = asr.transcribe(FakeModel([raw]), np.zeros(100, np.float32), language="ru")
    data = asr.to_dict(result)

    assert data["language"] == "ru"
    assert data["segments"][0]["words"][0]["text"] == "привет"
    assert data["segments"][0]["start"] == 1.0


def test_draft_srt_is_readable_by_pysubs2(tmp_path) -> None:
    raws = [
        segment("первая реплика", 0.5, 2.0),
        segment("вторая реплика", 2.5, 4.0),
    ]
    result = asr.transcribe(FakeModel(raws), np.zeros(100, np.float32), language="ru")

    path = subs.write(result.segments, tmp_path / "draft.srt")
    loaded = subs.pysubs2.SSAFile.load(str(path))

    assert len(loaded.events) == 2
    assert loaded.events[0].start == pytest.approx(500)
    assert loaded.events[0].end == pytest.approx(2000)
