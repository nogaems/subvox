"""Нарезка предложений: границы по пунктуации и паузам.

Слова подставные — важны тайминги и текст, не происхождение.
"""

from __future__ import annotations

import pytest

from subvox import sentences
from subvox.asr import Segment, Word


def segment(text: str, start: float, end: float) -> Segment:
    pieces = text.split(" ")
    step = (end - start) / max(len(pieces), 1)
    words = tuple(
        Word(
            start=start + i * step,
            end=start + (i + 1) * step - 0.1,
            text=piece,
            probability=1.0,
        )
        for i, piece in enumerate(pieces)
    )
    return Segment(
        start=start, end=end, text=text, words=words,
        avg_logprob=-0.3, no_speech_prob=0.05,
    )


def test_sentence_breaks_on_punctuation() -> None:
    segs = (segment("Один. Два. Три.", 0.0, 3.0),)
    parts = sentences.sentences(segs)

    assert [p.text for p in parts] == ["Один.", "Два.", "Три."]
    assert parts[0].start == pytest.approx(0.0)
    assert parts[2].end == pytest.approx(2.9, abs=0.05)


def test_sentence_breaks_on_pause() -> None:
    """Речь без точек, но с паузой — граница всё равно есть."""
    words = (
        Word(start=0.0, end=0.5, text="мы", probability=1.0),
        Word(start=0.6, end=1.1, text="ехали", probability=1.0),
        Word(start=3.5, end=4.0, text="долго", probability=1.0),
    )
    seg = Segment(start=0.0, end=4.0, text="мы ехали долго", words=words,
                  avg_logprob=-0.3, no_speech_prob=0.05)
    parts = sentences.sentences((seg,))

    assert len(parts) == 2
    assert parts[0].text == "мы ехали"
    assert parts[1].text == "долго"
    assert parts[1].start == pytest.approx(3.5)


def test_words_across_segments_flow_together() -> None:
    """Whisper разрезал фразу — предложение собирается через границу."""
    segs = (
        segment("и тогда он сказал что", 0.0, 1.0),
        segment("это конец", 1.0, 2.0),
    )
    parts = sentences.sentences(segs)

    assert len(parts) == 1
    assert parts[0].text == "и тогда он сказал что это конец"


def test_overlong_run_without_breaks_is_cut() -> None:
    """Стена текста без пауз и знаков режется, а не уходит в перевод как есть."""
    long = "слово " * 120
    segs = (segment(long.strip(), 0.0, 20.0),)
    parts = sentences.sentences(segs)

    assert len(parts) > 1
    assert all(len(p.text) <= sentences.MAX_SENTENCE_CHARS + 10 for p in parts)


def test_missing_word_timestamps_fall_back_to_segments() -> None:
    seg = Segment(start=1.0, end=3.0, text="просто фраза", words=(),
                  avg_logprob=-0.3, no_speech_prob=0.05)
    parts = sentences.sentences((seg,))

    assert len(parts) == 1
    assert parts[0].text == "просто фраза"
    assert parts[0].start == pytest.approx(1.0)


def test_empty_transcript_is_an_error() -> None:
    with pytest.raises(sentences.SentenceError):
        sentences.sentences(())
