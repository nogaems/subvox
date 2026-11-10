"""Нарезка реплик: переносы строк, длительности, CPS-предупреждения."""

from __future__ import annotations

import itertools

import pytest

from subvox import cues


def test_short_text_is_single_cue() -> None:
    result = cues.build([("Привет", 0.0, 1.5)])

    assert len(result) == 1
    assert result[0].text == "Привет"
    assert result[0].start == pytest.approx(0.0)
    assert result[0].end == pytest.approx(1.5)
    assert not result[0].too_fast


def test_two_lines_are_split_near_middle() -> None:
    text = "Это достаточно длинная реплика, которая не влезет в одну строку точно"
    result = cues.build([(text, 0.0, 4.0)])

    assert len(result) == 1
    assert "\n" in result[0].text
    first, second = result[0].text.split("\n")
    assert len(first) <= cues.LINE_CHARS
    assert len(second) <= cues.LINE_CHARS


def test_wall_of_text_is_split_into_multiple_cues() -> None:
    text = " ".join(["слово"] * 40)
    result = cues.build([(text, 0.0, 8.0)])

    assert len(result) >= 2
    for cue in result:
        assert len(cue.text) <= cues.LINE_CHARS * cues.MAX_LINES + 1
    assert result[0].start == pytest.approx(0.0)
    assert result[-1].end == pytest.approx(8.0)
    for previous, current in itertools.pairwise(result):
        assert current.start >= previous.start


def test_cue_can_extend_into_gap() -> None:
    """Реплика может продлиться в паузу после неё."""
    result = cues.build([("коротко", 0.0, 1.0)], next_start=5.0)

    # 1.0 минимум длительности + запас в паузу
    assert result[0].end >= 1.0


def test_cues_do_not_overlap() -> None:
    result = cues.build(
        [
            ("первая реплика тут", 0.0, 2.0),
            ("вторая реплика тут", 2.2, 4.5),
        ]
    )

    for previous, current in itertools.pairwise(result):
        assert current.start - previous.end >= 0.0


def test_fast_cue_is_flagged() -> None:
    text = "Очень длинная реплика без знаков препинания вообще"
    result = cues.build([(text, 0.0, 0.5)])

    assert result[0].too_fast


def test_translation_artifacts_are_cleaned() -> None:
    result = cues.build([("Привет ? ! Как дела ?", 0.0, 2.0)])

    assert result[0].text == "Привет?! Как дела?"
