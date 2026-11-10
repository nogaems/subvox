"""Нарезка реплик: переносы строк, длительности, CPS-предупреждения."""

from __future__ import annotations

import itertools

import pytest

from subvox import cues


def test_short_text_is_single_cue() -> None:
    result, skipped = cues.build([("Привет", 0.0, 1.5)])

    assert skipped == 0
    assert len(result) == 1
    assert result[0].text == "Привет"
    assert result[0].start == pytest.approx(0.0)
    assert result[0].end == pytest.approx(1.5)
    assert not result[0].too_fast


def test_two_lines_are_split_near_middle() -> None:
    text = "Это достаточно длинная реплика, которая не влезет в одну строку точно"
    result, _ = cues.build([(text, 0.0, 4.0)])

    assert len(result) == 1
    assert "\n" in result[0].text
    first, second = result[0].text.split("\n")
    assert len(first) <= cues.LINE_CHARS
    assert len(second) <= cues.LINE_CHARS


def test_wall_of_text_is_split_into_multiple_cues() -> None:
    text = " ".join(["слово"] * 40)
    result, _ = cues.build([(text, 0.0, 8.0)])

    assert len(result) >= 2
    for cue in result:
        assert len(cue.text) <= cues.LINE_CHARS * cues.MAX_LINES + 1
    assert result[0].start == pytest.approx(0.0)
    assert result[-1].end == pytest.approx(8.0)
    for previous, current in itertools.pairwise(result):
        assert current.start >= previous.start


def test_text_without_valid_break_is_not_left_on_one_line() -> None:
    """84 знака без границы ровно по середине — две реплики, не одна строка.

    Старый гейт резал только при ``len > 84``; текст на 84 знака с
    неудобной границей уходил в одну строку и мог обрезаться в кадре.
    """
    text = "а" * 43 + " " + "б" * 40
    result, skipped = cues.build([(text, 0.0, 4.0)])

    assert skipped == 0
    assert len(result) == 2
    assert all("\n" not in cue.text for cue in result)
    assert max(len(cue.text) for cue in result) < cues.UNRENDERABLE


def test_short_unbreakable_token_is_kept_as_overflow() -> None:
    """Токен 43–84 знака — переполнение строки, но не повод терять текст."""
    text = "а" * 50
    result, skipped = cues.build([(text, 0.0, 2.0)])

    assert skipped == 0
    assert len(result) == 1
    assert result[0].text == text


def test_unbreakable_token_longer_than_two_lines_is_skipped() -> None:
    """Бред модели без пробелов не лезет в субтитр — реплика пропускается."""
    noise = "ку-" * 40
    result, skipped = cues.build([(noise, 0.0, 2.0)])

    assert skipped == 1
    assert result == []


def test_skipped_cue_does_not_shift_neighbours() -> None:
    noise = "ку-" * 40
    result, skipped = cues.build(
        [
            ("привет", 0.0, 1.0),
            (noise, 2.0, 3.0),
            ("пока", 4.0, 5.0),
        ]
    )

    assert skipped == 1
    assert [cue.text for cue in result] == ["привет", "пока"]
    assert result[1].start == pytest.approx(4.0)


def test_cue_can_extend_into_gap() -> None:
    """Реплика может продлиться в паузу после неё."""
    result, _ = cues.build([("коротко", 0.0, 1.0)], next_start=5.0)

    # 1.0 минимум длительности + запас в паузу
    assert result[0].end >= 1.0


def test_cues_do_not_overlap() -> None:
    result, _ = cues.build(
        [
            ("первая реплика тут", 0.0, 2.0),
            ("вторая реплика тут", 2.2, 4.5),
        ]
    )

    for previous, current in itertools.pairwise(result):
        assert current.start - previous.end >= 0.0


def test_fast_cue_is_flagged() -> None:
    text = "Очень длинная реплика без знаков препинания вообще"
    result, _ = cues.build([(text, 0.0, 0.5)])

    assert result[0].too_fast


def test_translation_artifacts_are_cleaned() -> None:
    result, _ = cues.build([("Привет ? ! Как дела ?", 0.0, 2.0)])

    assert result[0].text == "Привет?! Как дела?"
