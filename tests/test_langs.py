"""Инварианты таблицы языков.

Тесты держат то, что легко сломать незаметно: полноту покрытия Whisper
и отсутствие «тихого» превращения непереводимого языка в переводимый.
"""

from __future__ import annotations

import pytest

from subvox import langs

# 99 языков Whisper плюс отдельный nn рядом с no.
WHISPER_LANGUAGES = 100


def test_every_whisper_language_is_present() -> None:
    assert len(langs.LANGUAGES) == WHISPER_LANGUAGES


def test_targets_are_translatable() -> None:
    assert langs.translatable("en")
    assert langs.translatable("ru")
    assert "en" in langs.targets()
    assert "ru" in langs.targets()


def test_untranslatable_are_reported_not_silently_english() -> None:
    assert langs.untranslatable() == ["br", "haw", "la", "mg"]


def test_flores_codes_are_unique() -> None:
    codes = [lang.flores for lang in langs.LANGUAGES.values() if lang.flores]
    assert len(codes) == len(set(codes))


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("sv", "sv"),
        ("sv-SE", "sv"),
        ("sv_SE", "sv"),
        ("  RU  ", "ru"),
        ("zh-Hans", "zh"),
        ("nb", "no"),
        ("iw", "he"),
        ("jv", "jw"),
    ],
)
def test_normalize(given: str, expected: str) -> None:
    assert langs.normalize(given) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("sv", "swe_Latn"),
        ("no", "nob_Latn"),
        ("nn", "nno_Latn"),
        ("zh", "zho_Hans"),
        ("yue", "yue_Hant"),
        ("lv", "lvs_Latn"),
        ("sq", "als_Latn"),
        ("ar", "arb_Arab"),
    ],
)
def test_flores_mapping(code: str, expected: str) -> None:
    assert langs.flores(code) == expected


def test_no_translation_is_an_error_not_a_fallback() -> None:
    with pytest.raises(langs.NoTranslation):
        langs.flores("la")


def test_unknown_language_is_an_error() -> None:
    with pytest.raises(langs.UnknownLanguage):
        langs.flores("klingon")
