"""Команды CLI, которым не нужны модели: list-langs.

Остальные команды требуют живых Whisper/NLLB и покрыты дымовыми
прогонами, а не тестами. list-langs чистая — проверяется честно,
через typer.testing: реальный разбор аргументов, реальный вывод.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from subvox import langs
from subvox.cli import app

runner = CliRunner()


def test_list_langs_table_has_all_languages() -> None:
    result = runner.invoke(app, ["list-langs"])

    assert result.exit_code == 0
    assert str(len(langs.LANGUAGES)) in result.output  # счётчик в заголовке
    assert "Russian" in result.output
    assert "Cantonese" in result.output


def test_list_langs_marks_untranslatable() -> None:
    result = runner.invoke(app, ["list-langs"])

    assert result.exit_code == 0
    # бретонский без перевода — в таблице и в строке-подвале
    assert "Breton" in result.output
    assert "br" in result.output


def test_list_langs_json_is_parseable() -> None:
    result = runner.invoke(app, ["list-langs", "--json"])

    assert result.exit_code == 0
    rows = json.loads(result.output)
    codes = {row["code"] for row in rows}
    assert {"ru", "en", "br"} <= codes
    by_code = {row["code"]: row for row in rows}
    assert by_code["ru"]["translatable"] is True
    assert by_code["br"]["translatable"] is False
    assert by_code["ru"]["name"] == "Russian"
