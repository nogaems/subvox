"""Команды CLI, которым не нужны модели: list-langs и пустой субтитр.

Остальные команды требуют живых Whisper/NLLB и покрыты дымовыми
прогонами, а не тестами. list-langs чистая — проверяется честно,
через typer.testing: реальный разбор аргументов, реальный вывод.
Путь «речи нет» тоже проверяется без моделей: ``pipeline.prepare``
подменяется на ``vad.NoSpeech``.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from subvox import langs, vad
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


def test_run_without_speech_writes_empty_file(tmp_path, monkeypatch) -> None:
    """Речи нет: файл создаётся, но пустой — и об этом сказано в выводе."""
    video = tmp_path / "music.mkv"
    video.write_bytes(b"")

    def no_speech(*_args, **_kwargs):
        raise vad.NoSpeech("речь не найдена даже на минимальном пороге VAD")

    monkeypatch.setattr("subvox.cli.pipeline.prepare", no_speech)
    result = runner.invoke(app, ["run", str(video), "--to", "ru"])

    assert result.exit_code == 0
    output = tmp_path / "music.ru.srt"
    assert output.is_file()
    assert output.read_text(encoding="utf-8") == ""
    assert "0 предложений, 0 реплик" in result.output
    assert "речи не найдено" in result.output
    assert "файл записан без реплик" in result.output
