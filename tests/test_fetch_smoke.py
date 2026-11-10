"""Инструмент скачивания корзины (contrib/fetch_smoke.py).

Скрипт живёт в contrib, не в пакете — грузим по пути. Сеть не
трогаем: только чистые функции (парсер манифеста, готовность слота,
проба JS-runtime, сборка команды yt-dlp). Это охрана контракта
после живых багов сборки команды: uvx получал флаги yt-dlp,
внутри списка аргументов лежал вложенный список.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "contrib" / "fetch_smoke.py"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_which(*found: str):
    """shutil.which, который «видит» только перечисленные имена."""

    def which(name: str) -> str | None:
        return f"/bin/{name}" if name in found else None

    return which


def test_manifest_slots(tool: ModuleType) -> None:
    """Манифест парсится: слоты уникальны, без точек, ссылки youtu.be."""
    found = tool.slots()
    assert len(found) == 16
    names = [slot for slot, _ in found]
    assert len(set(names)) == len(names)
    for slot, url in found:
        assert "." not in slot and "`" not in slot
        assert url.startswith("https://youtu.be/")
        assert url.removeprefix("https://youtu.be/") != ""


def test_ready_vs_intermediate(tool: ModuleType, tmp_path: Path) -> None:
    """Промежуточные потоки и .part — не готовность; готовый файл — да."""
    (tmp_path / "en-ted.f397.mp4").touch()  # видеопоток без аудио
    (tmp_path / "ru-lecture.f251.webm.part").touch()  # недокачанное
    assert not tool.is_ready(tmp_path, "en-ted")
    assert not tool.is_ready(tmp_path, "ru-lecture")

    (tmp_path / "en-ted.mkv").touch()  # слияние завершено
    assert tool.is_ready(tmp_path, "en-ted")
    (tmp_path / "ru-lecture.webm").touch()  # однопоточная загрузка
    assert tool.is_ready(tmp_path, "ru-lecture")


def test_js_runtime_priority(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime ищется в порядке приоритета yt-dlp; ничего нет — None."""
    monkeypatch.setattr(tool.shutil, "which", fake_which("node"))
    assert tool.js_runtime() == "node"
    monkeypatch.setattr(tool.shutil, "which", fake_which("bun", "node"))
    assert tool.js_runtime() == "node"
    monkeypatch.setattr(tool.shutil, "which", fake_which("deno", "node"))
    assert tool.js_runtime() == "deno"
    monkeypatch.setattr(tool.shutil, "which", fake_which())
    assert tool.js_runtime() is None


def test_select_keeps_manifest_order(tool: ModuleType) -> None:
    """--only сохраняет порядок манифеста, а не порядок аргументов."""
    entries = [("a", "u1"), ("b", "u2"), ("c", "u3")]
    assert tool.select(entries, []) == entries
    assert tool.select(entries, ["c", "a"]) == [("a", "u1"), ("c", "u3")]


def test_select_unknown_slot_is_an_error(tool: ModuleType) -> None:
    """Опечатка в слоте — явная ошибка, а не «уже скачано»."""
    with pytest.raises(SystemExit, match="нет таких слотов"):
        tool.select([("a", "u1")], ["a", "a-tokyo"])


def test_command_shape(tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Команда yt-dlp — плоский список, с runtime и шаблоном выхода."""
    monkeypatch.setattr(tool.shutil, "which", fake_which("uvx"))
    cmd = tool.command("node", "ja-tokyo", "https://youtu.be/X", tmp_path)

    assert cmd[:2] == ["/bin/uvx", "yt-dlp"]  # не uvx с флагами yt-dlp
    assert cmd[2:4] == ["--js-runtimes", "node"]  # runtime передан явно
    assert cmd[4:6] == ["--remote-components", "ejs:github"]  # солверы задач
    assert "--no-overwrites" in cmd
    assert cmd[-1] == "https://youtu.be/X"
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "ja-tokyo.%(ext)s")
