"""Пути по платформам: единственный источник знаний об ОС."""

from __future__ import annotations

from pathlib import Path

import pytest

from subvox import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты не зависят от окружения машины разработчика."""
    for var in ("SUBVOX_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)


def test_linux_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")

    assert paths.data_root() == Path.home() / ".local" / "share" / "subvox"


def test_linux_xdg_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")

    assert paths.data_root() == Path("/xdg/subvox")


def test_windows_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\u\AppData\Local")

    # Сепаратор подставляет pathlib: здесь проверяем, что взялась
    # база LOCALAPPDATA и пришилось имя приложения.
    assert paths.data_root() == Path(r"C:\Users\u\AppData\Local") / "subvox"


def test_windows_without_localappdata_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")

    assert paths.data_root() == Path.home() / "subvox"


def test_macos_application_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "darwin")

    assert paths.data_root() == Path.home() / "Library" / "Application Support" / "subvox"


def test_subvox_home_overrides_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\u\AppData\Local")
    monkeypatch.setenv("SUBVOX_HOME", "/portable")

    assert paths.data_root() == Path("/portable")


def test_layout_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUBVOX_HOME", "/base")

    assert paths.models_dir() == Path("/base/models")
    assert paths.work_dir() == Path("/base/work")
    assert paths.cache_dir() == Path("/base/cache")


def test_cache_linux_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")

    assert paths.cache_dir() == Path.home() / ".cache" / "subvox"


def test_cache_linux_xdg_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg-cache")

    assert paths.cache_dir() == Path("/xdg-cache/subvox")


def test_cache_windows_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\u\AppData\Local")

    assert paths.cache_dir() == Path(r"C:\Users\u\AppData\Local") / "subvox" / "cache"


def test_cache_macos_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "darwin")

    assert paths.cache_dir() == Path.home() / "Library" / "Caches" / "subvox"
