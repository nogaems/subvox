"""Рабочие каталоги: хеш содержимого, индекс путей, манифест."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subvox import paths, store


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Изолированный SUBVOX_HOME — тесты не трогают реальный каталог."""
    monkeypatch.setenv("SUBVOX_HOME", str(tmp_path))
    return tmp_path


def test_dir_is_named_by_content_not_path(home: Path) -> None:
    video = home / "a.mp4"
    video.write_bytes(b"same bytes")
    twin = home / "copy.mp4"
    twin.write_bytes(b"same bytes")

    assert store.media_dir(video) == store.media_dir(twin)


def test_changed_content_gets_new_dir(home: Path) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"version one")
    first = store.media_dir(video)

    video.write_bytes(b"version two")
    second = store.media_dir(video)

    assert first != second


def test_hash_is_cached_in_index(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"data")
    expected = store.media_hash(video)

    index = json.loads((paths.cache_dir() / "index.json").read_text(encoding="utf-8"))
    entry = index[str(video.resolve())]
    assert entry["size"] == 4

    def _no_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("хеш пересчитан вместо чтения индекса")

    monkeypatch.setattr(store.hashlib, "sha256", _no_recompute)
    assert store.media_hash(video) == expected


def test_corrupted_index_is_survived(home: Path) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"data")
    first = store.media_hash(video)

    (paths.cache_dir() / "index.json").write_text("{не json", encoding="utf-8")

    assert store.media_hash(video) == first  # пересчитал и не упал


def test_manifest_roundtrip(home: Path) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"x")
    directory = store.media_dir(video)

    store.save_manifest(directory, {"schema": 1, "language": "ru"})

    assert store.load_manifest(directory) == {"schema": 1, "language": "ru"}
    assert not list(directory.glob("*.tmp"))  # запись атомарна, мусора нет


def test_missing_manifest_is_none(home: Path) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"x")

    assert store.load_manifest(store.media_dir(video)) is None


def test_corrupted_manifest_is_none(home: Path) -> None:
    video = home / "v.mp4"
    video.write_bytes(b"x")
    directory = store.media_dir(video)

    (directory / "manifest.json").write_text("[1, 2]", encoding="utf-8")

    assert store.load_manifest(directory) is None
