"""Рабочие файлы, ассоциированные с вводом: каталог по хешу содержимого.

Каталог одного медиафайла — ``work/<media-hash>/``. Хеш — sha256
содержимого, а не путь: файл, перемещённый или скопированный без
изменений, попадает в свой прежний каталог, а два одинаковых файла
делят один. Это задел под упаковку прогона в бандл: каталог
самодостаточен и не зависит от того, где лежит видео.

Хеш содержимого на больших файлах стоит дорого, поэтому считается один
раз и запоминается в индексе по (путь, размер, mtime). Индекс —
best effort: битый или недоступный не ломает прогон, хеш просто
посчитается заново. Лежит он в кеш-каталоге (``paths.cache_dir``),
а не рядом с рабочими каталогами: регенерируем, привязан к путям
этой машины и в бандл прогона не попадает.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from subvox import paths

CHUNK_BYTES = 1 << 23
"""Порция чтения при хешировании: 8 МиБ — файлы бывают гигабайтными."""

MANIFEST_NAME = "manifest.json"

SCHEMA_VERSION = 1
"""Версия формата записей. Смена меняет все ключи: старое осиротеет
и пересчитается, а не отдастся как верное."""

RECORD_KINDS = ("lid", "vad", "transcript")
"""Виды записей; каждому виду — свой подкаталог рабочего каталога."""


def media_hash(video: Path) -> str:
    """Хеш содержимого файла — имя его рабочего каталога."""
    index = _read_index()
    key = str(video.resolve())
    stat = video.stat()

    entry: Any = index.get(key)
    if (
        isinstance(entry, dict)
        and entry.get("size") == stat.st_size
        and entry.get("mtime") == stat.st_mtime_ns
    ):
        return entry["hash"]

    digest = hashlib.sha256()
    with open(video, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    result = digest.hexdigest()[:32]

    index[key] = {"size": stat.st_size, "mtime": stat.st_mtime_ns, "hash": result}
    _write_index(index)
    return result


def media_dir(video: Path) -> Path:
    """Каталог рабочих файлов этого медиафайла. Создаётся по надобности."""
    directory = paths.work_dir() / media_hash(video)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _index_path() -> Path:
    # Регенерируемое и привязанное к путям этой машины — место в кеше,
    # не в данных: рабочий каталог остаётся чисто контентным.
    return paths.cache_dir() / "index.json"


def _read_index() -> dict[str, Any]:
    try:
        data = read_json(_index_path())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_index(index: dict[str, Any]) -> None:
    try:
        write_json(_index_path(), index)
    except OSError:
        pass  # индекс — оптимизация, а не обязанность


def load_manifest(directory: Path) -> dict[str, Any] | None:
    """Манифест прогона. Нет или битый — None, как промах."""
    try:
        data = read_json(directory / MANIFEST_NAME)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    write_json(directory / MANIFEST_NAME, manifest)


def record_key(kind: str, inputs: dict[str, Any]) -> str:
    """Ключ записи: детерминирован всеми входами стадии.

    Одинаковым входам — один и тот же ключ, любому изменению — другой.
    Ключ включает версию схемы: смена формата записей осиротит старые,
    не отдав их как верные.
    """
    blob = json.dumps(
        {"schema": SCHEMA_VERSION, "kind": kind, "inputs": inputs},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def record_path(directory: Path, kind: str, key: str) -> Path:
    return directory / kind / f"{key}.json"


def read_record(directory: Path, kind: str, key: str) -> dict[str, Any] | None:
    """Прочитать запись. Промах, недоступность или битость — None."""
    try:
        data = read_json(record_path(directory, kind, key))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_record(
    directory: Path, kind: str, key: str, inputs: dict[str, Any], data: Any
) -> None:
    """Записать запись вместе с её входами — файл сам себя описывает."""
    write_json(record_path(directory, kind, key), {"inputs": inputs, "data": data})


def register(directory: Path, kind: str, key: str) -> None:
    """Занести запись в манифест и прибрать осиротевшие файлы этого вида.

    Манифест — опись рабочего каталога для человека и будущего бандла;
    поиск по нему не ведётся, ключи детерминированы. Осиротевшим
    считается файл вида, не упомянутый в манифесте: его запись
    перезаписана другим кодом или ревизией.
    """
    manifest = load_manifest(directory) or {"schema": SCHEMA_VERSION, "records": []}
    records = [
        entry
        for entry in manifest.get("records", [])
        if isinstance(entry, dict)
        and not (entry.get("kind") == kind and entry.get("key") == key)
    ]
    records.append({"kind": kind, "key": key})
    manifest["records"] = records
    write_json(directory / MANIFEST_NAME, manifest)

    keep = {entry["key"] for entry in records if entry.get("kind") == kind}
    subdir = directory / kind
    if subdir.is_dir():
        for path in subdir.glob("*.json"):
            if path.stem not in keep:
                try:
                    path.unlink()
                except OSError:
                    pass  # прибраться — желание, а не обязанность


def read_json(path: Path) -> Any:
    """Строгое чтение; исключения — забота вызывающего."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Атомарная запись: tmp в том же каталоге, затем замена."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
