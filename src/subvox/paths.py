"""Единственный модуль, знающий про операционную систему.

Все постоянные пути subvox сходятся здесь; остальной код зовёт функции
этого модуля и не строит путей сам — иначе знание о платформе
расползается по кодовой базе, и Windows-сборка превращается в лотерею.

Все пути per-user, никогда системные. Две базы, и они разные:

* данные — то, что дорого пересчитывать и что жалко потерять:
  Linux — $XDG_DATA_HOME/subvox (~/.local/share/subvox),
  Windows — %LOCALAPPDATA%\\subvox,
  macOS — ~/Library/Application Support/subvox;
* кеш — то, что ОС или пользователь могут стереть без потери:
  Linux — $XDG_CACHE_HOME/subvox (~/.cache/subvox),
  Windows — %LOCALAPPDATA%\\subvox\\cache,
  macOS — ~/Library/Caches/subvox.

SUBVOX_HOME перекрывает обе базы целиком (данные — прямо в нём, кеш —
в его подкаталоге cache) — для переносной установки «всё в одном
каталоге» и для тестов.

Внутри базы данных три подкаталога:

* ``models/`` — модели Whisper и NLLB; качаются при первом запуске
  (см. ``model.py`` и ``nllb.py``), поэтому пакет гигабайтов не весит.
  Внутри — мини-хаб huggingface_hub: веса в общем ``blobs/`` с
  относительными симлинками в каталоги моделей; на Windows без прав
  на симлинки хаб сам кладёт копии. Мы этот каталог не обходим
  вручную и не парсим — он целиком под управлением huggingface_hub,
  и переносим между машинами как есть;
* ``work/`` — рабочие файлы прогонов, по каталогу на медиафайл;
  раскладкой владеет ``store.py``;
* ``smoke/`` — корзина живых видео для смоук-прогонов разработчика.
  Это данные, а не кеш: локальная копия может остаться единственной,
  если автор удалит видео с YouTube, а записи транскриптов в ``work/``
  привязаны к её хешу содержимого. Качается ``contrib/fetch-smoke.sh``
  по манифесту ``contrib/smoke-videos.md``, идемпотентно; в релизный
  пакет не входит.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "subvox"


def data_root() -> Path:
    """База данных. Всегда per-user, никогда системная."""
    override = os.environ.get("SUBVOX_HOME")
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / APP_NAME


def cache_dir() -> Path:
    """База регенерируемого: индекс путей и прочее, что можно стереть."""
    override = os.environ.get("SUBVOX_HOME")
    if override:
        return Path(override) / "cache"

    if sys.platform == "win32":
        # Отдельного системного каталога кешей у Windows нет; конвенция —
        # подкаталог cache внутри данных приложения.
        return data_root() / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP_NAME
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / APP_NAME


def models_dir() -> Path:
    """Где живут модели. Каталог создаётся при первом скачивании."""
    return data_root() / "models"


def work_dir() -> Path:
    """Корень рабочих файлов; по подкаталогу на медиафайл."""
    return data_root() / "work"


def smoke_dir() -> Path:
    """Корзина живых видео для смоук-прогонов (см. contrib/smoke-videos.md).

    Стандартное системное per-user место — то же, где модели и рабочие
    файлы, поэтому перезапуск прогона всегда находит одну и ту же
    корзину, а SUBVOX_HOME уводит и её.
    """
    return data_root() / "smoke"
