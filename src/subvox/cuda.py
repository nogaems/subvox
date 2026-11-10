"""Предзагрузка CUDA-библиотек из venv.

CTranslate2 собран под CUDA 12 и ищет ``libcublas.so.12`` и ``libcudnn.so.9``,
а в системе может стоять CUDA 13. Библиотеки правильных версий кладутся
в venv колёсами (``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12``) — но
``LD_LIBRARY_PATH``, заданный после старта процесса, загрузчик не читает.

Решение — загрузить ``.so`` напрямую через ``ctypes`` с ``RTLD_GLOBAL``:
дальше любой ``dlopen`` по имени находит уже загруженную библиотеку.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_done = False


def preload() -> None:
    """Подгрузить CUDA-библиотеки из venv, если они там есть. Идемпотентно."""
    global _done
    if _done or sys.platform != "linux":
        return
    _done = True

    try:
        import nvidia
    except ImportError:
        return

    roots = [Path(entry) for entry in list(nvidia.__path__)]
    libs = [p for root in roots for p in root.glob("*/lib/*.so*") if p.is_file()]
    if not libs:
        return

    # Порядок зависимостей заранее неизвестен, поэтому несколько проходов.
    for _ in range(4):
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(libs):
            break
        libs = failed
