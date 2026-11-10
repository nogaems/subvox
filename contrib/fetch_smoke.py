"""Скачивание корзины смоук-видео (contrib/smoke-videos.md).

Мультиплатформенное: Python-скрипт работает и на Linux/macOS,
и на Windows из-под того же venv — никаких отдельных веток
на каждую ОС. Ссылки живут в манифесте и только там: смена ссылки —
правка одного файла, этот скрипт не трогается.

Идемпотентно: существующие слоты пропускаются, ничего не
перезаписывается. Повторный запуск докачивает только недостающее.

Потолок 480p осознан: пайплайн ест только аудиодорожку, видео нужно
лишь для проверки синхронности субтитров глазами в mpv.

Вызов (из каталога проекта):

    uv run python contrib/fetch_smoke.py

yt-dlp не ставится в проект: скрипт дергает свежий ``uvx``-выпуск
(YouTube-брейкеры чинятся релизами yt-dlp, свежесть важнее стабильности).
ffmpeg нужен на хосте — yt-dlp использует его для слияния потоков
в mkv; он же обязателен и для самого прогона.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "contrib" / "smoke-videos.md"

ROW = re.compile(r"^\| *`([a-z0-9-]+)` *\| *(https://youtu\.be/[A-Za-z0-9_-]+) *\|")


def slots() -> list[tuple[str, str]]:
    """Слоты и ссылки из манифеста — таблицы победной и фейловой корзины."""
    found: list[tuple[str, str]] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def ytdlp() -> list[str]:
    """Команда запуска yt-dlp: [uvx, yt-dlp]. На Windows uvx.exe.

    yt-dlp не ставится в проект: свежий выпуск подтягивается uvx'ом —
    YouTube-брейкеры чинятся релизами, свежесть важнее стабильности.
    """
    name = "uvx.exe" if sys.platform == "win32" else "uvx"
    path = shutil.which(name) or shutil.which("uvx")
    if path is None:
        raise SystemExit("нет uv — установи: https://docs.astral.sh/uv/getting-started/")
    return [path, "yt-dlp"]


def fetch(slot: str, url: str, directory: Path) -> bool:
    """Скачать один слот. True — ок (включая «уже есть»)."""
    if any(directory.glob(f"{slot}.*")):
        print(f"  есть:   {slot}")
        return True

    print(f"  качаю:  {slot}")
    result = subprocess.run(  # noqa: PLW1510 — код возврата разбираем ниже сами
        ytdlp()
        + [
            ytdlp(),
            "-f",
            "bv*[height<=480]+ba/b[height<=480]/b",
            "--merge-output-format",
            "mkv",
            "--no-overwrites",
            "-o",
            str(directory / f"{slot}.%(ext)s"),
            url,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        tail = result.stderr.decode(errors="replace").strip().splitlines()[-3:]
        print(f"  ОШИБКА: {slot} ({url})")
        for line in tail:
            print(f"          {line}")
        return False
    return True


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "нет ffmpeg — поставь системным пакетным менеджером; "
            "он нужен yt-dlp'у для слияния потоков и обязателен для прогона"
        )

    entries = slots()
    if not entries:
        raise SystemExit(f"в {MANIFEST} не нашлось слотов — проверь формат таблиц")

    from subvox import paths

    directory = paths.smoke_dir()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"корзина:   {directory}")
    print(f"манифест:  {MANIFEST}")
    print(f"слотов:    {len(entries)}")
    print()

    failures = [slot for slot, url in entries if not fetch(slot, url, directory)]
    print()
    if failures:
        print(f"не скачалось: {len(failures)} — {', '.join(failures)}")
        raise SystemExit(1)

    print("готово, корзина целиком:")
    for path in sorted(directory.iterdir()):
        size = path.stat().st_size / 1e6
        print(f"  {size:8.1f} МБ  {path.name}")


if __name__ == "__main__":
    main()
