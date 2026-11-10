"""Скачивание корзины смоук-видео (contrib/smoke-videos.md).

Мультиплатформенное: Python-скрипт работает и на Linux/macOS,
и на Windows из-под того же venv — никаких отдельных веток
на каждую ОС. Ссылки живут в манифесте и только там: смена ссылки —
правка одного файла, этот скрипт не трогается.

Идемпотентно: готовый слот — файл ровно ``slot.mkv`` (или нативный
``.mp4``/``.webm`` однопоточной загрузки); промежуточные потоки
(``slot.f397.mp4``) и недокачанное (``*.part``) готовыми не
считаются — повторный запуск доделывает только их.

Потолок 480p осознан: пайплайн ест только аудиодорожку, видео нужно
лишь для проверки синхронности субтитров глазами в mpv.

Вызов (из каталога проекта):

    uv run python contrib/fetch_smoke.py
    uv run python contrib/fetch_smoke.py --only ja-tokyo,kk-almaty

Зависимости хоста — uv, ffmpeg и JS-runtime для yt-dlp (deno или
node). Свежий yt-dlp без JS-runtime не решает JS-задачи YouTube
(nsig) — CDN режет загрузку 503-ми посреди файла. Скрипт сам ищет
доступный runtime по PATH в порядке приоритета yt-dlp (deno, node,
quickjs, bun) и передаёт его флагом ``--js-runtimes`` каждой
загрузке; нет ни одного — честная ошибка до старта, как с ffmpeg.
Солвер-скрипты JS-задач подтягиваются с GitHub
(``--remote-components ejs:github``) по рекомендации самого
yt-dlp — без них часть форматов может отсутствовать.
shutil.which платформенно-нейтрален: на Windows находит и .exe,
поэтому обёртки ``.sh``/``.ps1`` получают это поведение без правок.

yt-dlp не ставится в проект: скрипт дергает свежий ``uvx``-выпуск
(YouTube-брейкеры чинятся релизами yt-dlp, свежесть важнее
стабильности). Вывод yt-dlp идёт в терминал напрямую, без
перехвата: прогресс и ошибки видны живьём, а не после конца
прогона. ffmpeg нужен yt-dlp'у для слияния потоков в mkv; он же
обязателен и для самого прогона.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "contrib" / "smoke-videos.md"

ROW = re.compile(r"^\| *`([a-z0-9-]+)` *\| *(https://youtu\.be/[A-Za-z0-9_-]+) *\|")

# Готовый слот — файл ровно slot.<ext>: слияние даёт mkv, однопоточная
# загрузка оставляет нативное расширение. Всё прочее (slot.f397.mp4,
# *.part) — промежуточные файлы yt-dlp, готовностью не являющиеся.
FINAL_EXTS = (".mkv", ".mp4", ".webm", ".m4a")

# JS-runtime'ы в порядке приоритета yt-dlp (см. --js-runtimes).
JS_RUNTIMES = ("deno", "node", "quickjs", "bun")


def slots() -> list[tuple[str, str]]:
    """Слоты и ссылки из манифеста — таблицы победной и фейловой корзины."""
    found: list[tuple[str, str]] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def select(entries: list[tuple[str, str]], only: list[str]) -> list[tuple[str, str]]:
    """Выбранные слоты в порядке манифеста; пусто — все.

    Неизвестный слот — ошибка, а не тихий пропуск: опечатка в имени
    не должна выглядеть как «уже скачано».
    """
    if not only:
        return entries
    known = {slot for slot, _ in entries}
    unknown = [slot for slot in only if slot not in known]
    if unknown:
        raise SystemExit(
            f"нет таких слотов в манифесте: {', '.join(unknown)}"
        )
    wanted = set(only)
    return [(slot, url) for slot, url in entries if slot in wanted]


def ytdlp() -> list[str]:
    """Команда запуска yt-dlp: [uvx, yt-dlp]. На Windows uvx.exe."""
    name = "uvx.exe" if sys.platform == "win32" else "uvx"
    path = shutil.which(name) or shutil.which("uvx")
    if path is None:
        raise SystemExit("нет uv — установи: https://docs.astral.sh/uv/getting-started/")
    return [path, "yt-dlp"]


def js_runtime() -> str | None:
    """Доступный JS-runtime для yt-dlp; None — нет ни одного.

    Ищется по PATH в порядке приоритета yt-dlp. shutil.which
    платформенно-нейтрален: на Windows находит и ``.exe``.
    """
    for name in JS_RUNTIMES:
        if shutil.which(name) is not None:
            return name
    return None


def is_ready(directory: Path, slot: str) -> bool:
    """Слот скачан целиком: есть готовый файл, не промежуточный."""
    return any((directory / f"{slot}{ext}").is_file() for ext in FINAL_EXTS)


def command(runtime: str, slot: str, url: str, directory: Path) -> list[str]:
    """Полная команда yt-dlp для одного слота — плоским списком."""
    return ytdlp() + [
        "--js-runtimes",
        runtime,
        "--remote-components",
        "ejs:github",
        "-f",
        "bv*[height<=480]+ba/b[height<=480]/b",
        "--merge-output-format",
        "mkv",
        "--no-overwrites",
        "-o",
        str(directory / f"{slot}.%(ext)s"),
        url,
    ]


def fetch(slot: str, url: str, directory: Path, runtime: str) -> bool:
    """Скачать один слот. True — ок (включая «уже есть»)."""
    if is_ready(directory, slot):
        print(f"  есть:   {slot}", flush=True)
        return True

    print(f"  качаю:  {slot}", flush=True)
    result = subprocess.run(  # noqa: PLW1510 — код возврата разбираем ниже сами
        command(runtime, slot, url, directory),
    )
    if result.returncode != 0:
        print(f"  ОШИБКА: {slot} ({url}) — детали выше, в выводе yt-dlp", flush=True)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать корзину смоук-видео.")
    parser.add_argument(
        "--only",
        default="",
        help="слоты через запятую; пусто — все из манифеста",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "нет ffmpeg — поставь системным пакетным менеджером; "
            "он нужен yt-dlp'у для слияния потоков и обязателен для прогона"
        )
    ytdlp()  # предпроверка uv: нет — SystemExit с текстом установки
    runtime = js_runtime()
    if runtime is None:
        raise SystemExit(
            "нет JS-runtime для yt-dlp — поставь deno или node: "
            "https://github.com/yt-dlp/yt-dlp/wiki/EJS — без него YouTube "
            "отдаёт не все форматы и режет загрузку 503-ми"
        )

    entries = slots()
    if not entries:
        raise SystemExit(f"в {MANIFEST} не нашлось слотов — проверь формат таблиц")
    only = [slot.strip() for slot in args.only.split(",") if slot.strip()]
    entries = select(entries, only)

    from subvox import paths

    directory = paths.smoke_dir()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"корзина:    {directory}")
    print(f"манифест:   {MANIFEST}")
    print(f"слотов:     {len(entries)}")
    print(f"JS-runtime: {runtime}")
    print()

    failures = [slot for slot, url in entries if not fetch(slot, url, directory, runtime)]
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
