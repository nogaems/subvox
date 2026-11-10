"""Смоук-прогоны по скачанной корзине.

Манифест — единственный источник слотов и ссылок; здесь он только
читается. Готовность — та же, что у скачивателя: файл ровно
``slot.mkv|.mp4|.webm|.m4a``, а не ``*.part`` и не отдельный поток.

Корзина по умолчанию должна быть полной: отсутствующий слот — это
ошибка с явным списком, а не тихий пропуск. ``--partial`` разрешает
неполную корзину — для контрибьютора, у которого часть видео
недоступна в регионе. Это отчёт для человека, а не pytest: ожидания
в манифесте записаны прозой, жёстких assert'ов здесь нет. Ненулевой
код у слота без префикса ``fail-`` — единственный сигнал «что-то не так».

Запуск:

    uv run python contrib/run_smoke.py
    uv run python contrib/run_smoke.py --only en-ted,ja-tokyo
    uv run python contrib/run_smoke.py --to ru --report smoke-report.md
    uv run python contrib/run_smoke.py --partial   # неполная корзина допустима
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CONTRIB = Path(__file__).resolve().parent


def fetch_tool():
    """Скачиватель как модуль: слоты и список финальных расширений."""
    spec = importlib.util.spec_from_file_location(
        "fetch_smoke", CONTRIB / "fetch_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class Result:
    slot: str
    file: str
    detected: str
    detect_code: int
    run_code: int
    cues: str
    cps: str
    device: str
    output: str
    seconds: float


def subvox_cmd() -> list[str]:
    """Консольный скрипт из того же venv, что и этот Python."""
    exe = Path(sys.executable)
    for name in ("subvox.exe", "subvox"):
        candidate = exe.with_name(name)
        if candidate.is_file():
            return [str(candidate)]
    return [sys.executable, "-c", "from subvox.cli import main; main()"]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: PLW1510 — код возврата разбираем сами
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def detected_language(output: str) -> str:
    """Код языка из вывода ``subvox detect`` (самая вероятная строка)."""
    best: tuple[float, str] | None = None
    for name, score in re.findall(r"^\s{2}(.+?)\s+(\d+\.\d+)\s*$", output, re.MULTILINE):
        code = re.search(r"\(([a-z]{2,3})\)", name)
        if code is None:
            continue
        value = float(score)
        if best is None or value > best[0]:
            best = (value, code.group(1))
    return best[1] if best else "?"


def final_file(directory: Path, slot: str, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        path = directory / f"{slot}{ext}"
        if path.is_file():
            return path
    return None


def output_file(path: Path, target: str) -> Path:
    return path.with_name(f"{path.stem}.{target}.srt")


def main() -> None:
    from subvox import paths

    parser = argparse.ArgumentParser(description="Смоук-прогоны по корзине.")
    parser.add_argument("--to", default="ru", help="целевой язык (по умолчанию ru)")
    parser.add_argument(
        "--only", default="", help="слоты через запятую; пусто — все из манифеста"
    )
    parser.add_argument("--report", type=Path, help="куда записать markdown-сводку")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="разрешить неполную корзину (по умолчанию отсутствие слота — ошибка)",
    )
    args = parser.parse_args()

    tool = fetch_tool()
    directory = paths.smoke_dir()
    entries = tool.slots()
    only = {slot.strip() for slot in args.only.split(",") if slot.strip()}
    if only:
        entries = [(slot, url) for slot, url in entries if slot in only]

    print(f"корзина: {directory}")
    print(f"слотов:  {len(entries)} · цель: {args.to}")
    print()

    results: list[Result] = []
    missing: list[str] = []
    unexpected: list[str] = []

    for slot, _url in entries:
        path = final_file(directory, slot, tool.FINAL_EXTS)
        if path is None:
            missing.append(slot)
            print(f"— {slot}: нет в корзине", flush=True)
            continue

        size = path.stat().st_size / 1e6
        print(f"=== {slot} ({path.name}, {size:.1f} МБ) ===", flush=True)
        started = time.monotonic()

        det = run([*subvox_cmd(), "detect", str(path)])
        print(det.stdout.rstrip())
        if det.stderr.strip():
            print(det.stderr.rstrip())
        language = detected_language(det.stdout)

        # Старый субтитр от прошлого прогона — не доказательство успеха:
        # упавший слот оставил бы его рядом с видео и соврал бы сводке.
        out = output_file(path, args.to)
        out.unlink(missing_ok=True)

        tr = run([*subvox_cmd(), "run", str(path), "--to", args.to])
        print(tr.stdout.rstrip())
        if tr.stderr.strip():
            print(tr.stderr.rstrip())

        cues = re.search(r"(\d+) предложений, (\d+) реплик", tr.stdout)
        cps = re.search(r"(\d+) реплик быстрее", tr.stdout)
        device = re.search(r"перевод (\w+)", tr.stdout)
        elapsed = time.monotonic() - started

        results.append(
            Result(
                slot=slot,
                file=path.name,
                detected=language,
                detect_code=det.returncode,
                run_code=tr.returncode,
                cues=f"{cues[1]}/{cues[2]}" if cues else "—",
                cps=cps[1] if cps else "0",
                device=device[1] if device else "—",
                output=out.name if out.is_file() else "нет",
                seconds=elapsed,
            )
        )

        expected_fail = slot.startswith("fail-")
        if det.returncode != 0 and not expected_fail:
            unexpected.append(f"{slot}: detect")
        if tr.returncode != 0 and not expected_fail:
            unexpected.append(f"{slot}: run")
        print(flush=True)

    print("=== сводка ===")
    header = (
        f"{'слот':<16} {'язык':<5} {'detect':>6} {'run':>4} "
        f"{'предл/репл':>10} {'cps':>4} {'устр':<5} {'файл':<24} {'с':>6}"
    )
    print(header)
    for r in results:
        print(
            f"{r.slot:<16} {r.detected:<5} {r.detect_code:>6} {r.run_code:>4} "
            f"{r.cues:>10} {r.cps:>4} {r.device:<5} {r.output:<24} {r.seconds:>6.0f}"
        )

    skipped = len(entries) - len(results)
    if skipped:
        print(f"\nне хватает в корзине: {skipped} — {', '.join(missing)}")
        if not args.partial:
            unexpected.append("корзина неполная")

    if args.report is not None:
        lines = [
            "| Слот | Язык | detect | run | Предл/репл | CPS | Устройство | Файл | Сек |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| `{r.slot}` | {r.detected} | {r.detect_code} | {r.run_code} "
            f"| {r.cues} | {r.cps} | {r.device} | {r.output} | {r.seconds:.0f} |"
            for r in results
        ]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"сводка: {args.report}")

    if unexpected:
        print(f"\nнеожиданные ошибки: {', '.join(unexpected)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
