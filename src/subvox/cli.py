"""Точка входа CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from subvox import __version__, asr, audio, langs, nllb, pipeline, sentences, subs, vad
from subvox import cues as cues_mod
from subvox.model import DEFAULT_WHISPER, ModelError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Офлайн-субтитры для видео на незнакомых языках.",
)

console = Console()
stderr = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"subvox {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """Офлайн-субтитры для видео на незнакомых языках."""


def _clock(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _fail(exc: Exception) -> None:
    stderr.print(f"[red]{exc}[/red]")
    raise typer.Exit(1) from None


def _gpu_notes(loaded: pipeline.Loaded | None, engine: nllb.Engine | None = None) -> None:
    """Честно сказать, если GPU не поднялся — молчаливый откат запрещён."""
    if loaded is not None and loaded.fallback_reason:
        stderr.print(
            f"[yellow]GPU не поднялся, работаю на {loaded.device}: "
            f"{loaded.fallback_reason}[/yellow]"
        )
    if engine is not None and engine.fallback_reason:
        stderr.print(
            f"[yellow]переводчик на GPU не поднялся, работаю на {engine.device}: "
            f"{engine.fallback_reason}[/yellow]"
        )


def _note(message: str) -> None:
    """Анонс долгих фоновых действий (скачивание модели) — в stderr."""
    stderr.print(f"[yellow]{message}[/yellow]")


NoCache = Annotated[
    bool,
    typer.Option("--no-cache", help="Посчитать заново, не читая и не записывая кеш прогона."),
]

SubsFormat = Annotated[
    str | None,
    typer.Option(
        "--format",
        help="Формат субтитров: srt (по умолчанию) или vtt — для браузеров. "
        "Действует на имя по умолчанию; с --out формат определяется его расширением.",
    ),
]


def _check_out(out: Path | None, fmt: str | None) -> None:
    """Ранний разбор --out: до моделей, а не после минут транскрипции."""
    if out is None:
        return
    actual = subs.format_of(out)
    if fmt is not None and fmt != actual:
        raise ValueError(f"--out указывает на .{actual}, а --format требует {fmt}")


def _dest(video: Path, base: str, out: Path | None, fmt: str | None) -> Path:
    """Куда писать: явный --out или имя по умолчанию video.<base>.<формат>."""
    if out is not None:
        return out
    return video.with_suffix(f".{base}.{fmt or 'srt'}")


@app.command("list-langs")
def list_langs(
    as_json: Annotated[bool, typer.Option("--json", help="Машинный вывод.")] = False,
) -> None:
    """Все языки: код для --from/--to, название, переводимость NLLB."""
    languages = sorted(langs.LANGUAGES.values(), key=lambda lang: lang.name.lower())

    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "code": lang.code,
                        "name": lang.name,
                        "translatable": lang.flores is not None,
                    }
                    for lang in languages
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    table = Table(title=f"Языки Whisper · {len(languages)}")
    table.add_column("код", style="bold")
    table.add_column("язык")
    table.add_column("перевод NLLB", justify="center")
    for lang in languages:
        table.add_row(lang.code, lang.name, "да" if lang.flores else "[dim]—[/dim]")
    console.print(table)

    silent = ", ".join(langs.untranslatable())
    aliases = ", ".join(f"{alias}={target}" for alias, target in langs.ALIASES.items())
    console.print(
        f"[dim]без перевода NLLB (только транскрипция): {silent}.[/dim]"
    )
    console.print(f"[dim]алиасы тоже принимаются: {aliases}.[/dim]")


@app.command()
def detect(
    video: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Видео или аудиофайл."),
    ],
    model: Annotated[str, typer.Option("--model", "-m", help="Модель Whisper.")] = DEFAULT_WHISPER,
    device: Annotated[
        str, typer.Option("--device", help="auto, cuda или cpu.")
    ] = "auto",
    track: Annotated[
        int, typer.Option("--track", "-t", help="Номер аудиодорожки, с нуля.")
    ] = 0,
    windows: Annotated[
        int, typer.Option("--windows", "-w", help="Сколько окон речи анализировать.")
    ] = 6,
    no_cache: NoCache = False,
    as_json: Annotated[bool, typer.Option("--json", help="Машинный вывод.")] = False,
) -> None:
    """Определить язык речи в видео и ничего больше не делать.

    Примеры: subvox detect film.mkv
             subvox detect film.mkv --json
    """
    try:
        stream = audio.probe(video).stream(track)
        result = pipeline.detect_language(
            video,
            track=track,
            model=model,
            device=device,
            windows=windows,
            use_cache=not no_cache,
            note=_note,
        )
    except (audio.AudioError, vad.NoSpeech, ModelError, ValueError) as exc:
        _fail(exc)

    detection = result.detection
    _gpu_notes(result.loaded)

    if as_json:
        payload: dict[str, Any] = {
            "file": str(video),
            "duration": round(detection.duration, 2),
            "speech_seconds": round(detection.speech_seconds, 2),
            "windows": detection.windows,
            "vad_threshold": detection.vad_threshold,
            "device": result.loaded.device if result.loaded else None,
            "cached": result.cached,
            "language": detection.language,
            "name": detection.name,
            "probability": round(detection.probability, 4),
            "confident": detection.confident,
            "translatable": langs.translatable(detection.language),
            "alternatives": [
                {
                    "code": alt.code,
                    "name": alt.name,
                    "probability": round(alt.probability, 4),
                }
                for alt in detection.alternatives
            ],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    cache_note = " [dim]· из кеша[/dim]" if result.cached else ""
    console.print(
        f"[bold]{video.name}[/bold] · {_clock(detection.duration)} · "
        f"дорожка {stream.index} [dim]({stream.label})[/dim]{cache_note}"
    )
    console.print()
    console.print(
        f"  [bold]{detection.name}[/bold] ({detection.language})   "
        f"[green]{detection.probability:.2f}[/green]"
    )
    if detection.alternatives:
        others = " · ".join(f"{alt.name} {alt.probability:.2f}" for alt in detection.alternatives)
        console.print(f"  [dim]{others}[/dim]")
    console.print(
        f"  [dim]речь {_clock(detection.speech_seconds)} из {_clock(detection.duration)}, "
        f"окон: {detection.windows}[/dim]"
    )

    if detection.vad_threshold < vad.THRESHOLDS[0]:
        console.print(
            f"  [yellow]речь поверх музыки или шума: VAD сработал только на пороге "
            f"{detection.vad_threshold}, результат может быть грубее обычного[/yellow]"
        )

    if not detection.confident:
        console.print(
            "  [yellow]уверенности мало: язык стоит задать вручную через --from[/yellow]"
        )
    if not langs.translatable(detection.language):
        console.print(
            f"  [yellow]{detection.name} распознаётся, но перевести его нечем: "
            "NLLB этот язык не поддерживает[/yellow]"
        )


@app.command()
def transcribe(
    video: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Видео или аудиофайл."),
    ],
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Язык оригинала. По умолчанию определяется. Полный список: subvox list-langs.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Модель Whisper.")] = DEFAULT_WHISPER,
    device: Annotated[str, typer.Option("--device", help="auto, cuda или cpu.")] = "auto",
    track: Annotated[
        int, typer.Option("--track", "-t", help="Номер аудиодорожки, с нуля.")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Куда писать субтитры (.srt/.vtt). По умолчанию video.<язык>.<формат>."),
    ] = None,
    fmt: SubsFormat = None,
    no_cache: NoCache = False,
    as_json: Annotated[bool, typer.Option("--json", help="Машинный вывод.")] = False,
) -> None:
    """Распознать речь и записать черновой SRT на языке оригинала.

    Примеры: subvox transcribe film.mkv
             subvox transcribe film.mkv --from en --out draft.vtt
    """
    try:
        _check_out(out, fmt)
        prepared = pipeline.prepare(
            video,
            track=track,
            source=source,
            model=model,
            device=device,
            use_cache=not no_cache,
            note=_note,
        )
    except (
        audio.AudioError,
        vad.NoSpeech,
        asr.TranscriptionError,
        langs.UnknownLanguage,
        ModelError,
        ValueError,
    ) as exc:
        _fail(exc)

    transcript = prepared.transcript
    destination = _dest(video, prepared.language, out, fmt)
    _gpu_notes(prepared.loaded)

    if as_json:
        payload = asr.to_dict(transcript)
        payload["file"] = str(video)
        payload["device"] = prepared.loaded.device if prepared.loaded else None
        payload["cached"] = prepared.transcript_cached
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    subs.write(transcript.segments, destination)

    notes = []
    if transcript.dropped_segments:
        notes.append(f"отброшено пустых/тихих: {transcript.dropped_segments}")
    if transcript.duplicated_segments:
        notes.append(f"свернуто повторов: {transcript.duplicated_segments}")
    notes.append(f"VAD {prepared.vad_threshold}")
    if prepared.transcript_cached:
        notes.append("из кеша")

    console.print(
        f"[bold]{video.name}[/bold] -> [bold]{destination.name}[/bold] "
        f"[dim](черновик)[/dim]"
    )
    console.print(
        f"  {langs.name(prepared.language)} ({prepared.language}): "
        f"{len(transcript.segments)} сегментов, {transcript.words} слов · {' · '.join(notes)}"
    )
    if prepared.vad_threshold < vad.THRESHOLDS[0]:
        console.print(
            "  [yellow]речь поверх музыки или шума: таймкоды и текст могут быть грубее обычного[/yellow]"
        )
    console.print(
        "  [dim]реплики пока равны сегментам Whisper; финальная нарезка появится после стадии перевода[/dim]"
    )


@app.command()
def run(
    video: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Видео или аудиофайл."),
    ],
    to: Annotated[
        str,
        typer.Option(
            "--to",
            help="Целевой язык, ISO-639-1: ru, en, sv... Полный список: subvox list-langs.",
        ),
    ],
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Язык оригинала. По умолчанию определяется. Полный список: subvox list-langs.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Модель Whisper.")] = DEFAULT_WHISPER,
    device: Annotated[str, typer.Option("--device", help="auto, cuda или cpu.")] = "auto",
    track: Annotated[
        int, typer.Option("--track", "-t", help="Номер аудиодорожки, с нуля.")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Куда писать субтитры (.srt/.vtt). По умолчанию video.<язык>.<формат>."),
    ] = None,
    fmt: SubsFormat = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Дополнительно сохранить черновик на языке оригинала."),
    ] = False,
    no_cache: NoCache = False,
) -> None:
    """Полный пайплайн: распознать, перевести, записать субтитры.

    Примеры: subvox run film.mkv --to ru
             subvox run film.mkv --to en --from ru --draft
             subvox run film.mkv --to en --format vtt
    """
    try:
        target = langs.require(to)
        if not langs.translatable(target.code):
            raise langs.NoTranslation(
                f"на {target.name} ({target.code}) переводить нечем: NLLB его не поддерживает"
            )

        _check_out(out, fmt)
        prepared = pipeline.prepare(
            video,
            track=track,
            source=source,
            model=model,
            device=device,
            use_cache=not no_cache,
            require_translatable=True,
            note=_note,
        )
        language = prepared.language
        parts = sentences.sentences(prepared.transcript.segments)

        engine: nllb.Engine | None = None
        if target.code == language:
            translated = [part.text for part in parts]
        else:
            engine_device = prepared.loaded.device if prepared.loaded else device
            if engine_device == "cpu" and device != "cpu":
                # Whisper упал на cpu после фолбэка — дать переводчику свой шанс.
                engine_device = "auto"
            engine = nllb.load(device=engine_device, note=_note)
            translated = nllb.translate(
                engine,
                [part.text for part in parts],
                source_lang=langs.flores(language),
                target_lang=target.flores,
            )
    except (
        audio.AudioError,
        vad.NoSpeech,
        asr.TranscriptionError,
        sentences.SentenceError,
        nllb.TranslationError,
        langs.UnknownLanguage,
        langs.NoTranslation,
        ModelError,
        ValueError,
    ) as exc:
        _fail(exc)

    entries = [
        (text, part.start, part.end)
        for text, part in zip(translated, parts, strict=True)
    ]
    cues = cues_mod.build(entries)

    destination = _dest(video, target.code, out, fmt)
    subs.write_cues(cues, destination)
    if draft:
        draft_path = _dest(video, f"{language}.draft", None, fmt)
        subs.write(prepared.transcript.segments, draft_path)

    _gpu_notes(prepared.loaded, engine)

    fast = sum(1 for cue in cues if cue.too_fast)
    notes = []
    if prepared.transcript_cached:
        notes.append("транскрипт из кеша")
    if engine is not None:
        notes.append(f"перевод {engine.device}")
    tail = f" · {' · '.join(notes)}" if notes else ""

    console.print(
        f"[bold]{video.name}[/bold] -> [bold]{destination.name}[/bold]"
    )
    console.print(
        f"  {langs.name(language)} ({language}) -> {target.name} ({target.code}): "
        f"{len(parts)} предложений, {len(cues)} реплик{tail}"
    )
    if fast:
        console.print(
            f"  [yellow]{fast} реплик быстрее {cues_mod.CPS_SOFT:.0f} симв/с — "
            "придётся ставить паузу[/yellow]"
        )
    if prepared.vad_threshold < vad.THRESHOLDS[0]:
        console.print(
            "  [yellow]речь поверх музыки или шума: текст и таймкоды могут быть грубее обычного[/yellow]"
        )


def main() -> None:
    """Точка входа для консольного скрипта."""
    app()


if __name__ == "__main__":
    main()
