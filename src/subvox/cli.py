"""Точка входа CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from subvox import __version__, asr, audio, langs, lid, nllb, sentences, subs, vad
from subvox import cues as cues_mod
from subvox.model import DEFAULT_WHISPER, ModelError, load_whisper

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
    as_json: Annotated[bool, typer.Option("--json", help="Машинный вывод.")] = False,
) -> None:
    """Определить язык речи в видео и ничего больше не делать."""
    try:
        info = audio.probe(video)
        stream = info.stream(track)
        samples = audio.extract(video, stream)
        loaded = load_whisper(model, device)
        result = lid.detect(loaded.model, samples, windows=windows)
    except (audio.AudioError, vad.NoSpeech, ModelError, ValueError) as exc:
        _fail(exc)

    if loaded.fallback_reason:
        stderr.print(
            f"[yellow]GPU не поднялся, работаю на {loaded.device}: "
            f"{loaded.fallback_reason}[/yellow]"
        )

    if as_json:
        payload: dict[str, Any] = {
            "file": str(video),
            "duration": round(result.duration, 2),
            "speech_seconds": round(result.speech_seconds, 2),
            "windows": result.windows,
            "vad_threshold": result.vad_threshold,
            "device": loaded.device,
            "language": result.language,
            "name": result.name,
            "probability": round(result.probability, 4),
            "confident": result.confident,
            "translatable": langs.translatable(result.language),
            "alternatives": [
                {
                    "code": alt.code,
                    "name": alt.name,
                    "probability": round(alt.probability, 4),
                }
                for alt in result.alternatives
            ],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    console.print(
        f"[bold]{video.name}[/bold] · {_clock(result.duration)} · "
        f"дорожка {stream.index} [dim]({stream.label})[/dim]"
    )
    console.print()
    console.print(
        f"  [bold]{result.name}[/bold] ({result.language})   "
        f"[green]{result.probability:.2f}[/green]"
    )
    if result.alternatives:
        others = " · ".join(f"{alt.name} {alt.probability:.2f}" for alt in result.alternatives)
        console.print(f"  [dim]{others}[/dim]")
    console.print(
        f"  [dim]речь {_clock(result.speech_seconds)} из {_clock(result.duration)}, "
        f"окон: {result.windows}[/dim]"
    )

    if result.vad_threshold < vad.THRESHOLDS[0]:
        console.print(
            f"  [yellow]речь поверх музыки или шума: VAD сработал только на пороге "
            f"{result.vad_threshold}, результат может быть грубее обычного[/yellow]"
        )

    if not result.confident:
        console.print(
            "  [yellow]уверенности мало: язык стоит задать вручную через --from[/yellow]"
        )
    if not langs.translatable(result.language):
        console.print(
            f"  [yellow]{result.name} распознаётся, но перевести его нечем: "
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
        typer.Option("--from", help="Язык оригинала. По умолчанию определяется."),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Модель Whisper.")] = DEFAULT_WHISPER,
    device: Annotated[str, typer.Option("--device", help="auto, cuda или cpu.")] = "auto",
    track: Annotated[
        int, typer.Option("--track", "-t", help="Номер аудиодорожки, с нуля.")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Куда писать .srt. По умолчанию video.<язык>.srt."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Машинный вывод.")] = False,
) -> None:
    """Распознать речь и записать черновой SRT на языке оригинала."""
    try:
        info = audio.probe(video)
        stream = info.stream(track)
        samples = audio.extract(video, stream)
        loaded = load_whisper(model, device)

        if source is not None:
            language = langs.require(source).code
            _regions, _speech, vad_threshold = vad.speech_regions(samples)
        else:
            detection = lid.detect(loaded.model, samples)
            language = detection.language
            vad_threshold = detection.vad_threshold

        transcript = asr.transcribe(
            loaded.model, samples, language=language, vad_threshold=vad_threshold
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

    destination = out or video.with_suffix(f".{language}.srt")
    if as_json:
        payload = asr.to_dict(transcript)
        payload["file"] = str(video)
        payload["device"] = loaded.device
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    subs.write(transcript.segments, destination)

    notes = []
    if transcript.dropped_segments:
        notes.append(f"отброшено пустых/тихих: {transcript.dropped_segments}")
    if transcript.duplicated_segments:
        notes.append(f"свернуто повторов: {transcript.duplicated_segments}")
    notes.append(f"VAD {vad_threshold}")

    console.print(
        f"[bold]{video.name}[/bold] -> [bold]{destination.name}[/bold] "
        f"[dim](черновик)[/dim]"
    )
    console.print(
        f"  {langs.name(language)} ({language}): {len(transcript.segments)} сегментов, "
        f"{transcript.words} слов · {' · '.join(notes)}"
    )
    if vad_threshold < vad.THRESHOLDS[0]:
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
        typer.Option("--to", help="Целевой язык, ISO-639-1: ru, en, sv..."),
    ],
    source: Annotated[
        str | None,
        typer.Option("--from", help="Язык оригинала. По умолчанию определяется."),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Модель Whisper.")] = DEFAULT_WHISPER,
    device: Annotated[str, typer.Option("--device", help="auto, cuda или cpu.")] = "auto",
    track: Annotated[
        int, typer.Option("--track", "-t", help="Номер аудиодорожки, с нуля.")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Куда писать .srt. По умолчанию video.<язык>.srt."),
    ] = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Дополнительно сохранить черновик на языке оригинала."),
    ] = False,
) -> None:
    """Полный пайплайн: распознать, перевести, записать субтитры."""
    try:
        target = langs.require(to)
        if not langs.translatable(target.code):
            raise langs.NoTranslation(
                f"на {target.name} ({target.code}) переводить нечем: NLLB его не поддерживает"
            )

        info = audio.probe(video)
        stream = info.stream(track)
        samples = audio.extract(video, stream)
        loaded = load_whisper(model, device)

        if source is not None:
            language = langs.require(source).code
            if not langs.translatable(language):
                raise langs.NoTranslation(
                    f"{langs.name(language)} ({language}) распознаётся, но перевести его нечем"
                )
            _regions, _speech, vad_threshold = vad.speech_regions(samples)
        else:
            detection = lid.detect(loaded.model, samples)
            language = detection.language
            vad_threshold = detection.vad_threshold
            if not langs.translatable(language):
                raise langs.NoTranslation(
                    f"{langs.name(language)} ({language}) распознаётся, но перевести его нечем"
                )

        transcript = asr.transcribe(
            loaded.model, samples, language=language, vad_threshold=vad_threshold
        )
        parts = sentences.sentences(transcript.segments)

        if target.code == language:
            translated = [part.text for part in parts]
        else:
            engine = nllb.load(device=loaded.device if loaded.device != "cpu" else "auto")
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

    destination = out or video.with_suffix(f".{target.code}.srt")
    subs.write_cues(cues, destination)
    if draft:
        draft_path = video.with_suffix(f".{language}.draft.srt")
        subs.write(transcript.segments, draft_path)

    fast = sum(1 for cue in cues if cue.too_fast)
    engine_note = ""
    if target.code != language:
        engine_note = f" · перевод {loaded.device}"

    console.print(
        f"[bold]{video.name}[/bold] -> [bold]{destination.name}[/bold]"
    )
    console.print(
        f"  {langs.name(language)} ({language}) -> {target.name} ({target.code}): "
        f"{len(parts)} предложений, {len(cues)} реплик{engine_note}"
    )
    if fast:
        console.print(
            f"  [yellow]{fast} реплик быстрее {cues_mod.CPS_SOFT:.0f} симв/с — "
            "придётся ставить паузу[/yellow]"
        )
    if vad_threshold < vad.THRESHOLDS[0]:
        console.print(
            "  [yellow]речь поверх музыки или шума: текст и таймкоды могут быть грубее обычного[/yellow]"
        )


def main() -> None:
    """Точка входа для консольного скрипта."""
    app()


if __name__ == "__main__":
    main()
