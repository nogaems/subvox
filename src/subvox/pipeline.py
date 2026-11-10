"""Связь стадий с рабочими файлами — общая для команд CLI.

Порядок стадий фиксирован: язык (LID или ``--from``), затем транскрипт.
Оба результата дорожатся в рабочем каталоге медиафайла (``store.py``).

Честность ключа: ключ записи — все входы стадии, включая константы
фильтров ASR и лестницы VAD. Любое изменение входа даёт другой ключ
и честный пересчёт; протухшего кеша не бывает. Запись хранит свои
входы при себе и сверяется с ними при чтении.

Стоимость полного попадания — ноль: Whisper не грузится, аудио не
извлекается, итерация по субтитрам занимает доли секунды вместо минут
ASR. Всё взаимодействие с рабочим каталогом — best effort: недоступный
или битый каталог не ломает прогон, откатываясь к полному подсчёту.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from subvox import asr, audio, langs, lid, store, vad
from subvox.model import DEFAULT_WHISPER, Loaded, load_whisper


@dataclass(frozen=True, slots=True)
class LanguageResult:
    """Итог стадии языка."""

    detection: lid.Detection
    loaded: Loaded | None
    cached: bool


@dataclass(frozen=True, slots=True)
class Prepared:
    """Язык и транскрипт — вход стадий ниже по течению."""

    language: str
    vad_threshold: float
    detection: lid.Detection | None
    transcript: asr.Transcript
    loaded: Loaded | None
    language_cached: bool
    transcript_cached: bool
    work_dir: Path | None


def _vad_block() -> dict:
    """Константы, от которых зависит порог VAD."""
    return {
        "thresholds": list(vad.THRESHOLDS),
        "min_speech_seconds": vad.MIN_SPEECH_SECONDS,
        "window_seconds": vad.WINDOW_SECONDS,
    }


def _lid_inputs(track: int, model: str, windows: int) -> dict:
    """Входы стадии языка: всё, от чего зависит её результат."""
    return {
        "track": track,
        "model": model,
        "windows": windows,
        "sample_rate": audio.SAMPLE_RATE,
        "vad": _vad_block(),
    }


def _vad_inputs(track: int) -> dict:
    """Входы стадии порога VAD, когда язык задан вручную."""
    return {"track": track, "sample_rate": audio.SAMPLE_RATE, "vad": _vad_block()}


def _transcript_inputs(
    track: int,
    model: str,
    language: str,
    vad_threshold: float,
    beam_size: int,
) -> dict:
    """Входы ASR: всё, от чего зависит транскрипт, включая фильтры."""
    return {
        "track": track,
        "model": model,
        "language": language,
        "vad_threshold": vad_threshold,
        "beam_size": beam_size,
        "sample_rate": audio.SAMPLE_RATE,
        "filters": {
            "no_speech_prob": asr.NO_SPEECH_PROB,
            "avg_logprob": asr.AVG_LOGPROB,
            "duplicate_start_epsilon": asr.DUPLICATE_START_EPSILON,
            "hallucination_silence": asr.HALLUCINATION_SILENCE,
        },
    }


def _work_dir(video: Path) -> Path | None:
    """Рабочий каталог медиафайла или None, если недоступен."""
    try:
        return store.media_dir(video)
    except OSError:
        return None


def _restore(directory: Path | None, kind: str, key: str, inputs: dict) -> Any:
    """Данные записи, если она есть и её входы совпали. Иначе None."""
    if directory is None:
        return None
    record = store.read_record(directory, kind, key)
    if record is None or record.get("inputs") != inputs:
        return None
    return record.get("data")


def _save(
    directory: Path | None,
    kind: str,
    key: str,
    inputs: dict,
    data: Any,
    *,
    use_cache: bool,
) -> None:
    """Записать запись, если кеш включён. Кеш — оптимизация, не обязанность."""
    if directory is None or not use_cache:
        return
    try:
        store.write_record(directory, kind, key, inputs, data)
        store.register(directory, kind, key)
    except OSError:
        pass


def _save_vad_threshold(
    directory: Path | None, track: int, threshold: float, *, use_cache: bool
) -> None:
    """Порог VAD — побочный продукт LID; храним и для заданного языка."""
    inputs = _vad_inputs(track)
    key = store.record_key("vad", inputs)
    _save(directory, "vad", key, inputs, {"threshold": threshold}, use_cache=use_cache)


def _language_by_lid(
    directory: Path | None,
    *,
    track: int,
    model: str,
    windows: int,
    use_cache: bool,
    audio_samples: Callable[[], np.ndarray],
    whisper_once: Callable[[], Loaded],
) -> tuple[lid.Detection, Loaded | None, bool]:
    """Стадия языка через LID: восстановить из записи или посчитать."""
    inputs = _lid_inputs(track, model, windows)
    key = store.record_key("lid", inputs)

    if use_cache:
        data = _restore(directory, "lid", key, inputs)
        if data is not None:
            try:
                return lid.from_dict(data), None, True
            except (KeyError, TypeError, ValueError):
                pass  # битая запись — как промах

    loaded = whisper_once()
    detection = lid.detect(loaded.model, audio_samples(), windows=windows)
    _save(directory, "lid", key, inputs, lid.to_dict(detection), use_cache=use_cache)
    _save_vad_threshold(directory, track, detection.vad_threshold, use_cache=use_cache)
    return detection, loaded, False


def detect_language(
    video: Path,
    *,
    track: int = 0,
    model: str = DEFAULT_WHISPER,
    device: str = "auto",
    windows: int = 6,
    use_cache: bool = True,
    note: Callable[[str], None] | None = None,
) -> LanguageResult:
    """Определить язык речи, дорожа результатом в рабочем каталоге."""
    directory = _work_dir(video)
    detection, loaded, cached = _language_by_lid(
        directory,
        track=track,
        model=model,
        windows=windows,
        use_cache=use_cache,
        audio_samples=lambda: audio.extract(video, audio.probe(video).stream(track)),
        whisper_once=lambda: load_whisper(model, device, note=note),
    )
    return LanguageResult(detection=detection, loaded=loaded, cached=cached)


def prepare(
    video: Path,
    *,
    track: int = 0,
    source: str | None = None,
    model: str = DEFAULT_WHISPER,
    device: str = "auto",
    windows: int = 6,
    beam_size: int = asr.BEAM_SIZE,
    use_cache: bool = True,
    require_translatable: bool = False,
    note: Callable[[str], None] | None = None,
) -> Prepared:
    """Язык и транскрипт, с кешем и ленивым аудио.

    ``require_translatable`` — ворота для команды ``run``: язык, который
    NLLB не переведёт, останавливает прогон до транскрипции, а не после
    неё. Аудио извлекается только когда оно действительно нужно: полное
    попадание обеих записей обходится без него совсем.
    """
    directory = _work_dir(video)

    samples: np.ndarray | None = None

    def audio_samples() -> np.ndarray:
        nonlocal samples
        if samples is None:
            stream = audio.probe(video).stream(track)
            samples = audio.extract(video, stream)
        return samples

    loaded: Loaded | None = None

    def whisper_once() -> Loaded:
        nonlocal loaded
        if loaded is None:
            loaded = load_whisper(model, device, note=note)
        return loaded

    detection: lid.Detection | None = None
    language_cached = False

    if source is not None:
        # Язык задан вручную: LID не нужен, порог VAD берётся из записи
        # или лестницы Silero.
        language = langs.require(source).code
    else:
        detection, loaded, language_cached = _language_by_lid(
            directory,
            track=track,
            model=model,
            windows=windows,
            use_cache=use_cache,
            audio_samples=audio_samples,
            whisper_once=whisper_once,
        )
        language = detection.language

    # Ворота переводимости — до порога VAD и транскрипции: непереводимый
    # язык останавливает прогон, не извлекая аудио и не грузя Whisper.
    if require_translatable and not langs.translatable(language):
        raise langs.NoTranslation(
            f"{langs.name(language)} ({language}) распознаётся, но перевести его нечем"
        )

    if source is not None:
        vad_inputs = _vad_inputs(track)
        vad_key = store.record_key("vad", vad_inputs)

        threshold: float | None = None
        if use_cache:
            data = _restore(directory, "vad", vad_key, vad_inputs)
            if isinstance(data, dict) and isinstance(
                data.get("threshold"), (int, float)
            ):
                threshold = float(data["threshold"])
                language_cached = True
        if threshold is None:
            # Лестница Silero — секунды, Whisper для неё не нужен.
            _regions, _speech, threshold = vad.speech_regions(audio_samples())
            _save(
                directory, "vad", vad_key, vad_inputs, {"threshold": threshold},
                use_cache=use_cache,
            )
        vad_threshold = threshold
    else:
        vad_threshold = detection.vad_threshold

    t_inputs = _transcript_inputs(track, model, language, vad_threshold, beam_size)
    t_key = store.record_key("transcript", t_inputs)

    if use_cache:
        data = _restore(directory, "transcript", t_key, t_inputs)
        if data is not None:
            try:
                cached_transcript = asr.from_dict(data)
            except (KeyError, TypeError, ValueError):
                cached_transcript = None  # битая запись — как промах
            if cached_transcript is not None:
                return Prepared(
                    language=language,
                    vad_threshold=vad_threshold,
                    detection=detection,
                    transcript=cached_transcript,
                    loaded=loaded,
                    language_cached=language_cached,
                    transcript_cached=True,
                    work_dir=directory,
                )

    transcript = asr.transcribe(
        whisper_once().model,
        audio_samples(),
        language=language,
        vad_threshold=vad_threshold,
        beam_size=beam_size,
    )
    _save(directory, "transcript", t_key, t_inputs, asr.to_dict(transcript), use_cache=use_cache)

    return Prepared(
        language=language,
        vad_threshold=vad_threshold,
        detection=detection,
        transcript=transcript,
        loaded=loaded,
        language_cached=language_cached,
        transcript_cached=False,
        work_dir=directory,
    )
