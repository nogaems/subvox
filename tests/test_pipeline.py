"""Связка стадий с рабочим каталогом: кеш честен и ленив.

Модели и ffmpeg подставные — проверяется поведение связки: что считается,
что берётся из кеша, и что смена любого входа (язык, beam, константы
фильтров) даёт честный промах, а не протухшее попадание.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subvox import asr, audio, langs, lid, pipeline, store, vad

SAMPLES = np.zeros(160_000, np.float32)


def _detection(language: str = "ru") -> lid.Detection:
    return lid.Detection(
        language=language,
        name=langs.name(language),
        probability=0.95,
        alternatives=(lid.LanguageScore("uk", langs.name("uk"), 0.03),),
        windows=2,
        speech_seconds=8.0,
        duration=10.0,
        vad_threshold=0.5,
    )


def _transcript(language: str = "ru") -> asr.Transcript:
    return asr.Transcript(
        language=language,
        duration=10.0,
        segments=(
            asr.Segment(
                start=0.5,
                end=2.0,
                text="привет",
                words=(asr.Word(0.5, 2.0, "привет", 0.9),),
                avg_logprob=-0.3,
                no_speech_prob=0.05,
            ),
        ),
        dropped_segments=0,
        duplicated_segments=0,
    )


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Изолированный SUBVOX_HOME и файл-заглушка вместо медиа."""
    monkeypatch.setenv("SUBVOX_HOME", str(tmp_path))
    path = tmp_path / "v.webm"
    path.write_bytes(b"fake media")
    return path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Подставные ffmpeg и модели со счётчиками вызовов."""
    counts: dict[str, Any] = {
        "probe": 0,
        "extract": 0,
        "whisper": 0,
        "lid": 0,
        "vad": 0,
        "transcribe": 0,
        "asr_calls": [],
    }

    class Stream:
        index = 0
        label = "rus"

    class Info:
        streams = (Stream(),)

        def stream(self, position: int = 0) -> Stream:
            return self.streams[position]

    class Loaded:
        model = object()
        device = "cuda"
        fallback_reason = None

    def fake_probe(path: Path) -> Info:
        counts["probe"] += 1
        return Info()

    def fake_extract(path: Path, stream: Stream, sample_rate: int = 16000) -> np.ndarray:
        counts["extract"] += 1
        return SAMPLES

    def fake_load_whisper(
        model: str, device: str, note: object = None
    ) -> Loaded:
        counts["whisper"] += 1
        counts["note"] = note
        return Loaded()

    def fake_lid_detect(model: object, samples: np.ndarray, **kwargs: Any) -> lid.Detection:
        counts["lid"] += 1
        return _detection()

    def fake_speech_regions(samples: np.ndarray, sample_rate: int = 16000) -> tuple:
        counts["vad"] += 1
        return [], 8.0, 0.5

    def fake_transcribe(
        model: object, samples: np.ndarray, *, language: str, vad_threshold: float, beam_size: int
    ) -> asr.Transcript:
        counts["transcribe"] += 1
        counts["asr_calls"].append(
            {"language": language, "vad_threshold": vad_threshold, "beam_size": beam_size}
        )
        return _transcript(language)

    monkeypatch.setattr(audio, "probe", fake_probe)
    monkeypatch.setattr(audio, "extract", fake_extract)
    monkeypatch.setattr(pipeline, "load_whisper", fake_load_whisper)
    monkeypatch.setattr(lid, "detect", fake_lid_detect)
    monkeypatch.setattr(vad, "speech_regions", fake_speech_regions)
    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    return counts


def test_first_run_computes_and_saves(video: Path, calls: dict[str, Any]) -> None:
    prepared = pipeline.prepare(video)

    assert prepared.language == "ru"
    assert not prepared.language_cached
    assert not prepared.transcript_cached
    assert calls["whisper"] == calls["extract"] == calls["transcribe"] == 1

    manifest = store.load_manifest(store.media_dir(video))
    assert {entry["kind"] for entry in manifest["records"]} == {"lid", "vad", "transcript"}


def test_second_run_is_full_hit(video: Path, calls: dict[str, Any]) -> None:
    first = pipeline.prepare(video)
    second = pipeline.prepare(video)

    assert second.language_cached
    assert second.transcript_cached
    assert calls["whisper"] == 1  # модель не грузилась повторно
    assert calls["extract"] == 1  # аудио не извлекалось повторно
    assert calls["transcribe"] == 1
    assert second.transcript.segments[0].text == first.transcript.segments[0].text
    assert second.transcript.words == first.transcript.words


def test_detect_language_hits_cache(video: Path, calls: dict[str, Any]) -> None:
    first = pipeline.detect_language(video)
    second = pipeline.detect_language(video)

    assert not first.cached
    assert second.cached
    assert calls["whisper"] == 1
    assert second.detection.language == "ru"


def test_source_matching_language_keeps_transcript(video: Path, calls: dict[str, Any]) -> None:
    pipeline.prepare(video)  # авто-LID сказал ru
    prepared = pipeline.prepare(video, source="ru")

    assert prepared.transcript_cached  # тот же язык — тот же транскрипт
    assert calls["transcribe"] == 1
    assert calls["whisper"] == 1  # язык задан вручную, LID не повторялся


def test_source_other_language_is_honest_miss(video: Path, calls: dict[str, Any]) -> None:
    pipeline.prepare(video)
    prepared = pipeline.prepare(video, source="en")

    assert not prepared.transcript_cached
    assert prepared.language == "en"
    assert calls["transcribe"] == 2
    assert calls["asr_calls"][-1]["language"] == "en"


def test_beam_size_change_recomputes(video: Path, calls: dict[str, Any]) -> None:
    pipeline.prepare(video)
    pipeline.prepare(video, beam_size=1)

    assert calls["transcribe"] == 2
    assert calls["asr_calls"][-1]["beam_size"] == 1
    assert calls["whisper"] == 2  # транскрипция заново требует модель


def test_filter_constant_change_invalidates(video: Path, calls: dict[str, Any],
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline.prepare(video)
    monkeypatch.setattr(asr, "NO_SPEECH_PROB", 0.9)  # правка фильтра галлюцинаций
    pipeline.prepare(video)

    assert calls["transcribe"] == 2


def test_vad_ladder_change_invalidates_lid(video: Path, calls: dict[str, Any],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline.prepare(video)
    monkeypatch.setattr(vad, "THRESHOLDS", (0.5, 0.3))
    pipeline.detect_language(video)

    assert calls["lid"] == 2


def test_corrupted_transcript_record_is_miss(video: Path, calls: dict[str, Any]) -> None:
    pipeline.prepare(video)
    record = next((store.media_dir(video) / "transcript").glob("*.json"))
    record.write_text("{не json", encoding="utf-8")

    prepared = pipeline.prepare(video)

    assert not prepared.transcript_cached
    assert calls["transcribe"] == 2


def test_no_cache_neither_reads_nor_writes(video: Path, calls: dict[str, Any]) -> None:
    pipeline.prepare(video, use_cache=False)
    assert store.load_manifest(store.media_dir(video)) is None  # ничего не записано

    pipeline.prepare(video, use_cache=False)
    assert calls["transcribe"] == 2  # и не читалось


def test_untranslatable_source_fails_before_any_work(video: Path, calls: dict[str, Any]) -> None:
    with pytest.raises(langs.NoTranslation):
        pipeline.prepare(video, source="br", require_translatable=True)

    assert calls["transcribe"] == 0
    assert calls["whisper"] == 0
    assert calls["extract"] == 0  # ворота стоят до извлечения аудио


def test_untranslatable_detected_language_stops_before_transcription(
    video: Path, calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lid, "detect", lambda model, samples, **kwargs: _detection("br"))

    with pytest.raises(langs.NoTranslation):
        pipeline.prepare(video, require_translatable=True)

    assert calls["transcribe"] == 0  # транскрипция не стартовала
    assert calls["whisper"] == 1  # язык всё же потребовал модель


def test_note_is_forwarded_to_model_loader(video: Path, calls: dict[str, Any]) -> None:
    """Анонс скачивания — забота CLI: пайплайн лишь пробрасывает колбэк."""
    seen: list[str] = []
    callback = seen.append  # один объект bound method, иначе is не сравнить
    pipeline.prepare(video, note=callback)

    assert calls["note"] is callback


def test_unavailable_store_does_not_break_run(
    video: Path, calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> Path:
        raise OSError("нет места")

    monkeypatch.setattr(store, "media_dir", broken)
    prepared = pipeline.prepare(video)

    assert prepared.work_dir is None
    assert not prepared.transcript_cached
    assert calls["transcribe"] == 1

    pipeline.prepare(video)
    assert calls["transcribe"] == 2  # без каталога кеша нет — честный пересчёт
