"""Достаём звук из видео.

ffmpeg здесь не костыль, а контракт: он единственный, кто одинаково
переваривает mp4, mkv, webm, mov, ts и всё остальное, что открывает mpv.

Декодируем сами, а не отдаём файл faster-whisper, по двум причинам:
нужен явный выбор аудиодорожки (в mkv их бывает пять) и нужен контроль
над частотой дискретизации и числом каналов.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
"""Whisper работает только с 16 кГц моно."""


class AudioError(RuntimeError):
    """Что-то не так с извлечением звука."""


class FFmpegNotFound(AudioError):
    """ffmpeg или ffprobe нет в PATH."""


class NoAudioStream(AudioError):
    """В файле нет звука."""


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int
    """Абсолютный индекс потока в контейнере — то, что идёт в ``-map 0:N``."""

    codec: str
    channels: int
    sample_rate: int
    language: str | None
    title: str | None

    @property
    def label(self) -> str:
        parts = [self.codec, f"{self.channels}ch", f"{self.sample_rate} Hz"]
        if self.language:
            parts.append(self.language)
        if self.title:
            parts.append(self.title)
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    duration: float
    streams: tuple[AudioStream, ...]

    @property
    def has_audio(self) -> bool:
        return bool(self.streams)

    def stream(self, position: int = 0) -> AudioStream:
        if not self.streams:
            raise NoAudioStream(f"в {self.path.name} нет аудиодорожек")
        try:
            return self.streams[position]
        except IndexError:
            raise AudioError(
                f"в {self.path.name} только {len(self.streams)} аудиодорожек, "
                f"запрошена {position}"
            ) from None


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FFmpegNotFound(f"{tool} не найден в PATH")


def probe(path: Path) -> MediaInfo:
    """Узнать длительность и список аудиодорожек."""
    _require("ffprobe")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AudioError(proc.stderr.strip() or f"ffprobe не смог открыть {path.name}")

    data = json.loads(proc.stdout or "{}")
    streams = []
    for raw in data.get("streams", []):
        if raw.get("codec_type") != "audio":
            continue
        tags = raw.get("tags") or {}
        streams.append(
            AudioStream(
                index=int(raw["index"]),
                codec=str(raw.get("codec_name", "?")),
                channels=int(raw.get("channels", 0)),
                sample_rate=int(raw.get("sample_rate", 0)),
                language=tags.get("language"),
                title=tags.get("title"),
            )
        )

    duration = 0.0
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        duration = float(fmt["duration"])

    return MediaInfo(path=path, duration=duration, streams=tuple(streams))


def extract(
    path: Path,
    stream: AudioStream | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Декодировать дорожку в моно float32.

    Возвращает массив в диапазоне [-1, 1] — ровно то, что ждёт Whisper.
    """
    _require("ffmpeg")
    index = 0 if stream is None else stream.index
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:{index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AudioError(proc.stderr.decode("utf-8", "replace").strip())

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioError(f"ffmpeg вернул пустой поток для {path.name}")
    return audio
