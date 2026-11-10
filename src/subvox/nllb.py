"""Перевод через NLLB-200 (CTranslate2).

Та же инфраструктура, что и у Whisper: CTranslate2, int8_float16 на CUDA,
int8 на CPU, тот же venv с cuDNN. Ни torch, ни transformers — токенизация
через ``tokenizers`` из файла ``tokenizer.json``.

Формат токенов — наследие fairseq:

* источник: ``spm-куски + ["</s>", src_lang]``;
* префикс декодера: ``[tgt_lang]``.

Проверено эмпирически на живой модели; ломать нельзя — модели без
языкового кода угадывают язык выхода вместо перевода.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subvox import paths

DEFAULT_MODEL = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"

MAX_INPUT_TOKENS = 200
"""NLLB учили на последовательностях до 512 токенов; предложение длиннее
200 уже подозрительно — обрезаем, а не молча портим перевод."""

BEAM_SIZE = 4
"""Лучше 5 у Whisper не нужно: у перевода нет отбраковки гипотез,
а цена растёт линейно."""

BATCH_TOKENS = 4096
"""Сколько токенов в батче на GPU. Подбирается под 8 ГБ VRAM
при загруженном Whisper (large-v3 float16 ~3.1 ГБ + NLLB int8 ~1.5 ГБ)."""


class TranslationError(RuntimeError):
    """Перевод не смог запуститься или ничего не вернул."""


@dataclass(frozen=True, slots=True)
class Engine:
    """Загруженный переводчик с токенизатором."""

    translator: Any
    tokenizer: Any
    vocab: dict[str, int]
    device: str
    fallback_reason: str | None = None


NLLB_GBYTES = 1.5
"""Приблизительный размер int8-конверсии для анонса скачивания."""


def _prefetch_nllb(
    model_id: str, note: Callable[[str], None] | None
) -> str:
    """Путь к модели в кеше; при промахе — скачивание с анонсом и барами."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache = str(paths.models_dir())
    try:
        return snapshot_download(model_id, cache_dir=cache, local_files_only=True)
    except LocalEntryNotFoundError:
        pass

    if note is not None:
        note(f"модель NLLB (~{NLLB_GBYTES:.1f} ГБ) качаю в {cache}")
    return snapshot_download(model_id, cache_dir=cache)


def load(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    note: Callable[[str], None] | None = None,
) -> Engine:
    """Загрузить NLLB с фолбэком CUDA -> CPU, как у Whisper."""
    import ctranslate2
    import tokenizers

    from subvox.cuda import preload as _preload_cuda

    _preload_cuda()

    try:
        path = _prefetch_nllb(model_id, note)
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(
            f"модель NLLB не скачалась в {paths.models_dir()}: {exc}"
        ) from exc
    tokenizer = tokenizers.Tokenizer.from_file(
        str(Path(path) / "tokenizer.json")
    )
    vocab = tokenizer.get_vocab()

    errors: list[str] = []
    for dev, compute in (
        ("cuda", "int8_float16"),
        ("cpu", "int8"),
    ):
        if device not in ("auto", dev):
            continue
        try:
            translator = ctranslate2.Translator(path, device=dev, compute_type=compute)
            # Прогон: загрузка переводчика, как и у Whisper, ничего не доказывает.
            translator.translate_batch(
                [["</s>", "eng_Latn"]],
                target_prefix=[["rus_Cyrl"]],
                beam_size=1,
                max_decoding_length=8,
            )
        except Exception as exc:  # noqa: BLE001 — причин может быть много
            errors.append(f"{dev}: {exc}")
            continue
        return Engine(
            translator=translator,
            tokenizer=tokenizer,
            vocab=vocab,
            device=dev,
            fallback_reason=errors[-1] if errors else None,
        )

    if errors:
        raise TranslationError("NLLB не загрузился — " + "; ".join(errors))
    raise ValueError(f"неизвестное устройство: {device!r}")


def _tokens(text: str, tokenizer: Any, lang: str) -> list[str]:
    """Собрать источник в fairseq-формате, обрезая сверхдлинное."""
    pieces = tokenizer.encode(text, add_special_tokens=False).tokens
    if len(pieces) > MAX_INPUT_TOKENS:
        pieces = pieces[:MAX_INPUT_TOKENS]
    return pieces + ["</s>", lang]


def translate(
    engine: Engine,
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[str]:
    """Перевести список предложений с FLORES-кода на FLORES-код."""
    if not texts:
        return []

    sources = [_tokens(text, engine.tokenizer, source_lang) for text in texts]
    prefixes = [[target_lang]] * len(sources)

    results = engine.translator.translate_batch(
        sources,
        target_prefix=prefixes,
        beam_size=BEAM_SIZE,
        max_batch_size=BATCH_TOKENS,
        batch_type="tokens",
    )

    translations: list[str] = []
    for result in results:
        # Гипотеза начинается с языкового кода и кончается </s>.
        # Токенизатор декодирует только id, поэтому словарь токен -> id.
        ids = [
            engine.vocab[token]
            for token in result.hypotheses[0]
            if token in engine.vocab and token not in (source_lang, target_lang, "<s>")
        ]
        text = engine.tokenizer.decode(ids, skip_special_tokens=True)
        if not text.strip():
            raise TranslationError(
                f"пустой перевод для {source_lang} -> {target_lang}; "
                "проверьте языковые коды"
            )
        translations.append(text.strip())

    return translations
