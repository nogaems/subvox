"""Перевод NLLB: контракты без модели.

Главное здесь — рантайм-фолбэк CUDA -> CPU. Транзиентный
``cudaErrorInvalidDevice`` не ловится пробным прогоном при загрузке:
он приходит уже на реальном батче, после готового транскрипта.
Падать в этот момент нельзя — перезагружаемся на CPU и продолжаем.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subvox import nllb

VOCAB = {"rus_Cyrl": 1, "ok": 2}


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(tokens=text.split())

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "перевод"


class BoomTranslator:
    def translate_batch(self, *args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("parallel_for failed: cudaErrorInvalidDevice")


class OkTranslator:
    def translate_batch(
        self, sources: list[list[str]], **kwargs: object
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(hypotheses=[["rus_Cyrl", "ok"]]) for _ in sources
        ]


def engine(translator: object, device: str) -> nllb.Engine:
    return nllb.Engine(
        translator=translator,
        tokenizer=FakeTokenizer(),
        vocab=VOCAB,
        device=device,
        model_id="fake/model",
    )


def test_empty_input_needs_no_model() -> None:
    out, used = nllb.translate(
        engine(BoomTranslator(), "cuda"), [], "eng_Latn", "rus_Cyrl"
    )
    assert out == []
    assert used.device == "cuda"


def test_cuda_failure_mid_run_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes: list[str] = []
    loaded: list[tuple[str, str]] = []
    cpu = engine(OkTranslator(), "cpu")

    def fake_load(model_id: str, device: str, note: object = None) -> nllb.Engine:
        loaded.append((model_id, device))
        return cpu

    monkeypatch.setattr(nllb, "load", fake_load)

    out, used = nllb.translate(
        engine(BoomTranslator(), "cuda"),
        ["hello world"],
        "eng_Latn",
        "rus_Cyrl",
        note=notes.append,
    )

    assert out == ["перевод"]
    assert used is cpu  # итоговая строка напечатает cpu, а не cuda
    assert loaded == [("fake/model", "cpu")]  # та же модель, только на CPU
    assert any("cuda" in line and "cpu" in line for line in notes)  # не молча


def test_cpu_failure_is_a_translation_error() -> None:
    with pytest.raises(nllb.TranslationError):
        nllb.translate(
            engine(BoomTranslator(), "cpu"),
            ["hello world"],
            "eng_Latn",
            "rus_Cyrl",
        )
