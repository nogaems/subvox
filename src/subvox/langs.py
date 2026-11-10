"""Соответствия языков между Whisper (ISO-639-1) и NLLB (FLORES-200).

Три пространства имён, которые надо сводить в одном месте:

* Whisper отдаёт двухбуквенный код ISO-639-1: ``sv``, ``no``, ``zh``.
* NLLB индексирует по FLORES-200: ``swe_Latn``, ``nob_Latn``, ``zho_Hans``.
  Это не просто трёхбуквенный код — суффикс задаёт письменность.
* Человеку нужно название: ``Swedish``.

Таблица сгенерирована из ``LANGUAGES`` в ``openai/whisper`` и
``FAIRSEQ_LANGUAGE_CODES`` в ``transformers`` (токенизатор NLLB), затем
выверена вручную для языков-макроязыков (арабский, персидский, китайский,
норвежский, латышский, албанский и др.), где ISO-639-3 не совпадает
с FLORES-200.

Покрытие неполное и это нормально: Whisper знает 99 языков, NLLB — 202,
но пересечение меньше. Для языков без перевода ``flores`` равен ``None`` —
вызывающий код обязан сообщить об этом явно, а не подставлять английский.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LANGUAGES",
    "Language",
    "NoTranslation",
    "UnknownLanguage",
    "flores",
    "get",
    "name",
    "normalize",
    "require",
    "targets",
    "translatable",
    "untranslatable",
]


class UnknownLanguage(ValueError):
    """Код языка не из тех, что умеет Whisper."""


class NoTranslation(ValueError):
    """Язык распознаётся, но переводить его нечем."""


@dataclass(frozen=True, slots=True)
class Language:
    code: str
    name: str
    flores: str | None


LANGUAGES: dict[str, Language] = {
    "af": Language("af", "Afrikaans", "afr_Latn"),
    "am": Language("am", "Amharic", "amh_Ethi"),
    "ar": Language("ar", "Arabic", "arb_Arab"),
    "as": Language("as", "Assamese", "asm_Beng"),
    "az": Language("az", "Azerbaijani", "azj_Latn"),
    "ba": Language("ba", "Bashkir", "bak_Cyrl"),
    "be": Language("be", "Belarusian", "bel_Cyrl"),
    "bg": Language("bg", "Bulgarian", "bul_Cyrl"),
    "bn": Language("bn", "Bengali", "ben_Beng"),
    "bo": Language("bo", "Tibetan", "bod_Tibt"),
    "br": Language("br", "Breton", None),
    "bs": Language("bs", "Bosnian", "bos_Latn"),
    "ca": Language("ca", "Catalan", "cat_Latn"),
    "cs": Language("cs", "Czech", "ces_Latn"),
    "cy": Language("cy", "Welsh", "cym_Latn"),
    "da": Language("da", "Danish", "dan_Latn"),
    "de": Language("de", "German", "deu_Latn"),
    "el": Language("el", "Greek", "ell_Grek"),
    "en": Language("en", "English", "eng_Latn"),
    "es": Language("es", "Spanish", "spa_Latn"),
    "et": Language("et", "Estonian", "est_Latn"),
    "eu": Language("eu", "Basque", "eus_Latn"),
    "fa": Language("fa", "Persian", "pes_Arab"),
    "fi": Language("fi", "Finnish", "fin_Latn"),
    "fo": Language("fo", "Faroese", "fao_Latn"),
    "fr": Language("fr", "French", "fra_Latn"),
    "gl": Language("gl", "Galician", "glg_Latn"),
    "gu": Language("gu", "Gujarati", "guj_Gujr"),
    "ha": Language("ha", "Hausa", "hau_Latn"),
    "haw": Language("haw", "Hawaiian", None),
    "he": Language("he", "Hebrew", "heb_Hebr"),
    "hi": Language("hi", "Hindi", "hin_Deva"),
    "hr": Language("hr", "Croatian", "hrv_Latn"),
    "ht": Language("ht", "Haitian creole", "hat_Latn"),
    "hu": Language("hu", "Hungarian", "hun_Latn"),
    "hy": Language("hy", "Armenian", "hye_Armn"),
    "id": Language("id", "Indonesian", "ind_Latn"),
    "is": Language("is", "Icelandic", "isl_Latn"),
    "it": Language("it", "Italian", "ita_Latn"),
    "ja": Language("ja", "Japanese", "jpn_Jpan"),
    "jw": Language("jw", "Javanese", "jav_Latn"),
    "ka": Language("ka", "Georgian", "kat_Geor"),
    "kk": Language("kk", "Kazakh", "kaz_Cyrl"),
    "km": Language("km", "Khmer", "khm_Khmr"),
    "kn": Language("kn", "Kannada", "kan_Knda"),
    "ko": Language("ko", "Korean", "kor_Hang"),
    "la": Language("la", "Latin", None),
    "lb": Language("lb", "Luxembourgish", "ltz_Latn"),
    "ln": Language("ln", "Lingala", "lin_Latn"),
    "lo": Language("lo", "Lao", "lao_Laoo"),
    "lt": Language("lt", "Lithuanian", "lit_Latn"),
    "lv": Language("lv", "Latvian", "lvs_Latn"),
    "mg": Language("mg", "Malagasy", None),
    "mi": Language("mi", "Maori", "mri_Latn"),
    "mk": Language("mk", "Macedonian", "mkd_Cyrl"),
    "ml": Language("ml", "Malayalam", "mal_Mlym"),
    "mn": Language("mn", "Mongolian", "khk_Cyrl"),
    "mr": Language("mr", "Marathi", "mar_Deva"),
    "ms": Language("ms", "Malay", "zsm_Latn"),
    "mt": Language("mt", "Maltese", "mlt_Latn"),
    "my": Language("my", "Myanmar", "mya_Mymr"),
    "ne": Language("ne", "Nepali", "npi_Deva"),
    "nl": Language("nl", "Dutch", "nld_Latn"),
    "nn": Language("nn", "Nynorsk", "nno_Latn"),
    "no": Language("no", "Norwegian", "nob_Latn"),
    "oc": Language("oc", "Occitan", "oci_Latn"),
    "pa": Language("pa", "Punjabi", "pan_Guru"),
    "pl": Language("pl", "Polish", "pol_Latn"),
    "ps": Language("ps", "Pashto", "pbt_Arab"),
    "pt": Language("pt", "Portuguese", "por_Latn"),
    "ro": Language("ro", "Romanian", "ron_Latn"),
    "ru": Language("ru", "Russian", "rus_Cyrl"),
    "sa": Language("sa", "Sanskrit", "san_Deva"),
    "sd": Language("sd", "Sindhi", "snd_Arab"),
    "si": Language("si", "Sinhala", "sin_Sinh"),
    "sk": Language("sk", "Slovak", "slk_Latn"),
    "sl": Language("sl", "Slovenian", "slv_Latn"),
    "sn": Language("sn", "Shona", "sna_Latn"),
    "so": Language("so", "Somali", "som_Latn"),
    "sq": Language("sq", "Albanian", "als_Latn"),
    "sr": Language("sr", "Serbian", "srp_Cyrl"),
    "su": Language("su", "Sundanese", "sun_Latn"),
    "sv": Language("sv", "Swedish", "swe_Latn"),
    "sw": Language("sw", "Swahili", "swh_Latn"),
    "ta": Language("ta", "Tamil", "tam_Taml"),
    "te": Language("te", "Telugu", "tel_Telu"),
    "tg": Language("tg", "Tajik", "tgk_Cyrl"),
    "th": Language("th", "Thai", "tha_Thai"),
    "tk": Language("tk", "Turkmen", "tuk_Latn"),
    "tl": Language("tl", "Tagalog", "tgl_Latn"),
    "tr": Language("tr", "Turkish", "tur_Latn"),
    "tt": Language("tt", "Tatar", "tat_Cyrl"),
    "uk": Language("uk", "Ukrainian", "ukr_Cyrl"),
    "ur": Language("ur", "Urdu", "urd_Arab"),
    "uz": Language("uz", "Uzbek", "uzn_Latn"),
    "vi": Language("vi", "Vietnamese", "vie_Latn"),
    "yi": Language("yi", "Yiddish", "ydd_Hebr"),
    "yo": Language("yo", "Yoruba", "yor_Latn"),
    "yue": Language("yue", "Cantonese", "yue_Hant"),
    "zh": Language("zh", "Chinese", "zho_Hans"),
}

# Устаревшие и региональные варианты, которые встречаются в тегах и в CLI.
ALIASES: dict[str, str] = {
    "nb": "no",
    "iw": "he",
    "in": "id",
    "ji": "yi",
    "jv": "jw",
    "mo": "ro",
    "cmn": "zh",
}


def normalize(code: str) -> str:
    """Привести тег к виду, в котором его понимает таблица.

    ``sv-SE`` -> ``sv``, ``zh_Hans`` -> ``zh``, ``nb`` -> ``no``.
    """
    base = code.strip().lower().replace("_", "-").split("-")[0]
    return ALIASES.get(base, base)


def get(code: str) -> Language | None:
    return LANGUAGES.get(normalize(code))


def require(code: str) -> Language:
    lang = get(code)
    if lang is None:
        known = ", ".join(sorted(LANGUAGES))
        raise UnknownLanguage(f"неизвестный язык {code!r}; известны: {known}")
    return lang


def flores(code: str) -> str:
    """Код FLORES-200 для NLLB. Бросает, если перевести нечем."""
    lang = require(code)
    if lang.flores is None:
        raise NoTranslation(
            f"{lang.name} ({lang.code}) распознаётся Whisper, но отсутствует в NLLB"
        )
    return lang.flores


def name(code: str) -> str:
    return require(code).name


def translatable(code: str) -> bool:
    lang = get(code)
    return lang is not None and lang.flores is not None


def targets() -> list[str]:
    """Все языки, на которые можно переводить."""
    return sorted(c for c, l in LANGUAGES.items() if l.flores)


def untranslatable() -> list[str]:
    """Языки, которые Whisper слышит, но NLLB не переводит."""
    return sorted(c for c, l in LANGUAGES.items() if not l.flores)
