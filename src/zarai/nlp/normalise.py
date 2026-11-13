"""Unicode clean-up for Urdu, Roman Urdu and English text. Changes characters, never wording."""

import re
import unicodedata

from zarai.kb.clean import normalize_letters

# Zero-width and bidi control characters, soft hyphen and the Arabic letter mark.
ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁤﻿؜­]")
# Vowel marks (zabar, zer, pesh, tanween, shadda, jazm …) and superscript alef.
# U+0653-U+0655 (madda, hamza above/below) are left alone: they build letters such as آ and ۂ.
DIACRITICS = re.compile("[ً-ْٖ-ٰٟ]")
ARABIC_LETTER = "؀-ۿݐ-ݿ"


def urdu_punctuation(text: str) -> str:
    """Latin ? ; , and sentence-final . become ؟ ؛ ، ۔ in Urdu-script text. Numbers keep 1,000 and 2.5."""
    text = text.replace("?", "؟").replace(";", "؛")
    text = re.sub(r"(?<!\d),|,(?!\d)", "،", text)
    # Only a full stop that follows an Urdu word (or its closing bracket) and ends a sentence.
    return re.sub(rf"(?<=[{ARABIC_LETTER})])\.(?=\s|$)", "۔", text)


def normalise_text(text: str, lang: str) -> str:
    """Normalise one cell. `lang` is "ur", "ur-Latn" or "en"; Urdu punctuation applies to "ur" only.

    Steps: NFKC + Arabic→Urdu letters (ي ى→ی, ك→ک, ه ة→ہ), drop zero-width characters and
    diacritics, Urdu punctuation for "ur", one space between words, no space before punctuation.
    """
    text = normalize_letters(str(text))
    text = ZERO_WIDTH.sub("", text)
    text = DIACRITICS.sub("", text)
    if lang == "ur":
        text = urdu_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?؟،؛۔)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def match_key(text: str) -> str:
    """Comparison key for duplicate checks: normalised, lower-cased, punctuation removed."""
    text = normalise_text(text, "en").lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()
