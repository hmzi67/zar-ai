"""Text cleaning shared by KB ingestion and retrieval tokenisation."""

import re
import unicodedata

# Arabic code points that look like Urdu letters but do not match them in search.
ARABIC_TO_URDU = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ہ", "ه": "ہ"})


def normalize_letters(text: str) -> str:
    """NFKC folds PDF presentation forms to base letters; then map Arabic letters to Urdu."""
    return unicodedata.normalize("NFKC", text).translate(ARABIC_TO_URDU)


def clean_text(text: str) -> str:
    text = normalize_letters(text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
