"""BM25 keyword retrieval over KB chunks."""

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from zarai.kb.clean import normalize_letters


def tokenize(text: str) -> list[str]:
    """Lowercase, fold Arabic/Urdu letter variants, strip diacritics, split on non-letters."""
    text = normalize_letters(text).lower()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[^\W_]+", text)


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-k chunks with a positive score, best first, each with a `score` field."""
        scores = self.index.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0]
