"""Dense retrieval with sentence-transformers (bge-m3 by default)."""

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"


def encode_texts(texts: list[str], model: str = DEFAULT_MODEL, batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True)


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict],
        model: str = DEFAULT_MODEL,
        batch_size: int = 32,
        index_file: str | Path | None = None,
    ):
        self.chunks = chunks
        self.model = model
        if index_file and Path(index_file).exists():
            self.embeddings = np.load(index_file)
        else:
            self.embeddings = encode_texts([c["text"] for c in chunks], model, batch_size)
        if len(self.embeddings) != len(chunks):
            raise ValueError(f"{index_file} is stale: rebuild it with zarai.kb.build_index")

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.embeddings @ encode_texts([query], self.model)[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
