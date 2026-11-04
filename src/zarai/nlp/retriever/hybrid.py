"""Hybrid retrieval (reciprocal rank fusion) and the retriever factory used by every pipeline."""

from zarai.common.io import read_jsonl
from zarai.nlp.retriever.bm25 import BM25Retriever
from zarai.nlp.retriever.dense import DenseRetriever
from zarai.nlp.retriever.rerank import Reranker


class HybridRetriever:
    def __init__(self, retrievers: list, rrf_k: int = 60, candidates: int = 50):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[dict]:
        fused: dict[str, dict] = {}
        for retriever in self.retrievers:
            for rank, chunk in enumerate(retriever.search(query, self.candidates), start=1):
                entry = fused.setdefault(chunk["chunk_id"], {**chunk, "score": 0.0})
                entry["score"] += 1.0 / (self.rrf_k + rank)
        return sorted(fused.values(), key=lambda c: -c["score"])[:k]


class RerankedRetriever:
    def __init__(self, retriever, reranker: Reranker, candidates: int = 20):
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[dict]:
        chunks = self.retriever.search(query, self.candidates)
        return self.reranker.rerank(query, chunks, k) if chunks else []


def build_retriever(cfg):
    """Build the retriever named by cfg.mode over cfg.chunks_file (configs/nlp/retriever.yaml)."""
    chunks = read_jsonl(cfg.chunks_file)

    def dense() -> DenseRetriever:
        return DenseRetriever(chunks, cfg.dense.model, cfg.dense.batch_size, cfg.dense.index_file)

    if cfg.mode == "bm25":
        retriever = BM25Retriever(chunks)
    elif cfg.mode == "dense":
        retriever = dense()
    elif cfg.mode == "hybrid":
        retriever = HybridRetriever(
            [BM25Retriever(chunks), dense()], cfg.hybrid.rrf_k, cfg.hybrid.candidates
        )
    else:
        raise ValueError(f"unknown retriever mode: {cfg.mode}")
    if cfg.rerank.enabled:
        return RerankedRetriever(retriever, Reranker(cfg.rerank.model), cfg.rerank.candidates)
    return retriever
