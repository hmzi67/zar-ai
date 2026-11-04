"""Cross-encoder reranking of retrieved chunks."""


class Reranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.encoder = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[dict], k: int = 5) -> list[dict]:
        scores = self.encoder.predict([(query, chunk["text"]) for chunk in chunks])
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: -pair[0])
        return [{**chunk, "score": float(score)} for score, chunk in ranked[:k]]
