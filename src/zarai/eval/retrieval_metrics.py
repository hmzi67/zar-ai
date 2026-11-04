"""Recall@k and MRR, plus a CLI that scores the configured retriever on the Q&A split."""

from zarai.common.config import parse_config
from zarai.common.io import read_jsonl, write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.nlp.retriever.hybrid import build_retriever


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant chunks found in the top k."""
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    return next((1.0 / i for i, c in enumerate(retrieved, start=1) if c in relevant_set), 0.0)


def evaluate_retrieval(runs: list[tuple[list[str], list[str]]], ks: list[int]) -> dict:
    """Mean metrics over (retrieved, relevant) pairs. Pairs with no relevant chunks are skipped."""
    scored = [(retrieved, relevant) for retrieved, relevant in runs if relevant]
    if not scored:
        raise ValueError("no queries with relevant chunks to score")
    metrics = {
        f"recall@{k}": sum(recall_at_k(a, b, k) for a, b in scored) / len(scored) for k in ks
    }
    metrics["mrr"] = sum(reciprocal_rank(a, b) for a, b in scored) / len(scored)
    metrics["n_queries"] = len(scored)
    return metrics


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"rag_eval_{cfg.mode}")
    log = get_logger(__name__, run_dir)
    retriever = build_retriever(cfg)
    ks = list(cfg.eval.ks)
    runs = [
        (
            [chunk["chunk_id"] for chunk in retriever.search(record["question"], max(ks))],
            record["kb_chunk_ids"],
        )
        for record in read_jsonl(cfg.eval.qa_file)
    ]
    write_json(run_dir / "metrics.json", evaluate_retrieval(runs, ks))
    log.info("wrote %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
