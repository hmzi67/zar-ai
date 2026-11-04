import pytest

from zarai.nlp.finetune.build_sft_data import build_examples
from zarai.nlp.retriever.bm25 import BM25Retriever, tokenize
from zarai.nlp.retriever.hybrid import HybridRetriever

CHUNKS = [
    {"chunk_id": "c1", "text": "wheat rust spreads in cool humid weather"},
    {"chunk_id": "c2", "text": "cotton whitefly control in kharif season"},
    {"chunk_id": "c3", "text": "گندم کی فصل کو پانی دینے کا طریقہ"},
    {"chunk_id": "c4", "text": "rice nursery sowing date"},
    {"chunk_id": "c5", "text": "potato storage temperature"},
]


def test_tokenize_folds_urdu_variants_and_diacritics():
    assert tokenize("كيا Wheat!") == ["کیا", "wheat"]
    assert tokenize("گندَم") == ["گندم"]


def test_bm25_ranks_matching_chunk_first_across_scripts():
    retriever = BM25Retriever(CHUNKS)
    assert retriever.search("wheat rust", 2)[0]["chunk_id"] == "c1"
    assert retriever.search("گندم پانی", 2)[0]["chunk_id"] == "c3"


def test_bm25_returns_only_positive_matches_with_metadata():
    retriever = BM25Retriever(CHUNKS)
    assert retriever.search("no such words", 3) == []
    assert retriever.search("", 3) == []
    results = retriever.search("wheat rust", 3)
    assert len(results) == 1 and results[0]["score"] > 0 and "text" in results[0]


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_hybrid_rrf_prefers_chunks_found_by_both():
    class Static:
        def __init__(self, ids):
            self.ids = ids

        def search(self, query, k):
            return [{"chunk_id": i, "score": 9.0} for i in self.ids[:k]]

    hybrid = HybridRetriever([Static(["a", "b"]), Static(["c", "b"])], rrf_k=60)
    results = hybrid.search("q", 3)
    assert results[0]["chunk_id"] == "b"
    assert results[0]["score"] == pytest.approx(2 / 62)


def qa(ids):
    return {"id": "qa_0001", "lang": "en", "question": "q?", "answer": "a.", "kb_chunk_ids": ids}


def test_build_sft_always_gold_includes_gold_and_distractors():
    [example] = build_examples([qa(["c1"])], CHUNKS, n_distractors=2, p_gold=1.0)
    user = example["messages"][1]["content"]
    assert "[c1]" in user and sum(f"[c{i}]" in user for i in range(2, 6)) == 2
    assert example["messages"][-1] == {"role": "assistant", "content": "a."}


def test_build_sft_never_gold_excludes_gold():
    [example] = build_examples([qa(["c1"])], CHUNKS, n_distractors=2, p_gold=0.0)
    assert "[c1]" not in example["messages"][1]["content"]


def test_build_sft_unknown_gold_raises():
    with pytest.raises(ValueError, match="missing"):
        build_examples([qa(["missing"])], CHUNKS)
