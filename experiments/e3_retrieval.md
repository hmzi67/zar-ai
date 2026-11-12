# E3 retrieval baselines (KB v2)

Which retriever finds the cited KB chunk for wheat questions in Urdu (`ur`), Roman Urdu (`ur-Latn`) and English (`en`)?

Setup: KB v2 is 42 chunks from `src_0001`, `src_0010` and `src_0011` in `kb/source_register.csv`. Queries are the 246 Q&A records with a cited chunk (82 per language; the 12 unsafe refusals have none). Gold is the record's `kb_chunk_ids`. Seed 42. The questions were written for this set and are not a held-out split, so treat these as early numbers.

## Recall@5 and MRR

| Retriever | ur R@5 | ur-Latn R@5 | en R@5 | ur MRR | ur-Latn MRR | en MRR | Run |
|---|---|---|---|---|---|---|---|
| Random | 0.159 | 0.122 | 0.110 | 0.136 | 0.110 | 0.110 | `e3_hybrid_kb_v2` |
| BM25 (repo `BM25Retriever`) | 0.780 | 0.098 | 0.177 | | | | `e3_bm25_kb_v2` |
| BM25 (rank_bm25, same folding) | 0.780 | 0.134 | 0.183 | 0.482 | 0.141 | 0.154 | `e3_hybrid_kb_v2` |
| BGE-M3 dense | 0.732 | 0.476 | 0.646 | 0.503 | 0.283 | 0.431 | `e3_bge_m3_kb_v2` |
| RRF (BM25 + BGE-M3, k=60) | 0.780 | 0.341 | 0.415 | 0.557 | 0.196 | 0.266 | `e3_hybrid_kb_v2` |
| Routed (ur: BM25, else BGE-M3) | 0.780 | 0.476 | 0.646 | 0.482 | 0.283 | 0.431 | `e3_hybrid_kb_v2` |

The script router sent 0 of 246 queries to the wrong retriever.

## Roman Urdu to Urdu script before retrieval

Each `ur-Latn` question was converted to Urdu script by `Qwen/Qwen2.5-7B-Instruct` (4-bit, greedy) and then retrieved. n = 82.

| Query form | Retriever | R@5 |
|---|---|---|
| Roman Urdu (as asked) | BGE-M3 | 0.476 |
| Converted to Urdu script | BGE-M3 | 0.634 |
| Converted to Urdu script | RRF | 0.573 |

Run: `e3_translit_kb_v2`. Some conversions are poor (some questions come back unchanged or with mixed scripts), so a better converter may help more.

## Runs

| Run | Where | Files |
|---|---|---|
| `e3_bm25_src0001` | local, repo `BM25Retriever` | `metrics.json`, `config.yaml`, `meta.json` |
| `e3_bm25_kb_v2` | local, repo `BM25Retriever` | `metrics.json`, `config.yaml`, `meta.json` |
| `e3_bge_m3_kb_v2` | Kaggle notebook `BGE-M3 benchmark` | `e3_bge_m3_kb_v2_metrics.json` |
| `e3_hybrid_kb_v2` | Kaggle notebook `zarai-e3-hybrid-retrieval` | `e3_hybrid_kb_v2_metrics.json` |
| `e3_translit_kb_v2` | Kaggle notebook `zarai-translit-rag-demo` | `e3_translit_rag_demo.json` |

Run outputs are gitignored; copies are kept by RA-3 and can be shared on request.
