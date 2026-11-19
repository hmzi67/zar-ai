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
| Routed v2 (ur: RRF, else BGE-M3) | 0.780 | 0.476 | 0.646 | 0.557 | 0.283 | 0.431 | `e3_rerank_chunks_kb_v2` |
| Routed v2 + reranker (`bge-reranker-v2-m3`, top 20) | 0.902 | 0.646 | 0.817 | 0.682 | 0.518 | 0.651 | `e3_rerank_chunks_kb_v2` |

The script router sent 0 of 246 queries to the wrong retriever.

## Held-out test

The 82 facts were shuffled (seed 42) and split in half. The setup for each language was chosen on the first half (dev, by MRR) and scored on the second half (test), which played no part in the choice. n = 41 per language per half, so each figure is uncertain by roughly 0.10.

| Language | Setup chosen on dev | Baseline test R@5 | Test R@1 | Test R@5 | Test MRR |
|---|---|---|---|---|---|
| ur | RRF + reranker | 0.805 | 0.463 | 0.951 | 0.648 |
| ur-Latn | Urdu script, RRF + reranker | 0.439 | 0.366 | 0.854 | 0.563 |
| en | BGE-M3 + reranker | 0.561 | 0.341 | 0.780 | 0.532 |

Roman Urdu was converted to Urdu script by `Qwen/Qwen2.5-7B-Instruct` (4-bit, greedy). The conversions are rough (misspellings, some Hindi script), so a better converter may help. Run: `e3_heldout_kb_v2`.

## Reranker and chunk size

The reranker re-scores the top 20 passages from Routed v2. Recall@1 rises from 0.378 to 0.537 (ur), 0.146 to 0.402 (ur-Latn) and 0.268 to 0.512 (en).

Smaller chunks (200-word windows, 40-word overlap, 112 chunks) did not help. With the reranker, Recall@5 was 0.829 (ur), 0.549 (ur-Latn) and 0.768 (en), below the current chunks in every language, so the chunk size stays as it is.

## Recall@5 by topic

Routed v2 without the reranker. Each topic has only 6 to 21 facts, so these are rough.

| Topic | Facts | ur | ur-Latn | en |
|---|---|---|---|---|
| disease | 17 | 0.588 | 0.294 | 0.647 |
| fertiliser | 15 | 0.800 | 0.467 | 0.600 |
| harvest_storage | 6 | 0.833 | 0.167 | 0.500 |
| irrigation | 10 | 1.000 | 0.300 | 0.900 |
| pest | 7 | 0.857 | 0.571 | 0.714 |
| sowing_varieties | 21 | 0.714 | 0.667 | 0.571 |
| weed | 6 | 1.000 | 0.833 | 0.667 |

Five farmer scenarios (late sowing, rust, irrigation, aphids, stubble burning) were checked in each language: Urdu found the right passage in 4 of 5, English in 2 of 5 and Roman Urdu in 1 of 5. Run: `e3_topics_scenarios_kb_v2`.

## Roman Urdu to Urdu script before retrieval

Each `ur-Latn` question was converted to Urdu script by `Qwen/Qwen2.5-7B-Instruct` (4-bit, greedy) and then retrieved. n = 82.

| Query form | Retriever | R@5 |
|---|---|---|
| Roman Urdu (as asked) | BGE-M3 | 0.476 |
| Converted to Urdu script | BGE-M3 | 0.634 |
| Converted to Urdu script | RRF | 0.573 |

Run: `e3_translit_kb_v2`. Some conversions are poor (some questions come back unchanged or with mixed scripts), so a better converter may help more.

## Better converter: Qwen2.5-14B

The 7B converter was replaced by `Qwen/Qwen2.5-14B-Instruct` (4-bit, greedy, 15 examples, one retry when the output is not pure Urdu script). Held-out test half, n = 41 per language.

| Language | Setup | R@1 | R@5 | MRR |
|---|---|---|---|---|
| ur | RRF + reranker | 0.463 | 0.951 | 0.648 |
| ur-Latn | Urdu script (14B), RRF + reranker | 0.439 | 0.902 | 0.624 |
| en | BGE-M3 + reranker | 0.341 | 0.780 | 0.532 |

Roman Urdu R@5 rises from 0.854 (7B) to 0.902. 16 of 82 conversions are still not pure Urdu script (mostly a few Latin letters, two are unusable). Run: `e3_final_kb_v2`, notebook `notebooks/01_ra3_e3_final_kb_v2.ipynb` (only the "Ours" rows of that run are reported here).

## Sarvam as the converter

`sarvamai/sarvam-translate` in place of Qwen2.5-14B, same retrieval, same held-out Roman Urdu questions (n = 41).

| Converter | Not pure Urdu (of 82) | R@1 | R@5 | MRR |
|---|---|---|---|---|
| Qwen2.5-14B | 16 | 0.439 | 0.902 | 0.624 |
| Sarvam | 29 | 0.366 | 0.707 | 0.515 |

Sarvam is worse on every measure, so Qwen2.5-14B stays. Run: `e3_sarvam_kb_v2`, notebook `notebooks/02_ra3_e3_sarvam_kb_v2.ipynb`.

## KB v3

KB v3 adds nine sources to KB v2 (`src_0013` to `src_0021` in `kb/source_register.csv`): 115 chunks from 12 sources. 390 Q&A records (130 facts in three languages), 378 with a cited chunk. Every question's evidence is an exact substring of its chunk, and numbers in answers are checked against the evidence (`scripts/kb/facts_v3.py`, `scripts/kb/build_qa_v3.py`). Same split rule as KB v2 (facts shuffled with seed 42, second half = test), n = 63 per language. The converter is Qwen2.5-14B (25 of 189 conversions not pure Urdu script).

| Method | ur R@5 | ur-Latn R@5 | en R@5 | ur MRR | ur-Latn MRR | en MRR |
|---|---|---|---|---|---|---|
| BM25 | 0.556 | 0.190 | 0.190 | 0.337 | 0.171 | 0.177 |
| BGE-M3 | 0.556 | 0.254 | 0.381 | 0.352 | 0.172 | 0.276 |
| Ours (routed + conversion + reranker) | 0.683 | 0.651 | 0.556 | 0.503 | 0.492 | 0.441 |

Ours leads in all three languages, most in Roman Urdu (+0.40 R@5 over BGE-M3). All scores are lower than on KB v2: there are almost three times as many chunks, several sources repeat the same advice, and English questions are matched against mostly Urdu sources.

Recall@5 of ours by source (all three languages together):

| Source | Questions | R@5 |
|---|---|---|
| src_0001 | 90 | 0.767 |
| src_0013 | 21 | 0.857 |
| src_0011 | 24 | 0.500 |
| src_0015 | 24 | 0.333 |
| src_0016 | 3 | 0.000 |
| src_0017 | 3 | 0.000 |

`src_0016` and `src_0017` are two-column magazine pages (Zarat Nama) whose extracted text mixes the columns; `src_0015` is an English guide. These are the next sources to clean. Run: `e3_kb_v3`, notebook `notebooks/03_ra3_e3_kb_v3.ipynb`.

## Gold audit on KB v3

Some facts are stated in more than one chunk, so a single gold chunk can mark a correct passage as a miss. Each test fact was checked and 35 extra correct chunks were added for 14 facts (`qa_multigold.jsonl`).

| Method | ur R@5 | ur-Latn R@5 | en R@5 |
|---|---|---|---|
| Ours, one gold chunk | 0.683 | 0.651 | 0.556 |
| Ours, all correct chunks | 0.683 | 0.651 | 0.635 |

Only English changes, so missing gold chunks explain little of the drop from KB v2. Run: `e3_kb_v3_multigold`, notebook `notebooks/04_ra3_e3_kb_v3_multigold.ipynb`.

## Runs

| Run | Where | Files |
|---|---|---|
| `e3_bm25_src0001` | local, repo `BM25Retriever` | `metrics.json`, `config.yaml`, `meta.json` |
| `e3_bm25_kb_v2` | local, repo `BM25Retriever` | `metrics.json`, `config.yaml`, `meta.json` |
| `e3_bge_m3_kb_v2` | Kaggle notebook `BGE-M3 benchmark` | `e3_bge_m3_kb_v2_metrics.json` |
| `e3_hybrid_kb_v2` | Kaggle notebook `zarai-e3-hybrid-retrieval` | `e3_hybrid_kb_v2_metrics.json` |
| `e3_translit_kb_v2` | Kaggle notebook `zarai-translit-rag-demo` | `e3_translit_rag_demo.json` |
| `e3_rerank_chunks_kb_v2` | Kaggle notebook `zarai-e3-rerank-chunks` | `e3_rerank_chunks_kb_v2_metrics.json` |
| `e3_topics_scenarios_kb_v2` | Kaggle notebook `zarai-e3-rerank-chunks` | `e3_topics_scenarios_kb_v2.json` |
| `e3_heldout_kb_v2` | Kaggle notebook `zarai-e3-heldout` | `e3_heldout_kb_v2.json` |
| `e3_final_kb_v2` | Kaggle notebook `zarai-e3-final` | `e3_final_kb_v2.json` |
| `e3_sarvam_kb_v2` | Kaggle, `notebooks/02_ra3_e3_sarvam_kb_v2.ipynb` | `e3_sarvam_kb_v2.json` |
| `e3_kb_v3` | Kaggle, `notebooks/03_ra3_e3_kb_v3.ipynb` | `e3_kb_v3.json` |
| `e3_kb_v3_multigold` | Kaggle, `notebooks/04_ra3_e3_kb_v3_multigold.ipynb` | `e3_kb_v3_multigold.json` |

Run outputs are gitignored; copies are kept by RA-3 and can be shared on request.
