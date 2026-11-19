# scripts/kb

Build the knowledge base chunks and the Q&A dataset from the KB source texts (RA-3).

| Script | Input | Output |
|---|---|---|
| `build_qa.py` | KB v2 chunks, `facts.py` | Q&A v2 (258 records: 82 facts in ur, ur-Latn, en, plus 12 safe refusals) |
| `chunk_new.py` | KB text for `src_0013` to `src_0021`, KB v2 chunks | KB v3 chunks (115) |
| `build_qa_v3.py` | KB v3 chunks, Q&A v2, `facts_v3.py` | Q&A v3 (390 records, 378 with a cited chunk) |

Every fact's evidence must be an exact substring of exactly one chunk, and every number in an answer must appear in the evidence; the scripts print any record that fails. Paths default to `data/interim/kb_text`, `data/processed/kb/` and `data/raw/qa/`, which are DVC-tracked, and can be passed as arguments. Running the scripts on the RA-3 copies of the inputs reproduces the KB v3 chunks and both Q&A files byte for byte.
