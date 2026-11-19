"""Append grounded Q&A for the new sources (src_0013 to src_0021) to the KB v2 Q&A: KB v3.

Usage: python scripts/kb/build_qa_v3.py [kb_v3_chunks] [qa_v2] [qa_v3_out]
"""

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from facts_v3 import F3  # noqa: E402

CHUNKS = sys.argv[1] if len(sys.argv) > 1 else "data/processed/kb/chunks_v3.jsonl"
QA_V2 = sys.argv[2] if len(sys.argv) > 2 else "data/raw/qa/qa_v2.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/raw/qa/qa_v3.jsonl"
with open(CHUNKS, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]
with open(QA_V2, encoding="utf-8") as f:
    old = [json.loads(line) for line in f]
num = re.compile(r"\d+(?:\.\d+)?")
n = len(old)
fact0 = max(
    int(r["notes"].split(";")[0].split("_")[1]) for r in old if r["notes"].startswith("fact_")
)
new, problems = [], []


def pages(c):
    p = c["pages"]
    return ast.literal_eval(p) if isinstance(p, str) else p


for i, (topic, ev, qs, ans) in enumerate(F3, fact0 + 1):
    hits = [c for c in chunks if ev in c["text"]]
    if len(hits) != 1:
        problems.append(f"fact {i}: evidence matches {len(hits)} chunks: {ev}")
        continue
    h = hits[0]
    for x in num.findall(ans[0]):
        if x not in h["text"]:
            problems.append(f"fact {i}: number {x} not in {h['chunk_id']}")
    pg = ",".join(pages(h))
    note = f"fact_{i:03d}; {h['source_id']} p.{pg}; evidence: {ev}"
    if h["source_id"] == "src_0013":
        note += "; KB text transcribed from page images, spot-check pending"
    if h["source_id"] == "src_0014":
        note += "; commercial source, confirm against government advice"
    for lang, k in [("ur", 0), ("ur-Latn", 1), ("en", 2)]:
        n += 1
        new.append(
            {
                "id": f"qa_{n:04d}",
                "lang": lang,
                "crop": "wheat",
                "topic": topic,
                "season": "rabi",
                "question": qs[k],
                "answer": ans[k],
                "kb_chunk_ids": [h["chunk_id"]],
                "source_type": "llm",
                "annotator": "RA-3",
                "expert_verified": False,
                "verifier": None,
                "notes": note,
            }
        )
qs_all = {r["question"] for r in old}
dups = [r["question"] for r in new if r["question"] in qs_all]
with open(OUT, "w", encoding="utf-8") as f:
    for r in old + new:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(F3), "facts ->", len(new), "new records; total", len(old) + len(new))
print("problems:", problems or "none")
print("duplicate questions:", dups or "none")
