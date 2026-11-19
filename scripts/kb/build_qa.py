"""Build the KB v2 Q&A (qa.jsonl) from KB-grounded facts and check grounding.

Usage: python scripts/kb/build_qa.py [kb_v2_chunks] [qa_out]
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from facts import UNSAFE, F  # noqa: E402

CHUNKS = sys.argv[1] if len(sys.argv) > 1 else "data/processed/kb/chunks_v2.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/raw/qa/qa_v2.jsonl"
with open(CHUNKS, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]
num = re.compile(r"\d+(?:\.\d+)?")
records, problems, n = [], [], 0
LANGS = [("ur", 0), ("ur-Latn", 1), ("en", 2)]


def add(lang, q, a, topic, season, kb, notes):
    global n
    n += 1
    records.append(
        {
            "id": f"qa_{n:04d}",
            "lang": lang,
            "crop": "wheat",
            "topic": topic,
            "season": season,
            "question": q,
            "answer": a,
            "kb_chunk_ids": kb,
            "source_type": "llm",
            "annotator": "RA-3",
            "expert_verified": False,
            "verifier": None,
            "notes": notes,
        }
    )


for i, (topic, ev, qs, ans) in enumerate(F, 1):
    evs = ev if isinstance(ev, tuple) else (ev,)
    hits = []
    for e in evs:
        found = [c for c in chunks if e in c["text"]]
        if not found:
            problems.append(f"fact {i}: evidence not found: {e}")
        elif found[0] not in hits:
            hits.append(found[0])
    if not hits:
        continue
    kb = [h["chunk_id"] for h in hits]
    pages = ",".join(
        sorted({p for h in hits for p in h["pages"]}, key=lambda x: int(x.split("-")[0]))
    )
    ev = " | ".join(evs)
    support = " ".join(c["text"] for c in hits)
    for x in num.findall(ans[0]):
        if x not in support:
            problems.append(f"fact {i}: number {x} not in cited chunk")
    for lang, k in LANGS:
        add(
            lang,
            qs[k],
            ans[k],
            topic,
            "rabi",
            kb,
            f"fact_{i:03d}; {hits[0]['source_id']} p.{pages}; evidence: {ev}"
            + (
                "; KB text transcribed from page images, spot-check pending"
                if hits[0]["source_id"] == "src_0001"
                else ""
            ),
        )

for j, (qs, ans) in enumerate(UNSAFE, 1):
    for lang, k in LANGS:
        add(
            lang,
            qs[k],
            ans[k],
            "off_topic_unsafe",
            "any",
            [],
            f"unsafe_{j:02d}; safe refusal, no KB chunk by design",
        )

with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(F), "facts", len(UNSAFE), "unsafe", len(records), "records")
print("problems:", problems or "none")
