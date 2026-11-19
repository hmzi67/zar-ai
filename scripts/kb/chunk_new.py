"""Chunk the new KB sources (src_0013 to src_0021) and append them to the KB v2 chunks: KB v3.

Usage: python scripts/kb/chunk_new.py [kb_text_dir] [kb_v2_chunks] [kb_v3_chunks]
"""

import json
import re
import sys
from collections import Counter

KB = sys.argv[1] if len(sys.argv) > 1 else "data/interim/kb_text"
KB_V2 = sys.argv[2] if len(sys.argv) > 2 else "data/processed/kb/chunks_v2.jsonl"
KB_V3 = sys.argv[3] if len(sys.argv) > 3 else "data/processed/kb/chunks_v3.jsonl"
MAXW = 300
META = {
    "src_0013": dict(
        title="گندم کی کاشت (Wheat Cultivation booklet)",
        publisher="Wheat Programme, Crop Sciences Institute, NARC, Pakistan Agricultural Research Council, Islamabad",
        year="2023",
        language="ur",
        url="TO ADD (PDF supplied by RA-3)",
        licence="TO CONFIRM (government booklet, no licence stated)",
    ),
    "src_0014": dict(
        title="Wheat Production Plan by GROWTECH",
        publisher="Growtech, Islamabad (private company)",
        year="2022",
        language="en",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM (company advisory, no licence stated; commercial source)",
    ),
    "src_0015": dict(
        title="Wheat Production Technology (Khyber Pakhtunkhwa)",
        publisher="TO CONFIRM (likely Agriculture Department, Khyber Pakhtunkhwa)",
        year="2024",
        language="en",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM",
    ),
    "src_0016": dict(
        title="آبپاش علاقوں میں گندم کی کاشت (Zarat Nama, 1 Nov 2025)",
        publisher="Directorate General Agriculture Information, Agriculture Department Punjab",
        year="2025",
        language="ur",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM (government magazine)",
    ),
    "src_0017": dict(
        title="موسمی تغیرات اور گندم کی آبپاشی (Zarat Nama, 1 Jan 2026)",
        publisher="Directorate General Agriculture Information, Agriculture Department Punjab",
        year="2026",
        language="ur",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM (government magazine)",
    ),
    "src_0018": dict(
        title="بارانی علاقوں میں گندم کی کاشت اور بیج سے منتقل ہونے والی بیماریاں (Zarat Nama, 1 Oct 2026)",
        publisher="Directorate General Agriculture Information, Agriculture Department Punjab",
        year="2026",
        language="ur",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM (government magazine)",
    ),
    "src_0019": dict(
        title="گندم کی کاشت بذریعہ ڈرل و سپر سیڈر (Zarat Nama, 1 Nov 2026)",
        publisher="Wheat Research Institute, AARI Faisalabad, in Zarat Nama (Agriculture Department Punjab)",
        year="2026",
        language="ur",
        url="TO ADD (text supplied by RA-3)",
        licence="TO CONFIRM (government magazine)",
    ),
    "src_0020": dict(
        title="گندم کا سست تیلہ (Plantwise Factsheet for Farmers)",
        publisher="CABI PlantwisePlus Knowledge Bank (M. Hasnain)",
        year="2013",
        language="ur",
        url="https://doi.org/10.1079/pwkb.20167801418",
        licence="Open access (CABI PlantwisePlus factsheet)",
    ),
    "src_0021": dict(
        title="گندم کے تنے کی کنگی (Plantwise Factsheet for Farmers)",
        publisher="CABI PlantwisePlus Knowledge Bank (M. U. Saleem)",
        year="2016",
        language="ur",
        url="https://doi.org/10.1079/pwkb.20167801416",
        licence="Open access (CABI PlantwisePlus factsheet)",
    ),
}


def units(sid):
    with open(f"{KB}/{sid}.txt", encoding="utf-8") as f:
        text = f.read()
    if sid == "src_0014":
        text = re.sub(r"Address: Plot 60-C.*\n", "", text)
    out, page = [], "-"
    for line in text.splitlines():
        m = re.match(r"=== page (\d+) ===", line)
        if m:
            page = m.group(1)
            continue
        if line.strip():
            out.append((page, line.strip()))
    return out


def chunk_source(sid):
    """Pack whole lines into chunks of at most MAXW words, keeping the pages each chunk covers."""
    out, buf, pages = [], [], []

    def emit():
        ps = sorted(set(pages), key=lambda p: int(p) if p.isdigit() else 0)
        out.append(
            {
                "source_id": sid,
                **META[sid],
                "chunk_id": f"{sid}_c{len(out) + 1:04d}",
                "text": "\n".join(buf),
                "pages": str(ps),
            }
        )

    for page, line in units(sid):
        w = len(line.split())
        if buf and sum(len(b.split()) for b in buf) + w > MAXW:
            emit()
            buf, pages = [], []
        buf.append(line)
        pages.append(page)
    if buf:
        emit()
    return out


chunks = [c for sid in META for c in chunk_source(sid)]
with open(KB_V2, encoding="utf-8") as f:
    old = [json.loads(line) for line in f]
with open(KB_V3, "w", encoding="utf-8") as f:
    for c in old + chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(len(old), "old +", len(chunks), "new", Counter(c["source_id"] for c in chunks))
print([len(c["text"].split()) for c in chunks])
