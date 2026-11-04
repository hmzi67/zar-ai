"""Clean and chunk ingested KB text into chunks.jsonl, optionally embedding it for dense search."""

import logging
from pathlib import Path

import numpy as np

from zarai.common.config import parse_config
from zarai.common.io import read_csv, read_jsonl, write_jsonl
from zarai.common.logging import get_logger
from zarai.kb.chunk import chunk_document
from zarai.kb.clean import clean_text
from zarai.nlp.retriever.dense import encode_texts

log = logging.getLogger(__name__)
CHUNK_METADATA = ["source_id", "title", "publisher", "year", "language", "url", "licence"]


def build_chunks(cfg) -> list[dict]:
    chunks = []
    for row in read_csv(cfg.register):
        text_path = Path(cfg.text_dir) / f"{row['source_id']}.txt"
        if not text_path.exists():
            log.warning("%s: no ingested text, run zarai.kb.ingest", row["source_id"])
            continue
        metadata = {key: row[key] for key in CHUNK_METADATA}
        text = clean_text(text_path.read_text(encoding="utf-8"))
        chunks += chunk_document(
            text,
            metadata,
            min_tokens=cfg.chunk.min_tokens,
            max_tokens=cfg.chunk.max_tokens,
            overlap_tokens=cfg.chunk.overlap_tokens,
        )
    return chunks


def main() -> None:
    cfg = parse_config(__doc__)
    get_logger(__name__)
    write_jsonl(cfg.chunks_file, build_chunks(cfg))
    chunks = read_jsonl(cfg.chunks_file)
    log.info("wrote %d chunks to %s", len(chunks), cfg.chunks_file)
    if cfg.dense.enabled:
        embeddings = encode_texts(
            [c["text"] for c in chunks], cfg.dense.model, cfg.dense.batch_size
        )
        Path(cfg.dense.index_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(cfg.dense.index_file, embeddings)
        log.info("wrote embeddings to %s", cfg.dense.index_file)


if __name__ == "__main__":
    main()
