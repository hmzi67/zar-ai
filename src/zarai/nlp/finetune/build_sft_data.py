"""Build RAFT-style fine-tuning data: each question sees its gold chunks plus distractors."""

import random

from zarai.common.config import parse_config
from zarai.common.io import read_jsonl, write_jsonl
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.nlp.generator.prompts import build_messages


def build_examples(
    records: list[dict],
    chunks: list[dict],
    n_distractors: int = 3,
    p_gold: float = 0.8,
    seed: int = 42,
) -> list[dict]:
    """With probability 1 - p_gold the gold chunks are left out, as RAFT does."""
    rng = random.Random(seed)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    examples = []
    for record in records:
        missing = [cid for cid in record["kb_chunk_ids"] if cid not in by_id]
        if missing:
            raise ValueError(f"{record['id']}: unknown kb_chunk_ids {missing}")
        pool = [c for c in chunks if c["chunk_id"] not in record["kb_chunk_ids"]]
        context = rng.sample(pool, min(n_distractors, len(pool)))
        if rng.random() < p_gold:
            context += [by_id[cid] for cid in record["kb_chunk_ids"]]
        rng.shuffle(context)
        messages = build_messages(record["question"], context, record["lang"])
        messages.append({"role": "assistant", "content": record["answer"]})
        examples.append({"id": record["id"], "lang": record["lang"], "messages": messages})
    return examples


def main() -> None:
    cfg = parse_config(__doc__)
    log = get_logger(__name__, start_run(cfg, "build_sft_data"))
    chunks = read_jsonl(cfg.sft.chunks_file)
    for qa_file, out_file in [(cfg.sft.qa_train, cfg.train_file), (cfg.sft.qa_val, cfg.val_file)]:
        examples = build_examples(
            read_jsonl(qa_file), chunks, cfg.sft.n_distractors, cfg.sft.p_gold, cfg.seed
        )
        write_jsonl(out_file, examples)
        log.info("wrote %d examples to %s", len(examples), out_file)


if __name__ == "__main__":
    main()
