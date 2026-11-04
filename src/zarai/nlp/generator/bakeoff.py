"""Generate answers from every candidate LLM under blinded system IDs, for expert rating."""

import random

from zarai.common.config import load_config, parse_config
from zarai.common.io import read_jsonl, write_json, write_jsonl
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.nlp.generator.llm import LLM
from zarai.nlp.generator.prompts import build_messages
from zarai.nlp.retriever.hybrid import build_retriever


def blind_candidates(candidates: list[str], seed: int) -> dict[str, str]:
    """Map shuffled blinded system IDs (S1, S2, ...) to model names."""
    shuffled = sorted(candidates)
    random.Random(seed).shuffle(shuffled)
    return {f"S{i}": model for i, model in enumerate(shuffled, start=1)}


def main() -> None:
    cfg = parse_config(__doc__)
    if not cfg.candidates:
        raise ValueError("set candidates in configs/nlp/generator_bakeoff.yaml")
    run_dir = start_run(cfg, "generator_bakeoff")
    log = get_logger(__name__, run_dir)

    retriever_cfg = load_config(cfg.retriever_config)
    retriever = build_retriever(retriever_cfg)
    questions = read_jsonl(cfg.qa_file)
    system_map = blind_candidates(list(cfg.candidates), cfg.seed)
    write_json(run_dir / "system_map.json", system_map)  # keep from raters until rating is done

    for system_id, model in system_map.items():
        llm = LLM(model, cfg.backend, cfg.max_new_tokens, cfg.temperature)
        answers = []
        for record in questions:
            chunks = retriever.search(record["question"], retriever_cfg.top_k)
            messages = build_messages(record["question"], chunks, record["lang"])
            answers.append(
                {"qid": record["id"], "system_id": system_id, "answer": llm.generate(messages)}
            )
        write_jsonl(run_dir / f"answers_{system_id}.jsonl", answers)
        log.info("%s: %d answers", system_id, len(answers))


if __name__ == "__main__":
    main()
