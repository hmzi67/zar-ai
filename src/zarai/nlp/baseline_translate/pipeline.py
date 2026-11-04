"""Baseline: translate the question to English, answer in English, translate the answer back."""

from zarai.common.config import load_config, parse_config
from zarai.common.io import read_jsonl, write_jsonl
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.nlp.baseline_translate.nllb import NLLBTranslator
from zarai.nlp.generator.llm import load_llm
from zarai.nlp.generator.prompts import build_messages
from zarai.nlp.retriever.hybrid import build_retriever
from zarai.nlp.safety import enforce
from zarai.nlp.script_detect import detect_script


def translate_answer(question: str, translator, retriever, llm, top_k: int = 5) -> str:
    lang = detect_script(question)
    english = question if lang == "en" else translator.translate(question, lang, "en")
    chunks = retriever.search(english, top_k)
    answer = llm.generate(build_messages(english, chunks, "en"))
    # Safety runs on the English answer because the translation could garble doses.
    answer = enforce(answer, chunks, "en").answer
    return answer if lang == "en" else translator.translate(answer, "en", lang)


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, "translate_baseline")
    log = get_logger(__name__, run_dir)
    retriever_cfg = load_config(cfg.retriever_config)
    retriever = build_retriever(retriever_cfg)
    translator, llm = NLLBTranslator(cfg.model), load_llm(cfg.generator)

    records = [r for r in read_jsonl(cfg.qa_file) if r["lang"] != "ur-Latn"]
    answers = [
        {
            "qid": r["id"],
            "answer": translate_answer(
                r["question"], translator, retriever, llm, retriever_cfg.top_k
            ),
        }
        for r in records
    ]
    write_jsonl(run_dir / "answers.jsonl", answers)
    log.info("wrote %d answers (Roman Urdu skipped: NLLB has no such code)", len(answers))


if __name__ == "__main__":
    main()
