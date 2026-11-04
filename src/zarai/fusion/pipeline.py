"""End-to-end answer: image -> classifier -> retrieval -> LLM -> safety. Abstaining asks to clarify."""

from zarai.common.config import load_config
from zarai.fusion.build_prompt import build_fusion_messages
from zarai.nlp.generator.llm import load_llm
from zarai.nlp.retriever.hybrid import build_retriever
from zarai.nlp.safety import OFF_TOPIC_REFUSAL, enforce, is_off_topic
from zarai.nlp.script_detect import detect_script
from zarai.vision.models import load_classifier

CLARIFY = {
    "ur": "تصویر سے مسئلہ واضح نہیں ہو رہا۔ براہ کرم قدرتی روشنی میں پتے کی قریب سے صاف تصویر بھیجیں اور علامات بتائیں۔",
    "ur-Latn": "Tasveer se masla wazeh nahi ho raha. Barah-e-karam qudrati roshni mein pattay ki qareeb se saaf tasveer bhejein aur alamaat batayein.",
    "en": "The photo does not make the problem clear. Please send a close, sharp photo of the leaf in natural light and describe the symptoms.",
}


class Pipeline:
    """Components load on first use, so a missing model only fails the endpoints that need it."""

    def __init__(self, cfg=None, retriever=None, llm=None, classifier=None):
        self.cfg = cfg
        self.top_k = load_config(cfg.retriever_config).top_k if cfg else 5
        self.min_score = cfg.min_score if cfg else 0.0
        self._retriever, self._llm, self._classifier = retriever, llm, classifier

    @classmethod
    def from_config(cls, path: str) -> "Pipeline":
        return cls(load_config(path))

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = build_retriever(load_config(self.cfg.retriever_config))
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = load_llm(self.cfg.generator)
        return self._llm

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = load_classifier(load_config(self.cfg.vision_config))
        return self._classifier

    def classify(self, image: bytes) -> dict:
        return self.classifier.classify(image)

    def answer(self, question: str, image: bytes | None = None) -> dict:
        lang = detect_script(question)
        vision = self.classify(image) if image is not None else None
        if vision and vision["abstain"]:
            return {"answer": CLARIFY[lang], "lang": lang, "status": "clarify", "sources": []}

        top = vision["top"] if vision else None
        query = f"{question} {top[0]['label']}" if top else question
        chunks = self.retriever.search(query, self.top_k)
        if is_off_topic(chunks, self.min_score):
            return {
                "answer": OFF_TOPIC_REFUSAL[lang],
                "lang": lang,
                "status": "refused",
                "sources": [],
            }

        raw = self.llm.generate(build_fusion_messages(question, top, chunks, lang))
        result = enforce(raw, chunks, lang)
        return {
            "answer": result.answer,
            "lang": lang,
            "status": "blocked" if result.blocked else "answered",
            "sources": [chunk["chunk_id"] for chunk in chunks],
        }
