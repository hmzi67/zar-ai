"""End-to-end answer: image -> classifier -> retrieval -> LLM -> safety.

The classifier's band decides the reply: a non-plant photo is turned away (status "not_plant"), an
unsure one asks for a clearer photo ("clarify"), a likely one is answered as "probably" with a
question to confirm the symptoms, and a confident one is answered plainly. Until the knowledge
base and the generator exist, a question with a photo gets the classifier's result on its own
(status "diagnosis"), so the image model can be tried in the chatbot.
"""

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

NOT_PLANT = {
    "ur": "براہ کرم گندم کے پتے کی تصویر بھیجیں۔",
    "ur-Latn": "Barah-e-karam gandum ke pattay ki tasveer bhejein.",
    "en": "Please send a photo of a wheat leaf.",
}

# Display names for the wheat_v2 classes. Unknown labels are shown as they are.
DISEASE_NAMES = {
    "healthy": {"ur": "صحت مند پتا", "ur-Latn": "Sehatmand patta", "en": "Healthy leaf"},
    "yellow_rust": {
        "ur": "پیلی کنگی (دھاری دار کنگی)",
        "ur-Latn": "Peeli kungi (dhaari daar kungi)",
        "en": "Yellow (stripe) rust",
    },
    "brown_rust": {
        "ur": "بھوری کنگی (پتے کی کنگی)",
        "ur-Latn": "Bhoori kungi (pattay ki kungi)",
        "en": "Brown (leaf) rust",
    },
    "black_rust": {
        "ur": "کالی کنگی (تنے کی کنگی)",
        "ur-Latn": "Kaali kungi (tanay ki kungi)",
        "en": "Black (stem) rust",
    },
    "septoria": {
        "ur": "سیپٹوریا (پتے کے دھبے)",
        "ur-Latn": "Septoria (pattay ke dhabbay)",
        "en": "Septoria leaf blotch",
    },
}

# A question about the symptoms that tells the predicted class apart, asked for a "likely" result.
CONFIRM = {
    "healthy": {
        "ur": "کیا پتا دھبوں، دھاریوں اور سفوف نما دانوں سے پاک ہے؟",
        "ur-Latn": "Kya patta dhabbon, dhaariyon aur safoof-numa daanon se paak hai?",
        "en": "Is the leaf free of spots, stripes and powdery pustules?",
    },
    "yellow_rust": {
        "ur": "کیا دانے پیلے ہیں اور پتے پر دھاریوں کی شکل میں لگے ہیں؟",
        "ur-Latn": "Kya daane peelay hain aur pattay par dhaariyon ki shakal mein lage hain?",
        "en": "Are the pustules yellow and lined up in stripes along the leaf?",
    },
    "brown_rust": {
        "ur": "کیا دانے نارنجی بھورے، چھوٹے اور گول ہیں اور پتے پر بکھرے ہوئے ہیں؟",
        "ur-Latn": "Kya daane naranji-bhooray, chhotay aur gol hain aur pattay par bikhray huay hain?",
        "en": "Are the pustules orange-brown, small and round, and scattered over the leaf?",
    },
    "black_rust": {
        "ur": "کیا دانے گہرے سرخی مائل بھورے سے کالے اور لمبے ہیں، اور تنے پر بھی ہیں؟",
        "ur-Latn": "Kya daane gehray surkhi-maail bhooray se kaalay aur lambay hain, aur tanay par bhi hain?",
        "en": "Are the pustules dark reddish-brown to black and long, and on the stem too?",
    },
    "septoria": {
        "ur": "کیا پتے پر بھورے دھبے ہیں جن کے اندر چھوٹے کالے نقطے ہیں؟",
        "ur-Latn": "Kya pattay par bhooray dhabbay hain jin ke andar chhotay kaalay nuqtay hain?",
        "en": "Are there tan or brown blotches with tiny black dots inside them?",
    },
}

DIAGNOSIS = {
    "ur": {
        "result": "تصویر کے ماڈل کا نتیجہ: {name}، {prob}۔",
        "likely": "تصویر کے ماڈل کا نتیجہ: غالباً {name}، {prob}۔",
        "confirm": "تصدیق کے لیے بتائیں: {question}",
        "others": "دیگر امکانات: {others}۔",
        "note": "علاج کے مشورے ابھی دستیاب نہیں، یہ صرف تصویر کے ماڈل کا اندازہ ہے۔ "
        "براہ کرم زرعی ماہر سے تصدیق کریں۔",
    },
    "ur-Latn": {
        "result": "Tasveer ke model ka nateeja: {name}, {prob}.",
        "likely": "Tasveer ke model ka nateeja: shayad {name}, {prob}.",
        "confirm": "Tasdeeq ke liye batayein: {question}",
        "others": "Deegar imkanaat: {others}.",
        "note": "Ilaaj ke mashware abhi dastiyab nahi, yeh sirf tasveer ke model ka andaza hai. "
        "Barah-e-karam zarai mahir se tasdeeq karein.",
    },
    "en": {
        "result": "Image model result: {name}, {prob}.",
        "likely": "Image model result: probably {name}, {prob}.",
        "confirm": "To confirm: {question}",
        "others": "Other possibilities: {others}.",
        "note": "Treatment advice isn't available yet, so this is only the image model's guess. "
        "Please confirm with an agriculture expert.",
    },
}

# Alternatives below this probability are left out of the diagnosis text.
MIN_SHOWN = 0.01

# What a missing knowledge base or generator raises.
NOT_READY = (NotImplementedError, FileNotFoundError, ImportError)


def disease_name(label: str, lang: str) -> str:
    return DISEASE_NAMES.get(label, {}).get(lang, label)


def confirm_text(label: str, lang: str) -> str | None:
    question = CONFIRM.get(label, {}).get(lang)
    return DIAGNOSIS[lang]["confirm"].format(question=question) if question else None


def diagnosis_text(top: list[dict], lang: str, band: str = "confident") -> str:
    text = DIAGNOSIS[lang]
    best = top[0]
    rest = [p for p in top[1:] if p["prob"] >= MIN_SHOWN]  # 0% alternatives say nothing
    head = text["likely" if band == "likely" else "result"]
    lines = [head.format(name=disease_name(best["label"], lang), prob=f"{best['prob']:.0%}")]
    if rest:
        others = ", ".join(f"{disease_name(p['label'], lang)} {p['prob']:.0%}" for p in rest)
        lines.append(text["others"].format(others=others))
    if band == "likely" and (confirm := confirm_text(best["label"], lang)):
        lines.append(confirm)
    lines.append(text["note"])
    return "\n".join(lines)


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
        if vision and vision.get("band") == "not_plant":
            return {
                "answer": NOT_PLANT[lang],
                "lang": lang,
                "status": "not_plant",
                "sources": [],
                "vision": vision,
            }
        if vision and vision["abstain"]:
            return {
                "answer": CLARIFY[lang],
                "lang": lang,
                "status": "clarify",
                "sources": [],
                "vision": vision,
            }

        top = vision["top"] if vision else None
        band = vision.get("band", "confident") if vision else None
        query = f"{question} {top[0]['label']}" if top else question
        try:
            chunks = self.retriever.search(query, self.top_k)
            if is_off_topic(chunks, self.min_score):
                return {
                    "answer": OFF_TOPIC_REFUSAL[lang],
                    "lang": lang,
                    "status": "refused",
                    "sources": [],
                    "vision": vision,
                }
            raw = self.llm.generate(build_fusion_messages(question, top, chunks, lang))
        except NOT_READY:
            if vision is None:
                raise
            return {
                "answer": diagnosis_text(top, lang, band),
                "lang": lang,
                "status": "diagnosis",
                "sources": [],
                "vision": vision,
            }
        result = enforce(raw, chunks, lang)
        answer = result.answer
        if (
            band == "likely"
            and not result.blocked
            and (confirm := confirm_text(top[0]["label"], lang))
        ):
            answer = f"{answer}\n\n{confirm}"
        return {
            "answer": answer,
            "lang": lang,
            "status": "blocked" if result.blocked else "answered",
            "sources": [chunk["chunk_id"] for chunk in chunks],
            "vision": vision,
        }
