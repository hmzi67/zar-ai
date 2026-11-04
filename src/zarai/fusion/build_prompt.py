"""Combine the image classifier's top-3, the question and retrieved chunks into one prompt."""

from zarai.nlp.generator.prompts import build_messages

VISION_HEADER = {
    "ur": "تصویر کے تجزیے کے ممکنہ نتائج (غلط بھی ہو سکتے ہیں)",
    "ur-Latn": "Tasveer ke tajziye ke mumkina nataij (ghalat bhi ho sakte hain)",
    "en": "Possible results from the image classifier (may be wrong)",
}


def build_fusion_messages(
    question: str, vision_top: list[dict] | None, chunks: list[dict], lang: str
) -> list[dict]:
    extra = None
    if vision_top:
        lines = "\n".join(f"- {item['label']}: {item['prob']:.2f}" for item in vision_top)
        extra = f"{VISION_HEADER[lang]}:\n{lines}"
    return build_messages(question, chunks, lang, extra)
