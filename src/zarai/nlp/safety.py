"""Answer safety: pesticide and dose grounding, extension-officer disclaimer, off-topic refusal."""

import re
from dataclasses import dataclass

from zarai.kb.clean import normalize_letters
from zarai.nlp.script_detect import detect_script

DISCLAIMER = {
    "ur": "نوٹ: یہ معلومات صرف عمومی رہنمائی کے لیے ہیں۔ کسی بھی زہر یا کھاد کے استعمال سے پہلے اپنے محکمہ زراعت (توسیع) کے افسر سے مشورہ کریں۔",
    "ur-Latn": "Note: Yeh maloomat sirf aam rehnumai ke liye hain. Kisi bhi zehar ya khad ke istemal se pehle apne mehkma zaraat (extension) ke afsar se mashwara karein.",
    "en": "Note: This is general guidance only. Consult your local agricultural extension officer before applying any pesticide or fertiliser.",
}
OFF_TOPIC_REFUSAL = {
    "ur": "معذرت، یہ نظام صرف فصلوں اور کھیتی باڑی سے متعلق سوالات کے جواب دیتا ہے۔",
    "ur-Latn": "Maazrat, yeh nizam sirf faslon aur kheti baari se mutaliq sawalat ke jawab deta hai.",
    "en": "Sorry, this system only answers questions about crops and farming.",
}
UNVERIFIED_DOSE = {
    "ur": "اس زہر یا مقدار کی تصدیق دستیاب معلومات سے نہیں ہو سکی، اس لیے خوراک نہیں بتائی جا رہی۔",
    "ur-Latn": "Is zehar ya miqdar ki tasdeeq dastiyab maloomat se nahi ho saki, is liye khurak nahi batai ja rahi.",
    "en": "This pesticide or dose could not be confirmed from the available sources, so no dose is being given.",
}

# Canonical unit -> spellings, all lowercase.
_UNIT_SPELLINGS = {
    "ml": ["ml", "cc", "millilitre", "milliliter", "ملی لیٹر", "سی سی"],
    "l": ["l", "ltr", "litre", "liter", "litres", "liters", "لیٹر"],
    "mg": ["mg", "ملی گرام"],
    "g": ["g", "gm", "gram", "grams", "گرام"],
    "kg": ["kg", "kilo", "kilogram", "kilograms", "کلو", "کلوگرام", "کلو گرام"],
    "bag": ["bag", "bags", "bori", "boriyan", "بوری", "بوریاں", "بوریوں"],
    "%": ["%", "percent", "فیصد"],
}
_UNIT = {spelling: unit for unit, spellings in _UNIT_SPELLINGS.items() for spelling in spellings}
_UNIT_PATTERN = "|".join(re.escape(s) for s in sorted(_UNIT, key=len, reverse=True))
_DOSE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})(?![^\W\d_])")

# shortcut: starter list of active ingredients, no brand names, numbers-as-words not caught. Get expert review before the farmer pilot.
_PESTICIDE_SPELLINGS = {
    "imidacloprid": ["imidacloprid", "امیڈاکلوپرڈ"],
    "acetamiprid": ["acetamiprid"],
    "chlorpyrifos": ["chlorpyrifos", "کلورپائریفاس"],
    "cypermethrin": ["cypermethrin", "سائپرمیتھرین"],
    "deltamethrin": ["deltamethrin", "ڈیلٹامیتھرین"],
    "bifenthrin": ["bifenthrin", "بائفینتھرین"],
    "lambda-cyhalothrin": ["lambda-cyhalothrin", "lambda cyhalothrin"],
    "emamectin": ["emamectin", "ایمامیکٹن"],
    "mancozeb": ["mancozeb", "مینکوزیب"],
    "carbendazim": ["carbendazim", "کاربینڈازم"],
    "tebuconazole": ["tebuconazole"],
    "propiconazole": ["propiconazole"],
    "metalaxyl": ["metalaxyl"],
    "copper oxychloride": ["copper oxychloride"],
    "glyphosate": ["glyphosate", "گلائفوسیٹ"],
    "atrazine": ["atrazine"],
    "pendimethalin": ["pendimethalin"],
}
_PESTICIDES = {
    canonical: re.compile("|".join(rf"(?<![^\W\d_]){re.escape(s)}(?![^\W\d_])" for s in spellings))
    for canonical, spellings in _PESTICIDE_SPELLINGS.items()
}
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٪٫", "01234567890123456789%.")


def find_risky_strings(text: str) -> set[str]:
    """Canonical pesticide names and doses in text, so "۲۵۰ ملی لیٹر" equals "250 ml"."""
    text = normalize_letters(text).translate(_DIGITS).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    found = {f"{float(number):g} {_UNIT[unit]}" for number, unit in _DOSE.findall(text)}
    return found | {name for name, pattern in _PESTICIDES.items() if pattern.search(text)}


def unsupported_strings(answer: str, chunks: list[dict]) -> list[str]:
    """Pesticides and doses in the answer that no retrieved chunk contains.

    Checks presence only: a real dose attached to the wrong pesticide still passes.
    """
    supported = find_risky_strings("\n".join(chunk["text"] for chunk in chunks))
    return sorted(find_risky_strings(answer) - supported)


def add_disclaimer(answer: str, lang: str) -> str:
    disclaimer = DISCLAIMER[lang]
    return answer if disclaimer in answer else f"{answer}\n\n{disclaimer}"


def is_off_topic(chunks: list[dict], min_score: float = 0.0) -> bool:
    """Off-topic if retrieval found nothing above min_score."""
    return not any(chunk.get("score", 0.0) > min_score for chunk in chunks)


@dataclass
class SafetyResult:
    answer: str
    unsupported: list[str]

    @property
    def blocked(self) -> bool:
        return bool(self.unsupported)


def enforce(answer: str, chunks: list[dict], lang: str | None = None) -> SafetyResult:
    """Replace the answer if it holds an ungrounded pesticide or dose; always add the disclaimer."""
    lang = lang or detect_script(answer)
    unsupported = unsupported_strings(answer, chunks)
    text = UNVERIFIED_DOSE[lang] if unsupported else answer
    return SafetyResult(add_disclaimer(text, lang), unsupported)
