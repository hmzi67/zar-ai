"""Turn a vision label into search words in the farmer's script, using the KB's own disease names."""

TERMS = {
    "aphid": {"ur": "سست تیلہ", "ur-Latn": "sust tela", "en": "aphid"},
    "black_rust": {
        "ur": "سیاہ کنگی تنے کی کنگی",
        "ur-Latn": "siyah kungi",
        "en": "black stem rust",
    },
    "blast": {
        "ur": "گندم کا بھبکا بلاسٹ",
        "ur-Latn": "gandum ka bhabka blast",
        "en": "wheat blast",
    },
    "brown_rust": {"ur": "بھوری کنگی", "ur-Latn": "bhoori kungi", "en": "brown leaf rust"},
    "common_root_rot": {"ur": "اکھیڑا جڑ کی سڑن", "ur-Latn": "ukhera", "en": "root rot foot rot"},
    "fusarium_head_blight": {
        "ur": "سٹے کی سڑن",
        "ur-Latn": "sittay ki sarran",
        "en": "fusarium head blight",
    },
    "healthy": None,
    "leaf_blight": {"ur": "پتوں کا جھلساؤ", "ur-Latn": "patton ka jhulsao", "en": "leaf blight"},
    "mildew": {"ur": "سفوفی پھپھوندی", "ur-Latn": "safoofi phaphoondi", "en": "powdery mildew"},
    "mite": {"ur": "جوئیں مائٹ", "ur-Latn": "mite", "en": "mite"},
    "septoria": {"ur": "سپٹوریا پتوں کے دھبے", "ur-Latn": "septoria", "en": "septoria leaf blotch"},
    "smut": {"ur": "کھلی کانگیاری", "ur-Latn": "khuli kangiyari", "en": "loose smut"},
    "stem_fly": {"ur": "تنے کی مکھی", "ur-Latn": "tanay ki makhi", "en": "stem fly"},
    "tan_spot": {
        "ur": "پتوں کے بھورے دھبے",
        "ur-Latn": "patton ke bhooray dhabbay",
        "en": "tan spot",
    },
    "yellow_rust": {"ur": "زرد کنگی", "ur-Latn": "zard kungi", "en": "yellow stripe rust"},
}


def vision_query(question: str, label: str | None, lang: str) -> str:
    """Add the predicted problem in the question's own script; healthy adds nothing, unmapped labels pass as is."""
    if not label or label == "healthy":
        return question
    terms = TERMS.get(label)
    return f"{question} {terms[lang] if terms else label}"
