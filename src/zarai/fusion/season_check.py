"""Flag a vision result that does not fit the month, using only timing rules stated in KB sources."""

from datetime import date

# label: (months the source says it is not active or not yet started, source chunk with the rule)
RULES = {
    "yellow_rust": ({11, 12}, "src_0011_c0001"),  # rust attack starts at the end of January
    "brown_rust": ({11, 12}, "src_0011_c0001"),
    "black_rust": ({11, 12}, "src_0011_c0001"),
}
# Diseases a KB source says are not seen in Pakistan; any detection needs expert checking.
NOT_REPORTED: dict[str, str] = {}

NOTE = {
    "season": {
        "ur": "یہ مسئلہ عام طور پر اس مہینے میں نہیں آتا۔ تصویر کی تصدیق محکمہ زراعت سے کروا لیں۔",
        "ur-Latn": "Yeh masla aam tor par is maheenay mein nahin aata. Tasveer ki tasdeeq mehkma zaraat se karwa lein.",
        "en": "This problem is not usually seen in this month. Please confirm the photo with the agriculture department.",
    },
    "not_reported": {
        "ur": "یہ بیماری ابھی پاکستان میں رپورٹ نہیں ہوئی۔ فوراً زرعی ماہرین سے رابطہ کریں تاکہ نمونہ ٹیسٹ ہو سکے۔",
        "ur-Latn": "Yeh bimari abhi Pakistan mein report nahin hui. Foran zarai mahireen se rabta karein taake namoona test ho sake.",
        "en": "This disease has not been reported in Pakistan. Contact agriculture experts at once so a sample can be tested.",
    },
}


def season_check(label: str | None, lang: str, today: date | None = None) -> dict | None:
    """Return a note and its KB source when the label is unlikely now, else None."""
    if label in NOT_REPORTED:
        return {"note": NOTE["not_reported"][lang], "source": NOT_REPORTED[label]}
    if label in RULES:
        months, source = RULES[label]
        if (today or date.today()).month in months:
            return {"note": NOTE["season"][lang], "source": source}
    return None
