"""Detect Urdu script (ur), Roman Urdu (ur-Latn) or English (en) from Unicode ranges and word lists."""

import re

ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]

# Words shared with English ("to", "is", "us", "main", "me", "hi") are left out on purpose.
ROMAN_URDU_WORDS = frozenset("""
    hai hain hay ho hoon tha thi thay ka ki ke ko mein mei se par pe aur ya nahi nahin nai
    kya kyun kyon kaise kese kab kahan kitna kitni kitne kaun konsa konsi kis mera meri mere
    hum ap aap tum yeh ye woh wo karna karein karen kar raha rahi rahe chahiye chahye lagta
    lagti gaya gayi liye wala wali acha achi bohat bahut kuch sab
    gandum kanak fasal kapas makai kamad ganna aloo tamatar chawal dhan kheti khet kisan
    khad pani bemari bimari keera keeray sundi patta patte zehar dawai beej
    """.split())
ENGLISH_WORDS = frozenset("""
    the a an is are was were be been am of in on at by for with from and or but not no
    what which when where why how who whom do does did have has had can could should would will
    i my we our you your it its they their this that these those there
    """.split())


def _is_arabic(char: str) -> bool:
    return any(low <= ord(char) <= high for low, high in ARABIC_RANGES)


def detect_script(text: str) -> str:
    """Return "ur", "ur-Latn" or "en". Text with no letters counts as "en"."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(map(_is_arabic, letters)) / len(letters) >= 0.5:
        return "ur"
    words = re.findall(r"[a-z']+", text.lower())
    roman = sum(w in ROMAN_URDU_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    # A tie goes to Roman Urdu: farmers code-mix English crop words into Urdu sentences.
    return "ur-Latn" if roman and roman >= english else "en"
