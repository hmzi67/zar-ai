"""NLLB translation between Urdu and English for the translate-then-answer baseline."""

# NLLB has no Roman Urdu code, so ur-Latn would need transliteration first.
NLLB_CODES = {"ur": "urd_Arab", "en": "eng_Latn"}


class NLLBTranslator:
    def __init__(self, model: str = "facebook/nllb-200-distilled-600M"):
        self.model = model

    def translate(self, text: str, source: str, target: str) -> str:
        unsupported = {source, target} - set(NLLB_CODES)
        if unsupported:
            raise ValueError(f"NLLB baseline does not support {sorted(unsupported)}")
        raise NotImplementedError(
            f"TODO: load {self.model}, translate {NLLB_CODES[source]} -> {NLLB_CODES[target]}"
        )
