"""Token-approximate chunking that keeps document metadata on every chunk."""

import math
import re

# shortcut: fixed words-to-tokens ratio; Urdu often runs higher. Use the real tokenizer if chunks get truncated.
TOKENS_PER_WORD = 1.3
SENTENCE_END = re.compile(r"[.!?۔؟]['\"”)]*\s*$")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


def chunk_text(
    text: str, min_tokens: int = 300, max_tokens: int = 500, overlap_tokens: int = 50
) -> list[str]:
    """Split into overlapping chunks of min_tokens..max_tokens, preferring sentence ends.

    A document shorter than min_tokens becomes one short chunk.
    """
    if not 0 <= overlap_tokens < min_tokens <= max_tokens:
        raise ValueError("need 0 <= overlap_tokens < min_tokens <= max_tokens")

    max_words = int(max_tokens / TOKENS_PER_WORD)
    min_words = math.ceil(min_tokens / TOKENS_PER_WORD)
    overlap_words = round(overlap_tokens / TOKENS_PER_WORD)
    words = re.findall(r"\S+\s*", text)  # keeps whitespace so chunks preserve newlines
    n = len(words)

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_words, n)
        if end < n:
            for i in range(end, start + min_words, -1):
                if SENTENCE_END.search(words[i - 1]):
                    end = i
                    break
        chunks.append("".join(words[start:end]).strip())
        if end == n:
            break
        # The last chunk takes extra overlap instead of falling below min_tokens.
        start = min(end - overlap_words, n - min_words)
    return chunks


def chunk_document(text: str, metadata: dict, **chunk_args) -> list[dict]:
    """Chunk one document. `metadata` must hold `source_id` and is copied onto every chunk."""
    return [
        {
            **metadata,
            "chunk_id": f"{metadata['source_id']}_c{i:04d}",
            "text": chunk,
            "n_tokens": estimate_tokens(chunk),
        }
        for i, chunk in enumerate(chunk_text(text, **chunk_args))
    ]
