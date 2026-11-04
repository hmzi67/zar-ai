import pytest

from zarai.kb.chunk import chunk_document, chunk_text, estimate_tokens
from zarai.kb.clean import clean_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("n", [231, 385, 386, 600, 1000, 5000])
def test_chunks_stay_in_token_range_and_cover_all_words(n):
    chunks = chunk_text(words(n))
    assert all(300 <= estimate_tokens(c) <= 500 for c in chunks)
    assert {w for c in chunks for w in c.split()} == set(words(n).split())


@pytest.mark.parametrize("overlap", [0, 50, 100])
def test_overlap_is_configurable(overlap):
    first, second = chunk_text(words(1000), overlap_tokens=overlap)[:2]
    assert len(set(first.split()) & set(second.split())) == round(overlap / 1.3)


def test_short_document_is_one_chunk():
    assert chunk_text("only a few words") == ["only a few words"]
    assert chunk_text("") == []


def test_prefers_urdu_sentence_end():
    sentence = " ".join(["لفظ"] * 299) + " ختم۔"
    chunks = chunk_text(sentence + " " + words(600))
    assert chunks[0].endswith("ختم۔")


def test_bad_limits_raise():
    with pytest.raises(ValueError):
        chunk_text("x", min_tokens=50, overlap_tokens=50)


def test_metadata_is_kept():
    meta = {"source_id": "s1", "title": "t", "language": "ur"}
    chunks = chunk_document(words(1000), meta)
    assert [c["chunk_id"] for c in chunks][:2] == ["s1_c0000", "s1_c0001"]
    assert all(c["title"] == "t" and c["language"] == "ur" and c["n_tokens"] for c in chunks)


def test_clean_text_folds_arabic_letters_and_whitespace():
    assert clean_text("كيا   ہے\n\n\n\nنہیں\x00") == "کیا ہے\n\nنہیں"
