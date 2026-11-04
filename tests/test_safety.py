import pytest

from zarai.nlp.safety import (
    DISCLAIMER,
    UNVERIFIED_DOSE,
    add_disclaimer,
    enforce,
    find_risky_strings,
    is_off_topic,
    unsupported_strings,
)


def chunks(*texts):
    return [{"chunk_id": f"c{i}", "text": t, "score": 1.0} for i, t in enumerate(texts)]


def test_grounded_answer_passes():
    source = chunks("Spray imidacloprid at 250 ml per acre.")
    assert unsupported_strings("Use 250ml of Imidacloprid.", source) == []


def test_invented_dose_and_pesticide_are_caught():
    source = chunks("Spray imidacloprid at 250 ml per acre.")
    assert unsupported_strings("Use 500 ml.", source) == ["500 ml"]
    assert unsupported_strings("Use mancozeb.", source) == ["mancozeb"]


def test_dose_is_not_matched_inside_a_larger_number():
    assert unsupported_strings("Use 2.5 kg.", chunks("Apply 12.5 kg.")) == ["2.5 kg"]


def test_urdu_digits_units_and_names_match_english_sources():
    source = chunks("Imidacloprid 250 ml per acre, 1,000 g urea")
    assert unsupported_strings("امیڈاکلوپرڈ ۲۵۰ ملی لیٹر فی ایکڑ", source) == []
    assert unsupported_strings("1000 گرام", source) == []
    assert unsupported_strings("۳۰۰ ملی لیٹر", source) == ["300 ml"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5 going home", set()),
        ("no numbers here", set()),
        ("2 bori DAP", {"2 bag"}),
        ("10% solution", {"10 %"}),
        ("۵ کلو گرام", {"5 kg"}),
    ],
)
def test_find_risky_strings(text, expected):
    assert find_risky_strings(text) == expected


@pytest.mark.parametrize("lang", list(DISCLAIMER))
def test_disclaimer_added_once_in_answer_language(lang):
    once = add_disclaimer("answer", lang)
    assert once.endswith(DISCLAIMER[lang])
    assert add_disclaimer(once, lang) == once


def test_enforce_blocks_unsupported_and_detects_language():
    result = enforce("۳۰۰ ملی لیٹر زہر ڈالیں", chunks("no dose here"))
    assert result.blocked and result.unsupported == ["300 ml"]
    assert result.answer.startswith(UNVERIFIED_DOSE["ur"])
    assert result.answer.endswith(DISCLAIMER["ur"])
    assert "۳۰۰" not in result.answer


def test_enforce_keeps_grounded_answer():
    result = enforce("Irrigate twice.", chunks("Irrigate twice."), "en")
    assert not result.blocked
    assert result.answer == f"Irrigate twice.\n\n{DISCLAIMER['en']}"


def test_off_topic_when_retrieval_finds_nothing():
    assert is_off_topic([])
    assert is_off_topic([{"score": 0.1}], min_score=0.5)
    assert not is_off_topic([{"score": 0.9}], min_score=0.5)
