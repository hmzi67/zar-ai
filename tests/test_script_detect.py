import pytest

from zarai.nlp.script_detect import detect_script


@pytest.mark.parametrize(
    "text,expected",
    [
        ("گندم کی فصل کو کتنا پانی دینا چاہیے؟", "ur"),
        ("گندم پر spray کب کریں", "ur"),
        ("Gandum ki fasal ko kitna pani dena chahiye?", "ur-Latn"),
        ("kapas ke patte peele ho rahe hain", "ur-Latn"),
        ("wheat ka rate kya hai", "ur-Latn"),
        ("What is the best time to sow wheat?", "en"),
        ("How much fertiliser does cotton need?", "en"),
        ("wheat rust", "en"),
        ("", "en"),
        ("12345 !!", "en"),
    ],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected
