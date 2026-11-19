from datetime import date

from zarai.fusion.season_check import season_check
from zarai.fusion.vision_terms import TERMS, vision_query


def test_query_uses_question_script():
    assert vision_query("پتے پیلے ہیں", "yellow_rust", "ur").endswith("زرد کنگی")
    assert vision_query("patte peelay hain", "yellow_rust", "ur-Latn").endswith("zard kungi")
    assert vision_query("leaves are yellow", "yellow_rust", "en").endswith("yellow stripe rust")


def test_healthy_adds_nothing_and_unmapped_label_passes_through():
    assert vision_query("q", "healthy", "en") == "q"
    assert vision_query("q", None, "en") == "q"
    assert vision_query("q", "new_class", "en") == "q new_class"


def test_all_classes_have_three_scripts():
    for label, terms in TERMS.items():
        assert terms is None or set(terms) == {"ur", "ur-Latn", "en"}, label


def test_season_flags_out_of_season_only():
    assert season_check("brown_rust", "en", date(2026, 12, 10))["source"] == "src_0011_c0001"
    assert season_check("aphid", "en", date(2026, 12, 10)) is None
    assert season_check("yellow_rust", "ur", date(2026, 11, 20)) is not None
    assert season_check("yellow_rust", "ur", date(2026, 2, 5)) is None
    assert season_check("septoria", "en", date(2026, 12, 1)) is None
