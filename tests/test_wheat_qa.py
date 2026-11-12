import pytest
from omegaconf import OmegaConf

from zarai.data.validate import validate_records
from zarai.data.wheat_qa import is_code_mixed, process, redact_pii
from zarai.nlp.normalise import match_key, normalise_text, urdu_punctuation

CFG = OmegaConf.load("configs/data/wheat_qa_v1.yaml")


# --------------------------------------------------------------------------- normaliser
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("كيه", "کیہ"),  # Arabic kaf, yeh, heh -> Urdu
        ("گَندُم", "گندم"),  # zabar and pesh removed
        ("اعلیٰ", "اعلی"),  # superscript alef removed
        ("آب", "آب"),  # madda is part of the letter: kept
        ("گن​دم‌", "گندم"),  # zero-width characters removed
        ("گندم   کی   فصل", "گندم کی فصل"),  # NBSP and runs of spaces
        ("یہ کیا ہے?", "یہ کیا ہے؟"),
        ("پانی دیں , کھاد دیں.", "پانی دیں، کھاد دیں۔"),
        ("1,000 روپے اور 2.5 کلو", "1,000 روپے اور 2.5 کلو"),  # numbers keep , and .
    ],
)
def test_normalise_urdu(raw, expected):
    assert normalise_text(raw, "ur") == expected


def test_latin_text_keeps_latin_punctuation():
    assert normalise_text("  Use 2.5 kg/acre , now ?", "en") == "Use 2.5 kg/acre, now?"
    assert normalise_text("Gandum ko paani kab dein ?", "ur-Latn") == "Gandum ko paani kab dein?"
    # Urdu words inside an English cell are letter-normalised but keep Latin punctuation.
    assert normalise_text("For Rust, فنگي سائیڈ.", "en") == "For Rust, فنگی سائیڈ."


def test_urdu_punctuation_leaves_latin_sentences_alone():
    assert urdu_punctuation("NARC-2009. Faisalabad") == "NARC-2009. Faisalabad"


def test_match_key_ignores_case_punctuation_and_spacing():
    assert match_key("When to irrigate  wheat?") == match_key("when to irrigate wheat")


def test_redact_pii():
    text, found = redact_pii("Call 0300-1234567 or ۰۳۰۰۱۲۳۴۵۶۷, CNIC 35202-1234567-1, a@b.pk")
    assert text == "Call [PHONE] or [PHONE], CNIC [CNIC], [EMAIL]"
    assert sorted(found) == ["cnic", "email", "phone", "phone"]


def test_code_mixed():
    assert is_code_mixed("wheat میں سفید پاؤڈر", "ur")
    assert is_code_mixed("For Rust, فنگی سائیڈ کا اسپرے کریں.", "en")
    assert not is_code_mixed(
        "گندم میں NARC-2009 اور فیصل آباد اور پاکستان کی اقسام بہترین ہیں", "ur"
    )


# --------------------------------------------------------------------------- mapping
def make_row(file, id_, q, a, row=None, **extra):
    """One trilingual Excel row; the Urdu and Roman cells are derived from the English text."""
    base = {
        "file": file, "sheet": "s", "excel_row": row or int(id_) + 1, "id": str(id_),
        "question_english": q, "answer_english": a,
        "question_roman_urdu": f"Sawal: {q}", "answer_roman_urdu": f"Jawab hai: {a}",
        "question_urdu": "گندم کا سوال؟", "answer_urdu": f"جواب {id_} ہے۔",
        "category": "disease_management", "season": "rabi", "source_type": "expert_knowledge",
    }  # fmt: skip
    return base | extra


def b_group(group, start, q, a, n_variants=2, **extra):
    rows = [make_row("B", start, q, a, group_id=group, variant_type="original", **extra)]
    rows += [
        make_row("B", start + i, f"Sir, {q} I have {i} acres.", a, group_id=group, variant_type="wrapped",
                 question_urdu=f"سوال {i}؟", **extra)
        for i in range(1, n_variants + 1)
    ]  # fmt: skip
    return rows


def notes_of(record):
    return record["notes"].split("; ")


def test_row_becomes_three_schema_valid_records():
    result = process(
        b_group("g1", 1, "How to identify yellow rust?", "Yellow stripes on leaves."), CFG
    )
    assert [r["lang"] for r in result.v1] == ["ur", "ur-Latn", "en"]
    assert validate_records(result.v1 + result.variants, "qa_record") == []
    for record in result.v1:
        assert record["source_type"] == "llm" and record["expert_verified"] is False
        assert notes_of(record)[0] == "group=B:g1"
        assert "provenance unverified; file label: expert_knowledge" in record["notes"]
        assert record["crop"] == "wheat" and record["topic"] == "disease"
    assert result.v1[2]["answer"] == "Yellow stripes on leaves."


def test_b_keeps_one_original_and_moves_the_rest_to_variants():
    rows = b_group("g1", 1, "How to identify yellow rust?", "Yellow stripes.", n_variants=3)
    rows.append(
        make_row(
            "B",
            9,
            "What does yellow rust look like?",
            "Yellow stripes.",
            group_id="g1",
            variant_type="original",
        )
    )
    result = process(rows, CFG)
    assert len(result.v1) == 3 and len(result.variants) == 4 * 3
    assert all(
        "augmentation_only" in notes_of(r) and notes_of(r)[0] == "group=B:g1"
        for r in result.variants
    )
    assert {r["id"] for r in result.v1}.isdisjoint(r["id"] for r in result.variants)


def test_case_templates_collapse_and_lose_their_suffix():
    rows = [
        make_row("A", 1, "How to treat tan spot (Case 31)?", "Spray fungicide.", question_roman_urdu="Ilaj kya hai? (Case 31)"),
        make_row("A", 2, "How to treat tan spot (Case 71)?", "Spray fungicide.", question_roman_urdu="Ilaj kya hai? (Case 71)"),
    ]  # fmt: skip
    result = process(rows, CFG)
    assert len(result.v1) == 3
    assert result.v1[2]["question"] == "How to treat tan spot?"
    assert result.v1[1]["question"] == "Ilaj kya hai?"
    assert "case_template" in notes_of(result.v1[0])
    assert [a["reason"] for a in result.log if a["action"] == "drop"] == [
        "case_template_collapsed_of_A"
    ]


def test_cross_file_duplicate_keeps_b_copy():
    rows = [make_row("A", 1, "When to irrigate wheat?", "Four irrigations.")]
    rows += b_group("g1", 10, "When to irrigate wheat?", "Four irrigations.")
    result = process(rows, CFG)
    assert {notes_of(r)[0] for r in result.v1} == {"group=B:g1"}
    assert any(a["reason"] == "exact_duplicate_of_B" and a["file"] == "A" for a in result.log)


def test_near_duplicate_needs_same_answer():
    rows = [
        make_row("A", 1, "What is the treatment for rust in wheat?", "Spray fungicide for rust."),
        make_row("A", 2, "What is the treatment for rot in wheat?", "Spray fungicide for rot."),
        make_row(
            "A", 3, "What is the best treatment for rust in wheat?", "Spray fungicide for rust."
        ),
    ]
    result = process(rows, CFG)
    assert len(result.v1) == 2 * 3
    assert [a["reason"] for a in result.log if a["action"] == "drop"] == ["near_duplicate_of_A"]


def test_conflicting_answers_quarantine_both_files_and_variants():
    rows = [
        make_row("A", 1, "Benefits of foliar urea?", "Quick nutrition."),
        make_row("A", 2, "Benefits of foliar urea?", "Provides quick nutrition."),
        make_row("A", 3, "When to sow wheat?", "Late October.", category="planting_calendar"),
    ]
    rows += b_group("g1", 10, "Benefits of foliar urea?", "Quick nutrition.")
    result = process(rows, CFG)
    assert len(result.v1) == 3 and result.variants == []
    assert sorted((r["reason"], r["file"]) for r in result.review) == [
        ("conflicting_answers", "A"), ("conflicting_answers", "A"), ("conflicting_answers", "B"),
    ]  # fmt: skip
    assert result.review[-1]["variants_excluded"] == 2


def test_unsafe_answer_quarantined():
    rows = [
        make_row(
            "A",
            1,
            "Wild oats control?",
            "Spray 2,4-D herbicide for wild oats.",
            category="weed_management",
        )
    ]
    result = process(rows, CFG)
    assert result.v1 == [] and result.review[0]["reason"] == "unsafe_answer"


@pytest.mark.parametrize(
    "season,expected,suspect",
    [("rabi", "rabi", False), ("all_seasons", "any", False), ("kharif", "any", True)],
)
def test_season_mapping(season, expected, suspect):
    record = process([make_row("A", 1, "q?", "a.", season=season)], CFG).v1[0]
    assert record["season"] == expected
    assert ("season_suspect" in notes_of(record)) is suspect


def test_topic_mapping_and_notes():
    rows = [
        make_row("A", 1, "Pesticide dose per acre?", "I cannot give a dose.", category="unsafe_request"),
        make_row("A", 2, "How much does yellow rust cut yield?", "Up to 30 percent.", category="yield_loss"),
        make_row("A", 3, "Best practices for weed management?", "Apply balanced NPK fertilizer and urea.", category="weed_management"),
        make_row("A", 4, "Use of drones?", "Drones help monitoring.", category="modern_technology"),
    ]  # fmt: skip
    en = [r for r in process(rows, CFG).v1 if r["lang"] == "en"]
    assert [r["topic"] for r in en] == ["off_topic_unsafe", "disease", "weed", "other"]
    assert "topic_from_keywords" in notes_of(en[1])
    assert "topic_keyword_disagrees=fertiliser" in notes_of(en[2])


def test_unknown_category_fails_loudly():
    with pytest.raises(ValueError, match="category"):
        process([make_row("A", 1, "q?", "a.", category="new_category")], CFG)


def test_flags_pesticide_and_code_mixed():
    row = make_row(
        "A", 1, "Treatment for powdery mildew?", "For mildew, فنگی سائیڈ کا اسپرے کریں.",
        answer_urdu="ٹیبوکونازول کا اسپرے کریں۔",
    )  # fmt: skip
    by_lang = {r["lang"]: notes_of(r) for r in process([row], CFG).v1}
    assert all(
        "pesticide_check" in n for n in by_lang.values()
    )  # any language naming one flags all
    assert "code_mixed" in by_lang["en"] and "code_mixed" not in by_lang["ur"]


def test_pii_is_redacted_in_output():
    row = make_row("A", 1, "My number is 0300-1234567, when to sow?", "Late October.")
    record = process([row], CFG).v1[2]
    assert record["question"] == "My number is [PHONE], when to sow?"
    assert "pii_redacted" in notes_of(record)


def test_generic_answer_on_other_topic_is_quarantined():
    generic = "Use balanced NPK fertilizer and timely sowing."
    rows = [
        make_row("A", 1, "Best practices for weed management?", generic, category="weed_management"),
        make_row("A", 2, "Best practices for fertilizer management?", generic, category="fertilizer_management"),
        # Specific answers that skip the topic word are not mismatches.
        make_row("A", 3, "How do I recognise yellow rust?", "Yellow stripes on the leaves."),
    ]  # fmt: skip
    result = process(rows, CFG)
    en = [r["question"] for r in result.v1 if r["lang"] == "en"]
    assert en == ["Best practices for fertilizer management?", "How do I recognise yellow rust?"]
    assert [(r["reason"], r["category"]) for r in result.review] == [
        ("question_answer_mismatch", "weed_management")
    ]
    assert result.review[0]["source_rows"] == 1


def test_mismatch_review_lists_each_question_once():
    generic = "Use balanced NPK fertilizer and timely sowing."
    rows = [
        make_row("A", i, f"Best practices for weed management (Case {i})?", generic, category="weed_management")
        for i in (1, 2, 3)
    ] + [make_row("A", 4, "Best practices for fertilizer management?", generic, category="fertilizer_management")]  # fmt: skip
    result = process(rows, CFG)
    assert len(result.review) == 1 and result.review[0]["source_rows"] == 3
    assert result.summary["quarantine"]["question_answer_mismatch"]["rows_excluded"] == {"A": 3}
