import json

import pytest

from zarai.data.validate import load_schema, validate_file, validate_records

QA = {
    "id": "qa_0001",
    "lang": "ur-Latn",
    "crop": "wheat",
    "topic": "fertiliser",
    "season": "rabi",
    "question": "q",
    "answer": "a",
    "kb_chunk_ids": ["s1_c0000"],
    "source_type": "farmer",
    "annotator": "ra3",
    "expert_verified": False,
    "verifier": None,
    "notes": "",
}
IMAGE = {
    "image_id": "img1",
    "file_path": "data/raw/images_field/img1.jpg",
    "crop": "cotton",
    "label": "healthy",
    "site_id": "site_a",
    "district": "Multan",
    "date": "2026-01-31",
    "growth_stage": "flowering",
    "view": "leaf_closeup",
    "phone_model": "phone",
    "labeller": "ra2",
    "expert_verified": True,
    "unsure": False,
}


@pytest.mark.parametrize("name", ["qa_record", "image_meta"])
def test_schemas_load(name):
    assert load_schema(name)["type"] == "object"


def test_valid_records_pass():
    assert validate_records([QA], "qa_record") == []
    assert validate_records([IMAGE], "image_meta") == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "qa_12"),
        ("lang", "ur-Arab"),
        ("crop", "mango"),
        ("topic", "x"),
        ("season", "spring"),
        ("source_type", "web"),
        ("expert_verified", "yes"),
        ("extra_field", 1),
    ],
)
def test_bad_qa_fields_fail(field, value):
    errors = validate_records([{**QA, field: value}], "qa_record")
    assert errors and field in errors[0] or "extra_field" in errors[0]


@pytest.mark.parametrize(
    "field,value",
    [("date", "31-01-2026"), ("date", "2026-02-31"), ("view", "top"), ("unsure", "maybe")],
)
def test_bad_image_fields_fail(field, value):
    assert validate_records([{**IMAGE, field: value}], "image_meta")


def test_missing_required_and_duplicate_ids():
    missing = {k: v for k, v in QA.items() if k != "answer"}
    assert any("answer" in e for e in validate_records([missing], "qa_record"))
    assert any("duplicate id" in e for e in validate_records([QA, QA], "qa_record"))


def test_validate_jsonl_and_csv_files(tmp_path):
    jsonl = tmp_path / "qa.jsonl"
    jsonl.write_text(json.dumps(QA, ensure_ascii=False) + "\n", encoding="utf-8")
    assert validate_file(jsonl, "qa_record") == []

    header = ",".join(IMAGE)
    row = ",".join(str(v).lower() if isinstance(v, bool) else v for v in IMAGE.values())
    csv_file = tmp_path / "images.csv"
    csv_file.write_text(f"{header}\n{row}\n", encoding="utf-8")
    assert validate_file(csv_file, "image_meta") == []

    csv_file.write_text(f"{header}\n{row.replace('leaf_closeup', '')}\n", encoding="utf-8")
    assert any("view" in e for e in validate_file(csv_file, "image_meta"))
