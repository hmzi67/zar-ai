import pytest

from zarai.data.split_qa import split_qa


def make_records(n_llm, n_field):
    source_types = ["llm"] * n_llm + ["farmer", "expert"] * (n_field // 2)
    return [{"id": f"qa_{i:04d}", "source_type": s} for i, s in enumerate(source_types)]


def test_sizes_disjoint_and_test_source_types():
    splits = split_qa(make_records(600, 400))
    assert [len(splits[s]) for s in ("train", "val", "test", "unused")] == [700, 100, 200, 0]
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == 1000
    assert {r["source_type"] for r in splits["test"]} <= {"farmer", "expert"}


def test_extra_records_are_kept_as_unused():
    assert len(split_qa(make_records(700, 400))["unused"]) == 100


def test_deterministic_regardless_of_input_order():
    records = make_records(600, 400)
    assert split_qa(records, seed=1) == split_qa(records[::-1], seed=1)
    assert split_qa(records, seed=1) != split_qa(records, seed=2)


def test_too_few_records_raises():
    with pytest.raises(ValueError, match="need 1000"):
        split_qa(make_records(100, 100))


def test_too_few_test_eligible_raises():
    with pytest.raises(ValueError, match="farmer/expert"):
        split_qa(make_records(900, 100))
