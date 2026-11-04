import pytest

from zarai.data.split_images import assert_no_site_overlap, split_images


def make_records(sites, per_site=5):
    return [
        {"image_id": f"{site}_{i}", "site_id": site, "label": "x", "unsure": False}
        for site in sites
        for i in range(per_site)
    ]


def sites_of(rows):
    return {r["site_id"] for r in rows}


def test_no_site_leakage_and_nothing_lost():
    records = make_records([f"site_{i}" for i in range(10)])
    splits = split_images(records, seed=7)
    assert_no_site_overlap(splits)
    assert all(splits.values())
    assert sum(len(rows) for rows in splits.values()) == len(records)


def test_overlap_assert_catches_leakage():
    with pytest.raises(AssertionError):
        assert_no_site_overlap({"train": make_records(["a"]), "val": make_records(["a", "b"])})


def test_public_sites_never_enter_field_test():
    records = make_records([f"site_{i}" for i in range(6)] + [f"public_ds_{i}" for i in range(6)])
    for seed in range(10):
        field_test = split_images(records, seed=seed)["field_test"]
        assert field_test and not any(s.startswith("public_") for s in sites_of(field_test))


def test_public_only_gives_empty_field_test():
    splits = split_images(make_records([f"public_ds_{i}" for i in range(5)]))
    assert splits["field_test"] == [] and splits["train"] and splits["val"]


def test_unsure_images_are_excluded():
    records = make_records([f"site_{i}" for i in range(5)])
    records[0]["unsure"] = True
    splits = split_images(records)
    assert sum(len(rows) for rows in splits.values()) == len(records) - 1


def test_deterministic_for_seed():
    records = make_records([f"site_{i}" for i in range(10)])
    assert split_images(records, seed=3) == split_images(records, seed=3)


def test_too_few_sites_raises():
    with pytest.raises(ValueError):
        split_images(make_records(["only_one"]))
