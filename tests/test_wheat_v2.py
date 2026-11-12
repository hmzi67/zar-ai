import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from zarai.common.io import read_csv
from zarai.data.clean_wheat import _write_csv
from zarai.data.leaf_mask import leaf_mask
from zarai.data.wheat_v2 import (
    EXCLUDED,
    POOLED_COLUMN,
    REVIEW_COLUMNS,
    LeakageError,
    assert_no_leakage,
    assign_splits,
    build_wheat_v2,
    classify_pair,
    eligible_folds,
    foreground_mad,
    is_conflict,
    is_dark_background,
    loso_column,
    map_label,
    select_tiles,
)

CONFIG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "data" / "wheat_v2.yaml")
CLASSES = list(CONFIG.classes)


# ---------------------------------------------------------------------------------------------
# Label mapping


@pytest.mark.parametrize(
    ("source", "raw", "expected"),
    [
        ("mendeley_wheat_leaf_ethiopia", "stripe_rust", "yellow_rust"),
        ("mendeley_wheat_leaf_ethiopia", "Stripe Rust", "yellow_rust"),
        ("mendeley_wheat_leaf_ethiopia", "Healthy", "healthy"),
        ("mendeley_wheat_leaf_ethiopia", "septoria", "septoria"),
        ("mendeley_iari_leaf_rust", "diseased", "brown_rust"),
        ("mendeley_iari_leaf_rust", "control", "healthy"),
        ("cgiar_wheat_rust", "leaf_rust", "brown_rust"),
        ("cgiar_wheat_rust", "Stem Rust", "black_rust"),
        ("cgiar_wheat_rust", "healthy_wheat", "healthy"),
        ("kaggle", "brown_rust", "brown_rust"),
        ("kaggle", "yellow_rust", "yellow_rust"),
        ("kaggle", "black_rust", "black_rust"),
        ("kaggle", "mildew", None),
        ("kaggle", "leaf_blight", None),
        ("kaggle", "fusarium_head_blight", None),
    ],
)
def test_source_labels_map_to_v2_classes(source, raw, expected):
    assert map_label(raw, CONFIG.sources[source].label_map, CLASSES) == expected


def test_every_label_map_targets_only_v2_classes():
    assert CLASSES == ["black_rust", "brown_rust", "healthy", "septoria", "yellow_rust"]
    for spec in CONFIG.sources.values():
        assert set(spec.get("label_map", {}).values()) <= set(CLASSES)


def test_a_label_map_into_a_dropped_class_is_rejected():
    with pytest.raises(ValueError, match="not a v2 class"):
        map_label("mildew", {"mildew": "mildew"}, CLASSES)


def test_conflicts_need_a_v2_label_on_at_least_one_side():
    def rec(label):
        return {"label": label}

    assert is_conflict([rec("brown_rust"), rec("yellow_rust")])
    assert is_conflict([rec(None), rec("brown_rust")])  # leaf_blight in one source, rust in other
    assert not is_conflict([rec(None), rec(None)])  # dropped either way
    assert not is_conflict([rec("healthy"), rec("healthy")])


DEDUPE = CONFIG.dedupe


@pytest.mark.parametrize(
    ("cosine", "mad", "both_dark", "has_phash", "expected"),
    [
        (1.0, 0.0, False, True, "same"),  # identical
        (0.75, 2.5, False, True, "same"),  # same photo with an overlay: pixels agree
        (0.98, 24.0, False, True, "same"),  # same scene, other exposure: embeddings agree
        (0.97, 20.0, False, False, "same"),  # embedding-only near-duplicate, as in wheat_v1
        (0.80, 20.0, False, True, "similar"),  # unconfirmed pHash: one split, not merged
        (0.97, 10.0, True, True, None),  # IARI: different leaves on black, embeddings ignored
        (0.97, 10.0, True, False, None),
        (0.50, 1.0, True, True, "same"),  # on black, pixels still decide
    ],
)
def test_pairs_are_one_photo_only_when_pixels_or_trusted_embeddings_agree(
    cosine, mad, both_dark, has_phash, expected
):
    assert classify_pair(cosine, mad, both_dark, has_phash, DEDUPE) == expected


def test_foreground_mad_ignores_a_shared_black_background():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy()
    a[:, 20:26] = (40, 160, 40)  # a narrow leaf
    b[:, 34:40] = (40, 160, 40)  # another leaf, elsewhere
    plain = float(np.abs(a.astype(float) - b.astype(float)).mean())
    assert plain < 20  # the black background hides the difference
    assert foreground_mad(a, b) == pytest.approx(80.0)  # (40 + 160 + 40) / 3
    assert foreground_mad(a, a) == 0
    assert is_dark_background(a) and not is_dark_background(np.full_like(a, 128))


# ---------------------------------------------------------------------------------------------
# NWRD tiling


def test_nwrd_tiles_need_five_percent_rust_and_empty_tiles_are_skipped():
    mask = np.zeros((20, 30), dtype=bool)  # three 10x10 tiles per row, two rows; 0-9 col strip
    mask[0:10, 0:10] = True  # tile (0, 0): 100%
    mask[0, 10:15] = True  # tile (10, 0): 5 px = 5%, kept
    mask[0, 20:24] = True  # tile (20, 0): 4 px = 4%, skipped
    kept, skipped = select_tiles(mask, tile_size=10, min_coverage=0.05)
    assert [box for box, _ in kept] == [(0, 0, 10, 10), (10, 0, 20, 10)]
    assert [coverage for _, coverage in kept] == pytest.approx([1.0, 0.05])
    assert skipped == Counter({"tiles: empty mask": 3, "tiles: rust below 5%": 1})


def test_nwrd_partial_tiles_at_the_edge_are_dropped():
    mask = np.ones((25, 25), dtype=bool)
    kept, _ = select_tiles(mask, tile_size=10, min_coverage=0.05)
    assert len(kept) == 4


# ---------------------------------------------------------------------------------------------
# Splits and leakage


def _rows(n_per_class=40):
    """Three training sources plus a test-only one. Pairs of images share a cluster."""
    rows, cluster = [], 0
    for source, labels in (
        ("a", ("healthy", "brown_rust")),
        ("b", ("healthy", "septoria")),
        ("c", ("healthy",)),
    ):
        for label in labels:
            for k in range(n_per_class):
                if k % 2 == 0:
                    cluster += 1
                rows.append({"source": source, "label": label, "cluster_id": cluster})
    for _ in range(6):
        cluster += 1
        rows.append({"source": "nwrd", "label": "yellow_rust", "cluster_id": cluster})
    for i, row in enumerate(rows):
        row["md5"] = row["pixel_md5"] = f"h{i}"
    return rows


def _with_splits(rows, folds):
    columns = assign_splits(
        rows, folds, {"nwrd"}, 42, {"train": 14, "val": 3, "test": 3}, {"train": 17, "val": 3}
    )
    for column, values in columns.items():
        for row, value in zip(rows, values, strict=True):
            row[column] = value
    return rows


def test_folds_need_two_classes_and_never_hold_out_a_test_only_source():
    rows = _rows()
    assert eligible_folds(rows, ["a", "b", "c", "nwrd"], {"nwrd"}, 2) == ["a", "b"]


def test_loso_holds_out_one_source_and_takes_val_from_the_others():
    rows = _rows()
    # One image of source c shares a cluster with source a, so it can't train fold a.
    rows[-7]["cluster_id"] = rows[0]["cluster_id"]
    rows = _with_splits(rows, ["a", "b"])
    column = loso_column("a")
    for row in rows:
        if row["source"] == "a":
            assert row[column] == "test"
        elif row["source"] == "nwrd":
            assert row[column] == "nwrd" and row[POOLED_COLUMN] == "nwrd"
    assert rows[-7][column] == EXCLUDED
    assert rows[-7][loso_column("b")] != EXCLUDED
    others = [r for r in rows if r["source"] in ("b", "c") and r[column] != EXCLUDED]
    val_share = sum(r[column] == "val" for r in others) / len(others)
    assert 0.10 <= val_share <= 0.20
    assert {r[column] for r in others} == {"train", "val"}
    assert {r["source"] for r in others if r[column] == "val"} == {"b", "c"}
    for clustered in Counter(r["cluster_id"] for r in rows).keys():
        assert len({r[column] for r in rows if r["cluster_id"] == clustered}) == 1 or any(
            r[column] == EXCLUDED for r in rows if r["cluster_id"] == clustered
        )
    assert_no_leakage(rows, {"nwrd"})


def test_pooled_split_is_70_15_15_and_never_trains_on_nwrd():
    rows = _with_splits(_rows(), ["a", "b"])
    counts = Counter(r[POOLED_COLUMN] for r in rows if r["source"] != "nwrd")
    total = sum(counts.values())
    assert counts["train"] / total == pytest.approx(0.70, abs=0.05)
    assert counts["val"] / total == pytest.approx(0.15, abs=0.05)
    assert not any(r["source"] == "nwrd" and r[POOLED_COLUMN] in ("train", "val") for r in rows)


def test_an_image_clustered_with_nwrd_is_excluded_everywhere():
    rows = _rows()
    rows[0]["cluster_id"] = rows[-1]["cluster_id"]
    rows = _with_splits(rows, ["a", "b"])
    assert rows[0][POOLED_COLUMN] == rows[0][loso_column("b")] == EXCLUDED
    assert_no_leakage(rows, {"nwrd"})


def test_leakage_check_passes_on_clean_splits():
    reports = assert_no_leakage(_with_splits(_rows(), ["a", "b"]), {"nwrd"})
    assert set(reports) == {POOLED_COLUMN, loso_column("a"), loso_column("b")}


@pytest.mark.parametrize("key", ["md5", "pixel_md5", "cluster_id"])
def test_leakage_check_fails_on_a_shared_hash_or_cluster(key):
    rows = _with_splits(_rows(), ["a", "b"])
    train = next(r for r in rows if r[POOLED_COLUMN] == "train")
    test = next(r for r in rows if r[POOLED_COLUMN] == "test")
    test[key] = train[key]
    with pytest.raises(LeakageError):
        assert_no_leakage(rows, {"nwrd"})


def test_leakage_check_fails_on_nwrd_in_train_or_val():
    rows = _with_splits(_rows(), ["a", "b"])
    rows[-1][loso_column("a")] = "val"
    with pytest.raises(LeakageError, match="test_only_in_train_val"):
        assert_no_leakage(rows, {"nwrd"})


def test_leakage_check_fails_when_a_held_out_image_trains():
    rows = _with_splits(_rows(), ["a", "b"])
    next(r for r in rows if r["source"] == "a")[loso_column("a")] = "train"
    with pytest.raises(LeakageError, match="heldout_in_train_val"):
        assert_no_leakage(rows, {"nwrd"})


# ---------------------------------------------------------------------------------------------
# Leaf mask


def test_leaf_mask_keeps_lesions_inside_a_leaf_and_drops_the_background():
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # black background, like IARI
    image[20:80, 30:70] = (40, 160, 40)  # green leaf
    image[45:55, 45:55] = (120, 60, 20)  # brown lesion inside it
    mask, fell_back = leaf_mask(image)
    assert not fell_back
    assert mask[50, 50] and mask[30, 35]
    assert not mask[5, 5] and not mask[50, 90]


def test_leaf_mask_falls_back_to_everything_when_it_finds_no_plant():
    mask, fell_back = leaf_mask(np.full((40, 40, 3), 128, dtype=np.uint8))
    assert fell_back and mask.all()


# ---------------------------------------------------------------------------------------------
# The whole build, on small synthetic sources


def _noise(path: Path, seed: int, size=(48, 48)) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)
    return path


def _fake_embedder(paths, pool):
    """Unrelated unit vectors, so only pixels decide what is a duplicate."""
    vectors = []
    for path in paths:
        seed = int(hashlib.md5(Path(path).read_bytes()).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).normal(size=64)
        vectors.append(vector / np.linalg.norm(vector))
    return np.array(vectors, dtype=np.float32)


@pytest.fixture
def sources(tmp_path):
    from PIL import Image

    raw = tmp_path / "raw"
    kaggle_rows, seed = [], 0
    for label in ("healthy", "brown_rust", "mildew"):
        for k in range(30):
            seed += 1
            path = _noise(raw / "kaggle" / label / f"{k}.png", seed)
            kaggle_rows.append({"label": label, "original_path": path.as_posix()})
    _write_csv(tmp_path / "wheat_v1.csv", ["label", "original_path"], kaggle_rows)

    eth = raw / "ethiopia" / "wheat_leaf"
    for label in ("Healthy", "septoria"):
        for k in range(30):
            seed += 1
            _noise(eth / label / f"{k}.jpg", seed)
    # The same photo as Kaggle brown_rust 0, labelled stripe rust: a conflict.
    stripe = eth / "stripe_rust" / "dup.png"
    stripe.parent.mkdir(parents=True)
    stripe.write_bytes(Path(kaggle_rows[30]["original_path"]).read_bytes())
    # The same photo as Kaggle healthy 0 with the same label: kept once, as Ethiopia's.
    (eth / "Healthy" / "dup.png").write_bytes(Path(kaggle_rows[0]["original_path"]).read_bytes())

    iari = raw / "iari" / "WheatLeafRust"
    for label in ("control", "diseased"):
        for k in range(30):
            seed += 1
            _noise(iari / ("train" if k % 2 else "test") / label / f"{k}.jpg", seed)

    nwrd = raw / "nwrd" / "train"
    for k in range(3):
        _noise(nwrd / "images" / f"{k}.jpg", 1000 + k, size=(16, 48))  # 3 tiles of 16 px
        mask = np.zeros((16, 48), dtype=np.uint8)
        mask[:8, :8] = 255  # tile 0: 25%, kept
        mask[0, 16:20] = 255  # tile 1: 4 px of 256, below 5%
        (nwrd / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(nwrd / "masks" / f"{k}.png")

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "classes": CLASSES,
            "sources": {
                "mendeley_wheat_leaf_ethiopia": {
                    "kind": "folders",
                    "root": (raw / "ethiopia").as_posix(),
                    "label_map": dict(CONFIG.sources.mendeley_wheat_leaf_ethiopia.label_map),
                },
                "mendeley_iari_leaf_rust": {
                    "kind": "folders",
                    "root": (raw / "iari").as_posix(),
                    "label_map": dict(CONFIG.sources.mendeley_iari_leaf_rust.label_map),
                },
                "kaggle": {
                    "kind": "wheat_v1_csv",
                    "splits_csv": (tmp_path / "wheat_v1.csv").as_posix(),
                    "label_map": dict(CONFIG.sources.kaggle.label_map),
                },
                "nwrd": {
                    "kind": "nwrd_masks",
                    "root": (raw / "nwrd").as_posix(),
                    "test_only": True,
                    "label": "yellow_rust",
                    "tile_size": 16,
                    "min_mask_coverage": 0.05,
                },
            },
            "dedupe": dict(CONFIG.dedupe),
            "splits": dict(CONFIG.splits),
            "preprocess": {"long_side": 32},
            "apply_label_review": None,
            "outputs": {
                "processed_dir": (tmp_path / "processed").as_posix(),
                "splits_csv": (tmp_path / "splits.csv").as_posix(),
                "interim_dir": (tmp_path / "interim").as_posix(),
                "review_csv": (tmp_path / "interim" / "label_review_v2.csv").as_posix(),
            },
        }
    )
    return cfg, kaggle_rows, stripe


def test_build_quarantines_conflicts_reassigns_duplicates_and_tiles_nwrd(sources):
    cfg, kaggle_rows, stripe = sources
    rows = build_wheat_v2(cfg, workers=2, embedder=_fake_embedder)
    by_original = {r["original_path"]: r for r in rows}
    conflict_kaggle = kaggle_rows[30]["original_path"]

    # The disputed photo is in no split, and both copies are listed for review.
    assert stripe.as_posix() not in by_original and conflict_kaggle not in by_original
    review = read_csv(cfg.outputs.review_csv)
    assert list(review[0]) == REVIEW_COLUMNS
    assert {(r["source"], r["current_label"], r["other_label"]) for r in review} == {
        ("kaggle", "brown_rust", "yellow_rust"),
        ("mendeley_wheat_leaf_ethiopia", "yellow_rust", "brown_rust"),
    }
    assert all(Path(r["thumbnail"]).is_file() for r in review)
    assert list((Path(cfg.outputs.interim_dir) / "review").glob("*.jpg"))

    # The agreeing duplicate is kept once, under the external source.
    assert kaggle_rows[0]["original_path"] not in by_original
    dup = next(r for r in rows if r["original_path"].endswith("Healthy/dup.png"))
    assert dup["source"] == "mendeley_wheat_leaf_ethiopia" and dup["label"] == "healthy"

    assert {r["label"] for r in rows} == {"healthy", "brown_rust", "septoria", "yellow_rust"}
    nwrd = [r for r in rows if r["source"] == "nwrd"]
    assert len(nwrd) == 3 and {r["label"] for r in nwrd} == {"yellow_rust"}
    folds = [c for c in rows[0] if c.startswith("split_loso_")]
    assert folds == [
        "split_loso_mendeley_wheat_leaf_ethiopia",
        "split_loso_mendeley_iari_leaf_rust",
        "split_loso_kaggle",
    ]
    assert all(r[c] == "nwrd" for r in nwrd for c in [POOLED_COLUMN, *folds])
    assert all(Path(r["path"]).is_file() and Path(r["mask_path"]).is_file() for r in rows)

    report = (Path(cfg.outputs.processed_dir) / "REPORT.md").read_text()
    assert "`mildew` | 30" in report
    assert "kaggle:brown_rust vs mendeley_wheat_leaf_ethiopia:yellow_rust" in report
    assert "## Leakage check" in report


def test_expert_labels_are_applied_on_the_next_build(sources):
    cfg, _, stripe = sources
    build_wheat_v2(cfg, workers=2, embedder=_fake_embedder)
    review_csv = Path(cfg.outputs.review_csv)
    review = read_csv(review_csv)
    for row in review:
        row["final_label"] = "yellow_rust"
    _write_csv(review_csv, REVIEW_COLUMNS, review)

    cfg.apply_label_review = review_csv.as_posix()
    rows = build_wheat_v2(cfg, workers=2, embedder=_fake_embedder)
    resolved = [r for r in rows if r["label_origin"] == "expert_review"]
    assert len(resolved) == 1  # both copies are the same photo, so one is kept
    assert resolved[0]["original_path"] == stripe.as_posix()
    assert resolved[0]["label"] == "yellow_rust"
    after = read_csv(review_csv)
    assert {r["status"] for r in after} == {"resolved"}
    assert {r["final_label"] for r in after} == {"yellow_rust"}


def test_an_unknown_expert_label_is_rejected(sources, tmp_path):
    cfg, _, _ = sources
    bad = tmp_path / "review.csv"
    _write_csv(bad, REVIEW_COLUMNS, [{"path": "x.png", "final_label": "rust"}])
    cfg.apply_label_review = bad.as_posix()
    with pytest.raises(ValueError, match="not a v2 class"):
        build_wheat_v2(cfg, workers=2, embedder=_fake_embedder)
