import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from zarai.common.io import read_csv
from zarai.data.clean_wheat import (
    CSV_COLUMNS,
    LONG_SIDE,
    WHEAT_CLASSES,
    assert_no_leakage,
    assert_pairs_in_one_split,
    clean_wheat,
    cluster_ids,
    collect_images,
    group_by_md5,
    near_duplicate_pairs,
    normalise_class_name,
    scaled_size,
    stratified_group_split,
)


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("Black Rust", "black_rust"),
        ("black_rust_test", "black_rust"),
        ("black_rust_valid", "black_rust"),
        ("Blast", "blast"),
        ("blast_test", "blast"),
        ("blast_test_valid", "blast"),
        ("Stem fly", "stem_fly"),
        ("stem_fly_valid", "stem_fly"),
        ("Tan spot", "tan_spot"),
        ("Fusarium Head Blight", "fusarium_head_blight"),
        ("  Common-Root  Rot_train ", "common_root_rot"),
    ],
)
def test_normalise_class_name(folder, expected):
    assert normalise_class_name(folder) == expected


def test_normalise_keeps_a_name_that_is_only_a_split_word():
    assert normalise_class_name("test") == "test"


def test_every_class_name_is_already_normal():
    assert {normalise_class_name(name) for name in WHEAT_CLASSES} == WHEAT_CLASSES


def test_collect_images_rejects_unknown_class(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split).mkdir()
    (tmp_path / "train" / "Leaf Rot").mkdir()
    with pytest.raises(ValueError, match="leaf_rot"):
        collect_images(tmp_path)


def record(md5, label, split="train", path=None):
    return {
        "md5": md5,
        "label": label,
        "original_split": split,
        "path": path or f"{split}/{label}/{md5}.png",
    }


def test_group_by_md5_keeps_one_copy_and_separates_conflicts():
    kept, conflicts = group_by_md5(
        [
            record("a", "smut", "test"),
            record("a", "smut", "train"),
            record("b", "smut"),
            record("b", "mite", "valid"),
            record("c", "mite"),
        ]
    )
    assert [(r["md5"], r["original_split"]) for r in kept] == [("a", "train"), ("c", "train")]
    assert [(g["md5"], g["labels"], len(g["copies"])) for g in conflicts] == [
        ("b", ["mite", "smut"], 2)
    ]


def test_near_duplicates_cluster_through_a_chain():
    hashes = np.array([[0b00000000], [0b00000011], [0b00001111], [0b11111111]], dtype=np.uint8)
    pairs = near_duplicate_pairs(hashes, max_distance=2)
    assert pairs == [(0, 1), (1, 2)]
    assert cluster_ids(4, pairs) == [0, 0, 0, 1]


def split_dataset(seed=7):
    """Three uneven classes, with clusters of 1 to 4 images and a few mixed-label clusters."""
    rng = np.random.default_rng(seed)
    labels, groups = [], []
    group = 0
    for label, n_groups in {"smut": 300, "mite": 150, "blast": 60}.items():
        for _ in range(n_groups):
            labels += [label] * int(rng.integers(1, 5))
            groups += [group] * (len(labels) - len(groups))
            group += 1
    for i in range(0, len(labels), 97):
        labels[i] = "mite"
    return labels, groups


def test_split_keeps_every_cluster_in_one_split():
    labels, groups = split_dataset()
    splits = stratified_group_split(labels, groups, seed=7)
    split_of_group = defaultdict(set)
    for group, split in zip(groups, splits, strict=True):
        split_of_group[group].add(split)
    assert all(len(found) == 1 for found in split_of_group.values())


def test_split_is_70_15_15_within_each_class():
    labels, groups = split_dataset()
    splits = stratified_group_split(labels, groups, seed=7)
    assert set(splits) == {"train", "val", "test"}
    counts = Counter(zip(labels, splits, strict=True))
    for label, n in Counter(labels).items():
        for split, share in {"train": 0.70, "val": 0.15, "test": 0.15}.items():
            assert counts[label, split] / n == pytest.approx(share, abs=0.05)


def test_split_is_reproducible_and_changes_with_seed():
    labels, groups = split_dataset()
    first = stratified_group_split(labels, groups, seed=7)
    assert first == stratified_group_split(labels, groups, seed=7)
    assert first != stratified_group_split(labels, groups, seed=8)


def rows_of(*rows):
    return [{"md5": md5, "cluster_id": cluster, "split": split} for md5, cluster, split in rows]


def test_leakage_check_passes_on_clean_rows():
    assert_no_leakage(rows_of(("a", 0, "train"), ("b", 0, "train"), ("c", 1, "test")))


@pytest.mark.parametrize(
    "rows",
    [
        rows_of(("a", 0, "train"), ("a", 1, "test")),  # md5 across splits
        rows_of(("a", 0, "train"), ("a", 1, "train")),  # md5 repeated in one split
        rows_of(("a", 0, "train"), ("b", 0, "val")),  # cluster across splits
    ],
)
def test_leakage_check_catches_overlap(rows):
    with pytest.raises(AssertionError):
        assert_no_leakage(rows)


def test_pair_check_catches_a_split_pair():
    assert_pairs_in_one_split([(0, 1)], ["train", "train"])
    with pytest.raises(AssertionError):
        assert_pairs_in_one_split([(0, 1)], ["train", "val"])


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1000, 500), (384, 192)),
        ((500, 1000), (192, 384)),
        ((96, 48), (384, 192)),
        ((5000, 2), (384, 1)),
    ],
)
def test_scaled_size(size, expected):
    assert scaled_size(*size) == expected


FOLDER_STYLES = {
    "train": lambda label: label.replace("_", " ").title(),
    "valid": lambda label: f"{label}_valid",
    "test": lambda label: f"{label}_test",
}


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (
            hashlib.md5(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_clean_wheat_end_to_end(tmp_path):
    pytest.importorskip("imagehash")
    image_module = pytest.importorskip("PIL.Image")

    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"

    def folder(split, label):
        path = raw / split / FOLDER_STYLES[split](label)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_noise(path, mode="RGB"):
        channels = len(mode)
        pixels = rng.integers(0, 256, (48, 64, channels), dtype=np.uint8)
        image_module.fromarray(pixels).save(path)
        return pixels

    for split in ("valid", "test"):
        folder(split, "smut")
    for label in sorted(WHEAT_CLASSES):
        for i in range(30):
            save_noise(folder("train", label) / f"{label}_{i}.png")

    # Exact duplicate in another split, under another spelling of the same class.
    exact = folder("train", "smut") / "smut_0.png"
    shutil.copy(exact, folder("test", "smut") / "copy.png")
    # Same bytes under two classes.
    save_noise(folder("train", "mite") / "conflict.png")
    shutil.copy(folder("train", "mite") / "conflict.png", folder("valid", "blast") / "conflict.png")
    # Slightly brighter, so the md5 differs but the pHash stays close.
    pixels = save_noise(folder("train", "aphid") / "near.png")
    image_module.fromarray(np.clip(pixels.astype(int) + 2, 0, 255).astype(np.uint8)).save(
        folder("test", "aphid") / "near_copy.png"
    )
    save_noise(folder("train", "healthy") / "transparent.png", mode="RGBA")

    before = snapshot(raw)
    out, conflicts_dir = tmp_path / "processed", tmp_path / "conflicts"
    splits_csv = tmp_path / "splits.csv"
    rows = clean_wheat(raw, out, conflicts_dir, splits_csv, seed=1, workers=2)

    assert snapshot(raw) == before

    assert list(rows[0]) == CSV_COLUMNS
    assert read_csv(splits_csv) == rows
    assert_no_leakage(rows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}

    exact_md5 = hashlib.md5(exact.read_bytes()).hexdigest()
    assert sum(r["md5"] == exact_md5 for r in rows) == 1

    near = [r for r in rows if Path(r["original_path"]).name in ("near.png", "near_copy.png")]
    assert len(near) == 2
    assert near[0]["cluster_id"] == near[1]["cluster_id"]
    assert near[0]["split"] == near[1]["split"]

    for row in rows[:5] + near:
        assert row["path"].endswith(f"{row['split']}/{row['label']}/{row['md5']}.jpg")
        with image_module.open(row["path"]) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert max(image.size) == LONG_SIDE

    conflicts = read_csv(conflicts_dir / "label_conflicts.csv")
    assert [c["labels"] for c in conflicts] == ["blast|mite"]
    assert Path(conflicts[0]["copied_to"]).is_file()
    assert conflicts[0]["md5"] not in {r["md5"] for r in rows}

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## Leakage check" in report
    assert "| Conflict groups (excluded) | 1 |" in report
    assert "| Images with transparency, flattened on white | 1 |" in report


def test_outputs_may_not_overlap_the_raw_data(tmp_path):
    pytest.importorskip("imagehash")
    image_module = pytest.importorskip("PIL.Image")
    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    for split, style in FOLDER_STYLES.items():
        for label in sorted(WHEAT_CLASSES):
            folder = raw / split / style(label)
            folder.mkdir(parents=True)
            if split == "train":
                for i in range(25):
                    pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    image_module.fromarray(pixels).save(folder / f"{i}.png")
    before = snapshot(raw)
    with pytest.raises(ValueError, match="overlaps the raw data"):
        clean_wheat(raw, raw / "out", tmp_path / "c", tmp_path / "s.csv", workers=2)
    assert snapshot(raw) == before
