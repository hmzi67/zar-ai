import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from zarai.common.io import read_csv
from zarai.data.clean_wheat import (
    CLUSTER_CONFLICT,
    CSV_COLUMNS,
    EMBEDDING_CLUSTER_CONFLICT,
    LONG_SIDE,
    MD5_CONFLICT,
    PIXEL_CONFLICT,
    WHEAT_CLASSES,
    assert_no_leakage,
    assert_pairs_in_one_split,
    clean_wheat,
    cluster_ids,
    collect_images,
    embedding_pairs,
    group_by_pixels,
    mixed_label_clusters,
    near_duplicate_pairs,
    normalise_class_name,
    scaled_size,
    select_review_pairs,
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


def record(md5, pixel_md5, label, split="train", name="x"):
    return {
        "md5": md5,
        "pixel_md5": pixel_md5,
        "label": label,
        "original_split": split,
        "path": f"{split}/{label}/{name}.png",
    }


def test_group_by_pixels_keeps_one_copy_preferring_train():
    kept, conflicts = group_by_pixels(
        [record("a", "p", "smut", "test"), record("b", "p", "smut", "train")]
    )
    assert not conflicts
    assert [(r["md5"], r["original_split"], len(r["copies"])) for r in kept] == [("b", "train", 2)]


def test_group_by_pixels_gives_each_copy_its_conflict_reason():
    _, conflicts = group_by_pixels(
        [
            # Same file bytes under two labels, plus a re-encoded copy of the same pixels.
            record("same", "p1", "smut"),
            record("same", "p1", "mite", name="y"),
            record("reencoded", "p1", "smut", name="z"),
            # Different file bytes, identical pixels, different labels.
            record("a", "p2", "smut", name="a"),
            record("b", "p2", "blast", name="b"),
        ]
    )
    reasons = {g["group"]: Counter(r["reason"] for r in g["copies"]) for g in conflicts}
    assert reasons == {
        "p1": {MD5_CONFLICT: 2, PIXEL_CONFLICT: 1},
        "p2": {PIXEL_CONFLICT: 2},
    }
    assert [g["labels"] for g in conflicts] == [["mite", "smut"], ["blast", "smut"]]


def test_near_duplicates_cluster_through_a_chain():
    hashes = np.array([[0b00000000], [0b00000011], [0b00001111], [0b11111111]], dtype=np.uint8)
    pairs = near_duplicate_pairs(hashes, max_distance=2)
    assert pairs == [(0, 1), (1, 2)]
    assert cluster_ids(4, pairs) == [0, 0, 0, 1]


def test_embedding_pairs_finds_each_pair_once_across_chunks():
    embeddings = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32
    )
    pairs = embedding_pairs(embeddings, min_cosine=0.95, chunk_size=2)
    assert [(i, j) for i, j, _ in pairs] == [(0, 2)]
    assert pairs[0][2] == pytest.approx(1.0)


def test_embedding_and_phash_pairs_join_one_cluster():
    assert cluster_ids(4, [(0, 1)] + [(1, 2)]) == [0, 0, 0, 1]


def test_mixed_label_clusters():
    assert mixed_label_clusters(["a", "a", "b", "c"], [0, 0, 1, 1]) == {1}


def test_select_review_pairs_takes_both_sides_of_the_threshold():
    pairs = [(i, i + 1, 0.93 + i * 0.002) for i in range(20)]
    pairs.append((98, 99, 0.80))
    pairs.append((97, 99, 0.999))
    selected = select_review_pairs(pairs, threshold=0.95, margin=0.02, n=6)
    cosines = [p[2] for p in selected]
    assert sum(c >= 0.95 for c in cosines) == 3
    assert sum(c < 0.95 for c in cosines) == 3
    assert all(0.93 <= c <= 0.97 for c in cosines)


def test_select_review_pairs_handles_too_few_pairs():
    assert select_review_pairs([(0, 1, 0.95)], 0.95, 0.02, 20) == [(0, 1, 0.95)]
    assert select_review_pairs([], 0.95, 0.02, 20) == []


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
    return [
        {"md5": md5, "pixel_md5": pixels, "cluster_id": cluster, "split": split}
        for md5, pixels, cluster, split in rows
    ]


def test_leakage_check_passes_on_clean_rows():
    assert_no_leakage(
        rows_of(("a", "p", 0, "train"), ("b", "q", 0, "train"), ("c", "r", 1, "test"))
    )


@pytest.mark.parametrize(
    "rows",
    [
        rows_of(("a", "p", 0, "train"), ("a", "q", 1, "test")),  # md5 across splits
        rows_of(("a", "p", 0, "train"), ("a", "q", 1, "train")),  # md5 repeated in one split
        rows_of(("a", "p", 0, "train"), ("b", "p", 1, "test")),  # pixels across splits
        rows_of(("a", "p", 0, "train"), ("b", "p", 0, "train")),  # pixels repeated in one split
        rows_of(("a", "p", 0, "train"), ("b", "q", 0, "val")),  # cluster across splits
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
        ((5000, 2), (384, 1)),
        ((385, 100), (384, 100)),
        ((384, 200), (384, 200)),
        ((96, 48), (96, 48)),
    ],
)
def test_scaled_size_only_downscales(size, expected):
    assert scaled_size(*size) == expected


def test_outputs_may_not_overlap_the_raw_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for out, conflicts, csv_path in (
        (raw / "out", tmp_path / "c", tmp_path / "s.csv"),
        (tmp_path / "out", raw / "c", tmp_path / "s.csv"),
        (tmp_path, tmp_path / "c", tmp_path / "s.csv"),
        (tmp_path / "out", tmp_path / "c", raw / "s.csv"),
    ):
        with pytest.raises(ValueError, match="overlaps the raw data"):
            clean_wheat(raw, out, conflicts, csv_path)


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


def fake_embedder(paths, _pool):
    """Unrelated unit vectors, except that each embed_*_a/b.png pair has cosine 0.96."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(paths), 384)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    names = [Path(path).name for path in paths]
    for stem in ("embed_same", "embed_mixed"):
        if f"{stem}_a.png" not in names:
            continue
        a, b = names.index(f"{stem}_a.png"), names.index(f"{stem}_b.png")
        other = embeddings[b] - (embeddings[b] @ embeddings[a]) * embeddings[a]
        embeddings[b] = 0.96 * embeddings[a] + 0.28 * other / np.linalg.norm(other)
    return embeddings


def test_clean_wheat_end_to_end(tmp_path):
    pytest.importorskip("imagehash")
    image_module = pytest.importorskip("PIL.Image")
    png_module = pytest.importorskip("PIL.PngImagePlugin")

    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"

    def folder(split, label):
        path = raw / split / FOLDER_STYLES[split](label)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_noise(path, channels=3, height=48, width=64):
        pixels = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
        image_module.fromarray(pixels).save(path)
        return pixels

    for label in sorted(WHEAT_CLASSES):
        for i in range(30):
            save_noise(folder("train", label) / f"{label}_{i}.png")

    # Exact duplicate in another split, under another spelling of the same class.
    exact = folder("train", "smut") / "smut_0.png"
    shutil.copy(exact, folder("test", "smut") / "copy.png")

    # Same bytes under two classes.
    save_noise(folder("train", "mite") / "md5_conflict.png")
    shutil.copy(
        folder("train", "mite") / "md5_conflict.png", folder("valid", "blast") / "md5_conflict.png"
    )

    # Same pixels but different bytes, with one consistent label: a duplicate the md5 can't see.
    metadata = png_module.PngInfo()
    metadata.add_text("note", "different bytes")
    pixels = save_noise(folder("train", "healthy") / "pixels_a.png")
    image_module.fromarray(pixels).save(
        folder("test", "healthy") / "pixels_b.png", pnginfo=metadata
    )

    # Same pixels but different bytes, under two classes.
    pixels = save_noise(folder("train", "smut") / "pixel_conflict.png")
    image_module.fromarray(pixels).save(
        folder("test", "aphid") / "pixel_conflict.png", pnginfo=metadata
    )

    # Near-duplicates (md5 and pixels differ, pHash close) with one label stay together.
    def brighter(values):
        return image_module.fromarray(np.clip(values.astype(int) + 2, 0, 255).astype(np.uint8))

    pixels = save_noise(folder("train", "aphid") / "near.png")
    brighter(pixels).save(folder("test", "aphid") / "near_copy.png")

    # Near-duplicates under two classes.
    pixels = save_noise(folder("train", "tan_spot") / "mixed.png")
    brighter(pixels).save(folder("test", "leaf_blight") / "mixed_copy.png")

    # Linked only by the embeddings, and from different original splits.
    save_noise(folder("train", "mildew") / "embed_same_a.png")
    save_noise(folder("test", "mildew") / "embed_same_b.png")

    # Linked only by the embeddings, under two classes.
    save_noise(folder("train", "septoria") / "embed_mixed_a.png")
    save_noise(folder("train", "brown_rust") / "embed_mixed_b.png")

    save_noise(folder("train", "healthy") / "transparent.png", channels=4)
    save_noise(folder("train", "mite") / "large.png", height=400, width=500)

    before = snapshot(raw)
    out, conflicts_dir = tmp_path / "processed", tmp_path / "conflicts"
    splits_csv = tmp_path / "splits.csv"
    rows = clean_wheat(
        raw, out, conflicts_dir, splits_csv, seed=1, workers=2, embedder=fake_embedder
    )

    assert snapshot(raw) == before

    assert list(rows[0]) == CSV_COLUMNS
    assert read_csv(splits_csv) == rows
    assert_no_leakage(rows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}

    def kept(*names):
        return [r for r in rows if Path(r["original_path"]).name in names]

    assert len(kept("smut_0.png", "copy.png")) == 1
    assert len(kept("pixels_a.png", "pixels_b.png")) == 1

    for pair in (("near.png", "near_copy.png"), ("embed_same_a.png", "embed_same_b.png")):
        found = kept(*pair)
        assert len(found) == 2
        assert found[0]["cluster_id"] == found[1]["cluster_id"]
        assert found[0]["split"] == found[1]["split"]

    for row in rows:
        assert row["path"].endswith(f"{row['split']}/{row['label']}/{row['pixel_md5']}.jpg")
        assert Path(row["path"]).is_file()
    with image_module.open(kept("large.png")[0]["path"]) as image:
        assert (image.format, image.mode, image.size) == ("JPEG", "RGB", (LONG_SIDE, 307))
    with image_module.open(kept("near.png")[0]["path"]) as image:
        assert image.size == (64, 48)

    conflicts = read_csv(conflicts_dir / "label_conflicts.csv")
    reasons = {(Path(c["original_path"]).name, c["label"]): c["reason"] for c in conflicts}
    assert reasons == {
        ("md5_conflict.png", "mite"): MD5_CONFLICT,
        ("md5_conflict.png", "blast"): MD5_CONFLICT,
        ("pixel_conflict.png", "smut"): PIXEL_CONFLICT,
        ("pixel_conflict.png", "aphid"): PIXEL_CONFLICT,
        ("mixed.png", "tan_spot"): CLUSTER_CONFLICT,
        ("mixed_copy.png", "leaf_blight"): CLUSTER_CONFLICT,
        ("embed_mixed_a.png", "septoria"): EMBEDDING_CLUSTER_CONFLICT,
        ("embed_mixed_b.png", "brown_rust"): EMBEDDING_CLUSTER_CONFLICT,
    }
    assert all(Path(c["copied_to"]).is_file() for c in conflicts)
    assert len({c["copied_to"] for c in conflicts}) == 6
    assert not {c["pixel_md5"] for c in conflicts} & {r["pixel_md5"] for r in rows}
    assert not kept(
        "mixed.png",
        "mixed_copy.png",
        "pixel_conflict.png",
        "md5_conflict.png",
        "embed_mixed_a.png",
        "embed_mixed_b.png",
    )

    assert (out / "review" / "threshold_pairs.jpg").is_file()
    review = read_csv(out / "review" / "threshold_pairs.csv")
    assert {Path(p).name for row in review for p in (row["path_a"], row["path_b"])} == {
        Path(r["path"]).name for r in kept("embed_same_a.png", "embed_same_b.png")
    }

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## Leakage check" in report
    assert "## Known issues" in report
    assert "| `leaf_blight\\|tan_spot` | 0 | 1 | 0 |" in report
    assert "| `md5_conflict` | 1 | 2 | 1 |" in report
    assert "| `pixel_conflict` | 1 | 2 | 1 |" in report
    assert "| `cluster_conflict` | 1 | 2 | 2 |" in report
    assert "| `embedding_cluster_conflict` | 1 | 2 | 2 |" in report
    assert "| - new links | 2 |" in report
    assert "| New links crossing the old splits | 1 |" in report
    assert "| Downscaled to 384 px | 1 |" in report
    assert "| Images with transparency, flattened on white | 1 |" in report
