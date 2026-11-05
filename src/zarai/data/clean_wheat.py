"""Deduplicate the public wheat disease dataset and rebuild its splits without leakage.

The original train/valid/test folders share images, so they are ignored. Exact duplicates are
reduced to one copy, images with conflicting labels are copied out for review, and near-duplicates
are clustered so that each cluster stays in one new split. The raw data is only ever read.
"""

import argparse
import csv
import hashlib
import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from zarai.common.io import read_csv
from zarai.common.logging import get_logger

WHEAT_CLASSES = frozenset(
    {
        "aphid",
        "black_rust",
        "blast",
        "brown_rust",
        "common_root_rot",
        "fusarium_head_blight",
        "healthy",
        "leaf_blight",
        "mildew",
        "mite",
        "septoria",
        "smut",
        "stem_fly",
        "tan_spot",
        "yellow_rust",
    }
)
ORIGINAL_SPLITS = ("train", "valid", "test")
SPLIT_SUFFIXES = ("_train", "_valid", "_val", "_test")
# 20 folds give exactly 70/15/15.
FOLDS_PER_SPLIT = {"train": 14, "val": 3, "test": 3}
MAX_HAMMING_DISTANCE = 6
LONG_SIDE = 384
JPEG_QUALITY = 92
CSV_COLUMNS = ["path", "label", "split", "cluster_id", "md5", "original_split", "original_path"]
CONFLICT_COLUMNS = ["md5", "labels", "n_copies", "copied_to", "original_paths"]

log = logging.getLogger(__name__)


def normalise_class_name(name: str) -> str:
    """Map a folder name to its class: "Black Rust", "blast_test_valid" -> black_rust, blast."""
    label = "_".join(name.lower().replace("-", " ").replace("_", " ").split())
    while label.endswith(SPLIT_SUFFIXES):
        label = label.rsplit("_", 1)[0]
    return label


def collect_images(raw_dir: Path) -> list[dict]:
    """List raw/<split>/<class>/*.png, sorted by path. Unknown class folders are an error."""
    records = []
    for original_split in ORIGINAL_SPLITS:
        for class_dir in sorted(p for p in (raw_dir / original_split).iterdir() if p.is_dir()):
            label = normalise_class_name(class_dir.name)
            if label not in WHEAT_CLASSES:
                raise ValueError(f"{class_dir}: unknown class {label!r}")
            records += [
                {"path": p.as_posix(), "label": label, "original_split": original_split}
                for p in sorted(class_dir.iterdir())
                if p.suffix.lower() == ".png"
            ]
    if not records:
        raise ValueError(f"no PNG images found in {raw_dir}")
    return records


def group_by_md5(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (one record per consistently labelled md5, conflict groups), sorted by md5.

    The kept copy is the first by original split (train, valid, test), then by path.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record["md5"]].append(record)

    kept, conflicts = [], []
    for md5 in sorted(groups):
        copies = sorted(
            groups[md5], key=lambda r: (ORIGINAL_SPLITS.index(r["original_split"]), r["path"])
        )
        labels = sorted({r["label"] for r in copies})
        if len(labels) > 1:
            conflicts.append({"md5": md5, "labels": labels, "copies": copies})
        else:
            kept.append(copies[0])
    return kept, conflicts


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int) -> list[tuple[int, int]]:
    """Return index pairs within max_distance bits. `hashes` is uint8, shape (n, hash_bytes)."""
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    pairs = []
    for i in range(len(hashes) - 1):
        distances = bit_counts[hashes[i + 1 :] ^ hashes[i]].sum(axis=1)
        pairs += [(i, i + 1 + int(j)) for j in np.nonzero(distances <= max_distance)[0]]
    return pairs


def cluster_ids(n: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    """Union-find over n items. Clusters are numbered by first appearance."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)

    numbers: dict[int, int] = {}
    return [numbers.setdefault(find(i), len(numbers)) for i in range(n)]


def stratified_group_split(labels: list[str], groups: list[int], seed: int = 42) -> list[str]:
    """Assign each sample to train/val/test, 70/15/15 by label, keeping each group in one split."""
    folds = StratifiedGroupKFold(
        n_splits=sum(FOLDS_PER_SPLIT.values()), shuffle=True, random_state=seed
    )
    fold_samples = (samples for _, samples in folds.split(np.zeros(len(labels)), labels, groups))

    splits = [""] * len(labels)
    for name, n_folds in FOLDS_PER_SPLIT.items():
        for _ in range(n_folds):
            for sample in next(fold_samples):
                splits[sample] = name
    return splits


def split_overlaps(rows: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
    """Count the md5s and cluster ids shared by each pair of splits."""
    values = {
        key: {name: {r[key] for r in rows if r["split"] == name} for name in FOLDS_PER_SPLIT}
        for key in ("md5", "cluster_id")
    }
    return {
        (a, b): {key: len(values[key][a] & values[key][b]) for key in values}
        for a, b in combinations(FOLDS_PER_SPLIT, 2)
    }


def assert_no_leakage(rows: list[dict]) -> None:
    """Fail if an md5 repeats anywhere, or an md5 or cluster appears in more than one split."""
    md5s = Counter(r["md5"] for r in rows)
    repeated = [md5 for md5, count in md5s.items() if count > 1]
    assert not repeated, f"{len(repeated)} md5s appear more than once: {repeated[:5]}"
    for (a, b), shared in split_overlaps(rows).items():
        assert not any(shared.values()), f"leakage between {a} and {b}: {shared}"


def assert_pairs_in_one_split(pairs: Iterable[tuple[int, int]], splits: list[str]) -> None:
    """Check the near-duplicate pairs directly, independently of the cluster ids."""
    split_pairs = [(i, j) for i, j in pairs if splits[i] != splits[j]]
    assert not split_pairs, f"{len(split_pairs)} near-duplicate pairs cross splits"


def scaled_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _open_rgb(path: str):
    """Load an image upright as RGB, flattening any transparency onto white."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
    has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB"), False
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB"), True


def _md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()


def _phash(path: str) -> str | None:
    """Return the 64-bit pHash as hex, or None if the image can't be decoded."""
    import imagehash

    try:
        image, _ = _open_rgb(path)
    except OSError:
        return None
    return str(imagehash.phash(image))


def _export_jpeg(paths: tuple[str, str]) -> tuple[int, bool]:
    """Save a resized JPEG. Return the source's long side and whether it had transparency."""
    from PIL import Image

    source, destination = paths
    image, has_alpha = _open_rgb(source)
    long_side = max(image.size)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    image.resize(scaled_size(*image.size), Image.Resampling.LANCZOS).save(
        destination, "JPEG", quality=JPEG_QUALITY
    )
    return long_side, has_alpha


def _reset_dir(path: Path, raw_dir: Path) -> None:
    """Empty an output folder, refusing any path that overlaps the raw data."""
    resolved, raw = path.resolve(), raw_dir.resolve()
    if resolved.is_relative_to(raw) or raw.is_relative_to(resolved):
        raise ValueError(f"output folder {path} overlaps the raw data in {raw_dir}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_conflicts(conflicts: list[dict], conflicts_dir: Path) -> None:
    """Copy one image per conflict group for review and list its labels. Raw files stay put."""
    rows = []
    for group in conflicts:
        source = Path(group["copies"][0]["path"])
        destination = conflicts_dir / f"{group['md5']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        rows.append(
            {
                "md5": group["md5"],
                "labels": "|".join(group["labels"]),
                "n_copies": len(group["copies"]),
                "copied_to": destination.as_posix(),
                "original_paths": "|".join(r["path"] for r in group["copies"]),
            }
        )
    _write_csv(conflicts_dir / "label_conflicts.csv", CONFLICT_COLUMNS, rows)


def _table(header: list[str], rows: Iterable[Iterable]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def render_report(
    records: list[dict],
    conflicts: list[dict],
    unreadable: list[dict],
    rows: list[dict],
    n_pairs: int,
    n_upscaled: int,
    n_alpha: int,
    seed: int,
) -> str:
    """Build REPORT.md. It has no timestamps, so a rerun gives an identical file."""
    n_raw = len(records)
    n_unique = len({r["md5"] for r in records})
    n_conflict_images = sum(len(g["copies"]) for g in conflicts)
    n_conflict_extra = n_conflict_images - len(conflicts)

    conflict_paths = {r["path"] for g in conflicts for r in g["copies"]}
    kept_paths = {r["original_path"] for r in rows}
    unreadable_paths = {r["path"] for r in unreadable}
    raw_counts = Counter((r["label"], r["original_split"]) for r in records)
    conflict_counts = Counter(r["label"] for r in records if r["path"] in conflict_paths)
    unreadable_counts = Counter(r["label"] for r in records if r["path"] in unreadable_paths)
    kept_counts = Counter((r["label"], r["split"]) for r in rows)
    duplicate_counts = Counter(
        r["label"]
        for r in records
        if r["path"] not in conflict_paths | kept_paths | unreadable_paths
    )

    class_rows = []
    for label in sorted({r["label"] for r in records}):
        raw = [raw_counts[label, split] for split in ORIGINAL_SPLITS]
        after = [kept_counts[label, split] for split in FOLDS_PER_SPLIT]
        class_rows.append(
            [
                label,
                *raw,
                sum(raw),
                duplicate_counts[label],
                conflict_counts[label],
                unreadable_counts[label],
                sum(after),
                *after,
            ]
        )
    totals = [sum(row[i] for row in class_rows) for i in range(1, len(class_rows[0]))]
    class_rows.append(["**total**", *totals])

    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row)
    sizes = [len(members) for members in clusters.values()]
    mixed = sorted(
        (cid for cid, members in clusters.items() if len({m["label"] for m in members}) > 1),
        key=int,
    )
    n_mixed_images = sum(len(clusters[cid]) for cid in mixed)

    split_counts = Counter(r["split"] for r in rows)
    split_rows = [
        [name, split_counts[name], f"{split_counts[name] / len(rows):.1%}"]
        for name in FOLDS_PER_SPLIT
    ]
    overlap_rows = [
        [f"{a} / {b}", shared["md5"], shared["cluster_id"]]
        for (a, b), shared in split_overlaps(rows).items()
    ]

    lines = [
        "# Wheat dataset v1",
        "",
        "Generated by `src/zarai/data/clean_wheat.py`. The original train/valid/test folders are "
        "ignored.",
        "",
        f"Seed {seed}. pHash near-duplicates at Hamming distance <= {MAX_HAMMING_DISTANCE}, computed "
        "on the source images before resizing. "
        f"JPEG quality {JPEG_QUALITY}, {LONG_SIDE} px on the long side.",
        "",
        "## Summary",
        "",
        *_table(
            ["Item", "Count"],
            [
                ["Raw images", n_raw],
                ["Unique md5s", n_unique],
                ["Exact-duplicate copies (raw minus unique)", n_raw - n_unique],
                ["- removed from consistently labelled groups", sum(duplicate_counts.values())],
                ["- extra copies inside conflict groups", n_conflict_extra],
                ["Conflict groups (excluded)", len(conflicts)],
                ["Raw images in conflict groups", n_conflict_images],
                ["Unreadable images (excluded)", len(unreadable)],
                ["Images in wheat_v1", len(rows)],
                ["Near-duplicate pairs", n_pairs],
                ["Near-duplicate clusters, including singletons", len(clusters)],
                ["Clusters with 2 or more images", sum(size > 1 for size in sizes)],
                ["Images in those clusters", sum(size for size in sizes if size > 1)],
                ["Largest cluster", max(sizes)],
                ["Clusters with more than one label (kept)", len(mixed)],
                ["Images in those clusters", n_mixed_images],
                [f"Images upscaled (long side under {LONG_SIDE} px)", n_upscaled],
                ["Images with transparency, flattened on white", n_alpha],
            ],
        ),
        "",
        "Near-duplicates are not removed: they stay in the dataset, together in one split.",
        "",
        "## Per-class counts",
        "",
        "`dups` are exact-duplicate copies removed from consistently labelled groups. `conflict` "
        "counts every raw copy in a conflict group.",
        "",
        *_table(
            [
                "class",
                "raw train",
                "raw valid",
                "raw test",
                "raw total",
                "dups",
                "conflict",
                "unreadable",
                "kept",
                "train",
                "val",
                "test",
            ],
            class_rows,
        ),
        "",
        "## New splits",
        "",
        *_table(["split", "images", "share"], split_rows),
        "",
        "## Leakage check",
        "",
        "Asserted on the rows read back from the written CSV. A failure stops the run before this "
        "report is written.",
        "",
        *_table(["splits", "shared md5s", "shared clusters"], overlap_rows),
        "",
        "No md5 appears twice, and every near-duplicate pair is in one split.",
        "",
        "## Clusters with more than one label",
        "",
        "These images look almost identical but are labelled differently, so some labels may be "
        "wrong. They are kept, and each cluster stays in one split.",
        "",
        "Cluster ids: " + (", ".join(mixed) if mixed else "none"),
        "",
    ]
    return "\n".join(lines)


def clean_wheat(
    raw_dir: Path,
    out_dir: Path,
    conflicts_dir: Path,
    splits_csv: Path,
    seed: int = 42,
    workers: int | None = None,
) -> list[dict]:
    """Run the whole pipeline and return the rows of the splits CSV."""
    records = collect_images(raw_dir)
    log.info("found %d images", len(records))

    with Pool(workers) as pool:
        paths = [r["path"] for r in records]
        for record, md5 in zip(records, pool.imap(_md5, paths, chunksize=32), strict=True):
            record["md5"] = md5
        kept, conflicts = group_by_md5(records)
        log.info("%d unique, consistently labelled; %d conflicts", len(kept), len(conflicts))

        hashes = pool.map(_phash, [r["path"] for r in kept], chunksize=32)
        unreadable = [r for r, h in zip(kept, hashes, strict=True) if h is None]
        kept = [r for r, h in zip(kept, hashes, strict=True) if h is not None]
        hash_bytes = np.array(
            [list(bytes.fromhex(h)) for h in hashes if h is not None], dtype=np.uint8
        )

        pairs = near_duplicate_pairs(hash_bytes, MAX_HAMMING_DISTANCE)
        clusters = cluster_ids(len(kept), pairs)
        splits = stratified_group_split([r["label"] for r in kept], clusters, seed)
        assert_pairs_in_one_split(pairs, splits)

        rows = [
            {
                "path": (out_dir / split / r["label"] / f"{r['md5']}.jpg").as_posix(),
                "label": r["label"],
                "split": split,
                "cluster_id": cluster,
                "md5": r["md5"],
                "original_split": r["original_split"],
                "original_path": r["path"],
            }
            for r, split, cluster in zip(kept, splits, clusters, strict=True)
        ]
        assert_no_leakage(rows)

        # Everything is checked before any earlier output is deleted.
        _reset_dir(out_dir, raw_dir)
        _reset_dir(conflicts_dir, raw_dir)
        _write_conflicts(conflicts, conflicts_dir)
        exports = pool.map(
            _export_jpeg, [(r["original_path"], r["path"]) for r in rows], chunksize=16
        )

    _write_csv(splits_csv, CSV_COLUMNS, rows)
    written = read_csv(splits_csv)
    assert_no_leakage(written)
    missing = [r["path"] for r in written if not Path(r["path"]).is_file()]
    assert not missing, f"{len(missing)} listed images are missing: {missing[:5]}"

    report = render_report(
        records,
        conflicts,
        unreadable,
        written,
        n_pairs=len(pairs),
        n_upscaled=sum(long_side < LONG_SIDE for long_side, _ in exports),
        n_alpha=sum(has_alpha for _, has_alpha in exports),
        seed=seed,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    log.info("wrote %d images and %s", len(written), out_dir / "REPORT.md")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/wheat"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wheat_v1"))
    parser.add_argument("--conflicts-dir", type=Path, default=Path("data/interim/label_conflicts"))
    parser.add_argument("--splits-csv", type=Path, default=Path("data/splits/wheat_v1.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="default: all CPUs")
    args = parser.parse_args()
    get_logger(__name__)
    clean_wheat(
        args.raw_dir, args.out_dir, args.conflicts_dir, args.splits_csv, args.seed, args.workers
    )


if __name__ == "__main__":
    main()
