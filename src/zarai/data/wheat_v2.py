"""Build wheat_v2: five classes from five sources, with leave-one-source-out splits.

Sources are combined into one manifest with a `source` column. The same photo found in two sources
is kept once, under the source listed first in the config, so Kaggle copies of external photos take
the external dataset's name. Photos whose labels disagree are never guessed: they are quarantined
for expert review and left out of every split. NWRD is tiled by its rust masks and is test-only.

Raw data is only read. Everything is checked, including the leakage assertions, before any earlier
output is deleted, and the run fails if a leakage check fails.
"""

import argparse
import logging
import re
from collections import Counter, defaultdict
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from zarai.common.io import read_csv
from zarai.common.logging import get_logger
from zarai.data.clean_wheat import (
    DARK_PIXEL,
    DARK_SHARE,
    JPEG_QUALITY,
    _check_outputs,
    _hash_array,
    _hash_image,
    _open_rgb,
    _reset_dir,
    _table,
    _write_csv,
    cluster_ids,
    dinov2_embeddings,
    embedding_pairs,
    near_duplicate_pairs,
    scaled_size,
)
from zarai.vision.eval_external import _image_files, normalise_folder, tile_boxes

SPLITS = ("train", "val", "test", "nwrd")
EXCLUDED = "excluded"
POOLED_COLUMN = "split_pooled"
LOSO_PREFIX = "split_loso_"
EXCLUDE_LABEL = "exclude"
THUMBNAIL_SIDE = 64
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
REVIEW_THUMBNAIL = 256
SHEET_GROUPS_PER_PAGE = 25
BASE_COLUMNS = [
    "path",
    "mask_path",
    "label",
    "source",
    "raw_label",
    "label_origin",
    "cluster_id",
    "md5",
    "pixel_md5",
    "original_path",
]
REVIEW_COLUMNS = [
    "path",
    "source",
    "current_label",
    "other_label",
    "other_source",
    "conflict_type",
    "group",
    "status",
    "thumbnail",
    "final_label",
]

log = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """A hash, cluster or test-only image is shared where it must not be."""


# ---------------------------------------------------------------------------------------------
# Labels and sources


def map_label(raw: str, label_map, classes) -> str | None:
    """Return the v2 class for a source's own label, or None if that class isn't in v2."""
    label = label_map.get(normalise_folder(raw))
    if label is not None and label not in classes:
        raise ValueError(f"label_map sends {raw!r} to {label!r}, which is not a v2 class")
    return label


def display_label(record: dict) -> str:
    return record["label"] or f"{record['raw_label']} (not in v2)"


def _record(path, source, raw_label, label, **extra) -> dict:
    return {
        "path": Path(path).as_posix(),
        "source": source,
        "raw_label": raw_label,
        "label": label,
        "label_origin": "source",
        "parent": Path(path).as_posix(),
        **extra,
    }


def collect_folders(root: Path, name: str, spec, classes) -> tuple[list[dict], Counter]:
    """Label each image by its nearest parent folder in the label map. Returns (records, skipped).

    An image under no mapped folder is kept as a record with label None, so the report can count
    it as a dropped class under its own folder name.
    """
    label_map = dict(spec.label_map)
    only_under = spec.get("only_under")
    records, skipped = [], Counter()
    for path in _image_files(root):
        parts = [normalise_folder(part) for part in path.relative_to(root).parts[:-1]]
        if only_under and only_under not in parts:
            skipped[f"outside {only_under}/"] += 1
            continue
        raw = next((p for p in reversed(parts) if p in label_map), parts[-1] if parts else ".")
        records.append(_record(path, name, raw, map_label(raw, label_map, classes)))
    return records, skipped


def collect_kaggle(name: str, spec, classes) -> tuple[list[dict], Counter]:
    """The cleaned wheat_v1 images, from their original files before any resizing."""
    label_map = dict(spec.label_map)
    records = [
        _record(
            row["original_path"], name, row["label"], map_label(row["label"], label_map, classes)
        )
        for row in read_csv(spec.splits_csv)
    ]
    return records, Counter()


def select_tiles(
    mask: np.ndarray, tile_size: int, min_coverage: float
) -> tuple[list[tuple[tuple[int, int, int, int], float]], Counter]:
    """Return ([(box, rust share)], skipped) for full tiles of a boolean HxW rust mask.

    An empty mask only means nobody marked rust, not that the leaf is healthy, so those tiles are
    skipped, as are tiles with less rust than min_coverage.
    """
    kept, skipped = [], Counter()
    for box in tile_boxes(mask.shape[1], mask.shape[0], tile_size):
        left, top, right, bottom = box
        coverage = float(mask[top:bottom, left:right].mean())
        if coverage == 0:
            skipped["tiles: empty mask"] += 1
        elif coverage < min_coverage:
            skipped[f"tiles: rust below {min_coverage:.0%}"] += 1
        else:
            kept.append((box, coverage))
    return kept, skipped


def _tile_image(job: tuple) -> tuple[list[dict], Counter]:
    """Cut one NWRD image into tiles by its mask. Not EXIF-rotated: the masks match stored pixels."""
    from PIL import Image

    image_path, mask_path, out_dir, name, label, tile_size, min_coverage = job
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) > 0
    with Image.open(image_path) as source:
        if (mask.shape[1], mask.shape[0]) != source.size:
            return [], Counter({"images: mask size differs": 1})
        image = source.convert("RGB")
    kept, stats = select_tiles(mask, tile_size, min_coverage)
    stats["images"] += 1
    records = []
    for (left, top, right, bottom), coverage in kept:
        path = Path(out_dir) / f"{Path(image_path).stem}_{left}_{top}.png"
        image.crop((left, top, right, bottom)).save(path)
        records.append(
            {
                **_record(path, name, "stripe_rust_mask", label),
                "parent": Path(image_path).as_posix(),
                "coverage": coverage,
            }
        )
    stats["tiles: kept"] += len(records)
    return records, stats


def collect_nwrd(root: Path, name: str, spec, classes, tiles_dir: Path, pool):
    """Tile every NWRD image that has a mask. Tiles are written to tiles_dir as PNG."""
    if spec.label not in classes:
        raise ValueError(f"{name}: label {spec.label!r} is not a v2 class")
    _reset_dir(tiles_dir)
    jobs, skipped = [], Counter()
    for path in _image_files(root):
        if path.parent.name != "images":
            continue
        mask = path.parent.parent / "masks" / f"{path.stem}.png"
        if not mask.exists():
            skipped["images: no mask"] += 1
            continue
        jobs.append(
            (path, mask, tiles_dir, name, spec.label, spec.tile_size, spec.min_mask_coverage)
        )
    records = []
    for found, stats in pool.imap(_tile_image, jobs):
        records += found
        skipped += stats
    return records, skipped


def collect_all(cfg, pool, tiles_dir: Path) -> tuple[list[dict], dict[str, Counter]]:
    classes = list(cfg.classes)
    records, stats = [], {}
    for name, spec in cfg.sources.items():
        if spec.kind == "folders":
            found, skipped = collect_folders(Path(spec.root), name, spec, classes)
        elif spec.kind == "wheat_v1_csv":
            found, skipped = collect_kaggle(name, spec, classes)
        elif spec.kind == "nwrd_masks":
            found, skipped = collect_nwrd(Path(spec.root), name, spec, classes, tiles_dir, pool)
        else:
            raise ValueError(f"{name}: unknown source kind {spec.kind!r}")
        if not found:
            raise ValueError(f"{name}: no images found; run scripts/check_external_datasets.py")
        log.info("%s: %d images", name, len(found))
        records += found
        stats[name] = skipped
    return records, stats


# ---------------------------------------------------------------------------------------------
# Expert review


def read_review(path: str | Path | None, classes) -> dict[str, str]:
    """{path: final_label} for rows an expert has filled in. Unknown labels are an error."""
    if not path:
        return {}
    labels = {}
    for row in read_csv(path):
        final = (row.get("final_label") or "").strip()
        if not final:
            continue
        if final not in classes and final != EXCLUDE_LABEL:
            raise ValueError(f"{path}: final_label {final!r} for {row['path']} is not a v2 class")
        labels[row["path"]] = final
    return labels


def apply_review(records: list[dict], final_labels: dict[str, str]) -> list[dict]:
    """Relabel reviewed images in place. Returns the records an expert chose to exclude."""
    excluded = []
    for record in records:
        final = final_labels.get(record["path"])
        if final == EXCLUDE_LABEL:
            excluded.append(record)
        elif final:
            record["label"] = final
            record["label_origin"] = "expert_review"
    return excluded


# ---------------------------------------------------------------------------------------------
# Duplicates and conflicts


def _describe(path: str) -> tuple[str, str | None, str | None, bytes | None]:
    """(file md5, pixel md5, pHash, 64x64 RGB thumbnail). Undecodable images give None.

    The hashes come from wheat_v1's _hash_image, so they match wheat_v1's exactly.
    """
    file_md5, pixels, phash = _hash_image(path)
    if pixels is None:
        return file_md5, None, None, None
    image, _ = _open_rgb(path)
    thumb = np.asarray(image.resize((THUMBNAIL_SIDE, THUMBNAIL_SIDE)), dtype=np.uint8)
    return file_md5, pixels, phash, thumb.tobytes()


def is_conflict(records: list[dict]) -> bool:
    """True when copies of one photo carry different labels, and at least one is a v2 class.

    Two labels that are both outside v2 aren't a conflict, as the photo is dropped either way.
    """
    votes = {record["label"] for record in records}
    return len(votes) > 1 and any(votes)


def _keep_key(rank: dict[str, int]):
    return lambda record: (rank[record["source"]], record["path"])


def group_exact(records: list[dict], rank: dict[str, int]):
    """Reduce identical pixels to one copy. Returns (kept, conflict groups, removed copies).

    removed lists (kept record, removed record) pairs.
    """
    by_pixels = defaultdict(list)
    for record in records:
        by_pixels[record["pixel_md5"]].append(record)
    kept, conflicts, removed = [], [], []
    for key in sorted(by_pixels):
        copies = sorted(by_pixels[key], key=_keep_key(rank))
        if is_conflict(copies):
            conflicts.append({"kind": "exact", "members": copies})
            continue
        kept.append(copies[0])
        removed += [(copies[0], copy) for copy in copies[1:]]
    return kept, conflicts, removed


def _thumbnail_array(record: dict) -> np.ndarray:
    return np.frombuffer(record["thumb"], np.uint8).reshape(THUMBNAIL_SIDE, THUMBNAIL_SIDE, 3)


def is_dark_background(thumb: np.ndarray) -> bool:
    """wheat_v1's rule: at least 40% of pixels below 25/255 in luminance."""
    return bool(((thumb.astype(np.float32) @ LUMA) < DARK_PIXEL).mean() >= DARK_SHARE)


def foreground_mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute RGB difference of two thumbnails, over pixels where either isn't near-black.

    A plain mean is dominated by a shared black background, so it can't tell two masked leaves
    apart. Without any foreground, it falls back to every pixel.
    """
    a, b = a.astype(np.float32), b.astype(np.float32)
    foreground = ((a @ LUMA) >= DARK_PIXEL) | ((b @ LUMA) >= DARK_PIXEL)
    if not foreground.any():
        foreground[:] = True
    return float(np.abs(a - b)[foreground].mean())


def classify_pair(
    cosine: float, mad: float, both_dark: bool, has_phash: bool, dedupe
) -> str | None:
    """ "same" (one photo), "similar" (kept in one split only) or None (not linked).

    The pixel check always counts. The embedding check counts unless both images are on dark
    backgrounds: DINOv2 and pHash both match different leaves on black (IARI), so there only
    pixels decide. An unconfirmed pHash match is a "similar" link, except on dark backgrounds.
    """
    if mad <= dedupe.max_foreground_mad:
        return "same"
    if cosine >= dedupe.min_cosine and not both_dark:
        return "same"
    if has_phash and not both_dark:
        return "similar"
    return None


def link_pairs(images: list[dict], embeddings: np.ndarray, dedupe) -> dict:
    """pHash and embedding pairs, split into same-photo links and similar-only links."""
    phash = set(near_duplicate_pairs(_hash_array(images), dedupe.max_hamming_distance))
    by_embedding = {(i, j) for i, j, _ in embedding_pairs(embeddings, dedupe.min_cosine)}
    dark = [is_dark_background(_thumbnail_array(r)) for r in images]
    found = {"same": set(), "similar": set(), None: set()}
    for i, j in sorted(phash | by_embedding):
        mad = foreground_mad(_thumbnail_array(images[i]), _thumbnail_array(images[j]))
        kind = classify_pair(
            float(embeddings[i] @ embeddings[j]), mad, dark[i] and dark[j], (i, j) in phash, dedupe
        )
        found[kind].add((i, j))
    return {
        "phash": phash,
        "embedding": by_embedding,
        "confirmed": found["same"],
        "unconfirmed": found["similar"],
        "dropped_dark": found[None],
        "dark": dark,
    }


def resolve_near_duplicates(images: list[dict], confirmed, rank: dict[str, int]):
    """Handle clusters of confirmed near-duplicates. Returns (kept indices, conflicts, removed).

    A cluster with disagreeing labels is quarantined whole. A cluster that spans sources keeps only
    the members from its highest-ranked source. Near-duplicates within one source all stay, and are
    kept in one split later, as in wheat_v1.
    """
    members = defaultdict(list)
    for index, cluster in enumerate(cluster_ids(len(images), confirmed)):
        members[cluster].append(index)
    kept, conflicts, removed = [], [], []
    for indices in members.values():
        group = [images[i] for i in indices]
        if len(group) > 1 and is_conflict(group):
            conflicts.append({"kind": "near_duplicate", "members": group})
            continue
        best = min(rank[r["source"]] for r in group)
        for index, record in zip(indices, group, strict=True):
            if rank[record["source"]] == best:
                kept.append(index)
            else:
                winner = next(r for r in group if rank[r["source"]] == best)
                removed.append((winner, record))
    return sorted(kept), conflicts, removed


# ---------------------------------------------------------------------------------------------
# Splits and leakage


def loso_column(source: str) -> str:
    return f"{LOSO_PREFIX}{source}"


def eligible_folds(rows: list[dict], sources: list[str], test_only, min_classes: int) -> list[str]:
    """Training sources with at least min_classes v2 classes, in config order."""
    classes = defaultdict(set)
    for row in rows:
        classes[row["source"]].add(row["label"])
    return [s for s in sources if s not in test_only and len(classes[s]) >= min_classes]


def fold_split(labels: list[str], groups: list, seed: int, folds_per_split: dict) -> list[str]:
    """Like wheat_v1's stratified_group_split, with the fold count per split as a parameter.

    {"train": 14, "val": 3, "test": 3} is wheat_v1's 70/15/15.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(
        n_splits=sum(folds_per_split.values()), shuffle=True, random_state=seed
    )
    fold_samples = (samples for _, samples in folds.split(np.zeros(len(labels)), labels, groups))
    splits = [""] * len(labels)
    for name, n_folds in folds_per_split.items():
        for _ in range(n_folds):
            for sample in next(fold_samples):
                splits[sample] = name
    return splits


def _group_split(rows, indices, seed, folds) -> dict[int, str]:
    strata = [f"{rows[i]['source']}|{rows[i]['label']}" for i in indices]
    groups = [rows[i]["cluster_id"] for i in indices]
    return dict(zip(indices, fold_split(strata, groups, seed, folds), strict=True))


def assign_splits(
    rows: list[dict], folds: list[str], test_only, seed: int, pooled_folds, loso_val_folds
) -> dict[str, list[str]]:
    """Return {column: split per row} for the pooled protocol and each LOSO fold.

    Test-only rows are `nwrd` everywhere, and any other image in a cluster with one is excluded
    everywhere. In a fold, the held-out source is all test, other images sharing a cluster with it
    are excluded from that fold, and the rest are split train/val by stratified group folds.
    """
    pooled_folds, loso_val_folds = dict(pooled_folds), dict(loso_val_folds)
    clusters_of = defaultdict(set)
    for row in rows:
        clusters_of[row["source"]].add(row["cluster_id"])
    test_only_clusters = set().union(*(clusters_of[s] for s in test_only))
    base = [
        (
            "nwrd"
            if row["source"] in test_only
            else EXCLUDED if row["cluster_id"] in test_only_clusters else None
        )
        for row in rows
    ]

    def column(fixed: list[str | None], folds_per_split) -> list[str]:
        open_rows = [i for i, value in enumerate(fixed) if value is None]
        assigned = _group_split(rows, open_rows, seed, folds_per_split) if open_rows else {}
        return [value if value is not None else assigned[i] for i, value in enumerate(fixed)]

    columns = {POOLED_COLUMN: column(base, pooled_folds)}
    for source in folds:
        fixed = [
            (
                value
                if value is not None
                else (
                    "test"
                    if row["source"] == source
                    else EXCLUDED if row["cluster_id"] in clusters_of[source] else None
                )
            )
            for row, value in zip(rows, base, strict=True)
        ]
        columns[loso_column(source)] = column(fixed, loso_val_folds)
    return columns


def split_columns(rows: list[dict]) -> list[str]:
    return [key for key in rows[0] if key == POOLED_COLUMN or key.startswith(LOSO_PREFIX)]


def leakage_report(rows: list[dict], column: str, test_only) -> dict:
    """Counts of everything that would be leakage in one protocol column."""
    values = {
        key: {name: {r[key] for r in rows if r[column] == name} for name in SPLITS}
        for key in ("md5", "pixel_md5", "cluster_id")
    }
    report = {
        "pairs": {
            (a, b): {key: len(values[key][a] & values[key][b]) for key in values}
            for a, b in combinations(SPLITS, 2)
        },
        "test_only_in_train_val": sum(
            r["source"] in test_only and r[column] in ("train", "val") for r in rows
        ),
        "unknown_split": sum(r[column] not in (*SPLITS, EXCLUDED) for r in rows),
        "heldout_in_train_val": 0,
    }
    if column.startswith(LOSO_PREFIX):
        source = column[len(LOSO_PREFIX) :]
        report["heldout_in_train_val"] = sum(
            r["source"] == source and r[column] in ("train", "val") for r in rows
        )
    return report


def assert_no_leakage(rows: list[dict], test_only) -> dict[str, dict]:
    """Raise LeakageError on any shared hash or cluster between splits, in any protocol column.

    Also fails if a file or pixel hash appears twice, if a test-only image is in train or val, or
    if a fold's held-out source has an image outside its test split. Returns the reports.
    """
    problems = []
    for key in ("md5", "pixel_md5"):
        repeated = [v for v, n in Counter(r[key] for r in rows).items() if n > 1]
        if repeated:
            problems.append(f"{len(repeated)} {key} values appear more than once")
    reports = {}
    for column in split_columns(rows):
        report = leakage_report(rows, column, test_only)
        reports[column] = report
        for (a, b), shared in report["pairs"].items():
            if any(shared.values()):
                problems.append(f"{column}: {a}/{b} share {shared}")
        for key in ("test_only_in_train_val", "heldout_in_train_val", "unknown_split"):
            if report[key]:
                problems.append(f"{column}: {key} = {report[key]}")
    if problems:
        raise LeakageError("leakage check failed:\n  " + "\n  ".join(problems))
    return reports


def assert_pairs_together(pairs, rows: list[dict]) -> None:
    """Check linked pairs directly, independently of the cluster ids."""
    for column in split_columns(rows):
        crossing = [
            (i, j)
            for i, j in pairs
            if EXCLUDED not in (rows[i][column], rows[j][column])
            and rows[i][column] != rows[j][column]
        ]
        if crossing:
            raise LeakageError(f"{column}: {len(crossing)} near-duplicate pairs cross splits")


# ---------------------------------------------------------------------------------------------
# Outputs


def _export_v2(job: tuple) -> tuple[int, bool, bool, float]:
    """Save the JPEG and its leaf mask. Returns (source long side, had alpha, mask fell back, share)."""
    from PIL import Image

    from zarai.data.leaf_mask import leaf_mask

    source, image_path, mask_path, long_side = job
    image, has_alpha = _open_rgb(source)  # as wheat_v1's _export_jpeg, with a long side setting
    source_long_side = max(image.size)
    size = scaled_size(*image.size, long_side)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(image_path, "JPEG", quality=JPEG_QUALITY)
    with Image.open(image_path) as saved:
        mask, fell_back = leaf_mask(np.asarray(saved.convert("RGB")))
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    return source_long_side, has_alpha, fell_back, float(mask.mean())


def _thumbnail(job: tuple[str, str]) -> None:
    source, destination = job
    image, _ = _open_rgb(source)
    image.thumbnail((REVIEW_THUMBNAIL, REVIEW_THUMBNAIL))
    image.save(destination, "JPEG", quality=90)


def conflict_type(members: list[dict]) -> str:
    return " vs ".join(sorted({f"{r['source']}:{display_label(r)}" for r in members}))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def review_rows(conflicts: list[dict], review_dir: Path, previous: dict[str, dict]) -> list[dict]:
    """One row per quarantined image. Keeps any final_label already filled in for its path."""
    rows = []
    for group in conflicts:
        members = group["members"]
        group_id = f"{group['kind']}_{min(r['pixel_md5'] for r in members)[:12]}"
        kind = conflict_type(members)
        for record in members:
            others = [r for r in members if r is not record]
            rows.append(
                {
                    "path": record["path"],
                    "source": record["source"],
                    "current_label": display_label(record),
                    "other_label": "|".join(sorted({display_label(r) for r in others})),
                    "other_source": "|".join(sorted({r["source"] for r in others})),
                    "conflict_type": kind,
                    "group": group_id,
                    "status": "quarantined",
                    "thumbnail": (
                        review_dir / "thumbs" / f"{record['source']}_{record['pixel_md5']}.jpg"
                    ).as_posix(),
                    "final_label": previous.get(record["path"], {}).get("final_label", ""),
                }
            )
    current = {row["path"] for row in rows}
    # Rows an expert already resolved stay listed, so their labels survive the next rebuild.
    for path, old in sorted(previous.items()):
        final = (old.get("final_label") or "").strip()
        if path not in current and final:
            status = "excluded" if final == EXCLUDE_LABEL else "resolved"
            rows.append({**{k: old.get(k, "") for k in REVIEW_COLUMNS}, "status": status})
    return sorted(rows, key=lambda r: (r["conflict_type"], r["group"], r["source"], r["path"]))


def write_contact_sheets(rows: list[dict], review_dir: Path) -> dict[str, list[str]]:
    """One sheet (or several pages) per conflict type, a row per group. Returns {type: paths}."""
    from PIL import Image, ImageDraw

    thumb, caption, per_group = 200, 30, 4
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["status"] == "quarantined":
            groups[row["conflict_type"]][row["group"]].append(row)
    sheets = {}
    for kind, by_group in sorted(groups.items()):
        ids = sorted(by_group)
        pages = [
            ids[i : i + SHEET_GROUPS_PER_PAGE] for i in range(0, len(ids), SHEET_GROUPS_PER_PAGE)
        ]
        sheets[kind] = []
        for number, page in enumerate(pages, 1):
            height = len(page) * (thumb + caption) + 24
            sheet = Image.new("RGB", (per_group * (thumb + 8) + 8, height), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 4), f"{kind} (page {number} of {len(pages)})", fill="black")
            for row_number, group_id in enumerate(page):
                y = 24 + row_number * (thumb + caption)
                for k, member in enumerate(by_group[group_id][:per_group]):
                    x = 8 + k * (thumb + 8)
                    draw.text(
                        (x, y), f"{member['source']}\n{member['current_label']}", fill="firebrick"
                    )
                    with Image.open(member["thumbnail"]) as image:
                        image.thumbnail((thumb, thumb))
                        sheet.paste(image, (x, y + caption - 4))
            suffix = f"_p{number}" if len(pages) > 1 else ""
            path = review_dir / f"{_slug(kind)}{suffix}.jpg"
            sheet.save(path, quality=88)
            sheets[kind].append(path.as_posix())
    return sheets


def render_report(summary: dict) -> str:
    """Build REPORT.md from the run summary. No timestamps, so a rerun gives an identical file."""
    s = summary
    classes, sources, rows = s["classes"], s["sources"], s["rows"]
    count = Counter((r["source"], r["label"]) for r in rows)
    lines = [
        "# Wheat dataset v2",
        "",
        "Generated by `src/zarai/data/wheat_v2.py` from `configs/data/wheat_v2.yaml`. Classes: "
        + ", ".join(f"`{c}`" for c in classes)
        + ".",
        "",
        f"Seed {s['seed']}. Exact duplicates share decoded RGB pixels. Near-duplicates are pHash "
        f"within Hamming distance {s['dedupe'].max_hamming_distance} or DINOv2 "
        f"(`{s['embedding_model']}`) CLS cosine >= {s['dedupe'].min_cosine}. A pair is the same "
        "photo when a pixel check (mean absolute RGB difference of 64x64 thumbnails, over "
        f"pixels where either isn't near-black, <= {s['dedupe'].max_foreground_mad}) or the "
        "embedding check confirms it. When both images are on dark backgrounds (wheat_v1's rule: "
        f"{DARK_SHARE:.0%} of pixels below {DARK_PIXEL}/255), only the pixel check counts, because "
        "pHash and DINOv2 both match different leaves on black: this affects "
        + (
            ", ".join(f"{src} ({n} images)" for src, n in sorted(s["links"]["dark_images"].items()))
            or "no images"
        )
        + ". A pHash match that isn't confirmed only keeps the pair in one split. Images are "
        f"downscaled to {s['long_side']} px on the long side only if larger, and "
        f"saved as RGB JPEG quality {JPEG_QUALITY} with transparency flattened on white.",
        "",
        "## Images per source and class",
        "",
        *_table(
            ["source", *classes, "total"],
            [
                [src, *[count[src, c] for c in classes], sum(count[src, c] for c in classes)]
                for src in sources
            ]
            + [
                [
                    "**total**",
                    *[sum(count[src, c] for src in sources) for c in classes],
                    len(rows),
                ]
            ],
        ),
        "",
        "## From raw images to wheat_v2",
        "",
        *_table(
            ["source", "found", "unreadable", "dropped class", "excluded by review", "quarantined"]
            + ["duplicate copies removed", "kept"],
            [
                [
                    src,
                    s["found"][src],
                    s["unreadable"][src],
                    sum(n for (k, _), n in s["dropped"].items() if k == src),
                    s["review_excluded"][src],
                    s["quarantined"][src],
                    s["removed_from"][src],
                    sum(count[src, c] for c in classes),
                ]
                for src in sources
            ],
        ),
        "",
        "Not counted as found: "
        + (
            "; ".join(
                f"{src}: " + ", ".join(f"{k} {n}" for k, n in sorted(stats.items()))
                for src, stats in s["skipped"].items()
                if stats
            )
            or "nothing"
        )
        + ".",
        "",
        "## Dropped classes",
        "",
        "Labels that aren't v2 classes, by source and the source's own label. These images are "
        "left out.",
        "",
        *_table(
            ["source", "label", "images"],
            [[src, f"`{label}`", n] for (src, label), n in sorted(s["dropped"].items())],
        ),
        "",
        "## Duplicates removed",
        "",
        "One copy is kept per photo, from the source listed first in the config. Counts are removed "
        "copies by the source that kept the photo and the source that lost its copy.",
        "",
        *_table(
            ["kept in", "removed from", "exact pixels", "confirmed near-duplicate"],
            [
                [kept, lost, s["removed_exact"][kept, lost], s["removed_near"][kept, lost]]
                for kept, lost in sorted(set(s["removed_exact"]) | set(s["removed_near"]))
            ],
        ),
        "",
        "Kaggle images now named after their external source, by label: "
        + (
            ", ".join(
                f"{src} `{label}` {n}"
                for (src, label), n in sorted(s["kaggle_reassigned"].items())
                if n
            )
            or "none"
        )
        + ".",
        "",
        "## Near-duplicates",
        "",
        *_table(
            ["item", "count"],
            [
                [
                    f"pHash pairs (distance <= {s['dedupe'].max_hamming_distance})",
                    s["links"]["phash"],
                ],
                [f"Embedding pairs (cosine >= {s['dedupe'].min_cosine})", s["links"]["embedding"]],
                ["- of these, pairs found by both", s["links"]["both"]],
                ["Same-photo links", s["links"]["same"]],
                ["- between different sources", s["links"]["same_cross"]],
                ["Similar-only links (kept in one split, not merged)", s["links"]["similar"]],
                ["- between different sources", s["links"]["similar_cross"]],
                [
                    "Dark-background pairs left unlinked (pixels disagree)",
                    s["links"]["dropped_dark"],
                ],
                ["Final clusters, including singletons", len({r["cluster_id"] for r in rows})],
                ["Largest cluster", max(Counter(r["cluster_id"] for r in rows).values())],
            ],
        ),
        "",
        "## Quarantine",
        "",
        "Images whose copies disagree on the label are excluded from training and testing. Each is "
        "listed in `data/interim/label_review_v2.csv` with a thumbnail, and each conflict type has "
        "a contact sheet under `data/interim/wheat_v2/review/`. Once an expert fills in "
        "`final_label` (a v2 class, or `exclude`), set `apply_label_review` in "
        "`configs/data/wheat_v2.yaml` to that CSV and rebuild.",
        "",
        *_table(
            ["conflict", "groups", "images", "contact sheet"],
            [
                [
                    kind.replace("|", "\\|"),
                    n_groups,
                    s["conflict_images"][kind],
                    ", ".join(f"`{p}`" for p in s["sheets"].get(kind, [])) or "-",
                ]
                for kind, n_groups in sorted(
                    s["conflict_groups"].items(),
                    key=lambda item: (-s["conflict_images"][item[0]], item[0]),
                )
            ],
        ),
        "",
        f"Expert labels applied this run: {s['review_applied']}. "
        f"Still in conflict after review: {s['review_unresolved']}.",
        "",
        "## NWRD tiles",
        "",
        f"{s['nwrd_tile_size']} px tiles at full resolution. A tile is `yellow_rust` only if its "
        f"mask covers at least {s['nwrd_min_coverage']:.0%} of it. Tiles with an empty mask are "
        "skipped, not treated as healthy. NWRD is test-only.",
        "",
        *_table(
            ["item", "count"], [[k, n] for k, n in sorted(s["skipped"].get("nwrd", {}).items())]
        ),
        "",
        "## Pooled split (development only)",
        "",
        *_split_table(rows, POOLED_COLUMN, sources),
        "",
        "## Leave-one-source-out folds",
        "",
        f"A source gets a fold only if it has at least {s['min_classes']} v2 classes. "
        + "Without a fold: "
        + (
            ", ".join(
                f"{src} ({len(s['source_classes'][src])} class)"
                for src in sources
                if src not in s["folds"] and src not in s["test_only"]
            )
            or "none"
        )
        + ". In each fold the held-out source is the whole test set, val is about 15% of the "
        "other sources, and NWRD is an extra test set. `excluded` are images sharing a cluster with "
        "the held-out source or with NWRD.",
        "",
    ]
    for fold in s["folds"]:
        column = loso_column(fold)
        missing = set(classes) - {r["label"] for r in rows if r[column] == "train"}
        lines += [f"### Held out: {fold}", "", *_split_table(rows, column, sources), ""]
        if missing:
            lines += [f"Classes with no training images in this fold: {sorted(missing)}.", ""]
    lines += [
        "## Leakage check",
        "",
        "Asserted on the rows read back from the written CSV; the build fails if any number below "
        "isn't zero. It also asserts that no file or pixel hash appears twice and that every "
        "linked pair is in one split (or excluded).",
        "",
        *_table(
            ["protocol", "shared md5s", "shared pixel hashes", "shared clusters"]
            + ["NWRD in train or val", "held-out source outside test"],
            [
                [
                    column,
                    sum(v["md5"] for v in report["pairs"].values()),
                    sum(v["pixel_md5"] for v in report["pairs"].values()),
                    sum(v["cluster_id"] for v in report["pairs"].values()),
                    report["test_only_in_train_val"],
                    report["heldout_in_train_val"] if column != POOLED_COLUMN else "-",
                ]
                for column, report in s["leakage"].items()
            ],
        ),
        "",
        "Shared counts are summed over every pair of train, val, test and nwrd.",
        "",
        "## Preprocessing",
        "",
        *_table(
            ["item", "count"],
            [
                [f"Downscaled to {s['long_side']} px", s["downscaled"]],
                [f"Kept at original size (long side <= {s['long_side']} px)", s["original_size"]],
                ["Images with transparency, flattened on white", s["had_alpha"]],
                [
                    "Leaf masks that found almost no plant (mask covers everything)",
                    s["mask_fell_back"],
                ],
                ["Median leaf-mask share of the image", f"{s['mask_median']:.2f}"],
            ],
        ),
        "",
        "Leaf masks (`mask_path`) come from a colour heuristic (`src/zarai/data/leaf_mask.py`), "
        "used by `background: remove` and the strong augmentation's background replacement. Check "
        "them in the sample grids from `dataloader.py` before trusting either.",
        "",
    ]
    return "\n".join(lines)


def _split_table(rows: list[dict], column: str, sources: list[str]) -> list[str]:
    counts = Counter((r["source"], r[column]) for r in rows)
    names = [*SPLITS, EXCLUDED]
    return _table(
        ["source", *names],
        [[src, *[counts[src, n] for n in names]] for src in sources]
        + [["**total**", *[sum(counts[src, n] for src in sources) for n in names]]],
    )


# ---------------------------------------------------------------------------------------------
# Pipeline


def _previous_review(cfg) -> dict[str, dict]:
    rows = {}
    for path in (cfg.outputs.review_csv, cfg.get("apply_label_review")):
        if path and Path(path).is_file():
            rows.update({row["path"]: row for row in read_csv(path)})
    return rows


def build_wheat_v2(cfg, workers: int | None = None, embedder=None, embedding_model="unspecified"):
    """Run the whole pipeline and return the rows of the splits CSV."""
    classes = list(cfg.classes)
    sources = list(cfg.sources)
    test_only = {name for name, spec in cfg.sources.items() if spec.get("test_only")}
    rank = {name: i for i, name in enumerate(sources)}
    out_dir = Path(cfg.outputs.processed_dir)
    interim = Path(cfg.outputs.interim_dir)
    splits_csv = Path(cfg.outputs.splits_csv)
    review_csv = Path(cfg.outputs.review_csv)
    review_dir = interim / "review"
    for spec in cfg.sources.values():
        if spec.get("root"):
            _check_outputs(Path(spec.root), out_dir, interim, splits_csv, review_csv)
    final_labels = read_review(cfg.get("apply_label_review"), classes)
    previous = _previous_review(cfg)

    with Pool(workers) as pool:
        records, skipped = collect_all(cfg, pool, interim / "nwrd_tiles")
        found = Counter(r["source"] for r in records)
        excluded_by_review = apply_review(records, final_labels)
        excluded_paths = {r["path"] for r in excluded_by_review}
        records = [r for r in records if r["path"] not in excluded_paths]

        described = pool.map(_describe, [r["path"] for r in records], chunksize=16)
        for record, (md5, pixels, phash, thumb) in zip(records, described, strict=True):
            record.update(md5=md5, pixel_md5=pixels, phash=phash, thumb=thumb)
        unreadable = Counter(r["source"] for r in records if r["pixel_md5"] is None)
        records = [r for r in records if r["pixel_md5"] is not None]

        images, conflicts, removed_exact = group_exact(records, rank)
        log.info("%d distinct images, %d exact conflicts", len(images), len(conflicts))
        embeddings = embedder([r["path"] for r in images], pool)
        links = link_pairs(images, embeddings, cfg.dedupe)
        kept, near_conflicts, removed_near = resolve_near_duplicates(
            images, links["confirmed"], rank
        )
        conflicts += near_conflicts
        log.info("%d near-duplicate conflicts; %d images kept", len(near_conflicts), len(kept))

        dropped = Counter((images[i]["source"], images[i]["raw_label"]) for i in kept)
        final_index = [i for i in kept if images[i]["label"] is not None]
        dropped -= Counter((images[i]["source"], images[i]["raw_label"]) for i in final_index)
        position = {old: new for new, old in enumerate(final_index)}
        final = [images[i] for i in final_index]

        pairs = {
            (position[i], position[j])
            for i, j in links["confirmed"] | links["unconfirmed"]
            if i in position and j in position
        }
        by_parent = defaultdict(list)
        for index, record in enumerate(final):
            by_parent[record["parent"]].append(index)
        pairs |= {
            (a, b) for group in by_parent.values() for a, b in zip(group, group[1:], strict=False)
        }
        clusters = cluster_ids(len(final), sorted(pairs))

        rows = [
            {
                "path": (out_dir / "images" / f"{r['pixel_md5']}.jpg").as_posix(),
                "mask_path": (out_dir / "masks" / f"{r['pixel_md5']}.png").as_posix(),
                "label": r["label"],
                "source": r["source"],
                "raw_label": r["raw_label"],
                "label_origin": r["label_origin"],
                "cluster_id": cluster,
                "md5": r["md5"],
                "pixel_md5": r["pixel_md5"],
                "original_path": r["path"],
            }
            for r, cluster in zip(final, clusters, strict=True)
        ]
        folds = eligible_folds(rows, sources, test_only, cfg.splits.min_classes_for_fold)
        columns = assign_splits(
            rows,
            folds,
            test_only,
            cfg.seed,
            cfg.splits.pooled_folds,
            cfg.splits.loso_val_folds,
        )
        for column, values in columns.items():
            for row, value in zip(rows, values, strict=True):
                row[column] = value
        assert_no_leakage(rows, test_only)
        assert_pairs_together(pairs, rows)
        quarantined = {r["pixel_md5"] for g in conflicts for r in g["members"]}
        if quarantined & {r["pixel_md5"] for r in rows}:
            raise LeakageError("a quarantined image is in wheat_v2")

        # Everything is checked before any earlier output is deleted.
        _reset_dir(out_dir)
        (out_dir / "images").mkdir()
        (out_dir / "masks").mkdir()
        _reset_dir(review_dir)
        (review_dir / "thumbs").mkdir()
        exports = pool.map(
            _export_v2,
            [
                (r["original_path"], r["path"], r["mask_path"], cfg.preprocess.long_side)
                for r in rows
            ],
            chunksize=8,
        )
        reviews = review_rows(conflicts, review_dir, previous)
        pool.map(
            _thumbnail,
            [(row["path"], row["thumbnail"]) for row in reviews if row["status"] == "quarantined"],
            chunksize=4,
        )

    _write_csv(review_csv, REVIEW_COLUMNS, reviews)
    sheets = write_contact_sheets(reviews, review_dir)
    _write_csv(splits_csv, [*BASE_COLUMNS, *columns], rows)
    written = read_csv(splits_csv)
    leakage = assert_no_leakage(written, test_only)
    missing = [r["path"] for r in written if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed images are missing: {missing[:5]}")

    members = [r for g in conflicts for r in g["members"]]
    conflict_groups = Counter(conflict_type(g["members"]) for g in conflicts)
    conflict_images = Counter()
    for group in conflicts:
        conflict_images[conflict_type(group["members"])] += len(group["members"])
    source_classes = defaultdict(set)
    for row in written:
        source_classes[row["source"]].add(row["label"])
    removed_from = Counter(lost["source"] for _, lost in removed_exact + removed_near)
    kaggle_matches = Counter(
        (kept_copy["source"], display_label(lost))
        for kept_copy, lost in removed_exact + removed_near
        if lost["source"] == "kaggle" and kept_copy["source"] != "kaggle"
    )
    long_side = cfg.preprocess.long_side
    nwrd_spec = next(spec for spec in cfg.sources.values() if spec.kind == "nwrd_masks")
    summary = {
        "classes": classes,
        "sources": sources,
        "rows": written,
        "seed": cfg.seed,
        "dedupe": cfg.dedupe,
        "embedding_model": embedding_model,
        "long_side": long_side,
        "found": found,
        "skipped": skipped,
        "unreadable": unreadable,
        "dropped": dropped,
        "review_excluded": Counter(r["source"] for r in excluded_by_review),
        "quarantined": Counter(r["source"] for r in members),
        "removed_exact": Counter((k["source"], lost["source"]) for k, lost in removed_exact),
        "removed_near": Counter((k["source"], lost["source"]) for k, lost in removed_near),
        "removed_from": removed_from,
        "kaggle_reassigned": kaggle_matches,
        "links": {
            "phash": len(links["phash"]),
            "embedding": len(links["embedding"]),
            "both": len(links["phash"] & links["embedding"]),
            "same": len(links["confirmed"]),
            "same_cross": sum(
                images[i]["source"] != images[j]["source"] for i, j in links["confirmed"]
            ),
            "similar": len(links["unconfirmed"]),
            "similar_cross": sum(
                images[i]["source"] != images[j]["source"] for i, j in links["unconfirmed"]
            ),
            "dropped_dark": len(links["dropped_dark"]),
            "dark_images": Counter(
                images[i]["source"] for i, is_dark in enumerate(links["dark"]) if is_dark
            ),
        },
        "conflict_groups": conflict_groups,
        "conflict_images": conflict_images,
        "sheets": sheets,
        "review_applied": sum(r["label_origin"] == "expert_review" for r in written)
        + len(excluded_by_review),
        "review_unresolved": sum(1 for r in members if r["path"] in final_labels),
        "nwrd_tile_size": nwrd_spec.tile_size,
        "nwrd_min_coverage": nwrd_spec.min_mask_coverage,
        "folds": folds,
        "test_only": test_only,
        "min_classes": cfg.splits.min_classes_for_fold,
        "source_classes": source_classes,
        "leakage": leakage,
        "downscaled": sum(e[0] > long_side for e in exports),
        "original_size": sum(e[0] <= long_side for e in exports),
        "had_alpha": sum(e[1] for e in exports),
        "mask_fell_back": sum(e[2] for e in exports),
        "mask_median": float(np.median([e[3] for e in exports])),
    }
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    log.info("wrote %d images, %d folds and %s", len(written), len(folds), out_dir / "REPORT.md")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/wheat_v2.yaml"))
    parser.add_argument("--workers", type=int, default=None, help="default: all CPUs")
    parser.add_argument("--device", default="cuda", help="device for the DINOv2 embeddings")
    args = parser.parse_args()
    get_logger(__name__)
    cfg = OmegaConf.load(args.config)
    embedding = cfg.dedupe.embedding
    build_wheat_v2(
        cfg,
        args.workers,
        partial(
            dinov2_embeddings,
            model_name=embedding.model,
            revision=embedding.hf_revision,
            device=args.device,
        ),
        f"{embedding.model}@{embedding.hf_revision}",
    )


if __name__ == "__main__":
    main()
