"""Deduplicate the public wheat disease dataset and rebuild its splits without leakage.

The original train/valid/test folders share images, so they are ignored. Images with identical
decoded pixels are reduced to one copy. Images with conflicting labels are copied out for review.
Near-duplicates (pHash and DINOv2 embeddings) are clustered so that each cluster stays in one new
split. The raw data is only ever read.
"""

import argparse
import csv
import hashlib
import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from functools import partial
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
EMBEDDING_MODEL = "vit_small_patch14_dinov2.lvd142m"
EMBEDDING_RESIZE = 256
EMBEDDING_CROP = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_COSINE = 0.95
REVIEW_MARGIN = 0.02
N_REVIEW_PAIRS = 20
LONG_SIDE = 384
JPEG_QUALITY = 92
# An image counts as dark-background when 40% of its pixels are below 25/255 in luminance.
DARK_PIXEL = 25
DARK_SHARE = 0.4
SMALL_CLASS_SIZE = 200
FOCUS_CONFLICT_PATTERNS = ("black_rust|brown_rust", "leaf_blight|septoria")

MD5_CONFLICT = "md5_conflict"
PIXEL_CONFLICT = "pixel_conflict"
CLUSTER_CONFLICT = "cluster_conflict"
CSV_COLUMNS = [
    "path",
    "label",
    "split",
    "cluster_id",
    "md5",
    "pixel_md5",
    "original_split",
    "original_path",
]
CONFLICT_COLUMNS = [
    "reason",
    "conflict_group",
    "label",
    "labels",
    "md5",
    "pixel_md5",
    "original_split",
    "original_path",
    "copied_to",
]
REVIEW_COLUMNS = [
    "cosine",
    "linked",
    "label_a",
    "label_b",
    "split_a",
    "split_b",
    "path_a",
    "path_b",
]

log = logging.getLogger(__name__)

Embedder = Callable[[list[str], object], np.ndarray]


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


def group_by_pixels(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (one record per consistently labelled image, conflict groups), sorted by pixel hash.

    Each kept record lists its raw `copies`. It is the first copy by original split (train, valid,
    test), then by path. A conflict copy's reason is md5_conflict if its own file bytes appear
    under more than one label, otherwise pixel_conflict.
    """
    by_pixels = defaultdict(list)
    md5_labels = defaultdict(set)
    for record in records:
        by_pixels[record["pixel_md5"]].append(record)
        md5_labels[record["md5"]].add(record["label"])

    kept, conflicts = [], []
    for pixel_md5 in sorted(by_pixels):
        copies = sorted(
            by_pixels[pixel_md5],
            key=lambda r: (ORIGINAL_SPLITS.index(r["original_split"]), r["path"]),
        )
        labels = sorted({r["label"] for r in copies})
        if len(labels) == 1:
            kept.append({**copies[0], "copies": copies})
            continue
        conflicts.append(
            {
                "group": pixel_md5,
                "labels": labels,
                "copies": [
                    {
                        **r,
                        "reason": MD5_CONFLICT if len(md5_labels[r["md5"]]) > 1 else PIXEL_CONFLICT,
                    }
                    for r in copies
                ],
            }
        )
    return kept, conflicts


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int) -> list[tuple[int, int]]:
    """Return index pairs within max_distance bits. `hashes` is uint8, shape (n, hash_bytes)."""
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    pairs = []
    for i in range(len(hashes) - 1):
        distances = bit_counts[hashes[i + 1 :] ^ hashes[i]].sum(axis=1)
        pairs += [(i, i + 1 + int(j)) for j in np.nonzero(distances <= max_distance)[0]]
    return pairs


def embedding_pairs(
    embeddings: np.ndarray, min_cosine: float, chunk_size: int = 1024
) -> list[tuple[int, int, float]]:
    """Return (i, j, cosine) for i < j at or above min_cosine. Embeddings must be L2-normalised."""
    pairs = []
    for start in range(0, len(embeddings), chunk_size):
        similarities = embeddings[start : start + chunk_size] @ embeddings.T
        rows, columns = np.nonzero(similarities >= min_cosine)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if start + row < column:
                pairs.append((start + row, column, float(similarities[row, column])))
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


def mixed_label_clusters(labels: list[str], clusters: list[int]) -> set[int]:
    """Return the ids of clusters that hold more than one label."""
    found = defaultdict(set)
    for label, cluster in zip(labels, clusters, strict=True):
        found[cluster].add(label)
    return {cluster for cluster, members in found.items() if len(members) > 1}


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
    """Count the file md5s, pixel hashes and cluster ids shared by each pair of splits."""
    values = {
        key: {name: {r[key] for r in rows if r["split"] == name} for name in FOLDS_PER_SPLIT}
        for key in ("md5", "pixel_md5", "cluster_id")
    }
    return {
        (a, b): {key: len(values[key][a] & values[key][b]) for key in values}
        for a, b in combinations(FOLDS_PER_SPLIT, 2)
    }


def assert_no_leakage(rows: list[dict]) -> None:
    """Fail if a hash repeats anywhere, or a hash or cluster appears in more than one split."""
    for key in ("md5", "pixel_md5"):
        repeated = [value for value, count in Counter(r[key] for r in rows).items() if count > 1]
        assert not repeated, f"{len(repeated)} {key} values appear more than once: {repeated[:5]}"
    for (a, b), shared in split_overlaps(rows).items():
        assert not any(shared.values()), f"leakage between {a} and {b}: {shared}"


def assert_pairs_in_one_split(pairs: Iterable[tuple[int, int]], splits: list[str]) -> None:
    """Check the near-duplicate pairs directly, independently of the cluster ids."""
    split_pairs = [(i, j) for i, j in pairs if splits[i] != splits[j]]
    assert not split_pairs, f"{len(split_pairs)} near-duplicate pairs cross splits"


def scaled_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    """Return the size after downscaling to long_side. Smaller images keep their size."""
    if max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def select_review_pairs(
    pairs: list[tuple[int, int, float]], threshold: float, margin: float, n: int
) -> list[tuple[int, int, float]]:
    """Pick up to n pairs within margin of threshold, half on each side, evenly spaced by cosine."""
    pairs = sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))
    above = [p for p in pairs if threshold <= p[2] <= threshold + margin]
    below = [p for p in pairs if threshold - margin <= p[2] < threshold]
    selected = []
    for side in (above, below):
        count = min(n // 2, len(side))
        selected += [side[i] for i in np.linspace(0, len(side) - 1, count, dtype=int)]
    return selected


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


def _hash_image(path: str) -> tuple[str, str | None, str | None]:
    """Return (file md5, md5 of the RGB pixels and size, pHash). Undecodable: both hashes None."""
    import imagehash

    file_md5 = hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()
    try:
        image, _ = _open_rgb(path)
    except OSError:
        return file_md5, None, None
    pixels = hashlib.md5(f"{image.width}x{image.height}".encode(), usedforsecurity=False)
    pixels.update(image.tobytes())
    return file_md5, pixels.hexdigest(), str(imagehash.phash(image))


def _hash_array(records: list[dict]) -> np.ndarray:
    return np.array([list(bytes.fromhex(r["phash"])) for r in records], dtype=np.uint8)


def _embedding_input(path: str) -> np.ndarray:
    """Standard DINOv2 preprocessing up to the crop: returns uint8 HWC."""
    from PIL import Image

    image, _ = _open_rgb(path)
    scale = EMBEDDING_RESIZE / min(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    left, top = (width - EMBEDDING_CROP) // 2, (height - EMBEDDING_CROP) // 2
    return np.asarray(image.crop((left, top, left + EMBEDDING_CROP, top + EMBEDDING_CROP)))


def dinov2_embeddings(
    paths: list[str], pool, device: str = "cuda", batch_size: int = 128
) -> np.ndarray:
    """Return L2-normalised DINOv2 ViT-S/14 CLS embeddings, shape (n, 384)."""
    import timm
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check nvidia-smi, or pass --device cpu")
    model = timm.create_model(
        EMBEDDING_MODEL, pretrained=True, img_size=EMBEDDING_CROP, num_classes=0
    )
    model = model.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def embed(batch: list[np.ndarray]) -> np.ndarray:
        pixels = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).float() / 255
        cls_tokens = model.forward_features((pixels - mean) / std)[:, 0]
        return torch.nn.functional.normalize(cls_tokens, dim=1).cpu().numpy()

    chunks, batch = [], []
    with torch.inference_mode():
        for pixels in pool.imap(_embedding_input, paths, chunksize=8):
            batch.append(pixels)
            if len(batch) == batch_size:
                chunks.append(embed(batch))
                batch = []
        if batch:
            chunks.append(embed(batch))
    return np.concatenate(chunks)


def _export_jpeg(paths: tuple[str, str]) -> tuple[int, bool, bool]:
    """Save a JPEG, downscaling if needed. Return (source long side, had alpha, dark background)."""
    from PIL import Image

    source, destination = paths
    image, has_alpha = _open_rgb(source)
    luminance = np.asarray(image.convert("L").resize((64, 64)))
    is_dark = bool((luminance < DARK_PIXEL).mean() >= DARK_SHARE)
    long_side = max(image.size)
    size = scaled_size(*image.size)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=JPEG_QUALITY)
    return long_side, has_alpha, is_dark


def _check_outputs(raw_dir: Path, *outputs: Path) -> None:
    """Refuse any output that overlaps the raw data, because outputs get deleted."""
    raw = raw_dir.resolve()
    for output in outputs:
        resolved = output.resolve()
        if resolved.is_relative_to(raw) or raw.is_relative_to(resolved):
            raise ValueError(f"output {output} overlaps the raw data in {raw_dir}")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_conflicts(groups: list[dict], conflicts_dir: Path) -> None:
    """Copy one file per distinct image and write one CSV row per raw copy. Raw files stay put."""
    rows = []
    for group in groups:
        for image_md5 in sorted({r["pixel_md5"] for r in group["copies"]}):
            copies = [r for r in group["copies"] if r["pixel_md5"] == image_md5]
            source = Path(copies[0]["path"])
            destination = conflicts_dir / f"{image_md5}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            rows += [
                {
                    "reason": r["reason"],
                    "conflict_group": group["group"],
                    "label": r["label"],
                    "labels": "|".join(group["labels"]),
                    "md5": r["md5"],
                    "pixel_md5": r["pixel_md5"],
                    "original_split": r["original_split"],
                    "original_path": r["path"],
                    "copied_to": destination.as_posix(),
                }
                for r in copies
            ]
    _write_csv(conflicts_dir / "label_conflicts.csv", CONFLICT_COLUMNS, rows)


def _write_review(pairs: list[tuple[int, int, float]], rows: list[dict], review_dir: Path) -> None:
    """Save a contact sheet and CSV of pairs near the cosine threshold, for a visual check."""
    from PIL import Image, ImageDraw

    thumb, caption, per_row = 200, 34, 4
    cell_width = 2 * thumb + 20
    n_rows = max(1, (len(pairs) + per_row - 1) // per_row)
    sheet = Image.new("RGB", (per_row * cell_width, n_rows * (thumb + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    review_rows = []
    for n, (i, j, cosine) in enumerate(pairs):
        x, y = n % per_row * cell_width, n // per_row * (thumb + caption)
        linked = cosine >= MIN_COSINE
        a, b = rows[i], rows[j]
        status = "LINKED" if linked else "not linked"
        draw.text(
            (x + 2, y + 2),
            f"{cosine:.4f} {status}\n{a['label']} / {b['label']}",
            fill="black" if linked else "firebrick",
        )
        for k, row in enumerate((a, b)):
            with Image.open(row["path"]) as image:
                image.thumbnail((thumb, thumb))
                sheet.paste(image, (x + k * (thumb + 4), y + caption))
        review_rows.append(
            {
                "cosine": f"{cosine:.6f}",
                "linked": linked,
                "label_a": a["label"],
                "label_b": b["label"],
                "split_a": a["split"],
                "split_b": b["split"],
                "path_a": a["path"],
                "path_b": b["path"],
            }
        )
    review_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(review_dir / "threshold_pairs.jpg", quality=90)
    _write_csv(review_dir / "threshold_pairs.csv", REVIEW_COLUMNS, review_rows)


def _table(header: list[str], rows: Iterable[Iterable]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def _known_issues(
    rows: list[dict], groups: list[dict], dark_labels: Counter, embedding_stats: dict
) -> list[str]:
    """Build the Known issues section, using this run's numbers."""
    kept = Counter(r["label"] for r in rows)
    test = Counter(r["label"] for r in rows if r["split"] == "test")
    dark_share = {label: dark_labels[label] / count for label, count in kept.items()}
    others = {label: share for label, share in dark_share.items() if label != "mildew"}
    next_label = max(others, key=others.get)

    exact = Counter("|".join(g["labels"]) for g in groups if g["kind"] == "exact")
    cluster = Counter("|".join(g["labels"]) for g in groups if g["kind"] == "cluster")
    patterns = sorted(set(exact) | set(cluster), key=lambda p: (-(exact[p] + cluster[p]), p))[:6]
    focus = ", ".join(
        f"`{p}` ({exact[p]} exact, {cluster[p]} cluster)" for p in FOCUS_CONFLICT_PATTERNS
    )

    small = sorted((c for c in kept if kept[c] < SMALL_CLASS_SIZE), key=kept.get)
    small_text = ", ".join(f"`{c}` ({kept[c]} images, {test[c]} in test)" for c in small) or "none"
    largest = max(kept, key=kept.get)
    smallest = min(kept, key=kept.get)

    return [
        "## Known issues",
        "",
        "### Mildew watermark and background bias",
        "",
        f"{dark_share['mildew']:.1%} of `mildew` images have a mostly black background "
        f"({dark_labels['mildew']} of {kept['mildew']}), against {others[next_label]:.1%} for the "
        f"next-highest class (`{next_label}`). Samples viewed by eye also show the same text "
        "watermark across these black-background images and on some `fusarium_head_blight` images. "
        "The watermark itself is not measured "
        f'here. "Mostly black" means at least {DARK_SHARE:.0%} of pixels below {DARK_PIXEL}/255 '
        "in luminance.",
        "",
        "A model can learn the background or watermark instead of the disease, and the "
        "pHash and embedding checks don't keep images from one source together, so this bias is "
        "present in train, val and test. High `mildew` scores may therefore overstate real "
        "performance. Evaluate `mildew` on field images, or with the background masked.",
        "",
        "### Label conflict patterns",
        "",
        f'The main patterns are {focus}. "Exact" counts images with identical bytes or pixels '
        'under both labels, and "cluster" counts pHash near-duplicate clusters holding both. '
        "Patterns this concentrated may be a systematic mix-up rather than random error, so they "
        "need review by someone who can identify these diseases. All of these images are excluded "
        "and listed in `label_conflicts.csv`, but similar images labelled only one way can't be "
        "detected and may remain.",
        "",
        *_table(
            ["labels", "exact groups", "pHash clusters"],
            [["`" + p.replace("|", "\\|") + "`", exact[p], cluster[p]] for p in patterns],
        ),
        "",
        "### Small classes",
        "",
        f"Classes under {SMALL_CLASS_SIZE} images: {small_text}. The largest class, `{largest}`, "
        f"has {kept[largest]} images, {kept[largest] / kept[smallest]:.1f} times the smallest. "
        "Per-class scores for small classes rest on few test images and will be noisy, so report "
        "macro-averaged metrics with confidence intervals.",
        "",
        "### Embedding clusters with mixed labels",
        "",
        f"{embedding_stats['mixed_clusters']} clusters ({embedding_stats['mixed_images']} images) "
        "hold more than one label only after the embedding links were added. They stay in the "
        "dataset, each in one split, because they have not been reviewed. Only mixed-label pHash "
        "clusters are excluded.",
        "",
    ]


def render_report(
    records: list[dict],
    groups: list[dict],
    unreadable: list[dict],
    rows: list[dict],
    exports: list[tuple[int, bool, bool]],
    phash_stats: dict,
    embedding_stats: dict,
    seed: int,
) -> str:
    """Build REPORT.md. It has no timestamps, so a rerun gives an identical file."""
    readable = [r for r in records if r.get("pixel_md5")]
    conflict_copies = [r for g in groups for r in g["copies"]]
    conflict_paths = {r["path"] for r in conflict_copies}
    kept_paths = {r["original_path"] for r in rows}
    duplicates = [
        r for r in readable if r["path"] not in conflict_paths and r["path"] not in kept_paths
    ]

    raw_counts = Counter((r["label"], r["original_split"]) for r in records)
    duplicate_counts = Counter(r["label"] for r in duplicates)
    conflict_counts = Counter(r["label"] for r in conflict_copies)
    unreadable_counts = Counter(r["label"] for r in unreadable)
    kept_counts = Counter((r["label"], r["split"]) for r in rows)

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

    reasons = Counter(r["reason"] for r in conflict_copies)
    distinct = defaultdict(set)
    for row in conflict_copies:
        distinct[row["reason"]].add(row["pixel_md5"])
    conflict_rows = [
        [
            f"`{reason}`",
            count,
            reasons[reason],
            len(distinct[reason]),
        ]
        for reason, count in (
            (
                MD5_CONFLICT,
                len({r["md5"] for r in conflict_copies if r["reason"] == MD5_CONFLICT}),
            ),
            (
                PIXEL_CONFLICT,
                sum(
                    all(r["reason"] == PIXEL_CONFLICT for r in g["copies"])
                    for g in groups
                    if g["kind"] == "exact"
                ),
            ),
            (CLUSTER_CONFLICT, sum(g["kind"] == "cluster" for g in groups)),
        )
    ]

    clusters = Counter(r["cluster_id"] for r in rows)
    sizes = list(clusters.values())
    split_counts = Counter(r["split"] for r in rows)
    split_rows = [
        [name, split_counts[name], f"{split_counts[name] / len(rows):.1%}"]
        for name in FOLDS_PER_SPLIT
    ]
    overlap_rows = [
        [f"{a} / {b}", shared["md5"], shared["pixel_md5"], shared["cluster_id"]]
        for (a, b), shared in split_overlaps(rows).items()
    ]
    n_small = sum(long_side <= LONG_SIDE for long_side, _, _ in exports)
    dark_labels = Counter(
        row["label"] for row, (_, _, is_dark) in zip(rows, exports, strict=True) if is_dark
    )

    lines = [
        "# Wheat dataset v1",
        "",
        "Generated by `src/zarai/data/clean_wheat.py`. The original train/valid/test folders are "
        "ignored.",
        "",
        f"Seed {seed}. Images are compared by decoded RGB pixels after flattening transparency. "
        f"Near-duplicates are pHash at Hamming distance <= {MAX_HAMMING_DISTANCE} or DINOv2 "
        f"(`{EMBEDDING_MODEL}`) CLS cosine >= {MIN_COSINE}, all computed on the source images "
        f"before resizing. JPEG quality {JPEG_QUALITY}. Images are downscaled to {LONG_SIDE} px "
        "on the long side only if larger.",
        "",
        "## Summary",
        "",
        *_table(
            ["Item", "Count"],
            [
                ["Raw images", len(records)],
                ["Unique file md5s", len({r["md5"] for r in records})],
                ["Unique decoded images (pixel hash)", len({r["pixel_md5"] for r in readable})],
                ["Duplicate copies removed from consistently labelled images", len(duplicates)],
                ["Raw images excluded as label conflicts", len(conflict_copies)],
                ["Unreadable images (excluded)", len(unreadable)],
                ["Images in wheat_v1", len(rows)],
                [f"Downscaled to {LONG_SIDE} px", len(exports) - n_small],
                [f"Kept at original size (long side <= {LONG_SIDE} px)", n_small],
                [
                    "Images with transparency, flattened on white",
                    sum(has_alpha for _, has_alpha, _ in exports),
                ],
            ],
        ),
        "",
        "## Label conflicts",
        "",
        "All are excluded from wheat_v1 and listed per raw image in "
        "`data/interim/label_conflicts/label_conflicts.csv`, with one file per distinct image "
        "copied there. The raw data is unchanged.",
        "",
        *_table(
            ["reason", "conflicts", "raw images", "distinct images"],
            conflict_rows,
        ),
        "",
        "Conflicts are counted as file md5s for `md5_conflict`, pixel hashes for "
        "`pixel_conflict`, and pHash clusters for `cluster_conflict`.",
        "",
        "## Near-duplicates",
        "",
        "Near-duplicates are not removed: they stay in the dataset, together in one split.",
        "",
        *_table(
            ["Item", "Count"],
            [
                ["pHash pairs", phash_stats["pairs"]],
                ["Clusters from pHash alone, including singletons", phash_stats["clusters"]],
                [f"Embedding pairs at cosine >= {MIN_COSINE}", embedding_stats["pairs"]],
                ["- already in the same pHash cluster", embedding_stats["already_linked"]],
                ["- new links", embedding_stats["new_links"]],
                ["New links crossing the old splits", embedding_stats["cross_old_splits"]],
                ["New links between different labels", embedding_stats["new_mixed"]],
                ["Cluster merges from the new links", embedding_stats["merges"]],
                ["Final clusters, including singletons", len(clusters)],
                ["Final clusters with 2 or more images", sum(size > 1 for size in sizes)],
                ["Images in those clusters", sum(size for size in sizes if size > 1)],
                ["Largest cluster", max(sizes)],
            ],
        ),
        "",
        "A new link joins two images not already in the same pHash cluster. It crosses the old "
        "splits when the two images, counting all their raw copies, appear in different original "
        "splits.",
        "",
        f"`review/threshold_pairs.jpg` shows {embedding_stats['review_pairs']} new-link candidates "
        f"within {REVIEW_MARGIN} of the threshold, half on each side, for a visual check. "
        "`review/threshold_pairs.csv` lists the same pairs.",
        "",
        "## Per-class counts",
        "",
        "`dups` are duplicate copies removed from consistently labelled images. `conflict` counts "
        "every raw copy excluded as a label conflict.",
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
        *_table(["splits", "shared md5s", "shared pixel hashes", "shared clusters"], overlap_rows),
        "",
        "No file md5 or pixel hash appears twice, every pHash and embedding pair is in one split, "
        "and no excluded conflict image is in wheat_v1.",
        "",
        *_known_issues(rows, groups, dark_labels, embedding_stats),
    ]
    return "\n".join(lines)


def clean_wheat(
    raw_dir: Path,
    out_dir: Path,
    conflicts_dir: Path,
    splits_csv: Path,
    seed: int = 42,
    workers: int | None = None,
    embedder: Embedder = dinov2_embeddings,
) -> list[dict]:
    """Run the whole pipeline and return the rows of the splits CSV."""
    _check_outputs(raw_dir, out_dir, conflicts_dir, splits_csv)
    records = collect_images(raw_dir)
    log.info("found %d images", len(records))

    with Pool(workers) as pool:
        hashes = pool.map(_hash_image, [r["path"] for r in records], chunksize=32)
        for record, (md5, pixel_md5, phash) in zip(records, hashes, strict=True):
            record.update(md5=md5, pixel_md5=pixel_md5, phash=phash)
        unreadable = [r for r in records if r["pixel_md5"] is None]
        images, exact_groups = group_by_pixels([r for r in records if r["pixel_md5"]])
        groups = [{**g, "kind": "exact"} for g in exact_groups]
        log.info("%d consistently labelled images; %d exact conflicts", len(images), len(groups))

        # Mixed-label pHash clusters are excluded before the final clusters are built.
        pairs = near_duplicate_pairs(_hash_array(images), MAX_HAMMING_DISTANCE)
        clusters = cluster_ids(len(images), pairs)
        mixed = mixed_label_clusters([r["label"] for r in images], clusters)
        members = defaultdict(list)
        for image, cluster in zip(images, clusters, strict=True):
            members[cluster].append(image)
        groups += [
            {
                "kind": "cluster",
                "group": f"phash_cluster_{cluster}",
                "labels": sorted({r["label"] for r in members[cluster]}),
                "copies": [
                    {**copy, "reason": CLUSTER_CONFLICT}
                    for image in members[cluster]
                    for copy in image["copies"]
                ],
            }
            for cluster in sorted(mixed)
        ]
        kept = [i for i, c in zip(images, clusters, strict=True) if c not in mixed]
        log.info("excluded %d mixed-label pHash clusters; %d images left", len(mixed), len(kept))

        phash_pairs = near_duplicate_pairs(_hash_array(kept), MAX_HAMMING_DISTANCE)
        phash_clusters = cluster_ids(len(kept), phash_pairs)

        embeddings = embedder([r["path"] for r in kept], pool)
        candidates = embedding_pairs(embeddings, MIN_COSINE - REVIEW_MARGIN)
        linked = [p for p in candidates if p[2] >= MIN_COSINE]
        new_links = [p for p in linked if phash_clusters[p[0]] != phash_clusters[p[1]]]
        log.info("%d embedding pairs, %d new links", len(linked), len(new_links))

        all_pairs = phash_pairs + [(i, j) for i, j, _ in linked]
        final_clusters = cluster_ids(len(kept), all_pairs)
        labels = [r["label"] for r in kept]
        splits = stratified_group_split(labels, final_clusters, seed)
        assert_pairs_in_one_split(all_pairs, splits)

        old_splits = [{c["original_split"] for c in r["copies"]} for r in kept]
        mixed_final = mixed_label_clusters(labels, final_clusters)
        embedding_stats = {
            "pairs": len(linked),
            "already_linked": len(linked) - len(new_links),
            "new_links": len(new_links),
            "cross_old_splits": sum(
                len(old_splits[i] | old_splits[j]) > 1 for i, j, _ in new_links
            ),
            "new_mixed": sum(labels[i] != labels[j] for i, j, _ in new_links),
            "merges": len(set(phash_clusters)) - len(set(final_clusters)),
            "mixed_clusters": len(mixed_final),
            "mixed_images": sum(c in mixed_final for c in final_clusters),
        }
        phash_stats = {"pairs": len(phash_pairs), "clusters": len(set(phash_clusters))}

        rows = [
            {
                "path": (out_dir / split / r["label"] / f"{r['pixel_md5']}.jpg").as_posix(),
                "label": r["label"],
                "split": split,
                "cluster_id": cluster,
                "md5": r["md5"],
                "pixel_md5": r["pixel_md5"],
                "original_split": r["original_split"],
                "original_path": r["path"],
            }
            for r, split, cluster in zip(kept, splits, final_clusters, strict=True)
        ]
        assert_no_leakage(rows)
        excluded = {copy["pixel_md5"] for group in groups for copy in group["copies"]}
        assert not excluded & {r["pixel_md5"] for r in rows}, "a conflict image is in wheat_v1"

        # Everything is checked before any earlier output is deleted.
        _reset_dir(out_dir)
        _reset_dir(conflicts_dir)
        _write_conflicts(groups, conflicts_dir)
        exports = pool.map(
            _export_jpeg, [(r["original_path"], r["path"]) for r in rows], chunksize=16
        )

    review_pairs = select_review_pairs(
        [p for p in candidates if phash_clusters[p[0]] != phash_clusters[p[1]]],
        MIN_COSINE,
        REVIEW_MARGIN,
        N_REVIEW_PAIRS,
    )
    embedding_stats["review_pairs"] = len(review_pairs)
    _write_review(review_pairs, rows, out_dir / "review")

    _write_csv(splits_csv, CSV_COLUMNS, rows)
    written = read_csv(splits_csv)
    assert_no_leakage(written)
    missing = [r["path"] for r in written if not Path(r["path"]).is_file()]
    assert not missing, f"{len(missing)} listed images are missing: {missing[:5]}"

    report = render_report(
        records, groups, unreadable, written, exports, phash_stats, embedding_stats, seed
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    log.info("wrote %d images and %s", len(written), out_dir / "REPORT.md")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/images_public/wheat_plant_diseases")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wheat_v1"))
    parser.add_argument("--conflicts-dir", type=Path, default=Path("data/interim/label_conflicts"))
    parser.add_argument("--splits-csv", type=Path, default=Path("data/splits/wheat_v1.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="default: all CPUs")
    parser.add_argument("--device", default="cuda", help="device for the DINOv2 embeddings")
    args = parser.parse_args()
    get_logger(__name__)
    clean_wheat(
        args.raw_dir,
        args.out_dir,
        args.conflicts_dir,
        args.splits_csv,
        args.seed,
        args.workers,
        partial(dinov2_embeddings, device=args.device),
    )


if __name__ == "__main__":
    main()
