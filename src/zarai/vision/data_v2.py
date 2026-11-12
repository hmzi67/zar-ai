"""wheat_v2 DataLoaders for one protocol and fold, and the checks that dataloader.py runs.

Splits come from one column of data/splits/wheat_v2.csv: `split_pooled`, or `split_loso_<source>`
for the fold that holds that source out. Every protocol has train, val, test and nwrd splits.
NWRD is only ever an eval split.
"""

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from zarai.common.config import load_config
from zarai.common.io import read_csv
from zarai.data.wheat_v2 import (
    LOSO_PREFIX,
    POOLED_COLUMN,
    SPLITS,
    assert_no_leakage,
    loso_column,
)
from zarai.vision.datasets import WheatV2Dataset
from zarai.vision.train import balanced_loader, sequential_loader
from zarai.vision.transforms import build_train_transforms, build_transforms

SAMPLES_DIR = Path("data/processed/wheat_v2/samples")


def available_folds(splits_csv) -> list[str]:
    """Held-out sources with a LOSO column, in CSV order."""
    with open(splits_csv, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return [c[len(LOSO_PREFIX) :] for c in header if c.startswith(LOSO_PREFIX)]


def resolve_fold(splits_csv, protocol: str, fold: str | None) -> tuple[str, str]:
    """Return (CSV column, fold name). A fold may be any unique part of a source name."""
    if protocol == "pooled":
        return POOLED_COLUMN, "pooled"
    if protocol != "loso":
        raise ValueError(f"protocol must be pooled or loso, got {protocol!r}")
    folds = available_folds(splits_csv)
    if fold in folds:
        return loso_column(fold), fold
    matches = [f for f in folds if fold and fold in f]
    if len(matches) != 1:
        raise ValueError(f"fold {fold!r} matches {matches or 'nothing'}; folds are {folds}")
    return loso_column(matches[0]), matches[0]


def build_datasets(cfg, mean, std) -> dict[str, WheatV2Dataset]:
    """One dataset per split. Only train is augmented; the others use the eval transform."""
    column, _ = resolve_fold(cfg.data.splits_csv, cfg.data.protocol, cfg.data.get("fold"))
    augment = cfg.train.augment
    profile = augment.profile
    train_transform = build_train_transforms(profile, cfg.data.image_size, mean, std, augment)
    eval_transform = build_transforms(
        cfg.data.image_size, train=False, mean=mean, std=std, eval_resize=cfg.data.eval_resize
    )
    replace_p = augment.strong.replace_background_p if profile == "strong" else 0.0
    return {
        split: WheatV2Dataset(
            cfg.data.splits_csv,
            column,
            split,
            train_transform if split == "train" else eval_transform,
            list(cfg.data.classes),
            background=cfg.data.background,
            replace_background_p=replace_p if split == "train" else 0.0,
            max_samples=cfg.data.get("max_samples"),
            seed=cfg.seed,
        )
        for split in SPLITS
    }


def build_loaders(cfg, mean, std) -> dict:
    datasets = build_datasets(cfg, mean, std)
    return {
        split: (
            balanced_loader(dataset, cfg) if split == "train" else sequential_loader(dataset, cfg)
        )
        for split, dataset in datasets.items()
    }


def assert_loaders_leak_nothing(datasets: dict, column: str, test_only) -> None:
    """Rerun the build's leakage assertion on the rows the datasets actually hold."""

    def other_column(key: str) -> bool:
        return key == POOLED_COLUMN or key.startswith(LOSO_PREFIX)

    rows = [
        {**{k: v for k, v in row.items() if not other_column(k)}, column: split}
        for split, dataset in datasets.items()
        for row in dataset.rows
    ]
    assert_no_leakage(rows, test_only)
    for split in ("train", "val"):
        found = {s for s in datasets[split].sources if s in test_only}
        if found:
            raise AssertionError(f"{split} holds test-only sources {found}")


def count_table(datasets: dict, key: str, values: list[str]) -> str:
    counts = {split: Counter(row[key] for row in d.rows) for split, d in datasets.items()}
    width = max(len(v) for v in [*values, key]) + 2
    lines = [f"{key:<{width}}" + "".join(f"{s:>8}" for s in datasets)]
    for value in values:
        lines.append(f"{value:<{width}}" + "".join(f"{counts[s][value]:>8}" for s in datasets))
    lines.append(f"{'total':<{width}}" + "".join(f"{len(d):>8}" for d in datasets.values()))
    return "\n".join(lines)


def save_sample_grid(dataset, path: Path, mean, std, n: int = 32, seed: int = 0) -> Path:
    """A grid of n samples as the loader yields them, captioned with label and source."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    tile, caption, per_row = 160, 26, 8
    rows = (len(picks) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * tile, rows * (tile + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, index in enumerate(picks):
        image, _ = dataset[int(index)]
        array = image.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        x, y = k % per_row * tile, k // per_row * (tile + caption)
        sheet.paste(Image.fromarray(array).resize((tile, tile)), (x, y + caption))
        row = dataset.rows[int(index)]
        draw.text((x + 2, y + 1), f"{row['label']}\n{row['source'][:24]}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)
    return path


def time_loader(loader, n_batches: int = 200) -> tuple[float, int]:
    """Images per second over n_batches, after the first batch (worker start-up) is discarded."""
    images, started, batches = 0, None, 0
    while batches < n_batches + 1:
        for batch, _ in loader:
            if started is None:
                started = time.monotonic()
            else:
                images += len(batch)
            batches += 1
            if batches >= n_batches + 1:
                break
    return images / (time.monotonic() - started), batches - 1


def check_fold(cfg, protocol: str, fold: str | None, n_batches: int, log=print) -> dict:
    """Build the loaders for one fold, print counts, save sample grids, time and check them."""
    from zarai.vision import backbones

    cfg.data.protocol, cfg.data.fold = protocol, fold
    column, name = resolve_fold(cfg.data.splits_csv, protocol, fold)
    mean, std = backbones.normalisation(backbones.backbone_spec(cfg), cfg.data.image_size)
    datasets = build_datasets(cfg, mean, std)
    test_only = set(cfg.data.test_only_sources)
    assert_loaders_leak_nothing(datasets, column, test_only)
    sources = sorted({s for d in datasets.values() for s in d.sources})
    log(
        f"\n=== {protocol} / {name}  (column {column}, aug {cfg.train.augment.profile}, "
        f"background {cfg.data.background}) ==="
    )
    log(count_table(datasets, "label", list(cfg.data.classes)))
    log("")
    log(count_table(datasets, "source", sources))
    missing = set(cfg.data.classes) - {r["label"] for r in datasets["train"].rows}
    if missing:
        log(f"WARNING: classes with no training images: {sorted(missing)}")
    stem = f"{protocol}_{name}_{cfg.train.augment.profile}_{cfg.data.background}"
    grids = {
        split: save_sample_grid(d, SAMPLES_DIR / f"{stem}_{split}.jpg", mean, std, seed=cfg.seed)
        for split, d in datasets.items()
    }
    for split, path in grids.items():
        kind = "augmented" if split == "train" else "eval transform"
        log(f"samples ({kind}) {split}: {path}")
    loader = balanced_loader(datasets["train"], cfg)
    rate, timed = time_loader(loader, n_batches)
    log(
        f"train loader: {rate:.0f} images/s over {timed} batches of {cfg.train.batch_size} "
        f"({cfg.data.num_workers} workers)"
    )
    log("leakage checks on the loaders' rows: passed")
    return {"fold": name, "images_per_second": rate}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build wheat_v2 and check its DataLoaders.")
    parser.add_argument("--config", default="configs/vision/wheat_v2.yaml")
    parser.add_argument("--protocol", choices=["pooled", "loso"], default="loso")
    parser.add_argument("--fold", default=None, help="held-out source; default: every LOSO fold")
    parser.add_argument("--aug", choices=["basic", "strong"], default="basic")
    parser.add_argument("--background", choices=["none", "remove"], default="none")
    parser.add_argument("--skip-preprocess", action="store_true", help="don't run dvc repro")
    parser.add_argument("--batches", type=int, default=200, help="batches to time")
    args = parser.parse_args(argv)

    if not args.skip_preprocess:
        dvc = Path(sys.executable).parent / "dvc"
        # DVC reruns the stage only if its inputs changed since dvc.lock.
        subprocess.run([str(dvc), "repro", "wheat_v2"], check=True)
    cfg = load_config(args.config)
    cfg.train.augment.profile = args.aug
    cfg.data.background = args.background
    cfg.train.batch_size = cfg.train.effective_batch_size  # train_v2 sets it per backbone
    if not Path(cfg.data.splits_csv).is_file():
        raise SystemExit(f"{cfg.data.splits_csv} is missing: run without --skip-preprocess first")

    rows = read_csv(cfg.data.splits_csv)
    reports = assert_no_leakage(rows, set(cfg.data.test_only_sources))
    print(f"leakage checks on {cfg.data.splits_csv}: passed for {', '.join(reports)}")
    print(f"report: {Path(cfg.data.splits_csv).parents[1] / 'processed/wheat_v2/REPORT.md'}")
    if args.protocol == "pooled":
        folds = [None]
    else:
        folds = [args.fold] if args.fold else available_folds(cfg.data.splits_csv)
    results = [check_fold(cfg, args.protocol, fold, args.batches) for fold in folds]
    print("\nAll checks passed.")
    for result in results:
        print(f"  {result['fold']}: {result['images_per_second']:.0f} images/s")
