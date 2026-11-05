"""Image datasets for a torch DataLoader: a split CSV (wheat_v1) or a split JSONL (field images)."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from zarai.common.io import read_csv, read_jsonl


def load_classes(classes_file: str | Path) -> list[str]:
    return json.loads(Path(classes_file).read_text(encoding="utf-8"))


def classes_from_rows(rows: list[dict]) -> list[str]:
    """Class names in the fixed order used for every model output."""
    return sorted({row["label"] for row in rows})


def class_balanced_weights(labels: list[int] | np.ndarray, power: float = 0.5) -> np.ndarray:
    """Per-sample sampling weights: inverse class frequency, raised to `power` (0.5 is sqrt)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    return (1.0 / counts[labels]) ** power


class SplitCsvDataset:
    """One split of a CSV with `path`, `label` and `split` columns."""

    def __init__(
        self,
        splits_csv: str | Path,
        split: str,
        transform: Callable | None = None,
        classes: list[str] | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        exclude_paths: set[str] | None = None,
    ):
        rows = read_csv(splits_csv)
        self.classes = classes or classes_from_rows(rows)
        exclude_paths = exclude_paths or set()
        self.rows = [
            row for row in rows if row["split"] == split and row["path"] not in exclude_paths
        ]
        if not self.rows:
            raise ValueError(f"{splits_csv}: no rows in split {split!r}")
        if max_samples is not None and max_samples < len(self.rows):
            keep = np.random.default_rng(seed).permutation(len(self.rows))[:max_samples]
            self.rows = [self.rows[i] for i in sorted(keep.tolist())]
        class_to_idx = {name: i for i, name in enumerate(self.classes)}
        unknown = {row["label"] for row in self.rows} - set(class_to_idx)
        if unknown:
            raise ValueError(f"{splits_csv}: labels not in classes: {sorted(unknown)}")
        self.labels = [class_to_idx[row["label"]] for row in self.rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.rows[index]["path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class ImageMetaDataset:
    def __init__(self, split_file, classes_file, transform: Callable | None = None):
        self.records = read_jsonl(split_file)
        self.class_to_idx = {c: i for i, c in enumerate(load_classes(classes_file))}
        self.transform = transform
        unknown = {r["label"] for r in self.records} - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"{split_file}: labels not in {classes_file}: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record["file_path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[record["label"]]
