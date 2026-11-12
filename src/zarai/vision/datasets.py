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
        split_column: str = "split",
    ):
        rows = read_csv(splits_csv)
        self.classes = classes or classes_from_rows(rows)
        exclude_paths = exclude_paths or set()
        self.rows = [
            row for row in rows if row[split_column] == split and row["path"] not in exclude_paths
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


class WheatV2Dataset(SplitCsvDataset):
    """One split of wheat_v2 for one protocol column, with optional background handling.

    background="remove" blacks out everything outside the leaf mask. replace_background_p is the
    chance of filling the background with a random colour, colour noise, or another training
    image instead (the strong profile). Fallback masks that cover the whole image are left alone.
    """

    def __init__(
        self,
        splits_csv,
        split_column: str,
        split: str,
        transform: Callable | None,
        classes: list[str],
        background: str = "none",
        replace_background_p: float = 0.0,
        max_samples: int | None = None,
        seed: int = 0,
    ):
        if background not in ("none", "remove"):
            raise ValueError(f"background must be none or remove, got {background!r}")
        super().__init__(
            splits_csv, split, transform, classes, max_samples, seed, split_column=split_column
        )
        self.background = background
        self.replace_background_p = replace_background_p

    @property
    def sources(self) -> list[str]:
        return [row["source"] for row in self.rows]

    def _random_background(self, size):
        import torch
        from PIL import Image

        kind = int(torch.randint(3, (1,)))
        if kind == 0:
            colour = tuple(int(c) for c in torch.randint(256, (3,)))
            return Image.new("RGB", size, colour)
        if kind == 1:
            coarse = torch.randint(256, (8, 8, 3), dtype=torch.uint8).numpy()
            return Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC)
        other = self.rows[int(torch.randint(len(self.rows), (1,)))]
        with Image.open(other["path"]) as image:
            return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)

    def load_image(self, index: int):
        """The RGB image after background handling, before the transform."""
        import torch
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        replace = self.replace_background_p > 0 and float(torch.rand(1)) < self.replace_background_p
        if self.background == "none" and not replace:
            return image
        with Image.open(row["mask_path"]) as mask:
            mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        if np.asarray(mask).min() > 127:  # a fallback mask: nothing to replace
            return image
        fill = self._random_background(image.size) if replace else Image.new("RGB", image.size)
        return Image.composite(image, fill, mask)

    def __getitem__(self, index: int):
        image = self.load_image(index)
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
