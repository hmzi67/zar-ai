"""Map-style image dataset over a split JSONL; usable directly with a torch DataLoader."""

import json
from collections.abc import Callable
from pathlib import Path

from zarai.common.io import read_jsonl


def load_classes(classes_file: str | Path) -> list[str]:
    return json.loads(Path(classes_file).read_text(encoding="utf-8"))


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
