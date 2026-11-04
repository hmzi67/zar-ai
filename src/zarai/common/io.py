"""Small file helpers. All text is UTF-8 (Urdu must round-trip unescaped)."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:  # -sig: Excel adds a BOM
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
