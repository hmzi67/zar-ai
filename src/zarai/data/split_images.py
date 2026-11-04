"""Group-split image metadata by site_id, holding out unseen field sites as field_test.

Sites whose id starts with `public_` (public datasets) never enter field_test. A public dataset
named as one site moves whole into train or val, so give it finer ids (see data/README.md).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from zarai.common.io import read_jsonl, write_json, write_jsonl
from zarai.common.logging import get_logger
from zarai.common.run_meta import write_run_meta

PUBLIC_PREFIX = "public_"
log = logging.getLogger(__name__)


def _hold_out(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (kept, held_out); no site_id appears in both."""
    groups = [r["site_id"] for r in records]
    if len(set(groups)) < 2:
        raise ValueError(f"need at least 2 sites to split, got {sorted(set(groups))}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    keep, hold = next(splitter.split(np.zeros(len(records)), groups=groups))
    return [records[i] for i in keep], [records[i] for i in hold]


def assert_no_site_overlap(splits: dict[str, list[dict]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = {r["site_id"] for r in splits[a]} & {r["site_id"] for r in splits[b]}
            assert not shared, f"site leakage between {a} and {b}: {sorted(shared)}"


def split_images(
    records: list[dict], field_test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> dict[str, list[dict]]:
    """Return train/val/field_test. Sizes are fractions of sites. `unsure` images are excluded."""
    usable = [r for r in records if not r.get("unsure")]
    if len(usable) < len(records):
        log.info("excluded %d unsure images", len(records) - len(usable))

    field = [r for r in usable if not r["site_id"].startswith(PUBLIC_PREFIX)]
    public = [r for r in usable if r["site_id"].startswith(PUBLIC_PREFIX)]
    if field:
        field_kept, field_test = _hold_out(field, field_test_size, seed)
    else:
        log.warning("no field sites yet: field_test is empty")
        field_kept, field_test = [], []

    train, val = _hold_out(public + field_kept, val_size, seed)
    splits = {"train": train, "val": val, "field_test": field_test}
    assert_no_site_overlap(splits)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="image metadata JSONL file")
    parser.add_argument("--out-dir", default="data/splits/images")
    parser.add_argument("--field-test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    get_logger(__name__)

    records = read_jsonl(args.input)
    splits = split_images(records, args.field_test_size, args.val_size, args.seed)
    out_dir = Path(args.out_dir)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        log.info("%s: %d images, %d sites", name, len(rows), len({r["site_id"] for r in rows}))
    write_json(out_dir / "classes.json", sorted({r["label"] for r in splits["train"]}))
    write_run_meta(out_dir, vars(args), args.seed)


if __name__ == "__main__":
    main()
