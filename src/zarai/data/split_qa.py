"""Split Q&A records 700/100/200. The test set only holds farmer or expert questions."""

import argparse
import random
from pathlib import Path

from zarai.common.io import read_jsonl, write_jsonl
from zarai.common.logging import get_logger
from zarai.common.run_meta import write_run_meta

TEST_SOURCE_TYPES = {"farmer", "expert"}


def split_qa(
    records: list[dict], n_train: int = 700, n_val: int = 100, n_test: int = 200, seed: int = 42
) -> dict[str, list[dict]]:
    """Return train/val/test plus `unused`, which holds any records beyond the requested sizes."""
    needed = n_train + n_val + n_test
    if len(records) < needed:
        raise ValueError(f"need {needed} records, got {len(records)}")

    rng = random.Random(seed)
    # Sort first so the split depends only on the seed, not on file order.
    shuffled = sorted(records, key=lambda r: r["id"])
    rng.shuffle(shuffled)

    eligible = [r for r in shuffled if r["source_type"] in TEST_SOURCE_TYPES]
    if len(eligible) < n_test:
        raise ValueError(f"need {n_test} farmer/expert records for test, got {len(eligible)}")

    test = eligible[:n_test]
    test_ids = {r["id"] for r in test}
    rest = [r for r in shuffled if r["id"] not in test_ids]
    return {
        "train": rest[:n_train],
        "val": rest[n_train : n_train + n_val],
        "test": test,
        "unused": rest[n_train + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Q&A JSONL file")
    parser.add_argument("--out-dir", default="data/splits/qa")
    parser.add_argument("--n-train", type=int, default=700)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger(__name__)
    splits = split_qa(read_jsonl(args.input), args.n_train, args.n_val, args.n_test, args.seed)
    out_dir = Path(args.out_dir)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        log.info("%s: %d records", name, len(rows))
    write_run_meta(out_dir, vars(args), args.seed)


if __name__ == "__main__":
    main()
