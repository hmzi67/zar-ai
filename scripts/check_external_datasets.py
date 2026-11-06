#!/usr/bin/env python
"""Report which external datasets are in place, and how to get the rest. Missing data isn't an error.

Usage:
  .venv/bin/python scripts/check_external_datasets.py
  .venv/bin/python scripts/check_external_datasets.py --download nwrd cgiar_wheat_rust

cgiar_wheat_rust uses a Kaggle API token from KAGGLE_API_TOKEN or ~/.kaggle/access_token.
"""

import argparse
from pathlib import Path

from zarai.vision.eval_external import EXTERNAL_DATASETS, ROOT, check_dataset, download_all


def describe(root: Path) -> int:
    missing = 0
    for name, spec in EXTERNAL_DATASETS.items():
        state = check_dataset(root, spec)
        print(f"\n{name}: {state['status'].upper()}")
        print(f"  folder:   {root / name}/")
        print(f"  expected: {spec.layout}")
        print(f"  licence:  {spec.licence}")
        print(f"  page:     {spec.page}")
        for url, filename, algorithm, value in spec.files:
            print(f"  download: {url}")
            print(f"            save as {filename}, {algorithm} {value}")
        if spec.manual:
            print(f"  download: {spec.manual}")
        if spec.notes:
            print(f"  note:     {spec.notes}")
        for label, count in sorted(state["counts"].items()):
            known = spec.expected_images.get(label)
            suffix = f" (the source lists {known})" if known else ""
            print(f"  found:    {label} {count}{suffix}")
        if "tiles" in state:
            print(f"  found:    {state['tiles']} rust tiles, skipped {state['skipped_tiles']}")
        for reason, count in sorted(state["skipped"].items()):
            print(f"  skipped:  {reason}: {count}")
        if state["status"] != "ready":
            missing += 1
    print(f"\n{len(EXTERNAL_DATASETS) - missing} of {len(EXTERNAL_DATASETS)} datasets ready.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--download", nargs="*", default=[], choices=list(EXTERNAL_DATASETS))
    args = parser.parse_args()
    for message in download_all(args.root, args.download):
        print(message)
    describe(args.root)


if __name__ == "__main__":
    main()
