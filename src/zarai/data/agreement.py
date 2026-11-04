"""Cohen's kappa (plain or weighted) for double-annotated subsets."""

import argparse
from collections.abc import Sequence

from sklearn.metrics import cohen_kappa_score

from zarai.common.io import read_csv


def cohen_kappa(
    rater_a: Sequence,
    rater_b: Sequence,
    weights: str | None = None,
    labels: Sequence | None = None,
) -> float:
    """Kappa between two raters. `weights` is None, "linear" or "quadratic".

    Weighted kappa depends on label order, so pass ordinal string labels in order via `labels`.
    Undefined (nan) if both raters use one identical label throughout.
    """
    if len(rater_a) == 0:
        raise ValueError("no ratings given")
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater lengths differ: {len(rater_a)} vs {len(rater_b)}")
    return float(cohen_kappa_score(rater_a, rater_b, weights=weights, labels=labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with one column per rater")
    parser.add_argument("--rater-a", required=True, help="column name")
    parser.add_argument("--rater-b", required=True, help="column name")
    parser.add_argument("--weights", choices=["linear", "quadratic"], default=None)
    parser.add_argument("--labels", nargs="*", default=None, help="label order for weighted kappa")
    args = parser.parse_args()

    rows = read_csv(args.input)
    a, b = [r[args.rater_a] for r in rows], [r[args.rater_b] for r in rows]
    print(f"n={len(rows)} kappa={cohen_kappa(a, b, args.weights, args.labels):.4f}")


if __name__ == "__main__":
    main()
