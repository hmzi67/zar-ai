"""Aggregate blinded expert ratings (eval/rating_sheets/expert_rubric.csv).

Raters are averaged per question first, so bootstrap and Wilcoxon treat the question as the unit.
"""

import argparse

import numpy as np
from scipy.stats import wilcoxon

from zarai.common.io import read_csv, write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run

SCORE_METRICS = ["correctness_1_5", "actionability_1_5", "fluency_1_5"]
FLAG_METRICS = ["faithful_yn", "unsafe_yn"]


def _parse(value: str, metric: str) -> float:
    value = value.strip().lower()
    if metric in FLAG_METRICS:
        if value not in {"y", "n"}:
            raise ValueError(f"{metric}: expected y or n, got {value!r}")
        return float(value == "y")
    score = float(value)
    if not 1 <= score <= 5:
        raise ValueError(f"{metric}: expected 1-5, got {value!r}")
    return score


def question_means(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """{system_id: {qid: mean rating over raters}}. Blank cells count as not rated."""
    collected: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row[metric].strip():
            by_qid = collected.setdefault(row["system_id"], {})
            by_qid.setdefault(row["qid"], []).append(_parse(row[metric], metric))
    return {
        system: {qid: float(np.mean(values)) for qid, values in by_qid.items()}
        for system, by_qid in collected.items()
    }


def bootstrap_ci(
    values: list[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarise(rows: list[dict], seed: int = 42) -> dict:
    """{system_id: {metric: {mean, ci_low, ci_high, n_questions}}}."""
    summary: dict[str, dict] = {}
    for metric in SCORE_METRICS + FLAG_METRICS:
        for system, by_qid in question_means(rows, metric).items():
            values = list(by_qid.values())
            low, high = bootstrap_ci(values, seed=seed)
            summary.setdefault(system, {})[metric] = {
                "mean": float(np.mean(values)),
                "ci_low": low,
                "ci_high": high,
                "n_questions": len(values),
            }
    return summary


def compare_systems(rows: list[dict], system_a: str, system_b: str, metric: str) -> dict:
    """Paired Wilcoxon signed-rank test over the questions both systems were rated on."""
    means = question_means(rows, metric)
    shared = sorted(set(means[system_a]) & set(means[system_b]))
    if not shared:
        raise ValueError(f"{system_a} and {system_b} share no rated questions for {metric}")
    diffs = [means[system_a][q] - means[system_b][q] for q in shared]
    if not any(diffs):  # wilcoxon is undefined when every difference is zero
        return {"statistic": 0.0, "pvalue": 1.0, "n_questions": len(shared)}
    result = wilcoxon(diffs)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_questions": len(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="filled rating sheet CSV")
    parser.add_argument("--compare", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = start_run(vars(args), "expert_ratings")
    log = get_logger(__name__, run_dir)
    rows = read_csv(args.input)
    report = {"summary": summarise(rows, args.seed)}
    if args.compare:
        report["wilcoxon"] = {
            metric: compare_systems(rows, *args.compare, metric)
            for metric in SCORE_METRICS + FLAG_METRICS
        }
    write_json(run_dir / "expert_ratings.json", report)
    log.info("wrote %s", run_dir / "expert_ratings.json")


if __name__ == "__main__":
    main()
