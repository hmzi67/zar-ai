"""System Usability Scale scoring from eval/pilot/pilot_log_template.csv columns sus_q1..sus_q10."""

import argparse

import numpy as np

from zarai.common.io import read_csv, write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run

SUS_COLUMNS = [f"sus_q{i}" for i in range(1, 11)]


def sus_score(responses: list[int]) -> float:
    """Standard SUS score (0-100) from ten 1-5 responses in questionnaire order."""
    if len(responses) != 10:
        raise ValueError(f"SUS needs 10 responses, got {len(responses)}")
    if any(not 1 <= r <= 5 for r in responses):
        raise ValueError(f"SUS responses must be 1-5, got {responses}")
    # Odd items are positively worded (r - 1), even items negatively (5 - r); enumerate starts at 0.
    return 2.5 * sum(r - 1 if i % 2 == 0 else 5 - r for i, r in enumerate(responses))


def score_rows(rows: list[dict]) -> list[float]:
    """Score rows with a completed questionnaire. Partly filled rows are an error, blank ones skipped."""
    scores = []
    for row in rows:
        values = [row[column].strip() for column in SUS_COLUMNS]
        if any(values):
            scores.append(sus_score([int(v) for v in values]))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="filled pilot log CSV")
    args = parser.parse_args()

    run_dir = start_run({**vars(args), "seed": None}, "sus")
    log = get_logger(__name__, run_dir)
    scores = score_rows(read_csv(args.input))
    if not scores:
        raise ValueError(f"no completed SUS questionnaires in {args.input}")
    write_json(
        run_dir / "sus.json",
        {
            "n": len(scores),
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "scores": scores,
        },
    )
    log.info("wrote %s", run_dir / "sus.json")


if __name__ == "__main__":
    main()
