"""Evaluate a classifier on the field test split. Pass logits_file=x.npz to score saved logits."""

import numpy as np

from zarai.common.config import parse_config
from zarai.common.io import write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.eval.vision_metrics import classification_summary
from zarai.vision.datasets import load_classes


def predict_logits(cfg) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO: run the checkpoint over data.test_file; return logits, labels")


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_eval_{cfg.experiment}")
    log = get_logger(__name__, run_dir)
    if cfg.get("logits_file"):
        saved = np.load(cfg.logits_file)
        logits, labels = saved["logits"], saved["labels"]
    else:
        logits, labels = predict_logits(cfg)
    summary = classification_summary(logits, labels, load_classes(cfg.data.classes_file))
    write_json(run_dir / "metrics.json", summary)
    log.info("wrote %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
