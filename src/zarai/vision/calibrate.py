"""Temperature scaling on logits and abstention threshold search for a target coverage.

CLI input: logits_file=x.npz holding `logits` and `labels` from the validation split.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from zarai.common.config import parse_config
from zarai.common.io import write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.eval.vision_metrics import expected_calibration_error


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    log_probs = log_softmax(logits / temperature, axis=1)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature that minimises validation NLL."""
    result = minimize_scalar(
        lambda log_t: _nll(logits, labels, math.exp(log_t)),
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
    )
    return math.exp(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def threshold_for_coverage(probs: np.ndarray, target_coverage: float) -> float:
    """Highest confidence threshold that still answers at least target_coverage of samples."""
    if not 0 < target_coverage <= 1:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    confidence = np.sort(probs.max(axis=1))[::-1]
    k = math.ceil(target_coverage * len(confidence) - 1e-9)
    return float(confidence[k - 1])


def predict_with_abstention(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Predicted class per sample, or -1 where confidence is below threshold."""
    return np.where(probs.max(axis=1) >= threshold, probs.argmax(axis=1), -1)


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_calibrate_{cfg.experiment}")
    log = get_logger(__name__, run_dir)

    saved = np.load(cfg.logits_file)
    logits, labels = saved["logits"], saved["labels"]
    temperature = fit_temperature(logits, labels)
    probs = apply_temperature(logits, temperature)
    threshold = threshold_for_coverage(probs, cfg.calibration.target_coverage)

    preds = predict_with_abstention(probs, threshold)
    answered = preds != -1
    result = {
        "temperature": temperature,
        "threshold": threshold,
        "target_coverage": cfg.calibration.target_coverage,
        "coverage": float(answered.mean()),
        "accuracy_on_answered": float((preds[answered] == labels[answered]).mean()),
        "ece_before": expected_calibration_error(softmax(logits, axis=1), labels),
        "ece_after": expected_calibration_error(probs, labels),
    }
    write_json(run_dir / "calibration.json", result)
    log.info("wrote %s", run_dir / "calibration.json")


if __name__ == "__main__":
    main()
