"""Macro-F1, per-class report, expected calibration error and the coverage-accuracy curve."""

import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def per_class_report(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict:
    return classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Sample-weighted gap between confidence and accuracy over equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    # Right-closed bins: a confidence of exactly 1.0 lands in the last bin.
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def coverage_accuracy_curve(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict]:
    """Coverage and accuracy on answered samples per threshold; accuracy is None if none answered."""
    thresholds = np.linspace(0, 1, 21) if thresholds is None else thresholds
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    curve = []
    for threshold in thresholds:
        answered = confidence >= threshold
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(answered.mean()),
                "accuracy": float(correct[answered].mean()) if answered.any() else None,
            }
        )
    return curve


def classification_summary(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)
    return {
        "macro_f1": macro_f1(labels, preds),
        "per_class": per_class_report(labels, preds, class_names),
        "ece": expected_calibration_error(probs, labels),
        "coverage_accuracy": coverage_accuracy_curve(probs, labels),
    }
