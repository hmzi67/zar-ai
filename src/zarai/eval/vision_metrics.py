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


def bootstrap_cis(
    labels: np.ndarray, preds: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Macro-F1 and accuracy with percentile 95% CIs, resampling images with replacement."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    f1s, accuracies = np.empty(n_resamples), np.empty(n_resamples)
    n_classes = int(max(labels.max(), preds.max())) + 1
    all_labels = list(range(n_classes))
    for i in range(n_resamples):
        sample = rng.integers(0, n, n)
        sampled_labels, sampled_preds = labels[sample], preds[sample]
        present = np.unique(sampled_labels)
        # A class absent from a resample has no support, so it isn't scored as an F1 of zero.
        f1s[i] = f1_score(
            sampled_labels,
            sampled_preds,
            labels=[c for c in all_labels if c in present],
            average="macro",
            zero_division=0,
        )
        accuracies[i] = (sampled_labels == sampled_preds).mean()
    return {
        "macro_f1": {
            "value": macro_f1(labels, preds),
            "low": float(np.percentile(f1s, 2.5)),
            "high": float(np.percentile(f1s, 97.5)),
        },
        "accuracy": {
            "value": float((labels == preds).mean()),
            "low": float(np.percentile(accuracies, 2.5)),
            "high": float(np.percentile(accuracies, 97.5)),
        },
    }


def confusion_between(matrix: np.ndarray, classes: list[str], a: str, b: str) -> dict:
    """How often classes a and b are mistaken for each other. Rows of the matrix are true labels."""
    i, j = classes.index(a), classes.index(b)
    return {
        "a": a,
        "b": b,
        "a_support": int(matrix[i].sum()),
        "b_support": int(matrix[j].sum()),
        "a_as_b": int(matrix[i, j]),
        "b_as_a": int(matrix[j, i]),
    }


def top_confusions(matrix: np.ndarray, classes: list[str], n: int = 10) -> list[dict]:
    """The largest off-diagonal cells, as true label, predicted label and count."""
    errors = [
        {"true": classes[i], "predicted": classes[j], "count": int(matrix[i, j])}
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and matrix[i, j] > 0
    ]
    return sorted(errors, key=lambda e: (-e["count"], e["true"], e["predicted"]))[:n]


def classification_summary(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)
    return {
        "macro_f1": macro_f1(labels, preds),
        "per_class": per_class_report(labels, preds, class_names),
        "ece": expected_calibration_error(probs, labels),
        "coverage_accuracy": coverage_accuracy_curve(probs, labels),
    }


def _class_scores(cm: np.ndarray, classes: list[int]) -> tuple[float, float, np.ndarray]:
    """(macro-F1 over classes with support, accuracy, recall per class) from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    present = [c for c in classes if support[c] > 0]
    f1 = 2 * tp[present] / (support[present] + predicted[present])
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support[classes] > 0, tp[classes] / support[classes], np.nan)
    return float(f1.mean()) if present else float("nan"), float(tp.sum() / cm.sum()), recall


def bootstrap_class_scores(
    labels: np.ndarray,
    preds: np.ndarray,
    classes: list[int],
    n_classes: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Macro-F1 over `classes`, accuracy and per-class recall, each with a percentile 95% CI.

    Predictions of any of n_classes count, so predicting a class outside `classes` is an error.
    A class absent from a resample is left out of that resample's macro-F1, as in bootstrap_cis.
    Keys: macro_f1, accuracy, recall_<class index>.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)

    def confusion(index: np.ndarray) -> np.ndarray:
        cells = labels[index] * n_classes + preds[index]
        return np.bincount(cells, minlength=n_classes * n_classes).reshape(n_classes, n_classes)

    every = np.arange(len(labels))
    f1, accuracy, recall = _class_scores(confusion(every), classes)
    rng = np.random.default_rng(seed)
    samples = [
        _class_scores(confusion(rng.integers(0, len(labels), len(labels))), classes)
        for _ in range(n_resamples)
    ]

    def interval(value: float, values: np.ndarray) -> dict[str, float]:
        return {
            "value": value,
            "low": float(np.nanpercentile(values, 2.5)),
            "high": float(np.nanpercentile(values, 97.5)),
        }

    scores = {
        "macro_f1": interval(f1, np.array([s[0] for s in samples])),
        "accuracy": interval(accuracy, np.array([s[1] for s in samples])),
    }
    recalls = np.array([s[2] for s in samples])
    for k, c in enumerate(classes):
        if not np.isnan(recall[k]):
            scores[f"recall_{c}"] = interval(float(recall[k]), recalls[:, k])
    return scores


def confidently_wrong_share(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Share of all samples that are wrong with confidence at or above the abstention threshold."""
    wrong = probs.argmax(axis=1) != labels
    return float((wrong & (probs.max(axis=1) >= threshold)).mean())
