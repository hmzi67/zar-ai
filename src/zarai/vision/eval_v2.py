"""Per-run evaluation for wheat_v2, called by zarai.vision.train.fit inside the training run.

Temperature and the abstention threshold are fitted on val only. The held-out split (the fold's
held-out source, or the pooled test split) and the NWRD tiles are then scored with them:
macro-F1 and per-class recall with bootstrap 95% CIs, accuracy, ECE, the coverage-accuracy curve,
and the share of confidently wrong predictions (wrong at or above the abstention threshold).

Macro-F1 averages the classes present in that split, but a prediction of any of the five classes
counts, so predicting a class the split doesn't contain is an error. NWRD has one class
(yellow_rust), so its macro-F1 is that class's F1 and depends only on recall.
"""

import numpy as np
from scipy.special import softmax

from zarai.common.io import write_json
from zarai.eval.vision_metrics import (
    bootstrap_class_scores,
    confidently_wrong_share,
    coverage_accuracy_curve,
    expected_calibration_error,
)
from zarai.vision.calibrate import apply_temperature, fit_temperature, threshold_for_coverage
from zarai.vision.train import predict

EVALUATIONS = {"heldout": "test", "nwrd": "nwrd"}


def score_split(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    temperature: float,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict]]:
    """Metrics for one evaluation split, and its coverage-accuracy curve."""
    probs = apply_temperature(logits, temperature)
    preds = probs.argmax(axis=1)
    present = sorted(int(c) for c in np.unique(labels))
    scores = bootstrap_class_scores(labels, preds, present, len(classes), n_bootstrap, seed)
    metrics = {"n": float(len(labels))}
    for key, value in scores.items():
        name = f"recall_{classes[int(key.split('_')[1])]}" if key.startswith("recall_") else key
        metrics[name] = value["value"]
        metrics[f"{name}_low"] = value["low"]
        metrics[f"{name}_high"] = value["high"]
    for c in present:
        metrics[f"n_{classes[c]}"] = float((labels == c).sum())
    answered = probs.max(axis=1) >= threshold
    metrics.update(
        ece=expected_calibration_error(probs, labels),
        ece_before=expected_calibration_error(softmax(logits, axis=1), labels),
        coverage=float(answered.mean()),
        confident_wrong=confidently_wrong_share(probs, labels, threshold),
    )
    if answered.any():
        metrics["accuracy_answered"] = float((preds[answered] == labels[answered]).mean())
    curve = coverage_accuracy_curve(probs, labels, np.linspace(0, 1, 101))
    return metrics, curve


def evaluate_run(model, cfg, loaders, classes, run_dir, log) -> dict[str, float]:
    """Calibrate on val, score held-out and NWRD, and log everything to the active MLflow run."""
    import mlflow
    from sklearn.metrics import confusion_matrix

    from zarai.vision.plots import plot_coverage_accuracy

    device, limit = cfg.train.device, cfg.train.get("limit_eval_batches")
    val_logits, val_labels = predict(model, loaders["val"], device, limit=limit)
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = apply_temperature(val_logits, temperature)
    threshold = threshold_for_coverage(val_probs, cfg.calibration.target_coverage)
    metrics = {
        "temperature": temperature,
        "abstain_threshold": threshold,
        "val_ece": expected_calibration_error(val_probs, val_labels),
    }
    np.savez(run_dir / "val_logits.npz", logits=val_logits, labels=val_labels)
    mlflow.log_artifact(str(run_dir / "val_logits.npz"), "predictions")
    for name, split in EVALUATIONS.items():
        logits, labels = predict(model, loaders[split], device, limit=limit)
        scores, curve = score_split(
            logits, labels, classes, temperature, threshold, cfg.eval.n_bootstrap, cfg.seed
        )
        metrics.update({f"{name}_{key}": value for key, value in scores.items()})
        np.savez(run_dir / f"{name}_logits.npz", logits=logits, labels=labels)
        matrix = confusion_matrix(labels, logits.argmax(axis=1), labels=list(range(len(classes))))
        write_json(
            run_dir / f"{name}_confusion.json", {"classes": classes, "matrix": matrix.tolist()}
        )
        write_json(run_dir / f"{name}_coverage_accuracy.json", curve)
        selected = {
            "threshold": threshold,
            "coverage": scores["coverage"],
            "accuracy": scores.get("accuracy_answered"),
        }
        plot = plot_coverage_accuracy(
            curve,
            selected if selected["accuracy"] is not None else None,
            run_dir / f"{name}_coverage_accuracy.png",
        )
        for path in (
            run_dir / f"{name}_logits.npz",
            run_dir / f"{name}_confusion.json",
            run_dir / f"{name}_coverage_accuracy.json",
            plot,
        ):
            mlflow.log_artifact(str(path), f"eval/{name}")
        log.info(
            "%s: macro-F1 %.4f (%.4f-%.4f), ECE %.4f, confidently wrong %.3f",
            name,
            scores["macro_f1"],
            scores["macro_f1_low"],
            scores["macro_f1_high"],
            scores["ece"],
            scores["confident_wrong"],
        )
    mlflow.log_metrics(metrics)
    return metrics
