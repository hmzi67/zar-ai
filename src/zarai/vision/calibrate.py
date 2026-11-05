"""Temperature scaling and abstention, fitted on val and measured on test. Pass run_id=<id>.

Reads the logits that zarai.vision.eval logged to the same MLflow run.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from zarai.common.config import parse_config
from zarai.common.io import write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.eval.vision_metrics import coverage_accuracy_curve, expected_calibration_error


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


def calibrate(cfg, run_dir, log) -> None:
    """Fit on val, measure on test, and log everything to the MLflow run."""
    import mlflow

    from zarai.vision.plots import plot_coverage_accuracy, plot_reliability
    from zarai.vision.tracking import artifact_path, configure, require_run_id

    run_id = require_run_id(cfg)
    configure(cfg)
    with mlflow.start_run(run_id=run_id):
        val = np.load(artifact_path(run_id, "predictions/val_logits.npz"))
        test = np.load(artifact_path(run_id, "predictions/test_logits.npz"))
        temperature = fit_temperature(val["logits"], val["labels"])

        before = softmax(test["logits"], axis=1)
        after = apply_temperature(test["logits"], temperature)
        labels = test["labels"]
        ece_before = expected_calibration_error(before, labels)
        ece_after = expected_calibration_error(after, labels)

        # The threshold comes from val, so that test only measures it.
        val_probs = apply_temperature(val["logits"], temperature)
        threshold = threshold_for_coverage(val_probs, cfg.calibration.target_coverage)
        preds = predict_with_abstention(after, threshold)
        answered = preds != -1
        selected = {
            "threshold": threshold,
            "coverage": float(answered.mean()),
            "accuracy": (
                float((preds[answered] == labels[answered]).mean()) if answered.any() else None
            ),
        }
        curve = coverage_accuracy_curve(after, labels, np.linspace(0, 1, 101))

        result = {
            "temperature": temperature,
            "val_target_coverage": cfg.calibration.target_coverage,
            "selected": selected,
            "test_ece_before": ece_before,
            "test_ece_after": ece_after,
            "val_ece_before": expected_calibration_error(
                softmax(val["logits"], axis=1), val["labels"]
            ),
            "val_ece_after": expected_calibration_error(val_probs, val["labels"]),
        }
        write_json(run_dir / "calibration.json", result)
        write_json(run_dir / "coverage_accuracy.json", curve)
        mlflow.log_artifact(str(run_dir / "calibration.json"), "calibration")
        mlflow.log_artifact(str(run_dir / "coverage_accuracy.json"), "calibration")
        metrics = {
            "temperature": temperature,
            "test_ece_before": ece_before,
            "test_ece_after": ece_after,
            "val_ece_before": result["val_ece_before"],
            "val_ece_after": result["val_ece_after"],
            "abstain_threshold": threshold,
            "test_coverage_at_threshold": selected["coverage"],
        }
        if selected["accuracy"] is not None:
            metrics["test_accuracy_at_threshold"] = selected["accuracy"]
        mlflow.log_metrics(metrics)
        reliability = plot_reliability(
            before,
            after,
            labels,
            ece_before,
            ece_after,
            temperature,
            run_dir / "reliability.png",
        )
        coverage = plot_coverage_accuracy(curve, selected, run_dir / "coverage_accuracy.png")
        mlflow.log_artifact(str(reliability), "plots")
        mlflow.log_artifact(str(coverage), "plots")
        log.info("temperature %.4f; test ECE %.4f -> %.4f", temperature, ece_before, ece_after)


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_calibrate_{cfg.experiment}")
    calibrate(cfg, run_dir, get_logger(__name__, run_dir))


if __name__ == "__main__":
    main()
