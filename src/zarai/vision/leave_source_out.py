"""Score a model trained without the watermarked source on that source, as unseen data.

Pass run_id=<id> to score one run. Logs to that MLflow run. With report=true, instead fills
experiments/wheat_v1_leave_source_out.md from logged runs.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax

from zarai.common.config import parse_config
from zarai.common.io import read_csv, write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.eval.vision_metrics import macro_f1
from zarai.vision.eval import is_dark_background, load_checkpoint
from zarai.vision.results import replace_section, table
from zarai.vision.tracking import artifact_path, configure, require_run_id, set_tracking_uri
from zarai.vision.train import predict
from zarai.vision.transforms import build_transforms
from zarai.vision.watermark import WATERMARK_CLASSES

SEEDS = ("42", "0", "1", "2")
BASELINE_CONFIG = "wheat_v1_efficientnetv2"
TBD = "[TBD]"


def subset_recall(preds: np.ndarray, labels: np.ndarray, target: int, mask=None) -> float | None:
    """Recall of one class, optionally within a subset. None if the subset has none of it."""
    subset = labels == target if mask is None else (labels == target) & mask
    return float((preds[subset] == target).mean()) if subset.any() else None


def heldout_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    splits: np.ndarray,
    layouts: np.ndarray,
    dark: np.ndarray,
    temperature: float | None,
    threshold: float | None,
) -> dict[str, float]:
    """Recall on the held-out source: by class, by original split, by layout and by background."""
    preds = logits.argmax(axis=1)
    metrics = {"heldout_n": len(labels), "heldout_accuracy": float((preds == labels).mean())}
    probs = softmax(logits, axis=1)
    for name in WATERMARK_CLASSES:
        target = classes.index(name)
        members = labels == target
        metrics[f"heldout_n_{name}"] = int(members.sum())
        metrics[f"heldout_recall_{name}"] = subset_recall(preds, labels, target)
        metrics[f"heldout_true_prob_{name}"] = float(probs[members, target].mean())
        for split in ("train", "val", "test"):
            metrics[f"heldout_recall_{name}_from_{split}"] = subset_recall(
                preds, labels, target, splits == split
            )
        for layout in sorted(set(layouts[members].tolist())):
            metrics[f"heldout_recall_{name}_layout_{layout}"] = subset_recall(
                preds, labels, target, layouts == layout
            )
    mildew = classes.index("mildew")
    metrics["heldout_recall_mildew_dark"] = subset_recall(preds, labels, mildew, dark)
    metrics["heldout_recall_mildew_light"] = subset_recall(preds, labels, mildew, ~dark)
    metrics["heldout_n_mildew_dark"] = int(((labels == mildew) & dark).sum())
    metrics["heldout_n_mildew_light"] = int(((labels == mildew) & ~dark).sum())
    if temperature is not None and threshold is not None:
        calibrated = softmax(logits / temperature, axis=1).max(axis=1)
        answered = calibrated >= threshold
        metrics["heldout_coverage_at_threshold"] = float(answered.mean())
        if answered.any():
            metrics["heldout_accuracy_at_threshold"] = float(
                (preds[answered] == labels[answered]).mean()
            )
    return {key: value for key, value in metrics.items() if value is not None}


def clean_test_metrics(
    logits: np.ndarray, labels: np.ndarray, marked: np.ndarray, classes: list[str]
) -> dict[str, float]:
    """Test scores on images outside the held-out source, and recall on its test images."""
    preds = logits.argmax(axis=1)
    clean = ~marked
    metrics = {
        "test_macro_f1_clean": macro_f1(labels[clean], preds[clean]),
        "test_accuracy_clean": float((preds[clean] == labels[clean]).mean()),
    }
    for name in WATERMARK_CLASSES:
        target = classes.index(name)
        metrics[f"test_recall_{name}_clean"] = subset_recall(preds, labels, target, clean)
        metrics[f"test_recall_{name}_marked"] = subset_recall(preds, labels, target, marked)
    return {key: value for key, value in metrics.items() if value is not None}


def score_run(cfg, run_dir: Path, log) -> None:
    import mlflow
    from torch.utils.data import DataLoader

    from zarai.vision.watermark_check import EditedDataset

    run_id = require_run_id(cfg)
    configure(cfg)
    with mlflow.start_run(run_id=run_id) as run:
        model, checkpoint = load_checkpoint(run_id, cfg.train.device)
        classes = checkpoint["classes"]
        detections = [row for row in read_csv(cfg.data.exclude_csv) if row["layout"]]
        if len(detections) != cfg.data.exclude_expected:
            raise ValueError(f"expected {cfg.data.exclude_expected}, found {len(detections)}")
        labels = np.array([classes.index(row["label"]) for row in detections])
        transform = build_transforms(
            checkpoint["image_size"],
            train=False,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            eval_resize=cfg.data.eval_resize,
        )
        dataset = EditedDataset(detections, labels.tolist(), transform, {}, {}, None)
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.eval_batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        logits = predict(model, loader, cfg.train.device)[0]
        logged = run.data.metrics
        metrics = heldout_metrics(
            logits,
            labels,
            classes,
            np.array([row["split"] for row in detections]),
            np.array([row["layout"] for row in detections]),
            np.array([is_dark_background(row["path"]) for row in detections]),
            logged.get("temperature"),
            logged.get("abstain_threshold"),
        )

        test = np.load(artifact_path(run_id, "predictions/test_logits.npz"))
        marked_paths = {row["path"] for row in detections}
        test_rows = [row for row in read_csv(cfg.data.splits_csv) if row["split"] == "test"]
        marked = np.array([row["path"] in marked_paths for row in test_rows])
        metrics.update(clean_test_metrics(test["logits"], test["labels"], marked, classes))
        mlflow.log_metrics(metrics)

        preds = logits.argmax(axis=1)
        predicted_as = {
            name: dict(
                Counter(classes[p] for p in preds[labels == classes.index(name)]).most_common()
            )
            for name in WATERMARK_CLASSES
        }
        write_json(run_dir / "heldout_predictions.json", predicted_as)
        np.savez(run_dir / "heldout_logits.npz", logits=logits, labels=labels)
        mlflow.log_artifact(str(run_dir / "heldout_predictions.json"), "leave_source_out")
        mlflow.log_artifact(str(run_dir / "heldout_logits.npz"), "leave_source_out")
        log.info(
            "held-out recall: mildew %.4f, fusarium_head_blight %.4f",
            metrics["heldout_recall_mildew"],
            metrics["heldout_recall_fusarium_head_blight"],
        )


def mean_sd(values: list[float | None], digits: int = 4) -> str:
    """Format `mean ± sd`, or [TBD] unless every seed has the value."""
    if not values or any(value is None for value in values):
        return TBD
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def render(
    lso: dict[str, dict], baseline: dict[str, dict], check: dict[str, dict], predicted: dict
) -> list[str]:
    """Build the generated report. Seed dictionaries map a seed to {"run_id", "metrics"}."""

    def values(runs: dict, key: str) -> list[float | None]:
        return [runs[s]["metrics"].get(key) if s in runs else None for s in SEEDS]

    def run_ids(runs: dict) -> list:
        return [f"`{runs[s]['run_id']}`" if s in runs else TBD for s in SEEDS]

    lines = [
        "## Runs",
        "",
        *table(
            ["seed", "MLflow run ID", "val macro-F1", "best epoch / epochs", "test macro-F1"],
            [
                [
                    seed,
                    run_id,
                    (
                        mean_sd([lso[seed]["metrics"].get("best_val_macro_f1")])
                        if seed in lso
                        else TBD
                    ),
                    (
                        f"{lso[seed]['metrics']['best_epoch']:.0f} / "
                        f"{lso[seed]['metrics']['epochs_run']:.0f}"
                        if seed in lso
                        else TBD
                    ),
                    mean_sd([lso[seed]["metrics"].get("test_macro_f1")]) if seed in lso else TBD,
                ]
                for seed, run_id in zip(SEEDS, run_ids(lso), strict=True)
            ],
        ),
        "",
        "## Held-out source: recall per class",
        "",
        "All 923 watermarked images, which these models never saw in training or selection.",
        "",
        *table(
            ["class", "images", "recall", "true-class probability"],
            [
                [
                    name,
                    mean_sd(values(lso, f"heldout_n_{name}"), 0).split(" ")[0],
                    mean_sd(values(lso, f"heldout_recall_{name}")),
                    mean_sd(values(lso, f"heldout_true_prob_{name}")),
                ]
                for name in WATERMARK_CLASSES
            ],
        ),
        "",
        f"Accuracy over all 923: {mean_sd(values(lso, 'heldout_accuracy'))}.",
        "",
        "## Test scores compared with the baseline",
        "",
        "The baseline trained with the watermarked images. Both columns use the same test images.",
        "",
    ]
    rows = [
        ("Test macro-F1, all images", "test_macro_f1", "test_macro_f1", False),
        ("Test accuracy, all images", "test_accuracy", "test_accuracy", False),
        (
            "Test macro-F1, without watermarked images",
            "test_macro_f1_clean",
            "test_macro_f1_without_watermarked",
            True,
        ),
    ]
    compare = [
        [
            label,
            mean_sd(values(check if use_check else baseline, base_key)),
            mean_sd(values(lso, key)),
        ]
        for label, key, base_key, use_check in rows
    ]
    for name in WATERMARK_CLASSES:
        compare += [
            [
                f"`{name}` recall, watermarked test images",
                mean_sd(values(check, f"test_{name}_recall_watermarked")),
                mean_sd(values(lso, f"test_recall_{name}_marked")),
            ],
            [
                f"`{name}` recall, clean test images",
                mean_sd(values(check, f"test_{name}_recall_clean")),
                mean_sd(values(lso, f"test_recall_{name}_clean")),
            ],
        ]
    lines += table(["measure", "baseline", "trained without the source"], compare)
    lines += [
        "",
        "## Breakdown of the held-out source",
        "",
        *table(
            ["subset", "images", "recall"],
            [
                [
                    "`mildew`, black background",
                    mean_sd(values(lso, "heldout_n_mildew_dark"), 0).split(" ")[0],
                    mean_sd(values(lso, "heldout_recall_mildew_dark")),
                ],
                [
                    "`mildew`, other background",
                    mean_sd(values(lso, "heldout_n_mildew_light"), 0).split(" ")[0],
                    mean_sd(values(lso, "heldout_recall_mildew_light")),
                ],
                *[
                    [
                        f"`{name}`, from original {split} split",
                        TBD if not lso else "-",
                        mean_sd(values(lso, f"heldout_recall_{name}_from_{split}")),
                    ]
                    for name in WATERMARK_CLASSES
                    for split in ("train", "val", "test")
                ],
            ],
        ),
        "",
        f"Coverage at the val-chosen threshold: {mean_sd(values(lso, 'heldout_coverage_at_threshold'))}; "
        f"accuracy when answering: {mean_sd(values(lso, 'heldout_accuracy_at_threshold'))}.",
        "",
        "## What held-out images were predicted as",
        "",
        "Counts summed over the seeds.",
        "",
    ]
    if predicted:
        for name, counts in predicted.items():
            lines += [
                f"`{name}`: " + ", ".join(f"{k} {v}" for k, v in counts.most_common(6)) + ".",
                "",
            ]
    else:
        lines += [TBD, ""]
    return lines


def _latest(experiment: str, select) -> dict[str, dict]:
    import mlflow

    found = mlflow.search_runs(
        experiment_names=[experiment],
        output_format="list",
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    return {
        seed: {"run_id": run.info.run_id, "metrics": run.data.metrics}
        for run in found
        for seed in [select(run)]
        if seed is not None
    }


def report(cfg) -> None:
    import json

    set_tracking_uri(cfg.mlflow.tracking_uri)
    lso = _latest(
        cfg.mlflow.experiment,
        lambda run: (
            run.data.params["seed"] if run.data.params.get("experiment") == cfg.experiment else None
        ),
    )
    baseline = _latest(
        "wheat_v1_baselines",
        lambda run: (
            run.data.params["seed"]
            if run.data.params.get("experiment") == BASELINE_CONFIG
            else None
        ),
    )
    check = _latest(
        "wheat_v1_watermark_check",
        lambda run: (
            run.data.params["source_seed"]
            if run.data.params.get("source_experiment") == BASELINE_CONFIG
            else None
        ),
    )
    predicted = {}
    for seed in lso:
        try:
            path = artifact_path(lso[seed]["run_id"], "leave_source_out/heldout_predictions.json")
        except Exception:  # MLflow raises several types for a missing artifact.
            continue
        for name, counts in json.loads(Path(path).read_text(encoding="utf-8")).items():
            predicted.setdefault(name, Counter()).update(counts)
    doc = Path(cfg.get("doc") or "experiments/wheat_v1_leave_source_out.md")
    doc.write_text(
        replace_section(doc.read_text(encoding="utf-8"), render(lso, baseline, check, predicted)),
        encoding="utf-8",
    )


def main() -> None:
    cfg = parse_config(__doc__)
    if cfg.get("report"):
        get_logger(__name__)
        report(cfg)
        return
    run_dir = start_run(cfg, f"vision_lso_{cfg.experiment}")
    score_run(cfg, run_dir, get_logger(__name__, run_dir))


if __name__ == "__main__":
    main()
