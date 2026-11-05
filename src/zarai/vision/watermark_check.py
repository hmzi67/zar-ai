"""Measure how much trained models rely on the wheat_v1 text watermarks. No retraining.

For each finished run in the baselines experiment, on val and test:
- with and without: recall on watermarked and clean images of the same class.
- removal: inpaint the text on watermarked images.
- placebo: apply the same inpainting to clean images, to measure damage from the edit itself.
- injection: paint a layout onto clean images of the other classes.

Results go to a separate MLflow experiment, so the baseline runs are not changed. Layouts come from
train images only. Prints nothing to choose a model with: this is a check on the data.
"""

import argparse
import logging
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import softmax

from zarai.common.io import read_csv
from zarai.common.logging import get_logger
from zarai.eval.vision_metrics import macro_f1
from zarai.vision.results import BASELINES, replace_section, table
from zarai.vision.tracking import dvc_hashes, set_tracking_uri
from zarai.vision.watermark import (
    WATERMARK_CLASSES,
    add_layout,
    layout_mask,
    learn_layouts,
    load_high_pass,
    match_score,
    remove_layout,
)

# Scores fall into two groups, at most 0.21 and at least 0.61, so this isn't a sensitive choice.
DETECTION_THRESHOLD = 0.4
DETECTIONS_CSV = "data/interim/wheat_v1_watermarks.csv"
CHECK_EXPERIMENT = "wheat_v1_watermark_check"
SPLITS = ("val", "test")
DETECTION_COLUMNS = ["path", "label", "split", "score", "layout", "watermarked"]

log = logging.getLogger(__name__)


def detect(rows: list[dict], maps: np.ndarray, layouts: dict[str, np.ndarray]) -> list[dict]:
    """Score every image against every layout, and flag it above the threshold."""
    detections = []
    for row, image_map in zip(rows, maps, strict=True):
        scores = {name: match_score(image_map, layout) for name, layout in layouts.items()}
        best = max(scores, key=scores.get)
        flagged = scores[best] >= DETECTION_THRESHOLD
        detections.append(
            {
                "path": row["path"],
                "label": row["label"],
                "split": row["split"],
                "score": round(scores[best], 4),
                "layout": best if flagged else "",
                "watermarked": flagged,
            }
        )
    return detections


def text_color(rows: list[dict], layouts: dict[str, np.ndarray]) -> tuple[int, int, int]:
    """Median colour of text pixels, from train images where the text is on a near-black area."""
    from PIL import Image

    samples = []
    for row in rows:
        if row["split"] != "train" or not row["layout"]:
            continue
        with Image.open(row["path"]) as image:
            pixels = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
        mask = layout_mask(layouts[row["layout"]])
        # Text brighter than the surrounding dark frame gives its colour with little blending.
        if np.percentile(pixels.mean(axis=2), 30) < 25:
            text = pixels[mask]
            samples.append(np.median(text[text.mean(axis=1) > 60], axis=0))
    if not samples:
        raise ValueError("no watermarked train images on a dark background")
    return tuple(int(value) for value in np.median(samples, axis=0))


class EditedDataset:
    """Images with an optional edit per index, loaded in file order."""

    def __init__(self, rows, labels, transform, edits, layouts, color):
        self.rows, self.labels, self.transform = rows, labels, transform
        self.edits, self.layouts, self.color = edits, layouts, color

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.rows[index]["path"]) as image:
            image = image.convert("RGB")
        edit = self.edits.get(index)
        if edit is not None:
            mode, name = edit
            if mode == "remove":
                image = remove_layout(image, self.layouts[name])
            else:
                image = add_layout(image, self.layouts[name], self.color)
        return self.transform(image), self.labels[index]


def recall(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray, target: int) -> float | None:
    """Recall of a class within a subset, or None if the subset holds none of it."""
    subset = mask & (labels == target)
    return float((preds[subset] == target).mean()) if subset.any() else None


def split_metrics(
    labels: np.ndarray,
    classes: list[str],
    flagged: np.ndarray,
    assigned: list[str],
    layouts: dict[str, np.ndarray],
    predict_with,
) -> dict[str, float]:
    """All measurements for one split. `predict_with(edits)` returns logits."""
    names = list(layouts)
    base = predict_with({})
    base_preds = base.argmax(axis=1)
    metrics = {
        "macro_f1": macro_f1(labels, base_preds),
        "accuracy": float((base_preds == labels).mean()),
    }

    keep = ~flagged
    metrics["macro_f1_without_watermarked"] = macro_f1(labels[keep], base_preds[keep])
    metrics["accuracy_without_watermarked"] = float((base_preds[keep] == labels[keep]).mean())

    removed = predict_with({int(i): ("remove", assigned[i]) for i in np.nonzero(flagged)[0]})
    removed_preds = removed.argmax(axis=1)
    metrics["macro_f1_removed"] = macro_f1(labels, removed_preds)
    metrics["accuracy_removed"] = float((removed_preds == labels).mean())

    # Each clean image gets one layout in turn, so the edit covers the same kinds of area.
    placebo = predict_with(
        {int(i): ("remove", names[int(i) % len(names)]) for i in np.nonzero(keep)[0]}
    )
    placebo_preds = placebo.argmax(axis=1)
    metrics["accuracy_clean"] = float((base_preds[keep] == labels[keep]).mean())
    metrics["accuracy_clean_placebo"] = float((placebo_preds[keep] == labels[keep]).mean())

    base_probs, removed_probs = softmax(base, axis=1), softmax(removed, axis=1)
    for name in WATERMARK_CLASSES:
        target = classes.index(name)
        values = {
            "watermarked": recall(base_preds, labels, flagged, target),
            "clean": recall(base_preds, labels, keep, target),
            "watermarked_removed": recall(removed_preds, labels, flagged, target),
            "clean_placebo": recall(placebo_preds, labels, keep, target),
        }
        metrics.update({f"{name}_recall_{k}": v for k, v in values.items() if v is not None})
        subset = flagged & (labels == target)
        metrics[f"{name}_n_watermarked"] = int(subset.sum())
        metrics[f"{name}_n_clean"] = int((keep & (labels == target)).sum())
        if subset.any():
            metrics[f"{name}_true_prob_watermarked"] = float(base_probs[subset, target].mean())
            metrics[f"{name}_true_prob_removed"] = float(removed_probs[subset, target].mean())

    # Injection: clean images of the other classes.
    targets = {classes.index(name) for name in WATERMARK_CLASSES}
    others = keep & ~np.isin(labels, list(targets))
    metrics["n_injection_images"] = int(others.sum())
    for layout_name in names:
        target_name = next(c for c in WATERMARK_CLASSES if layout_name.startswith(c))
        target = classes.index(target_name)
        injected = predict_with({int(i): ("add", layout_name) for i in np.nonzero(others)[0]})
        injected_preds = injected.argmax(axis=1)
        metrics[f"inject_{layout_name}_rate_before"] = float((base_preds[others] == target).mean())
        metrics[f"inject_{layout_name}_rate_after"] = float(
            (injected_preds[others] == target).mean()
        )
        metrics[f"inject_{layout_name}_accuracy_after"] = float(
            (injected_preds[others] == labels[others]).mean()
        )
    metrics["accuracy_others_before"] = float((base_preds[others] == labels[others]).mean())
    return metrics


def _sample_sheet(rows, assigned, flagged, layouts, color, path: Path) -> Path:
    """Before and after examples, to check by eye that the edits look right."""
    from PIL import Image

    rng = np.random.default_rng(0)
    size, per_kind = 200, 5
    sheet = Image.new("RGB", (per_kind * 2 * size, 3 * size), "white")
    marked = rng.permutation(np.nonzero(flagged)[0])[:per_kind]
    clean = rng.permutation(np.nonzero(~flagged)[0])[: 2 * per_kind]
    names = list(layouts)

    def paste(image, column, line):
        image.thumbnail((size - 4, size - 4))
        sheet.paste(image, (column * size, line * size))

    for k, i in enumerate(marked):
        with Image.open(rows[i]["path"]) as image:
            image = image.convert("RGB")
        paste(image.copy(), 2 * k, 0)
        paste(remove_layout(image, layouts[assigned[i]]), 2 * k + 1, 0)
    for k, i in enumerate(clean):
        with Image.open(rows[i]["path"]) as image:
            image = image.convert("RGB")
        paste(add_layout(image, layouts[names[k % len(names)]], color), k, 1)
        paste(remove_layout(image, layouts[names[k % len(names)]]), k, 2)
    sheet.save(path, quality=88)
    return path


def check_run(source, layouts, color, detections, workers) -> dict[str, float]:
    """Run all measurements for one trained model."""
    from torch.utils.data import DataLoader

    from zarai.vision.eval import load_checkpoint
    from zarai.vision.train import predict
    from zarai.vision.transforms import build_transforms

    model, checkpoint = load_checkpoint(source.info.run_id, "cuda")
    classes = checkpoint["classes"]
    transform = build_transforms(
        checkpoint["image_size"],
        train=False,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        eval_resize=checkpoint["config"]["data"]["eval_resize"],
    )

    def predictor(rows: list[dict], labels: list[int]):
        def predict_with(edits: dict) -> np.ndarray:
            dataset = EditedDataset(rows, labels, transform, edits, layouts, color)
            loader = DataLoader(dataset, batch_size=128, num_workers=workers, pin_memory=True)
            return predict(model, loader, "cuda")[0]

        return predict_with

    results = {}
    for split in SPLITS:
        rows = [r for r in detections if r["split"] == split]
        labels = np.array([classes.index(r["label"]) for r in rows])
        flagged = np.array([r["layout"] != "" for r in rows])
        assigned = [r["layout"] for r in rows]
        metrics = split_metrics(
            labels, classes, flagged, assigned, layouts, predictor(rows, labels.tolist())
        )
        logged = source.data.metrics.get("test_macro_f1")
        if split == "test" and logged is not None and abs(metrics["macro_f1"] - logged) > 1e-3:
            raise ValueError(
                f"baseline macro-F1 {metrics['macro_f1']} differs from logged {logged}"
            )
        results.update({f"{split}_{key}": value for key, value in metrics.items()})
    return results


def _mean_sd(values: list[float | None], digits: int = 3) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "[TBD]"
    if len(present) == 1:
        return f"{present[0]:.{digits}f}"
    return f"{np.mean(present):.{digits}f} ± {np.std(present, ddof=1):.{digits}f}"


def render(
    runs: list, detections: list[dict], layouts, threshold_gap: tuple[float, float]
) -> list[str]:
    """Build the generated part of the report, from logged metrics and the detection table."""
    by_model = {}
    for run in runs:
        by_model.setdefault(run.data.params["source_experiment"], []).append(run.data.metrics)
    models = [c for c in BASELINES if c in by_model]

    counts = Counter((d["label"], d["split"], d["layout"] != "") for d in detections)
    labels = sorted({d["label"] for d in detections})
    detection_rows = []
    for label in labels:
        row = [label]
        for split in ("train", "val", "test"):
            marked, clean = counts[label, split, True], counts[label, split, False]
            total = marked + clean
            row.append(f"{marked} / {total}")
        total_marked = sum(counts[label, s, True] for s in ("train", "val", "test"))
        total_all = total_marked + sum(counts[label, s, False] for s in ("train", "val", "test"))
        row.append(f"{total_marked / total_all:.1%}")
        detection_rows.append(row)

    lines = [
        f"Layouts learned from train images: {len(layouts)}. Scores: highest unflagged "
        f"{threshold_gap[0]:.3f}, lowest flagged {threshold_gap[1]:.3f}, threshold "
        f"{DETECTION_THRESHOLD}.",
        "",
        "## Detection",
        "",
        *table(["class", "train", "val", "test", "share"], detection_rows),
        "",
        "Each model's row is the mean ± SD over its logged seeds. The DINOv2 probe has one seed.",
        "",
    ]

    def cell(config, key, digits=3):
        return _mean_sd([m.get(key) for m in by_model[config]], digits)

    for split in ("test", "val"):
        lines += [f"## Results on {split}", "", "### Overall", ""]
        lines += table(
            [
                "model",
                "macro-F1",
                "macro-F1, text removed",
                "macro-F1, watermarked images left out",
                "accuracy",
                "accuracy, text removed",
            ],
            [
                [
                    BASELINES[c],
                    cell(c, f"{split}_macro_f1", 4),
                    cell(c, f"{split}_macro_f1_removed", 4),
                    cell(c, f"{split}_macro_f1_without_watermarked", 4),
                    cell(c, f"{split}_accuracy", 4),
                    cell(c, f"{split}_accuracy_removed", 4),
                ]
                for c in models
            ],
        )
        lines += ["", "### Recall by class and watermark", ""]
        recall_rows = []
        for c in models:
            for name in WATERMARK_CLASSES:
                metrics = by_model[c][0]
                recall_rows.append(
                    [
                        BASELINES[c],
                        name,
                        f"{metrics.get(f'{split}_{name}_n_watermarked', 0):.0f} / "
                        f"{metrics.get(f'{split}_{name}_n_clean', 0):.0f}",
                        cell(c, f"{split}_{name}_recall_watermarked"),
                        cell(c, f"{split}_{name}_recall_watermarked_removed"),
                        cell(c, f"{split}_{name}_recall_clean"),
                        cell(c, f"{split}_{name}_recall_clean_placebo"),
                        cell(c, f"{split}_{name}_true_prob_watermarked"),
                        cell(c, f"{split}_{name}_true_prob_removed"),
                    ]
                )
        lines += table(
            [
                "model",
                "class",
                "n marked / clean",
                "recall, marked",
                "recall, text removed",
                "recall, clean",
                "recall, clean + placebo",
                "true-class prob, marked",
                "true-class prob, removed",
            ],
            recall_rows,
        )
        lines += ["", "### Placebo on all clean images", ""]
        lines += table(
            ["model", "accuracy, clean", "accuracy, clean + placebo"],
            [
                [
                    BASELINES[c],
                    cell(c, f"{split}_accuracy_clean", 4),
                    cell(c, f"{split}_accuracy_clean_placebo", 4),
                ]
                for c in models
            ],
        )
        lines += [
            "",
            "### Injection: text painted on clean images of the other 13 classes",
            "",
            "The share predicted as the layout's class, before and after painting.",
            "",
        ]
        inject_rows = []
        for c in models:
            for layout_name in layouts:
                inject_rows.append(
                    [
                        BASELINES[c],
                        f"`{layout_name}`",
                        cell(c, f"{split}_inject_{layout_name}_rate_before"),
                        cell(c, f"{split}_inject_{layout_name}_rate_after"),
                        cell(c, f"{split}_accuracy_others_before"),
                        cell(c, f"{split}_inject_{layout_name}_accuracy_after"),
                    ]
                )
        lines += table(
            ["model", "layout", "share before", "share after", "accuracy before", "accuracy after"],
            inject_rows,
        )
        lines.append("")
    return lines


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="wheat_v1_baselines")
    parser.add_argument("--splits-csv", default="data/splits/wheat_v1.csv")
    parser.add_argument("--doc", type=Path, default=Path("experiments/wheat_v1_watermark_check.md"))
    parser.add_argument("--out", type=Path, default=Path(DETECTIONS_CSV))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit-runs", type=int, default=None, help="for a quick trial only")
    args = parser.parse_args()
    get_logger(__name__)

    dvc = dvc_hashes(args.splits_csv)
    rows = read_csv(args.splits_csv)
    with Pool(args.workers) as pool:
        maps = np.stack(pool.map(load_high_pass, [r["path"] for r in rows], chunksize=32))
    train = np.array([r["split"] == "train" for r in rows])
    labels = np.array([r["label"] for r in rows])
    layouts = learn_layouts({c: maps[train & (labels == c)] for c in WATERMARK_CLASSES})
    log.info("learned layouts: %s", list(layouts))
    detections = detect(rows, maps, layouts)
    del maps
    color = text_color(detections, layouts)
    log.info("text colour: %s", color)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from zarai.data.clean_wheat import _write_csv

    _write_csv(args.out, DETECTION_COLUMNS, detections)
    scores = np.array([d["score"] for d in detections])
    flags = np.array([d["layout"] != "" for d in detections])
    gap = (float(scores[~flags].max()), float(scores[flags].min()))

    set_tracking_uri(args.tracking_uri)
    sources = mlflow.search_runs(
        experiment_names=[args.experiment],
        output_format="list",
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    sources = [
        run
        for run in sources
        if run.data.params.get("dvc.processed_md5") == dvc["dvc.processed_md5"]
        and run.data.metrics.get("test_macro_f1") is not None
    ][: args.limit_runs]

    mlflow.set_experiment(CHECK_EXPERIMENT)
    run_dir = args.out.parent
    test_rows = [d for d in detections if d["split"] == "test"]
    for source in sources:
        params = source.data.params
        name = f"{params['experiment']}_seed{params['seed']}"
        log.info("checking %s", name)
        results = check_run(source, layouts, color, detections, args.workers)
        with mlflow.start_run(run_name=f"watermark_{name}"):
            mlflow.log_params(
                {
                    "source_run_id": source.info.run_id,
                    "source_experiment": params["experiment"],
                    "source_seed": params["seed"],
                    "threshold": DETECTION_THRESHOLD,
                    "text_color": list(color),
                    **dvc,
                }
            )
            mlflow.log_metrics(results)
            mlflow.log_artifact(str(args.out), "detections")

    sheet = _sample_sheet(
        test_rows,
        [d["layout"] for d in test_rows],
        np.array([d["layout"] != "" for d in test_rows]),
        layouts,
        color,
        run_dir / "wheat_v1_watermark_edits.jpg",
    )
    log.info("sample edits: %s", sheet)

    logged = mlflow.search_runs(
        experiment_names=[CHECK_EXPERIMENT],
        output_format="list",
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    latest = {}
    for run in logged:
        latest[run.data.params["source_run_id"]] = run
    generated = render(list(latest.values()), detections, layouts, gap)
    args.doc.write_text(
        replace_section(args.doc.read_text(encoding="utf-8"), generated), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
