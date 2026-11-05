"""Evaluate an MLflow run's best val checkpoint on the test split. Pass run_id=<id>.

Also saves val and test logits for zarai.vision.calibrate. Test results are never used for tuning
or for choosing a model.
"""

import logging
from pathlib import Path

import numpy as np

from zarai.common.config import parse_config
from zarai.common.io import write_json
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.data.clean_wheat import DARK_PIXEL, DARK_SHARE
from zarai.eval.vision_metrics import (
    bootstrap_cis,
    confusion_between,
    per_class_report,
    top_confusions,
)
from zarai.vision.gradcam import cam_mass_on_mask, grad_cam
from zarai.vision.models import build_model
from zarai.vision.tracking import artifact_path, configure, require_run_id
from zarai.vision.train import eval_loader, predict

FOCUS_PAIRS = (
    ("black_rust", "brown_rust"),
    ("leaf_blight", "septoria"),
    ("leaf_blight", "tan_spot"),
)
GRADCAM_CLASSES = ("mildew", "fusarium_head_blight")


def load_checkpoint(run_id: str, device: str):
    """Rebuild a run's model from its best val checkpoint. Returns (model, checkpoint)."""
    import torch

    path = artifact_path(run_id, "checkpoints/best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def is_dark_background(path: str) -> bool:
    """Same rule as the dataset report: enough of the image is near-black."""
    from PIL import Image

    with Image.open(path) as image:
        luminance = np.asarray(image.convert("L").resize((64, 64)))
    return bool((luminance < DARK_PIXEL).mean() >= DARK_SHARE)


def _denormalise(images, mean, std) -> np.ndarray:
    """Convert a normalised NCHW tensor to NHWC floats in 0-1."""
    array = images.permute(0, 2, 3, 1).cpu().numpy()
    return np.clip(array * np.array(std) + np.array(mean), 0, 1)


def run_gradcam(model, cfg, checkpoint, device: str, out_dir: Path) -> tuple[Path, dict]:
    """Save the Grad-CAM grid, and measure how much activation falls on near-black pixels."""
    import torch

    from zarai.vision.plots import plot_gradcam_grid

    classes = checkpoint["classes"]
    mean, std = checkpoint["mean"], checkpoint["std"]
    dataset = eval_loader(cfg, "test", mean, std, classes).dataset
    rng = np.random.default_rng(cfg.seed)
    panels, metrics = [], {}
    for name in GRADCAM_CLASSES:
        target = classes.index(name)
        candidates = [i for i, label in enumerate(dataset.labels) if label == target]
        if not candidates:
            continue
        n = min(cfg.eval.gradcam_per_class, len(candidates))
        chosen = sorted(rng.choice(candidates, n, replace=False).tolist())
        images = torch.stack([dataset[i][0] for i in chosen])
        cams, preds, probs = grad_cam(model, images, device)
        pixels = _denormalise(images, mean, std)
        luminance = pixels @ np.array([0.299, 0.587, 0.114])
        dark = luminance < DARK_PIXEL / 255
        metrics[f"cam_{name}_mass_on_dark"] = float(
            np.mean([cam_mass_on_mask(c, d) for c, d in zip(cams, dark, strict=True)])
        )
        metrics[f"cam_{name}_dark_area"] = float(dark.mean())
        n_empty = int((cams.max(axis=(1, 2)) <= 0).sum())
        metrics[f"cam_{name}_n_empty"] = n_empty
        if n_empty:
            logging.getLogger(__name__).warning("%d empty Grad-CAM maps for %s", n_empty, name)
        for image, cam, pred, prob in zip(pixels, cams, preds, probs, strict=True):
            panels.append(
                {
                    "image": image,
                    "cam": cam,
                    "correct": int(pred) == target,
                    "title": f"{name}\npred {classes[pred]} {prob:.2f}",
                }
            )
    return plot_gradcam_grid(panels, out_dir / "gradcam_grid.png"), metrics


def mildew_background_recall(cfg, classes: list[str], preds: np.ndarray, dataset) -> dict:
    """Mildew recall on black-background images and on the rest, to expose a background shortcut."""
    target = classes.index("mildew")
    indices = [i for i, label in enumerate(dataset.labels) if label == target]
    dark = [is_dark_background(dataset.rows[i]["path"]) for i in indices]
    results = {}
    for name, flag in (("dark", True), ("other", False)):
        group = [i for i, is_dark in zip(indices, dark, strict=True) if is_dark == flag]
        results[f"test_mildew_n_{name}"] = len(group)
        if group:
            results[f"test_mildew_recall_{name}"] = float(np.mean(preds[group] == target))
    return results


def evaluate(cfg, run_dir: Path, log) -> None:
    import mlflow
    import torch
    from sklearn.metrics import confusion_matrix

    from zarai.vision.plots import plot_confusion_matrix

    device = cfg.train.device
    run_id = require_run_id(cfg)
    configure(cfg)
    with mlflow.start_run(run_id=run_id):
        model, checkpoint = load_checkpoint(run_id, device)
        classes = checkpoint["classes"]
        log.info("checkpoint from epoch %d", checkpoint["epoch"])
        mean, std = checkpoint["mean"], checkpoint["std"]

        for split in ("val", "test"):
            loader = eval_loader(cfg, split, mean, std, classes)
            logits, labels = predict(model, loader, device)
            np.savez(run_dir / f"{split}_logits.npz", logits=logits, labels=labels)
            mlflow.log_artifact(str(run_dir / f"{split}_logits.npz"), "predictions")

        test_dataset = loader.dataset
        preds = logits.argmax(axis=1)
        indices = list(range(len(classes)))
        cis = bootstrap_cis(labels, preds, cfg.eval.n_bootstrap, cfg.seed)
        metrics = {
            f"test_{name}{suffix}": values[key]
            for name, values in cis.items()
            for suffix, key in (("", "value"), ("_low", "low"), ("_high", "high"))
        }
        report = per_class_report(labels, preds, classes)
        for name in classes:
            metrics[f"test_precision_{name}"] = report[name]["precision"]
            metrics[f"test_recall_{name}"] = report[name]["recall"]
            metrics[f"test_f1_{name}"] = report[name]["f1-score"]

        matrix = confusion_matrix(labels, preds, labels=indices)
        pairs = [confusion_between(matrix, classes, a, b) for a, b in FOCUS_PAIRS]
        for pair in pairs:
            metrics[f"test_confusion_{pair['a']}_as_{pair['b']}"] = pair["a_as_b"]
            metrics[f"test_confusion_{pair['b']}_as_{pair['a']}"] = pair["b_as_a"]
        metrics.update(mildew_background_recall(cfg, classes, preds, test_dataset))

        write_json(run_dir / "per_class.json", {name: report[name] for name in classes})
        write_json(run_dir / "confusion_pairs.json", pairs)
        write_json(run_dir / "top_confusions.json", top_confusions(matrix, classes))
        write_json(
            run_dir / "test_errors.json",
            [
                {
                    "path": test_dataset.rows[i]["path"],
                    "true": classes[labels[i]],
                    "predicted": classes[preds[i]],
                }
                for i in np.nonzero(preds != labels)[0].tolist()
            ],
        )
        for name in ("per_class", "confusion_pairs", "top_confusions", "test_errors"):
            mlflow.log_artifact(str(run_dir / f"{name}.json"), "metrics")
        figure = plot_confusion_matrix(matrix, classes, run_dir / "confusion_matrix.png")
        mlflow.log_artifact(str(figure), "plots")

        with torch.enable_grad():
            grid, cam_metrics = run_gradcam(model, cfg, checkpoint, device, run_dir)
        metrics.update(cam_metrics)
        mlflow.log_artifact(str(grid), "plots")
        mlflow.log_metrics(metrics)
        log.info(
            "test macro-F1 %.4f (%.4f-%.4f), accuracy %.4f (%.4f-%.4f)",
            metrics["test_macro_f1"],
            metrics["test_macro_f1_low"],
            metrics["test_macro_f1_high"],
            metrics["test_accuracy"],
            metrics["test_accuracy_low"],
            metrics["test_accuracy_high"],
        )


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_eval_{cfg.experiment}")
    evaluate(cfg, run_dir, get_logger(__name__, run_dir))


if __name__ == "__main__":
    main()
