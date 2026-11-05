"""Plots logged with each run. matplotlib only loads when a plot is made."""

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(history: list[dict], path: Path) -> Path:
    plt = _pyplot()
    epochs = [h["epoch"] for h in history]
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(10, 4))
    loss_ax.plot(epochs, [h["train_loss"] for h in history], label="train (smoothed)")
    loss_ax.plot(epochs, [h["val_loss"] for h in history], label="val")
    loss_ax.set(xlabel="epoch", ylabel="loss")
    loss_ax.legend()
    f1_ax.plot(epochs, [h["val_macro_f1"] for h in history], label="macro-F1")
    f1_ax.plot(epochs, [h["val_accuracy"] for h in history], label="accuracy")
    f1_ax.set(xlabel="epoch", ylabel="validation score")
    f1_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], path: Path) -> Path:
    """Row-normalised colours, with raw counts printed in each cell."""
    plt = _pyplot()
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true", title="Test confusion matrix (counts)")
    for row in range(len(classes)):
        for column in range(len(classes)):
            if matrix[row, column]:
                color = "white" if shares[row, column] > 0.5 else "black"
                ax.text(
                    column, row, int(matrix[row, column]), ha="center", va="center", color=color
                )
    fig.colorbar(image, label="share of true class", fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append((confidence[mask].mean(), correct[mask].mean(), int(mask.sum())))
    return rows


def plot_reliability(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    ece_before: float,
    ece_after: float,
    temperature: float,
    path: Path,
    n_bins: int = 15,
) -> Path:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    panels = (
        (axes[0], before, f"Before scaling (ECE {ece_before:.4f})"),
        (axes[1], after, f"After, T={temperature:.3f} (ECE {ece_after:.4f})"),
    )
    for ax, probs, title in panels:
        rows = _reliability_bins(probs, labels, n_bins)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect")
        ax.scatter(
            [r[0] for r in rows],
            [r[1] for r in rows],
            s=[max(12, 200 * r[2] / len(labels)) for r in rows],
            label="bins (size = share of images)",
        )
        ax.set(xlabel="confidence", title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coverage_accuracy(curve: list[dict], selected: dict | None, path: Path) -> Path:
    plt = _pyplot()
    points = [c for c in curve if c["accuracy"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([c["coverage"] for c in points], [c["accuracy"] for c in points], marker=".")
    if selected:
        ax.scatter(
            [selected["coverage"]],
            [selected["accuracy"]],
            color="firebrick",
            zorder=3,
            label=f"threshold from val ({selected['threshold']:.3f})",
        )
        ax.legend(loc="lower left")
    ax.set(xlabel="coverage (share answered)", ylabel="accuracy on answered", xlim=(0, 1.02))
    ax.set_title("Test coverage and accuracy with abstention")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradcam_grid(panels: list[dict], path: Path, per_row: int = 8) -> Path:
    """Each panel has `image` (HWC 0-1), `cam` (HW 0-1), `title` and `correct`."""
    plt = _pyplot()
    n_rows = (len(panels) + per_row - 1) // per_row
    fig, axes = plt.subplots(n_rows, per_row, figsize=(per_row * 2.2, n_rows * 2.5))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, panel in zip(np.ravel(axes), panels, strict=False):
        ax.imshow(panel["image"])
        ax.imshow(panel["cam"], cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax.set_title(panel["title"], fontsize=7, color="black" if panel["correct"] else "firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
