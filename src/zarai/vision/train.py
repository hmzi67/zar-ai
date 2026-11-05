"""Train an image classifier from a configs/vision/*.yaml file and log the run to MLflow.

Prints the MLflow run ID as the last line of stdout.
"""

import math
import time
from pathlib import Path

import numpy as np

from zarai.common.config import parse_config
from zarai.common.io import read_csv
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.eval.vision_metrics import macro_f1
from zarai.vision.datasets import SplitCsvDataset, class_balanced_weights
from zarai.vision.models import build_model
from zarai.vision.tracking import configure, log_run_context
from zarai.vision.transforms import build_transforms


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warm-up to 1, then a cosine decay to 0 at total_steps."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class EarlyStopping:
    """Stop after `patience` epochs without a higher value."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = -math.inf
        self.stale = 0

    def update(self, value: float) -> bool:
        """Record a value and return True if it is the new best."""
        if value > self.best:
            self.best, self.stale = value, 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale >= self.patience


def set_train_mode(model, freeze_backbone: bool) -> None:
    """A frozen backbone stays in eval mode, so only the classifier head is in train mode."""
    if freeze_backbone:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def predict(model, loader, device: str, use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) for a whole loader."""
    import torch

    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(images.to(device, non_blocking=True))
            logits.append(outputs.float().cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def excluded_paths(cfg, split: str) -> set[str] | None:
    """Paths left out of this split by data.exclude_csv, for a leave-source-out run.

    Rows with a non-empty `layout` are left out of the splits in data.exclude_splits. The total
    must equal data.exclude_expected, so a changed detection file can't pass unnoticed.
    """
    exclude_csv = cfg.data.get("exclude_csv")
    if not exclude_csv or split not in cfg.data.exclude_splits:
        return None
    marked = [row for row in read_csv(exclude_csv) if row["layout"]]
    if len(marked) != cfg.data.exclude_expected:
        raise ValueError(
            f"{exclude_csv}: expected {cfg.data.exclude_expected} images, found {len(marked)}"
        )
    return {row["path"] for row in marked}


def eval_loader(cfg, split: str, mean, std, classes=None):
    """Loader over a split with the eval transform, in file order."""
    from torch.utils.data import DataLoader

    transform = build_transforms(
        cfg.data.image_size, train=False, mean=mean, std=std, eval_resize=cfg.data.eval_resize
    )
    dataset = SplitCsvDataset(
        cfg.data.splits_csv,
        split,
        transform,
        classes,
        cfg.data.get("max_samples"),
        cfg.seed,
        excluded_paths(cfg, split),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )


def _train_loader(cfg, mean, std):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    aug = cfg.train.augment
    transform = build_transforms(
        cfg.data.image_size,
        train=True,
        mean=mean,
        std=std,
        crop_scale=aug.crop_scale,
        randaugment_ops=aug.randaugment_ops,
        randaugment_magnitude=aug.randaugment_magnitude,
        jitter=aug.jitter,
        jitter_hue=aug.jitter_hue,
    )
    dataset = SplitCsvDataset(
        cfg.data.splits_csv,
        "train",
        transform,
        None,
        cfg.data.get("max_samples"),
        cfg.seed,
        excluded_paths(cfg, "train"),
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    weights = class_balanced_weights(dataset.labels, cfg.train.sampler_power)
    sampler = WeightedRandomSampler(
        weights.tolist(), num_samples=len(dataset), replacement=True, generator=generator
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
        generator=generator,
    )


def fit(cfg, run_dir: Path, log) -> str:
    """Train until early stopping, keep the best val macro-F1 checkpoint, return the run ID."""
    import mlflow
    import torch
    from omegaconf import OmegaConf

    from zarai.vision.plots import plot_training_curves

    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check nvidia-smi")

    configure(cfg)
    with mlflow.start_run(run_name=f"{cfg.experiment}_seed{cfg.seed}") as run:
        log_run_context(cfg, run_dir)
        probe = build_model(
            cfg.model.name,
            num_classes=1,
            pretrained=False,
            image_size=cfg.data.image_size,
        )
        mean, std = probe.pretrained_cfg["mean"], probe.pretrained_cfg["std"]
        del probe

        train_loader = _train_loader(cfg, mean, std)
        classes = train_loader.dataset.classes
        val_loader = eval_loader(cfg, "val", mean, std, classes)
        log.info(
            "%d classes; %d train, %d val",
            len(classes),
            len(train_loader.dataset),
            len(val_loader.dataset),
        )

        model = build_model(
            cfg.model.name,
            num_classes=len(classes),
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            revision=cfg.model.hf_revision,
            image_size=cfg.data.image_size,
        ).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        mlflow.log_params(
            {
                "model.n_params": sum(p.numel() for p in model.parameters()),
                "model.n_trainable": sum(p.numel() for p in trainable),
                "data.mean": list(mean),
                "data.std": list(std),
            }
        )
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        limit = cfg.train.get("limit_batches")
        steps_per_epoch = min(len(train_loader), limit) if limit else len(train_loader)
        warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
        total_steps = cfg.train.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine_factor(step, warmup_steps, total_steps)
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        stopper = EarlyStopping(cfg.train.early_stopping_patience)
        checkpoint_path = run_dir / "best.pt"
        history, best_epoch = [], 0
        started = time.monotonic()

        for epoch in range(1, cfg.train.epochs + 1):
            set_train_mode(model, cfg.model.freeze_backbone)
            total_loss, n_batches = 0.0, 0
            for batch, (images, targets) in enumerate(train_loader):
                if limit and batch >= limit:
                    break
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss is {loss.item()} at epoch {epoch}, batch {batch}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1

            logits, labels = predict(model, val_loader, device)
            preds = logits.argmax(axis=1)
            val_loss = torch.nn.functional.cross_entropy(
                torch.from_numpy(logits), torch.from_numpy(labels)
            ).item()
            metrics = {
                "train_loss": total_loss / n_batches,
                "val_loss": val_loss,
                "val_accuracy": float((preds == labels).mean()),
                "val_macro_f1": macro_f1(labels, preds),
                "lr": optimizer.param_groups[0]["lr"],
            }
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            improved = stopper.update(metrics["val_macro_f1"])
            log.info(
                "epoch %d/%d train_loss %.4f val_loss %.4f val_macro_f1 %.4f%s",
                epoch,
                cfg.train.epochs,
                metrics["train_loss"],
                val_loss,
                metrics["val_macro_f1"],
                " *" if improved else "",
            )
            if improved:
                best_epoch = epoch
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "model_name": cfg.model.name,
                        "image_size": cfg.data.image_size,
                        "mean": list(mean),
                        "std": list(std),
                        "epoch": epoch,
                        "val_macro_f1": metrics["val_macro_f1"],
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    checkpoint_path,
                )
            if stopper.should_stop:
                log.info("early stopping after epoch %d", epoch)
                break

        mlflow.log_metrics(
            {
                "best_val_macro_f1": stopper.best,
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "train_minutes": (time.monotonic() - started) / 60,
            }
        )
        mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        checkpoint_path.unlink()  # MLflow holds the only copy, to avoid storing it twice.
        curves = plot_training_curves(history, run_dir / "training_curves.png")
        mlflow.log_artifact(str(curves), "plots")
        return run.info.run_id


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_train_{cfg.experiment}")
    log = get_logger(__name__, run_dir)
    log.info("run directory: %s", run_dir)
    run_id = fit(cfg, run_dir, log)
    log.info("MLflow run ID: %s", run_id)
    print(run_id)


if __name__ == "__main__":
    main()
