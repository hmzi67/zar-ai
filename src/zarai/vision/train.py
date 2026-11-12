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
    """Stop after `patience` epochs without a higher value, but not before `min_epochs`."""

    def __init__(self, patience: int, min_epochs: int = 0):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best = -math.inf
        self.stale = 0
        self.epochs = 0

    def update(self, value: float) -> bool:
        """Record a value and return True if it is the new best."""
        self.epochs += 1
        if value > self.best:
            self.best, self.stale = value, 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale >= self.patience and self.epochs >= self.min_epochs


def set_train_mode(model, freeze_backbone: bool) -> None:
    """A frozen backbone stays in eval mode, so only the classifier head is in train mode."""
    if freeze_backbone:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def predict(
    model, loader, device: str, use_amp: bool = True, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) for a whole loader, or its first `limit` batches."""
    import torch

    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader):
            if limit and batch >= limit:
                break
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


def sequential_loader(dataset, cfg):
    """Loader in file order, for evaluation."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )


def balanced_loader(dataset, cfg):
    """Training loader with a class-balanced sampler (inverse frequency ** sampler_power)."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

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


def eval_loader(cfg, split: str, mean, std, classes=None):
    """Loader over a split with the eval transform, in file order."""
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
    return sequential_loader(dataset, cfg)


def _train_loader(cfg, mean, std):
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
    return balanced_loader(dataset, cfg)


def is_v2(cfg) -> bool:
    """wheat_v2 configs choose a backbone and mode instead of a timm model name."""
    return "backbone" in cfg.model


def _setup_v1(cfg, log, device):
    probe = build_model(
        cfg.model.name, num_classes=1, pretrained=False, image_size=cfg.data.image_size
    )
    mean, std = probe.pretrained_cfg["mean"], probe.pretrained_cfg["std"]
    del probe
    train_loader = _train_loader(cfg, mean, std)
    classes = train_loader.dataset.classes
    loaders = {"train": train_loader, "val": eval_loader(cfg, "val", mean, std, classes)}
    model = build_model(
        cfg.model.name,
        num_classes=len(classes),
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        revision=cfg.model.hf_revision,
        image_size=cfg.data.image_size,
    ).to(device)
    groups = [p for p in model.parameters() if p.requires_grad]
    return model, groups, loaders, classes, mean, std, cfg.model.freeze_backbone


def _setup_v2(cfg, log, device):
    from zarai.vision import backbones
    from zarai.vision.data_v2 import build_loaders

    spec = backbones.backbone_spec(cfg)
    mean, std = backbones.normalisation(spec, cfg.data.image_size)
    loaders = build_loaders(cfg, mean, std)
    classes = list(cfg.data.classes)
    model = backbones.build(cfg, len(classes)).to(device)
    groups = backbones.param_groups(model, cfg)
    return model, groups, loaders, classes, mean, std, cfg.model.mode == "linear"


def fit(cfg, run_dir: Path, log, run_name=None, tags=None, evaluate=None) -> str:
    """Train until early stopping, keep the best val macro-F1 checkpoint, return the run ID.

    For wheat_v2 configs, `evaluate(model, cfg, loaders, classes, run_dir, log)` is called inside
    the same MLflow run, with the best weights restored.
    """
    import mlflow
    import torch
    from omegaconf import OmegaConf

    from zarai.vision.plots import plot_training_curves

    device = cfg.train.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check nvidia-smi")

    v2 = is_v2(cfg)
    configure(cfg)
    with mlflow.start_run(run_name=run_name or f"{cfg.experiment}_seed{cfg.seed}") as run:
        if tags:
            mlflow.set_tags(tags)
        log_run_context(cfg, run_dir)
        setup = _setup_v2 if v2 else _setup_v1
        model, groups, loaders, classes, mean, std, frozen = setup(cfg, log, device)
        train_loader, val_loader = loaders["train"], loaders["val"]
        log.info(
            "%d classes; %d train, %d val",
            len(classes),
            len(train_loader.dataset),
            len(val_loader.dataset),
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        mlflow.log_params(
            {
                "model.n_params": sum(p.numel() for p in model.parameters()),
                "model.n_trainable": sum(p.numel() for p in trainable),
                "data.mean": list(mean),
                "data.std": list(std),
            }
        )
        optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        accum = cfg.train.get("grad_accum_steps") or 1
        limit = cfg.train.get("limit_batches")  # optimizer steps per epoch
        limit_eval = cfg.train.get("limit_eval_batches")
        steps_per_epoch = len(train_loader) // accum
        if limit:
            steps_per_epoch = min(steps_per_epoch, limit)
        warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
        total_steps = cfg.train.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine_factor(step, warmup_steps, total_steps)
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        stopper = EarlyStopping(cfg.train.early_stopping_patience, cfg.train.get("min_epochs", 0))
        checkpoint_path = run_dir / "best.pt"
        history, best_epoch, best_state, step_seconds = [], 0, None, []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()

        for epoch in range(1, cfg.train.epochs + 1):
            set_train_mode(model, frozen)
            epoch_started = time.monotonic()
            total_loss, n_batches, steps = 0.0, 0, 0
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for batch, (images, targets) in enumerate(train_loader):
                if steps >= steps_per_epoch:
                    break
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss is {loss.item()} at epoch {epoch}, batch {batch}")
                (loss / accum).backward()
                total_loss += loss.item()
                n_batches += 1
                if (batch + 1) % accum:
                    continue
                if v2 and epoch == 1 and steps == 0:
                    missing = [
                        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
                    ]
                    if missing:
                        raise RuntimeError(
                            f"{len(missing)} trainable tensors got no gradient: {missing[:3]}"
                        )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if device == "cuda":
                    torch.cuda.synchronize()
                step_seconds.append(time.monotonic() - step_started)
                step_started = time.monotonic()
            train_seconds = time.monotonic() - epoch_started

            logits, labels = predict(model, val_loader, device, limit=limit_eval)
            preds = logits.argmax(axis=1)
            val_loss = torch.nn.functional.cross_entropy(
                torch.from_numpy(logits), torch.from_numpy(labels)
            ).item()
            metrics = {
                "train_loss": total_loss / n_batches,
                "val_loss": val_loss,
                "val_accuracy": float((preds == labels).mean()),
                "val_macro_f1": macro_f1(labels, preds),
                "lr": max(group["lr"] for group in optimizer.param_groups),
            }
            if v2:
                metrics["epoch_train_seconds"] = train_seconds
                metrics["epoch_seconds"] = time.monotonic() - epoch_started
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
                checkpoint = {
                    "classes": classes,
                    "image_size": cfg.data.image_size,
                    "mean": list(mean),
                    "std": list(std),
                    "epoch": epoch,
                    "val_macro_f1": metrics["val_macro_f1"],
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                if v2:
                    from zarai.vision.backbones import checkpoint_state

                    best_state = checkpoint_state(model, cfg.model.mode)
                    checkpoint.update(
                        state_dict=best_state,
                        trainable_only=cfg.model.mode != "full",
                        backbone=cfg.model.backbone,
                        mode=cfg.model.mode,
                    )
                else:
                    checkpoint.update(state_dict=model.state_dict(), model_name=cfg.model.name)
                torch.save(checkpoint, checkpoint_path)
            if stopper.should_stop:
                log.info("early stopping after epoch %d", epoch)
                break

        summary = {
            "best_val_macro_f1": stopper.best,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_minutes": (time.monotonic() - started) / 60,
        }
        if v2:
            # The first step includes warm-up (cuDNN autotuning, worker start-up), so it's left out.
            timed = step_seconds[1:] or step_seconds
            summary["seconds_per_step"] = float(np.median(timed))
            summary["steps_per_full_epoch"] = len(train_loader) // accum
            summary["est_full_epoch_train_seconds"] = (
                summary["seconds_per_step"] * summary["steps_per_full_epoch"]
            )
            if device == "cuda":
                summary["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
                summary["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        mlflow.log_metrics(summary)
        if not v2 or cfg.train.get("log_checkpoint", True):
            mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        checkpoint_path.unlink()  # MLflow holds the only copy, to avoid storing it twice.
        curves = plot_training_curves(history, run_dir / "training_curves.png")
        mlflow.log_artifact(str(curves), "plots")
        if v2 and evaluate is not None:
            from zarai.vision.backbones import load_state

            load_state(model, best_state, cfg.model.mode)
            evaluate(model, cfg, loaders, classes, run_dir, log)
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
