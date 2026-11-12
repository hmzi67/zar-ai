"""Train and evaluate wheat_v2 models: one run per (fold, seed), logged to MLflow, then the
results table is rebuilt. Called by the repo-root train.py.
"""

import argparse
import copy

from omegaconf import OmegaConf

from zarai.common.config import load_config
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.vision.data_v2 import available_folds, resolve_fold

DRY_RUN = {
    "train.epochs": 1,
    "train.min_epochs": 0,
    "train.limit_batches": 2,  # optimizer steps, each a full effective batch
    "train.limit_eval_batches": 2,
    "train.log_checkpoint": False,
    "eval.n_bootstrap": 100,
}


def resolve_run_config(
    base, backbone, mode, aug, protocol, fold, seed, dry_run=False, overrides=()
):
    """A copy of the config for one run, with batch size, accumulation and LR resolved.

    key=value overrides are applied last, so they win over both the config and --dry-run.
    """
    cfg = copy.deepcopy(base)
    cfg.model.backbone, cfg.model.mode = backbone, mode
    cfg.train.augment.profile = aug
    cfg.data.protocol = protocol
    _, cfg.data.fold = resolve_fold(cfg.data.splits_csv, protocol, fold)
    cfg.seed = int(seed)
    if backbone not in cfg.backbones:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(cfg.backbones)}")
    spec = cfg.backbones[backbone]
    if mode not in spec.modes:
        raise ValueError(f"{backbone} supports modes {list(spec.modes)}, not {mode!r}")
    micro = spec.micro_batch[mode]
    effective = cfg.train.effective_batch_size
    if effective % micro:
        raise ValueError(f"effective batch {effective} isn't a multiple of micro batch {micro}")
    cfg.train.batch_size = micro
    cfg.train.grad_accum_steps = effective // micro
    cfg.train.lr = spec.get("lr", {}).get(mode, cfg.train.lr_by_mode[mode])
    cfg.experiment = f"wheat_v2_{backbone}_{mode}_{aug}"
    if dry_run:
        for key, value in DRY_RUN.items():
            OmegaConf.update(cfg, key, value)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run_tags(cfg, dry_run: bool) -> dict[str, str]:
    held_out = cfg.data.fold
    return {
        "protocol": cfg.data.protocol,
        "fold": held_out,
        "source": held_out,  # the held-out source, or "pooled"
        "backbone": cfg.model.backbone,
        "mode": cfg.model.mode,
        "aug": cfg.train.augment.profile,
        "background": cfg.data.background,
        "seed": str(cfg.seed),
        "dry_run": str(dry_run).lower(),
    }


def run_one(cfg, dry_run: bool) -> dict:
    import mlflow

    from zarai.vision.eval_v2 import evaluate_run
    from zarai.vision.train import fit

    name = f"{cfg.model.backbone}_{cfg.model.mode}_{cfg.train.augment.profile}_{cfg.data.fold}"
    name += f"_seed{cfg.seed}" + ("_dryrun" if dry_run else "")
    run_dir = start_run(cfg, f"vision_{name}")
    log = get_logger(__name__, run_dir)
    run_id = fit(
        cfg, run_dir, log, run_name=name, tags=run_tags(cfg, dry_run), evaluate=evaluate_run
    )
    return {"run_id": run_id, "name": name, "metrics": mlflow.get_run(run_id).data.metrics}


def _fmt(metrics: dict, key: str, digits: int = 4) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    low, high = metrics.get(f"{key}_low"), metrics.get(f"{key}_high")
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def print_summary(results: list[dict]) -> None:
    print("\nMLflow runs (experiment wheat_v2):")
    for r in results:
        m = r["metrics"]
        print(f"  {r['run_id']}  {r['name']}")
        print(
            f"      held-out macro-F1 {_fmt(m, 'heldout_macro_f1')}  "
            f"NWRD yellow_rust recall {_fmt(m, 'nwrd_recall_yellow_rust')}"
        )
        print(
            f"      held-out ECE {_fmt(m, 'heldout_ece')}  confidently wrong "
            f"{_fmt(m, 'heldout_confident_wrong', 3)}  val macro-F1 {_fmt(m, 'best_val_macro_f1')}"
        )
        if "peak_vram_mib" in m:
            print(
                f"      peak VRAM {m['peak_vram_mib']:.0f} MiB allocated "
                f"({m.get('peak_vram_reserved_mib', 0):.0f} reserved), "
                f"{m['seconds_per_step']:.2f} s/step, ~{m['est_full_epoch_train_seconds']:.0f} s "
                f"per full training epoch ({m['steps_per_full_epoch']:.0f} steps)"
            )


def finished_runs(cfg) -> set[tuple]:
    """(fold, backbone, mode, aug, background, seed) of finished non-dry runs on this dataset."""
    from zarai.vision.results_v2 import latest_by_seed, load_runs
    from zarai.vision.tracking import dvc_hashes

    grouped = latest_by_seed(load_runs(cfg, dvc_hashes(cfg.data.splits_csv)["dvc.splits_md5"]))
    return {(*key, seed) for key, seeds in grouped.items() for seed in seeds}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate wheat_v2 models.")
    parser.add_argument("--config", default="configs/vision/wheat_v2.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--mode", choices=["linear", "lora", "full"], default=None)
    parser.add_argument("--aug", choices=["basic", "strong"], default=None)
    parser.add_argument("--protocol", choices=["pooled", "loso"], default=None)
    parser.add_argument("--fold", default=None, help="held-out source, or `all` for every fold")
    parser.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,0,1")
    parser.add_argument("--background", choices=["none", "remove"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="1 epoch of 2 batches")
    parser.add_argument("--no-results", action="store_true", help="don't rebuild the results file")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip a (fold, seed) that already has a finished run on this dataset version",
    )
    args, overrides = parser.parse_known_args(argv)
    bad = [o for o in overrides if "=" not in o]
    if bad:
        parser.error(f"unrecognised arguments: {bad} (config overrides look like key=value)")

    base = load_config(args.config, overrides)
    if args.background:
        base.data.background = args.background
    backbone = args.backbone or base.model.backbone
    mode = args.mode or base.model.mode
    aug = args.aug or base.train.augment.profile
    protocol = args.protocol or base.data.protocol
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [base.seed]
    if protocol == "pooled":
        folds = ["pooled"]
    elif (args.fold or base.data.fold) == "all":
        folds = available_folds(base.data.splits_csv)
    else:
        folds = [args.fold or base.data.fold]

    done = finished_runs(base) if args.skip_done and not args.dry_run else set()
    results = []
    for fold in folds:
        for seed in seeds:
            cfg = resolve_run_config(
                base, backbone, mode, aug, protocol, fold, seed, args.dry_run, overrides
            )
            key = (cfg.data.fold, backbone, mode, aug, cfg.data.background, str(cfg.seed))
            if key in done:
                print(f"skipping {key}: already finished", flush=True)
                continue
            results.append(run_one(cfg, args.dry_run))
    print_summary(results)
    if not args.no_results:
        from zarai.vision.results_v2 import write_results

        print(f"\nresults table: {write_results(base)}")
