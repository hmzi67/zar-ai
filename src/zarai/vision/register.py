"""Register the best-val seed of the frozen model in the MLflow Model Registry.

The version points at the run's exported ONNX file and records what serving needs. Running this
again reuses the existing version for that run instead of adding a new one.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from zarai.common.logging import get_logger

log = logging.getLogger(__name__)


def best_seed(runs: dict[str, dict]) -> str:
    """The seed with the highest validation macro-F1. Test scores play no part."""
    return max(runs, key=lambda seed: runs[seed]["best_val_macro_f1"])


def version_tags(seed: str, metrics: dict, params: dict, checkpoint: dict, onnx_sha: str) -> dict:
    """Everything needed to use or trace this version, as MLflow tags."""
    return {
        "seed": seed,
        "best_val_macro_f1": f"{metrics['best_val_macro_f1']:.6f}",
        "test_macro_f1": f"{metrics['test_macro_f1']:.6f}",
        "temperature": f"{metrics['temperature']:.6f}",
        "abstain_threshold": f"{metrics['abstain_threshold']:.6f}",
        "onnx_max_abs_diff": f"{metrics['onnx_max_abs_diff']:.3e}",
        "onnx_sha256": onnx_sha,
        "classes": json.dumps(checkpoint["classes"]),
        "image_size": str(checkpoint["image_size"]),
        "eval_resize": str(checkpoint["config"]["data"]["eval_resize"]),
        "mean": json.dumps(checkpoint["mean"]),
        "std": json.dumps(checkpoint["std"]),
        "dvc_processed_md5": params["dvc.processed_md5"],
        "dvc_splits_md5": params["dvc.splits_md5"],
        "hf_revision": params["model.hf_revision"],
    }


def register(cfg, description: str) -> None:
    import mlflow
    import torch
    from mlflow import MlflowClient

    from zarai.vision.tracking import artifact_path, set_tracking_uri

    set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    sources = {seed: client.get_run(run_id) for seed, run_id in cfg.frozen.runs.items()}
    seed = best_seed({s: run.data.metrics for s, run in sources.items()})
    source = sources[seed]
    metrics, run_id = source.data.metrics, source.info.run_id
    if metrics.get("onnx_max_abs_diff", 1.0) > source_tolerance(source):
        raise ValueError(f"run {run_id} has no ONNX export that passed the check")

    onnx_path = Path(artifact_path(run_id, "onnx/model.onnx"))
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    checkpoint = torch.load(
        artifact_path(run_id, "checkpoints/best.pt"), map_location="cpu", weights_only=True
    )
    tags = version_tags(seed, metrics, source.data.params, checkpoint, digest)

    name, alias = cfg.frozen.registered_model, cfg.frozen.alias
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name, description=description)
    existing = [v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id]
    if existing:
        version = existing[0]
    else:
        version = client.create_model_version(
            name, source=f"runs:/{run_id}/onnx", run_id=run_id, tags=tags, description=description
        )
    client.set_registered_model_alias(name, alias, version.version)
    found = client.get_model_version_by_alias(name, alias)
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}")
    registered = hashlib.sha256((Path(local) / "model.onnx").read_bytes()).hexdigest()
    if registered != digest:
        raise ValueError("the registered ONNX file does not match the run's")
    log.info("%s@%s is version %s: seed %s, run %s", name, alias, found.version, seed, run_id)


def source_tolerance(source) -> float:
    return float(source.data.params.get("onnx.tolerance", 1e-4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vision/wheat_vision_v1.yaml"))
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    get_logger(__name__)
    cfg = OmegaConf.load(args.config)
    if args.tracking_uri:
        cfg.mlflow.tracking_uri = args.tracking_uri
    register(cfg, cfg.frozen.description)


if __name__ == "__main__":
    main()
