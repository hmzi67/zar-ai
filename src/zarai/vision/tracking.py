"""MLflow logging shared by training, evaluation, calibration and export."""

import hashlib
import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from zarai.common.config import REPO_ROOT
from zarai.common.run_meta import git_commit

DVC_STAGE = "clean_wheat"
RAW_DVC_FILE = "data/raw/images_public/wheat_plant_diseases.dvc"


def require_run_id(cfg) -> str:
    """Return run_id. Without this check, MLflow would quietly start a new, unrelated run."""
    run_id = cfg.get("run_id")
    if not run_id:
        raise ValueError("pass the training run's ID as run_id=<id>")
    return run_id


def flatten(values: dict, prefix: str = "") -> dict:
    """Flatten nested config values to dotted keys, as MLflow params need."""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def dvc_hashes(splits_csv: str | Path) -> dict[str, str]:
    """Return the dataset's DVC hashes, failing if the splits CSV has changed since dvc.lock."""
    lock = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "dvc.lock"))
    outs = {out["path"]: out["md5"] for out in lock["stages"][DVC_STAGE]["outs"]}
    raw = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / RAW_DVC_FILE))
    splits_key = Path(splits_csv).as_posix()
    if splits_key not in outs:
        raise ValueError(f"{splits_key} is not an output of DVC stage {DVC_STAGE}")
    actual = hashlib.md5((REPO_ROOT / splits_key).read_bytes(), usedforsecurity=False).hexdigest()
    if actual != outs[splits_key]:
        raise ValueError(f"{splits_key} does not match dvc.lock: run dvc repro or dvc checkout")
    return {
        "dvc.raw_md5": raw["outs"][0]["md5"],
        "dvc.processed_md5": outs["data/processed/wheat_v1"],
        "dvc.splits_md5": outs[splits_key],
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def set_tracking_uri(uri: str) -> None:
    """Point MLflow at a store. MLflow 3 refuses a local file store unless this opt-out is set."""
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)


def configure(cfg: DictConfig) -> None:
    import mlflow

    set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)


def log_run_context(cfg: DictConfig, run_dir: Path) -> None:
    """Log config, seed, git state and DVC hashes. An unclean tree also logs its diff."""
    import mlflow

    mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
    mlflow.log_params(dvc_hashes(cfg.data.splits_csv))
    exclude_csv = cfg.data.get("exclude_csv")
    if exclude_csv:
        # The detection file isn't DVC-tracked, so its exact version is logged with the run.
        digest = hashlib.md5(Path(exclude_csv).read_bytes(), usedforsecurity=False).hexdigest()
        mlflow.log_param("data.exclude_csv_md5", digest)
        mlflow.log_artifact(exclude_csv, "data")
    status = _git("status", "--porcelain")
    mlflow.set_tags(
        {
            "git_commit": git_commit(),
            "git_dirty": str(bool(status.strip())).lower(),
            "seed": str(cfg.seed),
            "model_name": cfg.model.name,
        }
    )
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    mlflow.log_artifact(str(run_dir / "resolved_config.yaml"), "config")
    if status.strip():
        (run_dir / "git_status.txt").write_text(status, encoding="utf-8")
        (run_dir / "git_diff.patch").write_text(_git("diff", "HEAD"), encoding="utf-8")
        mlflow.log_artifact(str(run_dir / "git_status.txt"), "git")
        mlflow.log_artifact(str(run_dir / "git_diff.patch"), "git")
        # `git diff` leaves out new files, so they are logged as they are.
        for path in _git("ls-files", "--others", "--exclude-standard").splitlines():
            parent = Path(path).parent.as_posix()
            mlflow.log_artifact(str(REPO_ROOT / path), f"git/untracked/{parent}")


def artifact_path(run_id: str, path: str) -> str:
    """Return a local path to one artifact of a run."""
    import mlflow

    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
