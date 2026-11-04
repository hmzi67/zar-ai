"""Record config, seed and git commit for every training or eval run."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from zarai.common.config import REPO_ROOT
from zarai.common.seed import set_seed


def git_commit() -> str:
    """HEAD commit hash, or "unknown" outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write_run_meta(run_dir: str | Path, config: DictConfig | dict, seed: int | None) -> Path:
    """Write config.yaml and meta.json (seed, commit, time, argv) into run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml", resolve=True)
    meta = {
        "seed": seed,
        "git_commit": git_commit(),
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def start_run(config: DictConfig | dict, name: str) -> Path:
    """Seed the RNGs, create experiments/runs/<time>_<name>/ and log the run metadata."""
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return write_run_meta(REPO_ROOT / "experiments" / "runs" / f"{stamp}_{name}", config, seed)
