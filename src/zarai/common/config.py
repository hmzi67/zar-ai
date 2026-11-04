"""YAML config loading with Hydra-style `key=value` CLI overrides."""

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config. configs/paths.yaml is merged in as `paths`, overrides win."""
    return OmegaConf.merge(
        {"paths": OmegaConf.load(REPO_ROOT / "configs" / "paths.yaml")},
        OmegaConf.load(REPO_ROOT / path),
        OmegaConf.from_dotlist(overrides or []),
    )


def parse_config(description: str | None = None) -> DictConfig:
    """Standard CLI: `--config file.yaml [key=value ...]`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.lr=0.001")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
