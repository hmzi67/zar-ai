"""Train an image classifier from a configs/vision/*.yaml file."""

from pathlib import Path

from zarai.common.config import parse_config
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run


def fit(cfg, run_dir: Path, log) -> None:
    raise NotImplementedError(
        "TODO: training loop (build_model, ImageMetaDataset, AdamW, checkpoint best val macro-F1)"
    )


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_train_{cfg.experiment}")
    log = get_logger(__name__, run_dir)
    log.info("run directory: %s", run_dir)
    fit(cfg, run_dir, log)


if __name__ == "__main__":
    main()
