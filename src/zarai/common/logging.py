"""Console logging, plus a run.log file when a run directory is given."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, run_dir: str | Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if run_dir is not None:
        handlers.append(logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)
