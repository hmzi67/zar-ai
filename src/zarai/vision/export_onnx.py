"""Export a trained classifier to ONNX."""

from zarai.common.config import parse_config
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run


def export(cfg, run_dir) -> None:
    raise NotImplementedError("TODO: load model.checkpoint and call torch.onnx.export")


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_export_{cfg.experiment}")
    get_logger(__name__, run_dir).info("run directory: %s", run_dir)
    export(cfg, run_dir)


if __name__ == "__main__":
    main()
