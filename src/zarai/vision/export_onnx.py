"""Export an MLflow run's best val checkpoint to ONNX and check it against PyTorch.

Pass run_id=<id>. Fails if any output differs by more than eval.onnx_tolerance.
"""

from pathlib import Path

import numpy as np

from zarai.common.config import parse_config
from zarai.common.logging import get_logger
from zarai.common.run_meta import start_run
from zarai.vision.eval import load_checkpoint
from zarai.vision.tracking import configure, require_run_id
from zarai.vision.train import eval_loader

OPSET = 18


def outputs_match(pytorch: np.ndarray, onnx: np.ndarray, tolerance: float) -> tuple[bool, float]:
    """Return whether all outputs are within tolerance, and the largest absolute difference."""
    difference = float(np.abs(pytorch - onnx).max())
    return difference <= tolerance, difference


def export(cfg, run_dir: Path, log) -> None:
    import mlflow
    import onnx
    import onnxruntime
    import torch

    run_id = require_run_id(cfg)
    configure(cfg)
    with mlflow.start_run(run_id=run_id):
        # On CPU in float32, because bf16 autocast is only a training and GPU inference choice.
        model, checkpoint = load_checkpoint(run_id, "cpu")
        size = checkpoint["image_size"]
        path = run_dir / "model.onnx"
        torch.onnx.export(
            model,
            (torch.randn(2, 3, size, size),),
            str(path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        onnx.checker.check_model(onnx.load(str(path)))

        # Compare on real validation images, at two batch sizes.
        loader = eval_loader(
            cfg, "val", checkpoint["mean"], checkpoint["std"], checkpoint["classes"]
        )
        images = next(iter(loader))[0][: cfg.eval.onnx_check_images]
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            pytorch = model(images).numpy()
        worst = 0.0
        for batch in (images, images[:1]):
            result = session.run(["logits"], {"image": batch.numpy()})[0]
            matched, difference = outputs_match(
                pytorch[: len(batch)], result, cfg.eval.onnx_tolerance
            )
            worst = max(worst, difference)
            if not matched:
                raise AssertionError(
                    f"ONNX differs from PyTorch by {difference} on a batch of {len(batch)}"
                )
        mlflow.log_metric("onnx_max_abs_diff", worst)
        mlflow.log_params({"onnx.opset": OPSET, "onnx.tolerance": cfg.eval.onnx_tolerance})
        mlflow.log_artifact(str(path), "onnx")
        log.info("ONNX matches PyTorch: largest logit difference %.2e", worst)


def main() -> None:
    cfg = parse_config(__doc__)
    run_dir = start_run(cfg, f"vision_export_{cfg.experiment}")
    export(cfg, run_dir, get_logger(__name__, run_dir))


if __name__ == "__main__":
    main()
