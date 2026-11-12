"""Serve a trained wheat_v2 run: image bytes in, calibrated top-k classes and a confidence band out.

The checkpoint and temperature come from one MLflow run, so what is served is what was evaluated.
A CLIP plant filter runs first, and non-plant photos never reach the disease model. The calibrated
top probability then picks a band: confident, likely or unsure (configs/vision/serving.yaml). Uploaded photos get the same preprocessing as the training
images (upright RGB, downscaled to the long side, saved as JPEG) and then the run's eval transform.
"""

import io

import numpy as np
from omegaconf import OmegaConf

# Heavy imports (torch, PIL, scipy) stay inside functions, so the API can import this module
# without the vision dependencies.


class InvalidImageError(ValueError):
    """The upload could not be decoded as an image."""


def load_run_model(run_id: str, device: str):
    """Rebuild a wheat_v2 run's best checkpoint. Returns (model, checkpoint, run config)."""
    import torch

    from zarai.vision import backbones
    from zarai.vision.tracking import artifact_path

    checkpoint = torch.load(
        artifact_path(run_id, "checkpoints/best.pt"), map_location="cpu", weights_only=True
    )
    cfg = OmegaConf.create(checkpoint["config"])
    # linear and lora checkpoints hold only the trained tensors, on top of the pinned weights.
    model = backbones.build(
        cfg, len(checkpoint["classes"]), pretrained=checkpoint["trainable_only"]
    )
    backbones.load_state(model, checkpoint["state_dict"], checkpoint["mode"])
    return model.to(device), checkpoint, cfg


def prepare_image(data: bytes, long_side: int, jpeg_quality: int):
    """Decode an upload the way wheat_v2 exported its training images."""
    from PIL import Image, UnidentifiedImageError

    from zarai.data.clean_wheat import _open_rgb, scaled_size

    try:
        image, _ = _open_rgb(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError("the upload is not a readable image") from error
    size = scaled_size(*image.size, long_side)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as saved:
        return saved.convert("RGB")


BANDS = ("confident", "likely", "unsure")


def band_of(prob: float, bands) -> str:
    """confident (>= bands.answer), likely (>= bands.likely) or unsure."""
    if prob >= bands["answer"]:
        return "confident"
    return "likely" if prob >= bands["likely"] else "unsure"


def rank(probs: np.ndarray, classes: list[str], bands, top_k: int) -> dict:
    """{"top": [{"label", "prob"}] best first, "band", "abstain": band is unsure}."""
    order = np.argsort(-probs)[:top_k]
    top = [{"label": classes[i], "prob": round(float(probs[i]), 4)} for i in order]
    band = band_of(float(probs[order[0]]), bands)
    return {"top": top, "band": band, "abstain": band == "unsure"}


def not_plant(plant_prob: float) -> dict:
    return {"top": [], "band": "not_plant", "abstain": True, "plant_prob": round(plant_prob, 4)}


class WheatClassifier:
    def __init__(self, cfg):
        import torch

        from zarai.vision.tracking import set_tracking_uri
        from zarai.vision.transforms import build_transforms

        set_tracking_uri(cfg.mlflow.tracking_uri)
        import mlflow

        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.temperature = float(mlflow.get_run(cfg.run_id).data.metrics["temperature"])
        self.bands = {"answer": float(cfg.bands.answer), "likely": float(cfg.bands.likely)}
        if not 0 < self.bands["likely"] <= self.bands["answer"] <= 1:
            raise ValueError(f"bands need 0 < likely <= answer <= 1, got {self.bands}")
        self.plant_filter = None
        if cfg.get("plant_filter") and cfg.plant_filter.enabled:
            from zarai.vision.plant_filter import PlantFilter

            self.plant_filter = PlantFilter(cfg.plant_filter, self.device)
        self.model, checkpoint, run_cfg = load_run_model(cfg.run_id, self.device)
        self.model.eval()
        self.classes = list(checkpoint["classes"])
        self.transform = build_transforms(
            run_cfg.data.image_size,
            train=False,
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            eval_resize=run_cfg.data.eval_resize,
        )
        self.long_side, self.jpeg_quality = cfg.long_side, cfg.jpeg_quality
        self.top_k = cfg.top_k
        self.run_id = cfg.run_id

    def classify(self, data: bytes) -> dict:
        import torch

        from zarai.vision.calibrate import apply_temperature

        image = prepare_image(data, self.long_side, self.jpeg_quality)
        plant_prob = None
        if self.plant_filter is not None:
            is_plant, plant_prob = self.plant_filter.check(image)
            if not is_plant:
                return not_plant(plant_prob)
        batch = self.transform(image).unsqueeze(0).to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"),
        ):
            logits = self.model(batch).float().cpu().numpy()
        probs = apply_temperature(logits, self.temperature)[0]
        result = rank(probs, self.classes, self.bands, self.top_k)
        if plant_prob is not None:
            result["plant_prob"] = round(plant_prob, 4)
        return result
