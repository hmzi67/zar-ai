"""Grad-CAM for the timm CNNs and ViTs used here, with no extra dependency."""

import numpy as np


def cam_layer(model):
    """Return the last layer that still has spatial structure."""
    if hasattr(model, "stages"):  # ConvNeXt
        return model.stages[-1]
    if hasattr(model, "num_prefix_tokens"):  # ViT
        # After the last block, the head reads only the CLS token, so patch tokens get no gradient.
        # Before that block's attention, they reach the CLS token and so have a real gradient.
        return model.blocks[-1].norm1
    if hasattr(model, "bn2"):  # EfficientNet: bn2 also applies the activation after conv_head.
        # conv_head's raw output comes before BatchNorm, and gives maps unrelated to the image.
        return model.bn2
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")


def _spatial(model, activations):
    """Turn ViT tokens into (batch, channels, height, width). CNN outputs already are."""
    if not hasattr(model, "num_prefix_tokens"):
        return activations
    tokens = activations[:, model.num_prefix_tokens :]
    side = int(tokens.shape[1] ** 0.5)
    if side * side != tokens.shape[1]:
        raise ValueError(f"{tokens.shape[1]} patch tokens do not form a square")
    return tokens.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)


def grad_cam(model, images, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cams scaled 0-1 at input size, predicted classes, probabilities of those classes)."""
    import torch

    model.eval()
    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["activations"] = output

    handle = cam_layer(model).register_forward_hook(hook)
    try:
        # The input needs a gradient so that a frozen backbone still produces one at the layer.
        inputs = images.to(device).requires_grad_(True)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        model.zero_grad(set_to_none=True)
        logits.gather(1, preds[:, None]).sum().backward()
    finally:
        handle.remove()

    activations = _spatial(model, captured["activations"])
    gradients = _spatial(model, captured["activations"].grad)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cams = torch.nn.functional.interpolate(
        cams, size=images.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    peak = cams.flatten(1).max(dim=1).values.clamp_min(1e-12)
    cams = cams / peak[:, None, None]
    probs = logits.softmax(dim=1).gather(1, preds[:, None])[:, 0]
    model.zero_grad(set_to_none=True)
    return cams.detach().cpu().numpy(), preds.cpu().numpy(), probs.detach().cpu().numpy()


def cam_mass_on_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of a CAM's total activation that falls inside a boolean mask."""
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else 0.0
