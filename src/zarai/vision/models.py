"""timm model factory, pinned to exact Hugging Face revisions, and the serving classifier loader."""


def create_pinned_model(name: str, revision: str, **kwargs):
    """Create a timm model with pretrained weights from one exact Hugging Face commit."""
    import timm
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(f"timm/{name}", "model.safetensors", revision=revision)
    return timm.create_model(
        name, pretrained=True, pretrained_cfg_overlay={"file": weights, "source": ""}, **kwargs
    )


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    revision: str | None = None,
    image_size: int | None = None,
):
    """Build a classifier. Pretrained weights need a revision, so a run can be repeated exactly."""
    import timm

    kwargs = {"num_classes": num_classes}
    if image_size is not None and "vit" in name:
        kwargs["img_size"] = image_size
    if pretrained:
        if revision is None:
            raise ValueError(f"{name}: pretrained weights need a Hugging Face revision")
        model = create_pinned_model(name, revision, **kwargs)
    else:
        model = timm.create_model(name, pretrained=False, **kwargs)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    return model


def load_classifier(cfg):
    """Return an object with classify(image_bytes) -> {"top": [{"label", "prob"}], "abstain": bool}."""
    raise NotImplementedError(
        "TODO: load trained checkpoint, apply plant filter, temperature and abstention threshold"
    )
