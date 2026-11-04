"""timm model factory and the serving classifier loader."""


def build_model(
    name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False
):
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
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
