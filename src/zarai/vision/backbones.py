"""wheat_v2 backbones and fine-tuning modes, from the `backbones` and `model` config sections.

Every backbone is a timm model with weights pinned to one Hugging Face commit (models.build_model).
Modes:
- linear: frozen backbone, only the classifier head trains.
- full: everything trains; ViTs use layer-wise LR decay (timm's param groups).
- lora: LoRA adapters (peft) on each block's attention qkv and proj, plus the head. ViTs only.
"""

MODES = ("linear", "lora", "full")
LORA_TARGETS = r"blocks\.\d+\.attn\.(qkv|proj)"


def backbone_spec(cfg):
    name = cfg.model.backbone
    if name not in cfg.backbones:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(cfg.backbones)}")
    spec = cfg.backbones[name]
    if cfg.model.mode not in spec.modes:
        raise ValueError(f"{name} supports modes {list(spec.modes)}, not {cfg.model.mode!r}")
    return spec


def normalisation(spec, image_size: int) -> tuple[list[float], list[float]]:
    """The backbone's own mean and std, read from timm without downloading weights."""
    from zarai.vision.models import build_model

    probe = build_model(spec.timm, num_classes=1, pretrained=False, image_size=image_size)
    return list(probe.pretrained_cfg["mean"]), list(probe.pretrained_cfg["std"])


def build(cfg, num_classes: int, pretrained: bool = True):
    """Create the model for cfg.model.backbone in cfg.model.mode, with trainable flags set."""
    from zarai.vision.models import build_model

    spec, mode = backbone_spec(cfg), cfg.model.mode
    model = build_model(
        spec.timm,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=mode == "linear",
        revision=spec.hf_revision,
        image_size=cfg.data.image_size,
    )
    if mode == "lora":
        from peft import LoraConfig, inject_adapter_in_model

        lora = cfg.model.lora
        config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=LORA_TARGETS,
            bias="none",
        )
        model = inject_adapter_in_model(config, model)
        head = {f"head.{n}" for n, _ in model.get_classifier().named_parameters()}
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name or name in head
        if not any("lora_" in name for name, _ in model.named_parameters()):
            raise ValueError(f"{spec.timm}: no attention qkv/proj layers for LoRA")
    if spec.get("grad_checkpointing") and mode != "linear":
        model.set_grad_checkpointing(True)
    return model


def param_groups(model, cfg) -> list[dict]:
    """Optimiser groups. Full ViT fine-tuning scales each block's LR by layer_decay per block."""
    from timm.optim import param_groups_layer_decay

    spec, mode = backbone_spec(cfg), cfg.model.mode
    lr, weight_decay = cfg.train.lr, cfg.train.weight_decay
    if mode == "full" and spec.vit and cfg.model.layer_decay:
        groups = param_groups_layer_decay(
            model,
            weight_decay=weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=cfg.model.layer_decay,
        )
        for group in groups:
            group["lr"] = lr * group.pop("lr_scale")
        return groups
    # As wheat_v1: one group, weight decay on every trainable parameter.
    return [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]


def checkpoint_state(model, mode: str) -> dict:
    """CPU copy of what a checkpoint needs: everything for full, else only the trained tensors.

    linear and lora checkpoints rebuild the pinned pretrained backbone and load these on top.
    """
    if mode == "full":
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trained = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trained}


def load_state(model, state: dict, mode: str) -> None:
    missing, unexpected = model.load_state_dict(state, strict=mode == "full")
    trained = {name for name, p in model.named_parameters() if p.requires_grad}
    if unexpected or (mode != "full" and trained - set(state)):
        raise ValueError(f"checkpoint mismatch: unexpected {unexpected[:3]}, missing {missing[:3]}")
