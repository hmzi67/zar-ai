"""Train and eval image transforms."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int,
    train: bool,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    eval_resize: int = 256,
    crop_scale=(0.6, 1.0),
    randaugment_ops: int = 2,
    randaugment_magnitude: int = 9,
    jitter: float = 0.1,
    jitter_hue: float = 0.02,
):
    """Train: random crop, flip, RandAugment, mild colour jitter. Eval: resize and centre crop."""
    import torch
    from torchvision.transforms import InterpolationMode, v2

    bicubic = InterpolationMode.BICUBIC
    if train:
        steps = [
            v2.RandomResizedCrop(image_size, scale=tuple(crop_scale), interpolation=bicubic),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude),
            v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter_hue),
        ]
    else:
        steps = [v2.Resize(eval_resize, interpolation=bicubic), v2.CenterCrop(image_size)]
    return v2.Compose(
        [
            *steps,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ]
    )
