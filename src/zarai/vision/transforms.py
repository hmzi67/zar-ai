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


def build_strong_transforms(
    image_size: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    crop_scale=(0.4, 1.0),
    jitter: float = 0.4,
    jitter_hue: float = 0.05,
    grayscale_p: float = 0.1,
    blur_p: float = 0.2,
    blur_sigma=(0.1, 2.0),
    jpeg_p: float = 0.3,
    jpeg_quality=(30, 90),
):
    """The `strong` train profile: wide crops, strong colour jitter with a hue shift, grayscale,
    blur and JPEG compression. Background replacement happens in the dataset, before this, because
    it needs the leaf mask at the image's own size.
    """
    import torch
    from torchvision.transforms import InterpolationMode, v2

    return v2.Compose(
        [
            v2.ToImage(),  # uint8 tensor: v2.JPEG needs one
            v2.RandomResizedCrop(
                image_size, scale=tuple(crop_scale), interpolation=InterpolationMode.BICUBIC
            ),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter_hue),
            v2.RandomGrayscale(p=grayscale_p),
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=tuple(blur_sigma))], p=blur_p),
            v2.RandomApply([v2.JPEG(quality=tuple(jpeg_quality))], p=jpeg_p),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_train_transforms(profile: str, image_size: int, mean, std, augment):
    """Train transform for an augmentation profile: `basic` is wheat_v1's, `strong` is above."""
    if profile == "basic":
        return build_transforms(image_size, train=True, mean=mean, std=std, **augment.basic)
    if profile == "strong":
        options = {k: v for k, v in augment.strong.items() if k != "replace_background_p"}
        return build_strong_transforms(image_size, mean=mean, std=std, **options)
    raise ValueError(f"unknown augmentation profile {profile!r}: use basic or strong")
