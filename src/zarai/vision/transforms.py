"""Train and eval image transforms."""


def build_transforms(image_size: int, train: bool):
    import torch
    from torchvision.transforms import v2

    if train:
        resize = [
            v2.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        resize = [v2.Resize(int(image_size * 1.14)), v2.CenterCrop(image_size)]
    return v2.Compose(
        [
            *resize,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
