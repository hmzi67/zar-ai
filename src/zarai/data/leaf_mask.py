"""A simple colour-based plant mask, for masking or replacing image backgrounds.

Green and yellow-green pixels are taken as plant. Lesions inside a leaf (brown, black or pale) are
kept by closing small gaps and filling enclosed holes. It is a heuristic, not a trained segmenter:
lesions on a leaf edge can be cut off, and brown soil or dead leaves count as background. When the
mask finds almost no plant, the whole image is kept instead, so a failed mask never blanks a photo.
"""

import numpy as np

# Hue in degrees. 35-170 covers yellow-green to cyan-green, which includes stripe rust's yellow.
MIN_HUE, MAX_HUE = 35, 170
MIN_SATURATION = 0.15
MIN_VALUE = 0.12
CLOSING_SHARE = 0.01  # closing radius, as a share of the long side
MIN_COMPONENT_SHARE = 0.005  # drop plant blobs smaller than this share of the image
MIN_PLANT_SHARE = 0.05  # below this, the mask is treated as failed and covers everything


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def leaf_mask(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (bool HxW mask, fell_back). `rgb` is uint8 HxWx3."""
    from PIL import Image
    from scipy import ndimage

    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"), dtype=np.float32) / 255
    hue, saturation, value = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    plant = (
        (hue >= MIN_HUE) & (hue <= MAX_HUE) & (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)
    )
    radius = max(1, round(CLOSING_SHARE * max(rgb.shape[:2])))
    plant = ndimage.binary_closing(plant, structure=_disk(radius), border_value=0)
    plant = ndimage.binary_fill_holes(plant)
    components, n = ndimage.label(plant)
    if n:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        plant = (sizes >= MIN_COMPONENT_SHARE * plant.size)[components]
    if plant.mean() < MIN_PLANT_SHARE:
        return np.ones(rgb.shape[:2], dtype=bool), True
    return plant, False
