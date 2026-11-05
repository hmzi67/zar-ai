"""Find the fixed-position text watermarks in wheat_v1, and remove or add them on an image.

Each layout is learned from train images only: images are averaged after a high-pass filter, so
that scene content cancels out and only text at a fixed position remains.
"""

import numpy as np

GRID = 256
HIGH_PASS_SIZE = 11
WATERMARK_CLASSES = ("mildew", "fusarium_head_blight")
CLUSTERS_PER_CLASS = {"mildew": 4, "fusarium_head_blight": 5}
# A cluster is a layout if its mean map has a peak far above the noise of averaged scene content.
MIN_LAYOUT_PEAK = 15.0
MASK_LEVEL = 0.25
SEARCH_PIXELS = 6


def high_pass(pixels: np.ndarray) -> np.ndarray:
    """Yellow-minus-blue detail of a GRID x GRID RGB array, which is where the yellow text shows."""
    from scipy.ndimage import uniform_filter

    pixels = pixels.astype(np.float32)
    yellow = (pixels[..., 0] + pixels[..., 1]) / 2 - pixels[..., 2]
    return (yellow - uniform_filter(yellow, HIGH_PASS_SIZE)).astype(np.float32)


def load_high_pass(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        resized = image.convert("RGB").resize((GRID, GRID), Image.Resampling.BILINEAR)
    return high_pass(np.asarray(resized))


def learn_layouts(maps_by_class: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Cluster each class's high-pass maps and keep the clusters that average to clear text."""
    from sklearn.cluster import KMeans

    layouts = {}
    for name, maps in maps_by_class.items():
        features = maps[:, ::2, ::2].reshape(len(maps), -1)
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
        clusters = KMeans(CLUSTERS_PER_CLASS[name], n_init=5, random_state=seed).fit_predict(
            features
        )
        means = [maps[clusters == c].mean(axis=0) for c in range(CLUSTERS_PER_CLASS[name])]
        kept = sorted(
            (mean for mean in means if mean.max() >= MIN_LAYOUT_PEAK),
            key=lambda mean: -float(mean.max()),
        )
        layouts.update({f"{name}_{i}": mean for i, mean in enumerate(kept)})
    return layouts


def layout_mask(layout: np.ndarray) -> np.ndarray:
    """Boolean GRID x GRID map of the text pixels of a layout."""
    return layout >= MASK_LEVEL * layout.max()


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1


def match_score(image_map: np.ndarray, layout: np.ndarray) -> float:
    """Best normalised correlation with the layout, allowing a small shift."""
    import cv2

    top, bottom, left, right = _bounds(layout_mask(layout))
    search = image_map[
        max(0, top - SEARCH_PIXELS) : min(GRID, bottom + SEARCH_PIXELS),
        max(0, left - SEARCH_PIXELS) : min(GRID, right + SEARCH_PIXELS),
    ]
    template = layout[top:bottom, left:right]
    return float(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).max())


def _mask_for(image_size: tuple[int, int], layout: np.ndarray, grow: int) -> np.ndarray:
    """Layout mask at an image's own size, grown by `grow` grid pixels, as uint8 0 or 255."""
    import cv2

    kernel = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    grown = cv2.dilate(layout_mask(layout).astype(np.uint8), kernel) * 255
    return cv2.resize(grown, image_size, interpolation=cv2.INTER_NEAREST)


def remove_layout(image, layout: np.ndarray, grow: int = 2):
    """Inpaint a layout's text area. Returns a new PIL image."""
    import cv2
    from PIL import Image

    cv2.setNumThreads(0)
    pixels = np.asarray(image.convert("RGB"))
    result = cv2.inpaint(pixels, _mask_for(image.size, layout, grow), 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result)


def add_layout(image, layout: np.ndarray, color: tuple[int, int, int]):
    """Paint a layout's text onto an image in the given colour. Returns a new PIL image."""
    import cv2
    from PIL import Image

    cv2.setNumThreads(0)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    strength = np.clip(layout / layout.max() * 1.5, 0, 1).astype(np.float32)
    alpha = cv2.resize(strength, image.size, interpolation=cv2.INTER_LINEAR)[..., None]
    result = pixels * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
