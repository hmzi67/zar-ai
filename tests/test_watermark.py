import numpy as np
import pytest

from zarai.vision.watermark_check import DETECTION_THRESHOLD, recall, split_metrics


def text_layout():
    """A fixed pattern of bright bars on a flat background, on the 256 x 256 grid."""
    layout = np.zeros((256, 256), dtype=np.float32)
    for row in range(100, 160, 12):
        layout[row : row + 4, 60:200] = 50.0
    return layout


def test_recall_uses_only_the_class_inside_the_subset():
    labels = np.array([0, 0, 0, 1])
    preds = np.array([0, 1, 0, 1])
    mask = np.array([True, True, False, True])
    assert recall(preds, labels, mask, 0) == 0.5
    assert recall(preds, labels, np.zeros(4, dtype=bool), 0) is None


def test_match_score_separates_marked_images_and_tolerates_a_small_shift():
    pytest.importorskip("cv2")
    from zarai.vision.watermark import match_score

    layout = text_layout()
    noise = np.random.default_rng(0).normal(0, 5, (256, 256)).astype(np.float32)
    assert match_score(noise, layout) < DETECTION_THRESHOLD
    assert match_score(noise + layout, layout) > DETECTION_THRESHOLD
    shifted = np.roll(layout, (3, -3), axis=(0, 1))
    assert match_score(noise + shifted, layout) > DETECTION_THRESHOLD


def test_learn_layouts_keeps_text_clusters_and_drops_scene_noise():
    pytest.importorskip("sklearn")
    from zarai.vision.watermark import learn_layouts

    rng = np.random.default_rng(0)
    layout = text_layout()
    noise = rng.normal(0, 5, (60, 256, 256)).astype(np.float32)
    marked = noise[:30] + layout
    maps = np.concatenate([marked, noise[30:]])
    found = learn_layouts({"mildew": maps})
    assert list(found) == ["mildew_0"]
    assert np.corrcoef(found["mildew_0"].ravel(), layout.ravel())[0, 1] > 0.95


def test_remove_and_add_change_only_the_text_area_and_keep_the_size():
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    from zarai.vision.watermark import add_layout, remove_layout

    layout = text_layout()
    flat = image_module.new("RGB", (512, 384), (40, 90, 40))
    painted = add_layout(flat, layout, (150, 160, 80))
    assert painted.size == flat.size
    pixels = np.asarray(painted, dtype=float)
    assert pixels[0, 0].tolist() == [40, 90, 40]  # Outside the text.
    assert pixels[int(101 * 384 / 256), 256, 2] > 60  # Inside the text.

    cleaned = np.asarray(remove_layout(painted, layout), dtype=float)
    base = np.asarray(flat, dtype=float)
    assert cleaned.shape == base.shape
    assert np.abs(cleaned - base).mean() < np.abs(pixels - base).mean() / 5


def test_split_metrics_detects_a_model_that_only_reads_the_watermark():
    """A fake model that predicts class 1 for marked or painted images, and class 0 otherwise."""
    classes = ["aphid", "fusarium_head_blight", "mildew"]
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 2])
    flagged = np.array([True, True, False, False, False, False, False, False])
    assigned = ["fusarium_head_blight_0", "fusarium_head_blight_0", "", "", "", "", "", ""]
    layouts = {"fusarium_head_blight_0": text_layout()}

    def predict_with(edits):
        logits = np.zeros((len(labels), 3))
        marked = flagged.copy()
        for index, (mode, _name) in edits.items():
            marked[index] = mode == "add"
        logits[marked, 1] = 5.0
        logits[~marked, 0] = 5.0
        return logits

    metrics = split_metrics(labels, classes, flagged, assigned, layouts, predict_with)
    assert metrics["fusarium_head_blight_recall_watermarked"] == 1.0
    assert metrics["fusarium_head_blight_recall_clean"] == 0.0
    assert metrics["fusarium_head_blight_recall_watermarked_removed"] == 0.0
    assert metrics["fusarium_head_blight_n_watermarked"] == 2
    assert metrics["n_injection_images"] == 3
    assert metrics["inject_fusarium_head_blight_0_rate_before"] == 0.0
    assert metrics["inject_fusarium_head_blight_0_rate_after"] == 1.0
    assert metrics["accuracy_clean"] == metrics["accuracy_clean_placebo"]
