import io

import numpy as np
import pytest

from zarai.fusion.pipeline import CONFIRM, DISEASE_NAMES, diagnosis_text
from zarai.vision.plant_filter import threshold_keeping
from zarai.vision.serving import InvalidImageError, band_of, prepare_image, rank
from zarai.vision.serving_bands import band_table

CLASSES = ["black_rust", "brown_rust", "healthy", "septoria", "yellow_rust"]


BANDS = {"answer": 0.9, "likely": 0.6}


def test_rank_sorts_and_bands_the_top_probability():
    probs = np.array([0.05, 0.7, 0.05, 0.1, 0.1])
    result = rank(probs, CLASSES, BANDS, top_k=5)
    assert [p["label"] for p in result["top"]][:2] == ["brown_rust", "septoria"]
    assert len(result["top"]) == 5
    assert (result["band"], result["abstain"]) == ("likely", False)


@pytest.mark.parametrize(
    "prob, band",
    [(0.95, "confident"), (0.9, "confident"), (0.75, "likely"), (0.6, "likely"), (0.59, "unsure")],
)
def test_band_edges(prob, band):
    assert band_of(prob, BANDS) == band


def test_band_table_counts_and_scores_each_band():
    probs = np.array([[0.95, 0.05], [0.7, 0.3], [0.2, 0.8], [0.55, 0.45]])
    labels = np.array([0, 1, 1, 0])
    rows = {r["band"]: r for r in band_table(probs, labels, BANDS)}
    assert (rows["confident"]["n"], rows["confident"]["accuracy"]) == (1, 1.0)
    assert (rows["likely"]["n"], rows["likely"]["accuracy"]) == (2, 0.5)
    assert (rows["unsure"]["n"], rows["unsure"]["accuracy"]) == (1, 1.0)


def test_plant_threshold_keeps_the_requested_share():
    probs = np.linspace(0, 1, 201)
    threshold = threshold_keeping(probs, 0.9)
    assert (probs >= threshold).mean() >= 0.9
    assert (probs >= threshold + 0.01).mean() < 0.9


def test_every_class_has_a_name_and_a_confirming_question_in_every_language():
    for label in CLASSES:
        assert set(DISEASE_NAMES[label]) == {"ur", "ur-Latn", "en"}
        assert set(CONFIRM[label]) == {"ur", "ur-Latn", "en"}


def test_a_likely_diagnosis_says_probably_and_asks_to_confirm():
    top = [{"label": "septoria", "prob": 0.72}, {"label": "brown_rust", "prob": 0.2}]
    text = diagnosis_text(top, "en", band="likely")
    assert text.startswith("Image model result: probably Septoria leaf blotch, 72%.")
    assert "To confirm: Are there tan or brown blotches" in text
    assert "To confirm" not in diagnosis_text(top, "en")


def test_diagnosis_text_names_the_top_class_in_the_question_language():
    top = [{"label": "brown_rust", "prob": 0.8}, {"label": "healthy", "prob": 0.15}]
    assert diagnosis_text(top, "ur-Latn").startswith("Tasveer ke model ka nateeja: Bhoori kungi")
    assert "بھوری کنگی" in diagnosis_text(top, "ur")
    assert diagnosis_text([{"label": "mystery", "prob": 1.0}], "en").startswith(
        "Image model result: mystery, 100%."
    )


def test_uploads_are_downscaled_like_the_training_images():
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGBA", (1024, 600), (0, 128, 0, 255)).save(buffer, "PNG")
    image = prepare_image(buffer.getvalue(), long_side=512, jpeg_quality=92)
    assert image.mode == "RGB" and image.size == (512, 300)


def test_an_unreadable_upload_is_rejected():
    pytest.importorskip("PIL.Image")
    with pytest.raises(InvalidImageError):
        prepare_image(b"not an image", long_side=512, jpeg_quality=92)


def test_negligible_alternatives_are_left_out_of_the_diagnosis():
    top = [{"label": "septoria", "prob": 0.99}, {"label": "healthy", "prob": 0.002}]
    assert "Other possibilities" not in diagnosis_text(top, "en")
