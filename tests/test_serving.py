import io

import numpy as np
import pytest

from zarai.fusion.pipeline import DISEASE_NAMES, diagnosis_text
from zarai.vision.serving import InvalidImageError, prepare_image, rank

CLASSES = ["black_rust", "brown_rust", "healthy", "septoria", "yellow_rust"]


def test_rank_sorts_and_abstains_below_the_threshold():
    probs = np.array([0.05, 0.6, 0.05, 0.1, 0.2])
    result = rank(probs, CLASSES, threshold=0.7, top_k=3)
    assert [p["label"] for p in result["top"]] == ["brown_rust", "yellow_rust", "septoria"]
    assert result["abstain"] is True
    assert rank(probs, CLASSES, threshold=0.5, top_k=3)["abstain"] is False


def test_every_class_has_a_name_in_every_language():
    for label in CLASSES:
        assert set(DISEASE_NAMES[label]) == {"ur", "ur-Latn", "en"}


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
