import numpy as np
import pytest

from zarai.vision import vlm_hybrid as vh

CLASSES = ["brown_rust", "healthy", "yellow_rust"]


def test_rule_answers_only_on_agreement():
    dino = np.array([0, 1, 2, 2])
    gemma = np.array([0, 2, 3, 2])  # 3 = unsure
    assert vh.rule(dino, gemma).tolist() == [0, vh.DEFER, vh.DEFER, 2]


def test_with_threshold_defers_low_confidence():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    assert vh.with_threshold(probs, 0.5).tolist() == [0, vh.DEFER]


def test_outcome_recall_answered_and_all():
    labels = np.array([2, 2, 2, 1])
    answer = np.array([2, vh.DEFER, 0, 1])
    o = vh.outcome(labels, answer, CLASSES, ["yellow_rust", "brown_rust"])
    assert o["coverage"] == 0.75 and o["accuracy"] == pytest.approx(2 / 3)
    assert o["yellow_rust_recall_answered"] == 0.5
    assert o["yellow_rust_recall_all"] == pytest.approx(1 / 3)
    assert o["brown_rust_recall_all"] is None


def test_fold_of():
    assert vh.fold_of("pooled_test") == ("pooled", "split_pooled", "heldout")
    assert vh.fold_of("nwrd") == ("pooled", "split_pooled", "nwrd")
    assert vh.fold_of("loso:kaggle") == ("kaggle", "split_loso_kaggle", "heldout")


def test_stacker_handles_classes_missing_from_training():
    pytest.importorskip("sklearn")
    x = np.array([[0.0], [0.1], [1.0], [0.9]])
    y = np.array([0, 0, 2, 2])
    probs = vh.fit_stacker(x, y, 3, type("S", (), {"C": 1.0, "max_iter": 100})())(x)
    assert probs.shape == (4, 3) and np.allclose(probs[:, 1], 0)
