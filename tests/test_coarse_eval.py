import numpy as np
import pytest

from zarai.vision import coarse_eval as ce

FINE = ["black_rust", "brown_rust", "healthy", "septoria", "yellow_rust"]
GROUPS = {
    "healthy": "healthy",
    "septoria": "septoria",
    "yellow_rust": "rust",
    "brown_rust": "rust",
    "black_rust": "rust",
}
COARSE = ["healthy", "rust", "septoria"]


def test_coarse_map_merges_rusts_and_keeps_no_label():
    m = ce.coarse_map(FINE, GROUPS, COARSE)
    assert m.tolist() == [1, 1, 0, 2, 1, 3]


def test_outcome_counts_no_label_and_deferred_as_unanswered():
    labels = np.array([1, 1, 0, 2])
    answer = np.array([1, 3, ce.DEFER, 0])  # 3 = no label
    o = ce.outcome(labels, answer, 3, COARSE)
    assert o["coverage"] == 0.5 and o["accuracy"] == 0.5
    assert o["recall_rust"] == 0.5 and o["recall_healthy"] == 0.0
    assert o["recall_septoria"] == pytest.approx(0.0)


def test_confusion_and_rust_stats():
    labels = np.array([0, 0, 1, 2, 2])
    answer = np.array([1, 0, 1, 1, ce.DEFER])
    o = ce.outcome(labels, answer, 3, COARSE)
    assert o["confusion"] == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1]]
    st = ce.rust_stats(np.array(o["confusion"], dtype=float), COARSE)
    assert st["rust_answers"] == 3 and st["rust_images"] == 1 and st["rust_ratio"] == 3
    assert st["rust_precision"] == pytest.approx(1 / 3)
    assert st["non_rust_called_rust"] == 0.5 and st["healthy_recall"] == 0.5
    assert st["septoria_to_rust"] == 0.5 and st["septoria_to_not_answered"] == 0.5
