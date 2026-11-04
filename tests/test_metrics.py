import math

import numpy as np
import pytest
from scipy.special import softmax

from zarai.data.agreement import cohen_kappa
from zarai.eval.expert_ratings import bootstrap_ci, compare_systems, question_means, summarise
from zarai.eval.retrieval_metrics import evaluate_retrieval, recall_at_k, reciprocal_rank
from zarai.eval.sus import score_rows, sus_score
from zarai.eval.vision_metrics import (
    classification_summary,
    coverage_accuracy_curve,
    expected_calibration_error,
    macro_f1,
)
from zarai.vision.calibrate import (
    apply_temperature,
    fit_temperature,
    predict_with_abstention,
    threshold_for_coverage,
)


def test_recall_and_mrr():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 1) == 0
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0


def test_evaluate_retrieval_skips_queries_without_relevant_chunks():
    metrics = evaluate_retrieval([(["a"], ["a"]), (["a"], ["b"]), (["a"], [])], [1])
    assert metrics == {"recall@1": 0.5, "mrr": 0.5, "n_queries": 2}
    with pytest.raises(ValueError):
        evaluate_retrieval([(["a"], [])], [1])


def test_macro_f1_and_ece():
    labels = np.array([0, 0, 1, 1])
    assert macro_f1(labels, labels) == 1.0
    assert macro_f1(labels, np.array([0, 0, 0, 0])) == pytest.approx((2 / 3 + 0) / 2)

    certain = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(certain, np.array([0, 1])) == 0
    assert expected_calibration_error(certain, np.array([1, 0])) == 1
    half = np.array([[0.75, 0.25]] * 4)
    assert expected_calibration_error(half, np.array([0, 0, 0, 1])) == pytest.approx(0)


def test_coverage_accuracy_curve():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    curve = coverage_accuracy_curve(probs, np.array([0, 1, 1]), np.array([0.0, 0.7, 0.95]))
    assert curve[0]["coverage"] == 1 and curve[0]["accuracy"] == pytest.approx(2 / 3)
    assert curve[1]["coverage"] == pytest.approx(2 / 3) and curve[1]["accuracy"] == 1
    assert curve[2] == {"threshold": 0.95, "coverage": 0, "accuracy": None}


def test_classification_summary_keys():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    summary = classification_summary(logits, np.array([0, 1]), ["a", "b"])
    assert summary["macro_f1"] == 1 and summary["per_class"]["a"]["f1-score"] == 1


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 4)) * 2
    probs = softmax(true_logits, axis=1)
    labels = np.array([rng.choice(4, p=p) for p in probs])
    temperature = fit_temperature(true_logits * 3, labels)
    assert temperature == pytest.approx(3, rel=0.1)

    before = expected_calibration_error(softmax(true_logits * 3, axis=1), labels)
    after = expected_calibration_error(apply_temperature(true_logits * 3, temperature), labels)
    assert after < before


@pytest.mark.parametrize("target", [0.1, 0.5, 0.8, 1.0])
def test_threshold_meets_target_coverage(target):
    probs = softmax(np.random.default_rng(1).normal(size=(100, 3)), axis=1)
    threshold = threshold_for_coverage(probs, target)
    preds = predict_with_abstention(probs, threshold)
    assert (preds != -1).mean() == pytest.approx(target)


def test_threshold_rejects_bad_coverage():
    with pytest.raises(ValueError):
        threshold_for_coverage(np.array([[0.5, 0.5]]), 0)


def test_kappa_plain_and_weighted():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1
    # Observed agreement 0.5, chance agreement 0.5.
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0)
    near = cohen_kappa([1, 2, 3, 4], [1, 2, 3, 3], weights="quadratic")
    far = cohen_kappa([1, 2, 3, 4], [1, 2, 3, 1], weights="quadratic")
    assert far < near
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])


def rating(qid, system, correctness, rater="r1", unsafe="n"):
    return {
        "qid": qid,
        "system_id": system,
        "correctness_1_5": str(correctness),
        "actionability_1_5": "3",
        "fluency_1_5": "",
        "faithful_yn": "y",
        "unsafe_yn": unsafe,
        "rater_id": rater,
    }


def test_expert_ratings_average_raters_per_question():
    rows = [rating("q1", "S1", 2), rating("q1", "S1", 4, rater="r2"), rating("q2", "S1", 5)]
    assert question_means(rows, "correctness_1_5") == {"S1": {"q1": 3.0, "q2": 5.0}}
    assert "S1" not in question_means(rows, "fluency_1_5")

    summary = summarise(rows)["S1"]
    assert (
        summary["correctness_1_5"]["mean"] == 4 and summary["correctness_1_5"]["n_questions"] == 2
    )
    assert summary["faithful_yn"]["mean"] == 1 and summary["unsafe_yn"]["mean"] == 0
    assert "fluency_1_5" not in summary


def test_expert_ratings_reject_bad_values():
    with pytest.raises(ValueError):
        question_means([rating("q1", "S1", 6)], "correctness_1_5")
    with pytest.raises(ValueError):
        question_means([rating("q1", "S1", 3, unsafe="maybe")], "unsafe_yn")


def test_bootstrap_ci_brackets_mean_and_collapses_on_constant():
    low, high = bootstrap_ci(list(np.random.default_rng(0).normal(3, 1, 200)))
    assert low < 3 < high
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)


def test_wilcoxon_paired_on_shared_questions():
    rows = [rating(f"q{i}", "A", 5) for i in range(10)]
    rows += [rating(f"q{i}", "B", 1 + i % 3) for i in range(10)]
    rows.append(rating("only_a", "A", 1))
    result = compare_systems(rows, "A", "B", "correctness_1_5")
    assert result["n_questions"] == 10 and result["pvalue"] < 0.05
    assert compare_systems(rows, "A", "A", "correctness_1_5")["pvalue"] == 1


def test_sus_score_known_values():
    assert sus_score([5, 1] * 5) == 100
    assert sus_score([1, 5] * 5) == 0
    assert sus_score([3] * 10) == 50
    with pytest.raises(ValueError):
        sus_score([3] * 9)
    with pytest.raises(ValueError):
        sus_score([6] + [3] * 9)
    assert math.isclose(sus_score([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]), 75)


def test_sus_rows_skip_blank_and_reject_partial():
    filled = {f"sus_q{i}": "3" for i in range(1, 11)}
    blank = {f"sus_q{i}": "" for i in range(1, 11)}
    assert score_rows([filled, blank]) == [50]
    with pytest.raises(ValueError):
        score_rows([{**filled, "sus_q3": ""}])
