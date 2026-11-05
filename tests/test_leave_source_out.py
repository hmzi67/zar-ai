import numpy as np
import pytest
from omegaconf import OmegaConf

from zarai.vision.datasets import SplitCsvDataset
from zarai.vision.leave_source_out import (
    TBD,
    clean_test_metrics,
    heldout_metrics,
    mean_sd,
    render,
    subset_recall,
)
from zarai.vision.train import excluded_paths

CLASSES = ["aphid", "fusarium_head_blight", "mildew"]


def one_hot(preds):
    logits = np.zeros((len(preds), 3))
    logits[np.arange(len(preds)), preds] = 5.0
    return logits


def test_dataset_leaves_out_excluded_paths_and_keeps_all_classes(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(
        "path,label,split\na.jpg,smut,train\nb.jpg,mite,train\nc.jpg,mite,train\n",
        encoding="utf-8",
    )
    dataset = SplitCsvDataset(csv_path, "train", exclude_paths={"a.jpg"})
    assert [row["path"] for row in dataset.rows] == ["b.jpg", "c.jpg"]
    assert dataset.classes == ["mite", "smut"]
    assert dataset.labels == [0, 0]


def exclusion_config(tmp_path, expected=2, splits=("train", "val")):
    path = tmp_path / "marks.csv"
    path.write_text("path,layout\na.jpg,x\nb.jpg,\nc.jpg,y\n", encoding="utf-8")
    return OmegaConf.create(
        {
            "data": {
                "exclude_csv": str(path),
                "exclude_splits": list(splits),
                "exclude_expected": expected,
            }
        }
    )


def test_excluded_paths_applies_only_to_the_chosen_splits(tmp_path):
    cfg = exclusion_config(tmp_path)
    assert excluded_paths(cfg, "train") == {"a.jpg", "c.jpg"}
    assert excluded_paths(cfg, "val") == {"a.jpg", "c.jpg"}
    assert excluded_paths(cfg, "test") is None
    assert excluded_paths(OmegaConf.create({"data": {}}), "train") is None


def test_excluded_paths_fails_if_the_count_changes(tmp_path):
    with pytest.raises(ValueError, match="expected 923"):
        excluded_paths(exclusion_config(tmp_path, expected=923), "train")


def test_subset_recall():
    preds, labels = np.array([1, 1, 0, 2]), np.array([1, 1, 1, 2])
    assert subset_recall(preds, labels, 1) == pytest.approx(2 / 3)
    assert subset_recall(preds, labels, 1, np.array([True, False, True, True])) == 0.5
    assert subset_recall(preds, labels, 0) is None


def test_heldout_metrics_split_recall_by_class_split_layout_and_background():
    labels = np.array([2, 2, 2, 1, 1])
    preds = np.array([2, 2, 0, 1, 0])
    metrics = heldout_metrics(
        one_hot(preds),
        labels,
        CLASSES,
        np.array(["train", "test", "test", "val", "test"]),
        np.array(["mildew_0"] * 3 + ["fusarium_head_blight_0"] * 2),
        np.array([True, True, False, False, False]),
        temperature=1.0,
        threshold=0.5,
    )
    assert metrics["heldout_n"] == 5
    assert metrics["heldout_accuracy"] == pytest.approx(3 / 5)
    assert metrics["heldout_recall_mildew"] == pytest.approx(2 / 3)
    assert metrics["heldout_recall_fusarium_head_blight"] == 0.5
    assert metrics["heldout_recall_mildew_from_train"] == 1.0
    assert metrics["heldout_recall_mildew_from_test"] == 0.5
    assert "heldout_recall_mildew_from_val" not in metrics
    assert metrics["heldout_recall_mildew_dark"] == 1.0
    assert metrics["heldout_recall_mildew_light"] == 0.0
    assert metrics["heldout_n_mildew_dark"] == 2
    assert metrics["heldout_recall_fusarium_head_blight_layout_fusarium_head_blight_0"] == 0.5
    assert metrics["heldout_coverage_at_threshold"] == 1.0


def test_clean_test_metrics_separate_the_source():
    labels = np.array([0, 1, 2, 1, 2])
    preds = np.array([0, 1, 2, 0, 0])
    marked = np.array([False, False, False, True, True])
    metrics = clean_test_metrics(one_hot(preds), labels, marked, CLASSES)
    assert metrics["test_macro_f1_clean"] == 1.0
    assert metrics["test_accuracy_clean"] == 1.0
    assert metrics["test_recall_mildew_clean"] == 1.0
    assert metrics["test_recall_mildew_marked"] == 0.0


def test_mean_sd_needs_every_seed():
    assert mean_sd([0.8, 0.9]) == "0.8500 ± 0.0707"
    assert mean_sd([0.8, None]) == TBD
    assert mean_sd([]) == TBD
    assert mean_sd([0.8]) == "0.8000"


def test_report_without_runs_has_no_numbers():
    text = "\n".join(render({}, {}, {}, {}))
    assert TBD in text
    held_out = text.split("## Test scores")[0].split("## Held-out source")[1]
    assert not any(ch.isdigit() for ch in held_out.replace("923", ""))
