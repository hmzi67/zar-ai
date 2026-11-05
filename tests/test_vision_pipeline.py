import numpy as np
import pytest

from zarai.eval.vision_metrics import bootstrap_cis, confusion_between, top_confusions
from zarai.vision.datasets import SplitCsvDataset, class_balanced_weights, classes_from_rows
from zarai.vision.export_onnx import outputs_match
from zarai.vision.gradcam import cam_mass_on_mask
from zarai.vision.results import END, START, TBD, render_results, replace_section, select_best
from zarai.vision.tracking import flatten, require_run_id
from zarai.vision.train import EarlyStopping, warmup_cosine_factor


def test_warmup_rises_then_cosine_decays_to_zero():
    factors = [warmup_cosine_factor(step, 10, 110) for step in range(111)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0)
    assert factors[60] == pytest.approx(0.5)
    assert factors[110] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(factors[9:], factors[10:], strict=False))


def test_early_stopping_counts_only_epochs_without_improvement():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(0.5) is True
    assert stopper.update(0.5) is False
    assert not stopper.should_stop
    assert stopper.update(0.6) is True
    assert stopper.update(0.4) is False
    assert stopper.update(0.4) is False
    assert stopper.should_stop
    assert stopper.best == 0.6


def test_sampler_weights_are_sqrt_inverse_frequency():
    weights = class_balanced_weights([0, 0, 0, 0, 1], power=0.5)
    assert weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0])
    assert weights[:4].sum() / weights[4] == pytest.approx(2.0)


def write_splits(path, rows):
    path.write_text(
        "path,label,split\n" + "".join(f"{p},{label},{s}\n" for p, label, s in rows),
        encoding="utf-8",
    )
    return path


def test_dataset_uses_one_split_and_keeps_class_order_from_all_splits(tmp_path):
    csv_path = write_splits(
        tmp_path / "s.csv",
        [("a.jpg", "smut", "train"), ("b.jpg", "mite", "val"), ("c.jpg", "aphid", "train")],
    )
    dataset = SplitCsvDataset(csv_path, "train")
    assert dataset.classes == ["aphid", "mite", "smut"]
    assert dataset.labels == [2, 0]
    assert len(dataset) == 2


def test_dataset_rejects_an_empty_split_and_unknown_labels(tmp_path):
    csv_path = write_splits(tmp_path / "s.csv", [("a.jpg", "smut", "train")])
    with pytest.raises(ValueError, match="no rows"):
        SplitCsvDataset(csv_path, "val")
    with pytest.raises(ValueError, match="labels not in classes"):
        SplitCsvDataset(csv_path, "train", classes=["mite"])


def test_classes_from_rows_is_sorted_and_unique():
    assert classes_from_rows([{"label": "b"}, {"label": "a"}, {"label": "b"}]) == ["a", "b"]


def test_bootstrap_cis_bracket_the_value_and_are_reproducible():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 300)
    preds = np.where(rng.random(300) < 0.8, labels, rng.integers(0, 3, 300))
    first = bootstrap_cis(labels, preds, n_resamples=200, seed=1)
    assert first == bootstrap_cis(labels, preds, n_resamples=200, seed=1)
    for values in first.values():
        assert values["low"] < values["value"] < values["high"]
    assert first["accuracy"]["value"] == pytest.approx((labels == preds).mean())


def test_bootstrap_is_exact_when_every_prediction_is_right():
    labels = np.array([0, 1, 2] * 20)
    cis = bootstrap_cis(labels, labels, n_resamples=50)
    assert cis["macro_f1"] == {"value": 1.0, "low": 1.0, "high": 1.0}


def test_confusion_helpers_read_rows_as_true_labels():
    matrix = np.array([[8, 2, 0], [1, 9, 0], [0, 3, 7]])
    classes = ["a", "b", "c"]
    pair = confusion_between(matrix, classes, "a", "b")
    assert pair == {"a": "a", "b": "b", "a_support": 10, "b_support": 10, "a_as_b": 2, "b_as_a": 1}
    assert top_confusions(matrix, classes, n=2) == [
        {"true": "c", "predicted": "b", "count": 3},
        {"true": "a", "predicted": "b", "count": 2},
    ]


def test_outputs_match_applies_the_tolerance():
    base = np.zeros((2, 3))
    assert outputs_match(base, base + 5e-5, 1e-4)[0]
    matched, difference = outputs_match(base, base + 2e-4, 1e-4)
    assert not matched
    assert difference == pytest.approx(2e-4)


def test_cam_mass_on_mask():
    cam = np.array([[1.0, 3.0], [0.0, 0.0]])
    assert cam_mass_on_mask(cam, np.array([[True, False], [False, False]])) == pytest.approx(0.25)
    assert cam_mass_on_mask(np.zeros((2, 2)), np.ones((2, 2), dtype=bool)) == 0.0


def test_flatten_and_require_run_id():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
    assert require_run_id({"run_id": "x"}) == "x"
    with pytest.raises(ValueError, match="run_id"):
        require_run_id({})


DVC = {"dvc.processed_md5": "p", "dvc.splits_md5": "s", "dvc.raw_md5": "r"}


def run(run_id, **metrics):
    return {"run_id": run_id, "metrics": metrics}


def test_select_best_uses_val_and_ignores_better_test_scores():
    runs = {
        ("wheat_v1_efficientnetv2", "42"): run("e", best_val_macro_f1=0.9, test_macro_f1=0.1),
        ("wheat_v1_convnext_t", "42"): run("c", best_val_macro_f1=0.8, test_macro_f1=0.99),
    }
    assert select_best(runs)[0] == "wheat_v1_efficientnetv2"
    assert select_best({}) is None


def test_select_best_ignores_other_seeds():
    runs = {
        ("wheat_v1_efficientnetv2", "42"): run("e", best_val_macro_f1=0.8),
        ("wheat_v1_convnext_t", "0"): run("c", best_val_macro_f1=0.99),
    }
    assert select_best(runs)[0] == "wheat_v1_efficientnetv2"


def test_results_with_no_runs_contain_no_numbers():
    text = "\n".join(render_results({}, None, None, DVC))
    assert TBD in text
    table_cells = [
        cell.strip()
        for line in text.splitlines()
        if line.startswith("| EfficientNetV2-S")
        for cell in line.strip("|").split("|")[1:]
    ]
    assert table_cells and set(table_cells) == {TBD}


def test_results_show_logged_values_and_mark_missing_ones():
    runs = {
        ("wheat_v1_efficientnetv2", "42"): run(
            "abc",
            best_val_macro_f1=0.91234,
            best_epoch=7,
            epochs_run=12,
            test_macro_f1=0.9,
            test_macro_f1_low=0.88,
            test_macro_f1_high=0.92,
        )
    }
    text = "\n".join(render_results(runs, None, None, DVC))
    line = next(row for row in text.splitlines() if row.startswith("| EfficientNetV2-S"))
    assert "0.9123" in line
    assert "7 / 12" in line
    assert "0.9000 (0.8800-0.9200)" in line
    assert "`abc`" in line
    assert TBD in line  # Calibration hasn't been logged in this run.
    convnext = next(row for row in text.splitlines() if row.startswith("| ConvNeXt-T"))
    assert "0.9" not in convnext


def test_replace_section_keeps_the_written_analysis():
    document = f"intro\n{START}\nold\n{END}\nanalysis"
    assert replace_section(document, ["new"]) == f"intro\n{START}\nnew\n{END}\nanalysis"
    with pytest.raises(ValueError, match="markers"):
        replace_section("no markers", ["new"])


def test_transforms_give_the_requested_size():
    pytest.importorskip("torchvision")
    image_module = pytest.importorskip("PIL.Image")
    from zarai.vision.transforms import build_transforms

    image = image_module.new("RGB", (300, 120), "green")
    for train in (True, False):
        assert build_transforms(224, train=train)(image).shape == (3, 224, 224)


@pytest.mark.parametrize("name", ["convnext_atto", "efficientnetv2_rw_t", "vit_tiny_patch16_224"])
def test_grad_cam_runs_for_each_architecture_with_a_frozen_backbone(name):
    timm = pytest.importorskip("timm")
    torch = pytest.importorskip("torch")
    from zarai.vision.gradcam import grad_cam

    model = timm.create_model(name, pretrained=False, num_classes=3)
    for param in model.parameters():
        param.requires_grad = False
    cams, preds, probs = grad_cam(model, torch.randn(2, 3, 224, 224), device="cpu")
    assert cams.shape == (2, 224, 224)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    # A hook on the wrong layer gives all-zero maps without any error.
    assert (cams.max(axis=(1, 2)) > 0).all()
    assert preds.shape == probs.shape == (2,)
