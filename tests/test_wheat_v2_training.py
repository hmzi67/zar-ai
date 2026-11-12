from pathlib import Path

import numpy as np
import pytest

from zarai.common.config import load_config
from zarai.eval.vision_metrics import bootstrap_class_scores, confidently_wrong_share
from zarai.vision.results import TBD
from zarai.vision.train import EarlyStopping

CFG = load_config("configs/vision/wheat_v2.yaml")


def test_early_stopping_waits_for_min_epochs():
    stopper = EarlyStopping(patience=2, min_epochs=5)
    for value in (0.5, 0.4, 0.4, 0.4):
        stopper.update(value)
    assert not stopper.should_stop  # patience is used up, but only 4 epochs have run
    stopper.update(0.4)
    assert stopper.should_stop


def test_bootstrap_class_scores_match_the_plain_scores_and_bracket_them():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 400)
    preds = np.where(rng.random(400) < 0.7, labels, rng.integers(0, 5, 400))
    scores = bootstrap_class_scores(labels, preds, [0, 1, 2], 5, n_resamples=200)
    expected = f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)
    assert scores["macro_f1"]["value"] == pytest.approx(expected)
    assert scores["recall_1"]["value"] == pytest.approx((preds[labels == 1] == 1).mean())
    for value in scores.values():
        assert value["low"] <= value["value"] <= value["high"]


def test_a_prediction_outside_the_held_out_classes_counts_as_an_error():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 4, 1, 1])  # class 4 isn't in this set
    scores = bootstrap_class_scores(labels, preds, [0, 1], 5, n_resamples=10)
    assert scores["recall_0"]["value"] == 0.5
    assert "recall_4" not in scores


def test_confidently_wrong_counts_only_wrong_answers_above_the_threshold():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.95, 0.05]])
    labels = np.array([1, 1, 1, 0])  # wrong at 0.9 and at 0.6
    assert confidently_wrong_share(probs, labels, threshold=0.7) == 0.25


def test_strong_transform_gives_normalised_tensors_of_the_right_size():
    from PIL import Image

    from zarai.vision.transforms import build_train_transforms

    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 200, 3), np.uint8))
    for profile in ("basic", "strong"):
        out = build_train_transforms(profile, 224, (0.5,) * 3, (0.5,) * 3, CFG.train.augment)(image)
        assert tuple(out.shape) == (3, 224, 224)
    with pytest.raises(ValueError, match="profile"):
        build_train_transforms("wild", 224, (0.5,) * 3, (0.5,) * 3, CFG.train.augment)


def _tiny_split(tmp_path: Path):
    from PIL import Image

    image = np.zeros((40, 40, 3), np.uint8)
    image[:, :] = (200, 50, 50)  # red background
    image[10:30, 10:30] = (40, 160, 40)  # green leaf
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    Image.fromarray(image).save(tmp_path / "a.png")
    Image.fromarray(mask).save(tmp_path / "a_mask.png")
    csv = tmp_path / "s.csv"
    csv.write_text(
        f"path,mask_path,label,source,split_pooled\n"
        f"{tmp_path / 'a.png'},{tmp_path / 'a_mask.png'},healthy,x,train\n",
        encoding="utf-8",
    )
    return csv


def test_background_remove_blacks_out_only_the_background(tmp_path):
    from zarai.vision.datasets import WheatV2Dataset

    csv = _tiny_split(tmp_path)
    kept = WheatV2Dataset(csv, "split_pooled", "train", None, ["healthy"])
    removed = WheatV2Dataset(csv, "split_pooled", "train", None, ["healthy"], background="remove")
    original, masked = np.asarray(kept.load_image(0)), np.asarray(removed.load_image(0))
    assert (original[0, 0] == (200, 50, 50)).all()
    assert (masked[0, 0] == 0).all()
    assert (masked[20, 20] == original[20, 20]).all()
    with pytest.raises(ValueError, match="background"):
        WheatV2Dataset(csv, "split_pooled", "train", None, ["healthy"], background="blur")


def test_background_replacement_keeps_the_leaf(tmp_path):
    from zarai.vision.datasets import WheatV2Dataset

    csv = _tiny_split(tmp_path)
    dataset = WheatV2Dataset(
        csv, "split_pooled", "train", None, ["healthy"], replace_background_p=1.0
    )
    image = np.asarray(dataset.load_image(0))
    assert (image[20, 20] == (40, 160, 40)).all()


def test_lora_trains_only_adapters_and_the_head():
    from omegaconf import OmegaConf

    from zarai.vision import backbones

    cfg = OmegaConf.merge(CFG, {"model": {"backbone": "dinov2_vit_s14", "mode": "lora"}})
    model = backbones.build(cfg, num_classes=5, pretrained=False)
    trained = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sum("lora_" in n for n in trained) == 2 * 2 * len(model.blocks)  # qkv and proj, A and B
    assert {n for n in trained if "lora_" not in n} == {"head.weight", "head.bias"}
    state = backbones.checkpoint_state(model, "lora")
    assert set(state) == set(trained)
    backbones.load_state(model, state, "lora")


def test_layer_decay_lowers_the_lr_of_early_blocks():
    from omegaconf import OmegaConf

    from zarai.vision import backbones

    cfg = OmegaConf.merge(
        CFG, {"model": {"backbone": "dinov2_vit_s14", "mode": "full"}, "train": {"lr": 1.0e-4}}
    )
    model = backbones.build(cfg, num_classes=5, pretrained=False)
    lrs = sorted({round(g["lr"], 12) for g in backbones.param_groups(model, cfg)})
    assert max(lrs) == pytest.approx(1.0e-4)
    assert min(lrs) == pytest.approx(1.0e-4 * 0.75 ** (len(model.blocks) + 1))


def test_lora_is_refused_for_a_backbone_without_attention():
    from omegaconf import OmegaConf

    from zarai.vision import backbones

    cfg = OmegaConf.merge(CFG, {"model": {"backbone": "efficientnetv2_s", "mode": "lora"}})
    with pytest.raises(ValueError, match="supports modes"):
        backbones.backbone_spec(cfg)


def test_run_config_accumulates_to_the_effective_batch(tmp_path):
    from zarai.vision.train_v2 import resolve_run_config

    csv = tmp_path / "s.csv"
    csv.write_text("path,split_pooled,split_loso_kaggle,split_loso_cgiar_wheat_rust\n")
    base = CFG.copy()
    base.data.splits_csv = str(csv)
    cfg = resolve_run_config(base, "dinov2_vit_l14", "full", "strong", "loso", "cgiar", 0)
    assert (cfg.train.batch_size, cfg.train.grad_accum_steps) == (16, 4)
    assert cfg.data.fold == "cgiar_wheat_rust" and cfg.seed == 0
    assert cfg.train.lr == CFG.train.lr_by_mode.full
    cfg = resolve_run_config(base, "efficientnetv2_s", "full", "basic", "pooled", None, 42, True)
    assert cfg.train.lr == 3.0e-4 and cfg.data.fold == "pooled"
    assert (cfg.train.epochs, cfg.train.limit_batches) == (1, 2)
    cfg = resolve_run_config(
        base,
        "efficientnetv2_s",
        "full",
        "basic",
        "pooled",
        None,
        42,
        True,
        ["train.limit_batches=9"],
    )
    assert cfg.train.limit_batches == 9


def test_results_table_marks_everything_not_run_as_tbd():
    from zarai.vision.results_v2 import grid_combos, render

    combos = grid_combos(CFG)
    assert ("efficientnetv2_s", "lora", "basic", "none") not in combos
    assert len(combos) == (1 * 2 + 4 * 3) * 2
    seeds = {
        "42": {"heldout_macro_f1": 0.6, "nwrd_recall_yellow_rust": 0.2},
        "0": {"heldout_macro_f1": 0.7},
    }
    grouped = {("kaggle", "dinov2_vit_b14", "lora", "strong", "none"): seeds}
    dvc = {"dvc.splits_md5": "s", "dvc.processed_md5": "p"}
    lines = render(CFG, grouped, ["kaggle", "nwrd_never"], {"kaggle": ["healthy"]}, dvc)
    text = "\n".join(lines)
    assert "| dinov2_vit_b14 | lora | strong | 2/3 | 0.6500 ± 0.0707 |" in text
    assert "| 1 | dinov2_vit_b14 | lora | strong | 1/2 | 0.6500 |" in text
    assert text.count(TBD) > 50
    assert "| efficientnetv2_s | full | basic | 0/3 | [TBD] |" in text
