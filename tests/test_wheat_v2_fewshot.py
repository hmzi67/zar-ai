from collections import Counter

from omegaconf import OmegaConf

from zarai.vision.analysis_v2 import class_sources, fold_training, source_class_counts
from zarai.vision.fewshot_nwrd import draw, nwrd_image_id, render, split_nwrd_images

CFG = OmegaConf.load("configs/vision/wheat_v2_fewshot.yaml")


def _tiles(images: dict[str, int], linked=()):
    rows, cluster = [], 0
    for image, n in images.items():
        for k in range(n):
            cluster += 1
            rows.append(
                {
                    "original_path": f"data/interim/wheat_v2/nwrd_tiles/{image}_{k * 384}_0.png",
                    "cluster_id": str(cluster),
                    "label": "yellow_rust",
                }
            )
    for a, b in linked:  # give one tile of b the cluster of a tile of a
        first = next(r for r in rows if nwrd_image_id(r) == a)
        next(r for r in rows if nwrd_image_id(r) == b)["cluster_id"] = first["cluster_id"]
    return rows


def test_image_id_strips_the_tile_offsets():
    assert nwrd_image_id({"original_path": "x/nwrd_tiles/IMG_0042_768_384.png"}) == "IMG_0042"


def test_nwrd_split_keeps_every_image_and_linked_images_on_one_side():
    rows = _tiles({f"im{i}": 1 + i % 4 for i in range(30)}, linked=[("im0", "im7")])
    side = split_nwrd_images(rows, adapt_share=0.5, seed=42)
    assert set(side) == {nwrd_image_id(r) for r in rows}
    assert side["im0"] == side["im7"]
    adapt = sum(side[nwrd_image_id(r)] == "adapt" for r in rows)
    assert 0.4 <= adapt / len(rows) <= 0.65
    assert side == split_nwrd_images(rows, adapt_share=0.5, seed=42)


def test_draws_are_balanced_reproducible_and_the_same_for_every_model():
    sets = {
        "adapt_yellow": [{"path": f"y{i}", "label": "yellow_rust"} for i in range(150)],
        "adapt_healthy": [{"path": f"h{i}", "label": "healthy"} for i in range(500)],
    }
    rows = draw(sets, 25, 3)
    assert Counter(r["label"] for r in rows) == {"yellow_rust": 25, "healthy": 25}
    assert len({r["path"] for r in rows}) == 50
    assert rows == draw(sets, 25, 3)
    assert rows != draw(sets, 25, 4)


def test_fewshot_report_shows_tbd_until_runs_exist():
    sizes = {"adapt_yellow": 211, "test_yellow": 198, "adapt_healthy": 1117, "test_healthy": 241}
    found = {
        ("dinov2_vit_b14/full/strong", 10): [
            {"nwrd_test_recall": 0.5, "healthy_fpr_yellow": 0.1, "healthy_fpr_any_disease": 0.2}
        ]
        * 5
    }
    text = "\n".join(render(CFG, found, sizes))
    assert "| dinov2_vit_b14/full/strong | 10 | 5/5 | 0.500 ± 0.000 | 0.100 ± 0.000 |" in text
    assert "| efficientnetv2_s/full/strong | 0 | 0/1 | [TBD] |" in text


def test_source_counts_and_single_source_classes():
    rows = [
        {"source": "a", "label": "black_rust", "split_loso_b": "train"},
        {"source": "a", "label": "healthy", "split_loso_b": "train"},
        {"source": "b", "label": "healthy", "split_loso_b": "test"},
        {"source": "b", "label": "septoria", "split_loso_b": "test"},
        {"source": "nwrd", "label": "yellow_rust", "split_loso_b": "nwrd"},
    ]
    classes = ["black_rust", "healthy", "septoria", "yellow_rust"]
    counts = source_class_counts(rows, ["a", "b", "nwrd"], classes)
    by_class = class_sources(counts, ["a", "b", "nwrd"], classes, exclude={"nwrd"})
    assert by_class["healthy"] == [("a", 1), ("b", 1)]
    assert by_class["yellow_rust"] == []  # test-only data isn't a training source
    train = fold_training(rows, "b", classes)
    assert train["septoria"] == Counter() and train["black_rust"] == Counter({"a": 1})
