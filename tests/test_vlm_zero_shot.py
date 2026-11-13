import numpy as np
import pytest

from zarai.vision import vlm_zero_shot as vz

LABELS = ["yellow_rust", "brown_rust", "black_rust", "septoria", "healthy", "unsure"]


@pytest.mark.parametrize(
    "text, label",
    [
        ('{"label": "septoria", "confidence": 0.7, "evidence": "dark dots"}', "septoria"),
        (
            '```json\n{"label": "Yellow_Rust", "confidence": 1, "evidence": "stripes"}\n```',
            "yellow_rust",
        ),
        ('Sure! {"label": "unsure", "confidence": 0.2, "evidence": "blurry"}', "unsure"),
    ],
)
def test_parse_accepts_valid_json(text, label):
    out = vz.parse(text, LABELS)
    assert out["valid"] and out["label"] == label


@pytest.mark.parametrize(
    "text",
    [
        "yellow_rust",
        '{"label": "stripe rust", "confidence": 0.9}',  # not one of the labels
        '{"label": "healthy", "confidence": 1.5}',  # out of range
        '{"label": "healthy", "confidence": "high"}',
        '{"label": "healthy", "confidence": true}',
        '{"label": "healthy", "confidence": 0.9',  # truncated
        "",
    ],
)
def test_parse_rejects_invalid(text):
    assert not vz.parse(text, LABELS)["valid"]


def _rows():
    rows = []
    for i in range(40):
        source = ["a", "b"][i % 2]
        rows.append(
            {
                "path": f"img{i}.jpg",
                "label": ["healthy", "yellow_rust"][(i // 4) % 2],
                "source": source,
                "split_pooled": ["train", "train", "val", "test"][i % 4],
                "split_loso_a": "test" if source == "a" else "train",
                "split_loso_b": "test" if source == "b" else "train",
            }
        )
    rows.append(
        {
            "path": "nw.jpg",
            "label": "yellow_rust",
            "source": "nwrd",
            "split_pooled": "nwrd",
            "split_loso_a": "nwrd",
            "split_loso_b": "nwrd",
        }
    )
    return rows


def test_test_sets_and_union():
    rows = _rows()
    sets = vz.test_sets(rows)
    assert list(sets) == ["pooled_test", "loso:a", "loso:b", "nwrd"]
    assert all(r["split_pooled"] == "test" for r in sets["pooled_test"])
    assert {r["source"] for r in sets["loso:a"]} == {"a"}
    assert len(vz.images_to_run(rows)) == 41  # every non-NWRD image is in a LOSO test set


def test_references_come_from_pooled_train_and_are_deterministic():
    rows = _rows()
    first = vz.references(rows, ["healthy", "yellow_rust"], 2, seed=0)
    assert first == vz.references(rows, ["healthy", "yellow_rust"], 2, seed=0)
    assert [lab for lab, _ in first] == ["healthy", "healthy", "yellow_rust", "yellow_rust"]
    assert all(r["split_pooled"] == "train" and r["label"] == lab for lab, r in first)


def test_agreement():
    labels = np.array([0, 0, 1, 1, 2])
    dino = np.array([0, 1, 1, 0, 2])
    gemma = np.array([0, 0, 1, 5, 1])  # 5 = no label
    a = vz.agreement(labels, dino, gemma, 5)
    assert a["agree"] == pytest.approx(2 / 5) and a["acc_agree"] == 1.0
    assert a["disagree"] == pytest.approx(2 / 5)
    assert a["dino_acc_disagree"] == 0.5 and a["gemma_acc_disagree"] == 0.5
    assert a["gemma_no_label"] == pytest.approx(1 / 5) and a["dino_acc_gemma_no_label"] == 0.0


def test_no_label_counts_as_wrong():
    pytest.importorskip("scipy")
    labels = np.array([0, 0, 1, 1])
    s = vz.scores(labels, np.array([0, 5, 1, 1]), 5)
    assert s["recall_0"]["value"] == 0.5 and s["recall_1"]["value"] == 1.0


def test_confusion_has_unsure_and_invalid_columns():
    classes = ["healthy", "yellow_rust"]
    recs = [
        {"valid": True, "label": "yellow_rust"},
        {"valid": True, "label": "unsure"},
        {"valid": False, "label": None},
        {"valid": True, "label": "yellow_rust"},
    ]
    cols = [vz.answer_column(r, classes) for r in recs]
    m = vz.confusion([0, 0, 1, 1], cols, 2)
    assert m == [[0, 1, 1, 0], [0, 1, 0, 1]]
    assert vz.called_as(m, 0, 1) == (1, 2)
    assert vz.precision_of(m, 1) == (1, 2)
