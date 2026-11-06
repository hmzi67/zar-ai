import numpy as np
import pytest

from zarai.vision.eval_external import (
    EXTERNAL_DATASETS,
    TBD,
    check_dataset,
    collect_folders,
    evaluation_metrics,
    evaluation_names,
    normalise_folder,
    overlap_with,
    render,
    safe_extract,
    tile_boxes,
)
from zarai.vision.watermark import WATERMARK_CLASSES

CLASSES = ["black_rust", "brown_rust", "healthy", "yellow_rust"]


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def logits_for(preds, n_classes=4, strength=5.0):
    logits = np.zeros((len(preds), n_classes))
    logits[np.arange(len(preds)), preds] = strength
    return logits


def test_label_maps_use_only_our_classes():
    from zarai.data.clean_wheat import WHEAT_CLASSES

    for spec in EXTERNAL_DATASETS.values():
        assert set(spec.label_map.values()) <= WHEAT_CLASSES
    assert set(WATERMARK_CLASSES) <= WHEAT_CLASSES


@pytest.mark.parametrize(
    ("dataset", "folder", "label"),
    [
        ("mendeley_wheat_leaf_ethiopia", "Stripe Rust", "yellow_rust"),
        ("mendeley_wheat_leaf_ethiopia", "Healthy", "healthy"),
        ("mendeley_wheat_leaf_ethiopia", "septoria", "septoria"),
        ("mendeley_iari_leaf_rust", "Diseased", "brown_rust"),
        ("mendeley_iari_leaf_rust", "control", "healthy"),
        ("cgiar_wheat_rust", "stem_rust", "black_rust"),
        ("cgiar_wheat_rust", "leaf_rust", "brown_rust"),
        ("cgiar_wheat_rust", "healthy_wheat", "healthy"),
    ],
)
def test_requested_label_maps(dataset, folder, label):
    assert EXTERNAL_DATASETS[dataset].label_map[normalise_folder(folder)] == label


def test_collect_folders_finds_classes_at_any_depth_and_reports_the_rest(tmp_path):
    spec = EXTERNAL_DATASETS["mendeley_iari_leaf_rust"]
    touch(tmp_path / "WheatLeafRust" / "train" / "Diseased" / "a.jpg")
    touch(tmp_path / "WheatLeafRust" / "test" / "control" / "b.PNG")
    touch(tmp_path / "other" / "c.jpg")
    touch(tmp_path / "__MACOSX" / "control" / "d.jpg")
    touch(tmp_path / "WheatLeafRust" / "train" / "Diseased" / "notes.txt")
    samples, skipped = collect_folders(tmp_path, spec)
    assert sorted(s["label"] for s in samples) == ["brown_rust", "healthy"]
    assert skipped == {"folder other": 1}


def test_cgiar_uses_only_the_train_folder(tmp_path):
    spec = EXTERNAL_DATASETS["cgiar_wheat_rust"]
    touch(tmp_path / "train" / "train" / "stem_rust" / "a.jpg")
    touch(tmp_path / "train" / "train" / "mystery" / "b.jpg")
    touch(tmp_path / "test" / "c.jpg")
    touch(tmp_path / "test" / "leaf_rust" / "d.jpg")
    samples, skipped = collect_folders(tmp_path, spec)
    assert [s["label"] for s in samples] == ["black_rust"]
    assert skipped == {"outside train/": 2, "folder mystery": 1}


def test_tile_boxes_are_full_and_do_not_overlap():
    assert tile_boxes(800, 400) == [(0, 0, 384, 384), (384, 0, 768, 384)]
    assert tile_boxes(300, 300) == []


def test_nwrd_labels_images_and_tiles_by_their_own_mask(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from zarai.vision.eval_external import collect_masked

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "masks").mkdir(parents=True)
    image_module.new("RGB", (800, 400)).save(tmp_path / "train" / "images" / "1.jpg")
    mask = np.zeros((400, 800), dtype=np.uint8)
    mask[10, 10] = 255  # Inside the first tile only.
    image_module.fromarray(mask).save(tmp_path / "train" / "masks" / "1.png")
    image_module.new("RGB", (800, 400)).save(tmp_path / "train" / "images" / "2.jpg")
    image_module.fromarray(np.zeros((400, 800), dtype=np.uint8)).save(
        tmp_path / "train" / "masks" / "2.png"
    )
    image_module.new("RGB", (800, 400)).save(tmp_path / "train" / "images" / "3.jpg")

    images, tiles, skip_images, skip_tiles = collect_masked(tmp_path)
    assert [s["label"] for s in images] == ["yellow_rust"]
    assert [s["box"] for s in tiles] == [(0, 0, 384, 384)]
    assert skip_images == {"empty mask": 1, "no mask": 1}
    assert skip_tiles == {"empty mask": 3}


def test_check_reports_a_missing_dataset_without_failing(tmp_path):
    state = check_dataset(tmp_path, EXTERNAL_DATASETS["nwrd"])
    assert state["status"] == "missing"
    (tmp_path / "mendeley_iari_leaf_rust" / "wrong").mkdir(parents=True)
    touch(tmp_path / "mendeley_iari_leaf_rust" / "wrong" / "a.jpg")
    state = check_dataset(tmp_path, EXTERNAL_DATASETS["mendeley_iari_leaf_rust"])
    assert state["status"] == "empty or unrecognised layout"
    assert state["skipped"] == {"folder wrong": 1}


def test_metrics_count_predictions_outside_the_mapped_classes_as_errors():
    labels = np.array([1, 1, 2, 2])
    preds = np.array([1, 3, 2, 2])  # One brown_rust image is predicted as yellow_rust.
    metrics, matrix = evaluation_metrics(
        logits_for(preds),
        labels,
        CLASSES,
        ["brown_rust", "healthy"],
        temperature=1.0,
        threshold=0.0,
        overlap=np.array([False, False, False, True]),
    )
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["out_of_set_share"] == 0.25
    assert metrics["recall_brown_rust"] == 0.5
    assert metrics["recall_healthy"] == 1.0
    # brown_rust: P=1, R=0.5, F1=2/3. healthy: F1=1.
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert metrics["overlap_n"] == 1
    assert matrix[1, 3] == 1
    assert matrix.shape == (4, 4)


def test_abstention_counts_a_declined_image_as_a_miss_for_recall():
    labels = np.array([1, 1, 2, 2])
    logits = np.array(
        [[0, 9, 0, 0], [0, 0.1, 0, 0], [0, 0, 9, 0], [0, 0, 0, 9]], dtype=float
    )  # The second is unsure and is declined. The last is sure and wrong.
    metrics, _ = evaluation_metrics(
        logits,
        labels,
        CLASSES,
        ["brown_rust", "healthy"],
        temperature=1.0,
        threshold=0.8,
        overlap=np.zeros(4, dtype=bool),
    )
    assert metrics["abstain_coverage"] == 0.75
    assert metrics["abstain_coverage_brown_rust"] == 0.5
    assert metrics["abstain_recall_brown_rust"] == 0.5
    assert metrics["abstain_recall_healthy"] == 0.5
    assert metrics["abstain_accuracy"] == pytest.approx(2 / 3)
    assert metrics["abstain_out_of_set_share"] == pytest.approx(1 / 3)


def test_macro_f1_ignores_a_mapped_class_with_no_images():
    labels = np.array([1, 1])
    metrics, _ = evaluation_metrics(
        logits_for(np.array([1, 1])),
        labels,
        CLASSES,
        ["brown_rust", "healthy"],
        temperature=1.0,
        threshold=0.0,
        overlap=np.zeros(2, dtype=bool),
    )
    assert metrics["macro_f1"] == 1.0
    assert "recall_healthy" not in metrics


def test_macro_f1_without_overlap_ignores_a_class_left_with_no_images():
    labels = np.array([1, 1, 2, 2])
    metrics, _ = evaluation_metrics(
        logits_for(labels),
        labels,
        CLASSES,
        ["brown_rust", "healthy"],
        temperature=1.0,
        threshold=0.0,
        overlap=np.array([True, True, False, False]),
    )
    assert metrics["macro_f1_no_overlap"] == 1.0


def test_overlap_finds_near_duplicates_and_skips_unreadable_images():
    reference = ["00000000000000ff", "ffffffffffffffff"]
    external = ["00000000000000fe", "0f0f0f0f0f0f0f0f", None]
    assert overlap_with(external, reference, 2).tolist() == [True, False, False]
    assert overlap_with([None], reference, 2).tolist() == [False]


def test_nearest_reference_returns_the_closest_match_or_minus_one():
    from zarai.vision.eval_external import nearest_reference

    reference = ["00000000000000ff", "00000000000000fe", "ffffffffffffffff"]
    assert nearest_reference(["00000000000000fe", "0f0f0f0f0f0f0f0f", None], reference, 2) == [
        1,
        -1,
        -1,
    ]


def test_overlap_breakdown_counts_each_image_once_by_both_labels():
    from zarai.vision.eval_external import overlap_breakdown

    samples = [
        {"image": "a", "label": "yellow_rust"},
        {"image": "a", "label": "yellow_rust"},  # A second tile of the same image.
        {"image": "b", "label": "yellow_rust"},
        {"image": "c", "label": "healthy"},
    ]
    matches = {
        "a": {"label": "brown_rust", "split": "train"},
        "b": {"label": "brown_rust", "split": "train"},
        "c": {"label": "healthy", "split": "test"},
    }
    assert overlap_breakdown(samples, matches) == [
        {"external": "yellow_rust", "wheat_v1": "brown_rust", "split": "train", "images": 2},
        {"external": "healthy", "wheat_v1": "healthy", "split": "test", "images": 1},
    ]


def test_safe_extract_refuses_paths_outside_the_target(tmp_path):
    import zipfile

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()


def status_missing():
    return {
        name: {"name": name, "status": "missing", "counts": {}, "skipped": {}}
        for name in EXTERNAL_DATASETS
    }


def test_report_without_runs_has_only_tbd_results():
    text = "\n".join(render([], {}, status_missing(), ["abc"]))
    for name in evaluation_names():
        assert f"| {name} | {TBD} |" in text
    results = text.split("## Results")[1].split("## Overlap")[0]
    assert not any(ch.isdigit() for ch in results.replace("15", "").replace("F1", ""))


def test_report_shows_measured_values_and_marks_other_datasets():
    runs = [
        {"nwrd_image__n": 90.0, "nwrd_image__macro_f1": 0.5},
        {"nwrd_image__n": 90.0, "nwrd_image__macro_f1": 0.7},
    ]
    text = "\n".join(render(runs, {}, status_missing(), ["a", "b"]))
    line = next(row for row in text.splitlines() if row.startswith("| nwrd_image | 90"))
    assert "0.6000 ± 0.1414" in line
    assert f"| cgiar_wheat_rust | {TBD} |" in text


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status, self.position = data, status, 0

    def read(self, size=-1):
        chunk = (
            self.data[self.position :]
            if size < 0
            else self.data[self.position : self.position + size]
        )
        self.position += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_resumes_from_a_partial_file_after_a_timeout(tmp_path, monkeypatch):
    import urllib.request

    from zarai.vision import eval_external

    calls = []

    def flaky(request, timeout):
        headers = dict(request.header_items())
        calls.append(headers.get("Range"))
        if len(calls) == 1:
            (tmp_path / "f.zip.part").write_bytes(b"abc")
            raise TimeoutError("The read operation timed out")
        return FakeResponse(b"def", status=206)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    eval_external._fetch("https://example.org/f", tmp_path / "f.zip")
    assert calls == [None, "bytes=3-"]
    assert (tmp_path / "f.zip").read_bytes() == b"abcdef"
    assert not (tmp_path / "f.zip.part").exists()


def test_fetch_starts_over_if_the_server_ignores_the_range(tmp_path, monkeypatch):
    import urllib.request

    from zarai.vision import eval_external

    (tmp_path / "f.zip.part").write_bytes(b"stale")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(b"whole"))
    eval_external._fetch("https://example.org/f", tmp_path / "f.zip")
    assert (tmp_path / "f.zip").read_bytes() == b"whole"


def test_fetch_keeps_the_partial_file_when_every_attempt_fails(tmp_path, monkeypatch):
    import urllib.request

    from zarai.vision import eval_external

    def failing(request, timeout):
        (tmp_path / "f.zip.part").write_bytes(b"ab")
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    with pytest.raises(OSError):
        eval_external._fetch("https://example.org/f", tmp_path / "f.zip", attempts=2)
    assert (tmp_path / "f.zip.part").exists()


def test_kaggle_token_is_not_sent_to_the_storage_host(monkeypatch):
    import email.message
    import urllib.error
    import urllib.request

    from zarai.vision import eval_external

    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append((request.full_url, dict(request.header_items())))
            headers = email.message.Message()
            headers["Location"] = "https://storage.example.com/signed"
            raise urllib.error.HTTPError(request.full_url, 302, "Found", headers, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    url = eval_external.kaggle_download_url("owner/data", "secret")
    assert url == "https://storage.example.com/signed"
    assert sent[0][0].startswith("https://www.kaggle.com/")
    assert sent[0][1]["Authorization"] == "Bearer secret"


def test_download_all_reports_a_failure_and_continues(tmp_path, monkeypatch):
    from zarai.vision import eval_external

    def fake_download(root, spec):
        if spec.name == "nwrd":
            raise OSError("HTTP Error 403: Forbidden")
        return f"{spec.name}: ok"

    monkeypatch.setattr(eval_external, "download", fake_download)
    messages = eval_external.download_all(tmp_path, ["nwrd", "mendeley_iari_leaf_rust"])
    assert messages == [
        "nwrd: FAILED: OSError: HTTP Error 403: Forbidden",
        "mendeley_iari_leaf_rust: ok",
    ]


def test_kaggle_token_prefers_the_environment(tmp_path, monkeypatch):
    from zarai.vision import eval_external

    token_file = tmp_path / "access_token"
    token_file.write_text("from-file\n")
    monkeypatch.setattr(eval_external, "KAGGLE_TOKEN_FILE", token_file)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    assert eval_external.kaggle_token() == "from-file"
    monkeypatch.setenv("KAGGLE_API_TOKEN", "from-env")
    assert eval_external.kaggle_token() == "from-env"
    token_file.unlink()
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    assert eval_external.kaggle_token() is None
