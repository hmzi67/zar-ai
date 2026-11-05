from zarai.vision.register import best_seed, version_tags


def test_best_seed_uses_val_only():
    runs = {
        "42": {"best_val_macro_f1": 0.85, "test_macro_f1": 0.99},
        "2": {"best_val_macro_f1": 0.86, "test_macro_f1": 0.10},
    }
    assert best_seed(runs) == "2"


def test_version_tags_record_what_serving_needs():
    tags = version_tags(
        "2",
        {
            "best_val_macro_f1": 0.86,
            "test_macro_f1": 0.84,
            "temperature": 0.7,
            "abstain_threshold": 0.75,
            "onnx_max_abs_diff": 4e-6,
        },
        {"dvc.processed_md5": "p", "dvc.splits_md5": "s", "model.hf_revision": "r"},
        {
            "classes": ["a", "b"],
            "image_size": 224,
            "mean": [0.5],
            "std": [0.5],
            "config": {"data": {"eval_resize": 256}},
        },
        "sha",
    )
    assert tags["temperature"] == "0.700000"
    assert tags["abstain_threshold"] == "0.750000"
    assert tags["classes"] == '["a", "b"]'
    assert tags["eval_resize"] == "256"
    assert tags["onnx_sha256"] == "sha"
    assert all(isinstance(value, str) for value in tags.values())
