from types import SimpleNamespace

from zarai.vision.register import best_seed, sync_descriptions, version_tags


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


class FakeClient:
    def __init__(self, model, version):
        self.model, self.version, self.updates = model, version, []

    def get_registered_model(self, name):
        return SimpleNamespace(description=self.model)

    def get_model_version(self, name, version):
        return SimpleNamespace(description=self.version)

    def update_registered_model(self, name, description):
        self.updates.append(("model", description))

    def update_model_version(self, name, version, description):
        self.updates.append(("version", version, description))


def test_sync_descriptions_updates_only_what_differs():
    client = FakeClient("old", "new")
    sync_descriptions(client, "wheat-vision", "1", "new")
    assert client.updates == [("model", "new")]
    unchanged = FakeClient("new", "new")
    sync_descriptions(unchanged, "wheat-vision", "1", "new")
    assert unchanged.updates == []
