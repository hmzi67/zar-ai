import pytest
from fastapi.testclient import TestClient

from app.backend.main import app, get_pipeline
from app.backend.settings import settings
from zarai.fusion.pipeline import Pipeline

client = TestClient(app)
IMAGE = {"image": ("leaf.jpg", b"fake", "image/jpeg")}


class NotReady:
    def search(self, query, k):
        raise FileNotFoundError("chunks.jsonl")


class ToyLLM:
    def generate(self, messages):
        return "Irrigate twice."


class ToyRetriever:
    def search(self, query, k):
        return [{"chunk_id": "c1", "text": "Irrigate twice.", "score": 1.0}]


class ToyKBRetriever:
    def search(self, query, k):
        chunk = {"chunk_id": "src_0011_c0001", "source_id": "src_0011", "pages": ["1"]}
        return [{**chunk, "text": "پہلا پانی بوائی کے 20 تا 25 دن بعد", "score": 0.8}]


@pytest.fixture
def pipeline(monkeypatch):
    """Point the API at an injected pipeline instead of real models."""

    def use(**components):
        monkeypatch.setattr("app.backend.main.get_pipeline", lambda: Pipeline(**components))

    return use


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


class NoModel:
    def classify(self, image):
        raise NotImplementedError("no run to serve")


class ToyClassifier:
    def __init__(self, band="confident"):
        self.result = {
            "top": [{"label": "yellow_rust", "prob": 0.9}, {"label": "healthy", "prob": 0.1}],
            "band": band,
            "abstain": band == "unsure",
        }

    def classify(self, image):
        return self.result


class BadImage:
    def classify(self, image):
        from zarai.vision.serving import InvalidImageError

        raise InvalidImageError("not an image")


def test_chat_returns_501_until_the_kb_exists():
    get_pipeline.cache_clear()
    assert client.post("/chat", json={"question": "kya hai"}).status_code == 501


def test_image_endpoints_return_501_without_an_image_model(pipeline):
    pipeline(classifier=NoModel(), retriever=NotReady())
    assert client.post("/classify", files=IMAGE).status_code == 501
    assert client.post("/ask", data={"question": "kya hai"}, files=IMAGE).status_code == 501
    assert client.post("/retrieve", json={"question": "kya hai"}).status_code == 501


def test_ask_answers_with_the_diagnosis_until_the_kb_exists(pipeline):
    pipeline(classifier=ToyClassifier(), retriever=NotReady())
    body = client.post("/ask", data={"question": "What is this?"}, files=IMAGE).json()
    assert body["status"] == "diagnosis" and body["sources"] == []
    assert body["vision"]["top"][0] == {"label": "yellow_rust", "prob": 0.9}
    assert body["answer"].startswith("Image model result: Yellow (stripe) rust, 90%.")
    assert "Healthy leaf 10%" in body["answer"]


def test_an_unsure_classifier_still_returns_its_guesses(pipeline):
    pipeline(classifier=ToyClassifier(band="unsure"), retriever=NotReady())
    body = client.post("/ask", data={"question": "kya hai"}, files=IMAGE).json()
    assert body["status"] == "clarify" and body["vision"]["band"] == "unsure"


def test_a_likely_result_is_answered_with_a_confirming_question(pipeline):
    pipeline(classifier=ToyClassifier(band="likely"), retriever=NotReady())
    body = client.post("/ask", data={"question": "What is this?"}, files=IMAGE).json()
    assert body["status"] == "diagnosis" and body["vision"]["band"] == "likely"
    assert "probably Yellow (stripe) rust" in body["answer"] and "To confirm:" in body["answer"]


def test_a_non_plant_photo_is_turned_away(pipeline):
    class NotPlant:
        def classify(self, image):
            return {"top": [], "band": "not_plant", "abstain": True, "plant_prob": 0.01}

    pipeline(classifier=NotPlant(), retriever=NotReady())
    body = client.post("/ask", data={"question": "ye kya hai?"}, files=IMAGE).json()
    assert body["status"] == "not_plant"
    assert body["answer"] == "Barah-e-karam gandum ke pattay ki tasveer bhejein."


def test_an_unreadable_image_is_a_400(pipeline):
    pipeline(classifier=BadImage())
    assert client.post("/classify", files=IMAGE).status_code == 400


def test_missing_kb_returns_501(pipeline):
    pipeline(retriever=NotReady())
    assert client.post("/chat", json={"question": "kya hai"}).status_code == 501


def test_chat_returns_answer_model(pipeline):
    pipeline(retriever=ToyRetriever(), llm=ToyLLM())
    body = client.post("/chat", json={"question": "How often to irrigate?"}).json()
    assert body["status"] == "answered" and body["lang"] == "en" and body["sources"] == ["c1"]


def test_request_validation():
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/classify").status_code == 422


def test_upload_type_and_size_limits(monkeypatch):
    text_file = {"image": ("a.txt", b"x", "text/plain")}
    assert client.post("/classify", files=text_file).status_code == 415
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    assert client.post("/classify", files=IMAGE).status_code == 413


def test_retrieve_returns_passages_without_llm(pipeline):
    pipeline(retriever=ToyKBRetriever())
    body = client.post("/retrieve", json={"question": "گندم کو پانی کب دیں؟"}).json()
    assert body["lang"] == "ur"
    assert body["passages"][0]["chunk_id"] == "src_0011_c0001"
    assert body["passages"][0]["source_id"] == "src_0011"
    assert body["passages"][0]["pages"] == ["1"]


def test_retrieve_validation():
    assert client.post("/retrieve", json={"question": ""}).status_code == 422
