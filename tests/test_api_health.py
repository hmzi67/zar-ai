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


def test_real_endpoints_return_501_until_models_exist():
    get_pipeline.cache_clear()
    assert client.post("/chat", json={"question": "kya hai"}).status_code == 501
    assert client.post("/classify", files=IMAGE).status_code == 501
    assert client.post("/ask", data={"question": "kya hai"}, files=IMAGE).status_code == 501


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
