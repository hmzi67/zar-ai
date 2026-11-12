"""API request and response models."""

from typing import Literal

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 2000


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Passage(BaseModel):
    chunk_id: str
    source_id: str | None = None
    pages: list[str] = []
    score: float
    text: str


class RetrieveResponse(BaseModel):
    lang: Literal["ur", "ur-Latn", "en"]
    passages: list[Passage]


class Prediction(BaseModel):
    label: str
    prob: float


class ClassifyResponse(BaseModel):
    top: list[Prediction]
    abstain: bool
    # confident >= 0.90, likely >= 0.60, unsure below (configs/vision/serving.yaml); not_plant when
    # the plant filter turned the photo away, and then top is empty.
    band: Literal["confident", "likely", "unsure", "not_plant"] = "confident"
    plant_prob: float | None = None


class AnswerResponse(BaseModel):
    answer: str
    lang: Literal["ur", "ur-Latn", "en"]
    # diagnosis: the image model's result alone, while the knowledge base or generator isn't ready.
    status: Literal["answered", "clarify", "refused", "blocked", "diagnosis", "not_plant"]
    sources: list[str]
    vision: ClassifyResponse | None = None