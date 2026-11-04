"""API request and response models."""

from typing import Literal

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 2000


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AnswerResponse(BaseModel):
    answer: str
    lang: Literal["ur", "ur-Latn", "en"]
    status: Literal["answered", "clarify", "refused", "blocked"]
    sources: list[str]


class Prediction(BaseModel):
    label: str
    prob: float


class ClassifyResponse(BaseModel):
    top: list[Prediction]
    abstain: bool
