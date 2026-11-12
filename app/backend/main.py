"""ZarAI API. /chat, /classify, /ask and /retrieve return 501 until the models and KB exist."""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend.schemas import (
    MAX_QUESTION_CHARS,
    AnswerResponse,
    ChatRequest,
    ClassifyResponse,
    HealthResponse,
    Passage,
    RetrieveResponse,
)
from app.backend.settings import settings
from zarai import __version__
from zarai.fusion.pipeline import Pipeline
from zarai.nlp.script_detect import detect_script

log = logging.getLogger(__name__)
app = FastAPI(title="ZarAI API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache
def get_pipeline() -> Pipeline:
    return Pipeline.from_config(settings.app_config)


def _call_pipeline(call: Callable[[Pipeline], dict]) -> dict:
    try:
        return call(get_pipeline())
    except (NotImplementedError, FileNotFoundError, ImportError) as error:
        log.warning("pipeline not ready: %s", error)
        raise HTTPException(status_code=501, detail="Model or knowledge base not ready") from error


def _read_image(image: UploadFile) -> bytes:
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image")
    limit = settings.max_upload_mb * 1024 * 1024
    data = image.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Image over {settings.max_upload_mb} MB")
    return data


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.post("/chat")
def chat(request: ChatRequest) -> AnswerResponse:
    return AnswerResponse(**_call_pipeline(lambda p: p.answer(request.question)))


@app.post("/retrieve")
def retrieve(request: ChatRequest) -> RetrieveResponse:
    """KB passages for a question, without the LLM, so retrieval can be checked on its own."""
    chunks = _call_pipeline(lambda p: {"chunks": p.retriever.search(request.question, p.top_k)})
    passages = [
        Passage(
            chunk_id=c["chunk_id"],
            source_id=c.get("source_id"),
            pages=[str(p) for p in c.get("pages", [])],
            score=c["score"],
            text=c["text"],
        )
        for c in chunks["chunks"]
    ]
    return RetrieveResponse(lang=detect_script(request.question), passages=passages)


@app.post("/classify")
def classify(image: Annotated[UploadFile, File()]) -> ClassifyResponse:
    data = _read_image(image)
    return ClassifyResponse(**_call_pipeline(lambda p: p.classify(data)))


@app.post("/ask")
def ask(
    question: Annotated[str, Form(min_length=1, max_length=MAX_QUESTION_CHARS)],
    image: Annotated[UploadFile, File()],
) -> AnswerResponse:
    data = _read_image(image)
    return AnswerResponse(**_call_pipeline(lambda p: p.answer(question, data)))
