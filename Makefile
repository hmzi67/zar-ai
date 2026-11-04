PY := .venv/bin/python
VISION_CONFIG ?= configs/vision/baseline_efficientnetv2.yaml

.PHONY: setup lint test kb-build vision-train vision-eval rag-eval app-dev

setup:
	bash scripts/setup_env.sh dev

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

test:
	$(PY) -m pytest

# kb-build needs the nlp extra: bash scripts/setup_env.sh dev,nlp
kb-build:
	$(PY) -m zarai.kb.ingest --config configs/kb.yaml
	$(PY) -m zarai.kb.build_index --config configs/kb.yaml

vision-train:
	$(PY) -m zarai.vision.train --config $(VISION_CONFIG)

vision-eval:
	$(PY) -m zarai.vision.eval --config $(VISION_CONFIG)

rag-eval:
	$(PY) -m zarai.eval.retrieval_metrics --config configs/nlp/retriever.yaml

app-dev:
	trap 'kill 0' EXIT; \
	$(PY) -m uvicorn app.backend.main:app --reload --port 8000 & \
	cd app/frontend && npm install && npm run dev
