PY := .venv/bin/python
VISION_CONFIG ?= configs/vision/wheat_v1_efficientnetv2.yaml

.PHONY: setup lint test kb-build vision-train vision-eval vision-baselines rag-eval app-dev data train results mlflow-ui grid

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
	$(PY) -m zarai.vision.eval --config $(VISION_CONFIG) run_id=$(RUN_ID)

vision-baselines:
	bash scripts/run_wheat_v1_baselines.sh

rag-eval:
	$(PY) -m zarai.eval.retrieval_metrics --config configs/nlp/retriever.yaml

app-dev:
	trap 'kill 0' EXIT; \
	$(PY) -m uvicorn app.backend.main:app --reload --port 8000 & \
	cd app/frontend && npm install && npm run dev

# wheat_v2. Pass flags through ARGS, e.g. make train ARGS="--backbone dinov2_vit_b14 --mode lora --fold all"
data:
	$(PY) dataloader.py $(ARGS)

train:
	$(PY) train.py $(ARGS)

results:
	$(PY) -m zarai.vision.results_v2

mlflow-ui:
	MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000

# The first grid (or GRID=full), resumable; see scripts/run_wheat_v2_grid.sh
grid:
	bash scripts/run_wheat_v2_grid.sh $(or $(GRID),first)
