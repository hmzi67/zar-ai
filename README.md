# ZarAI

ZarAI is a multilingual (Urdu, Roman Urdu, English) RAG chatbot and crop disease/pest image classifier for small farmers in Pakistan. It is a 6-month research project.

**Status:** starting skeleton. There are no results yet.

| Component | Metric | Value |
|---|---|---|
| Vision on unseen sites | Macro-F1 | [F1] |
| Vision with abstention | Accuracy at target coverage | [ACC]% |
| Retrieval | Recall@5 | [R@5] |
| Pilot | SUS | [SUS] |

## Team

| Person | Owns | Main folders |
|---|---|---|
| Hamza (lead) | LLM/RAG, fusion, papers | `src/zarai/nlp/`, `src/zarai/fusion/`, `src/zarai/eval/`, `paper/`, `experiments/` |
| RA-2 | Field images, vision model, farmer pilot | `src/zarai/vision/`, `data/raw/images_*`, `eval/pilot/`, `docs/field_collection_protocol.md` |
| RA-3 | Knowledge base, Q&A dataset, web app | `src/zarai/kb/`, `kb/`, `data/raw/qa/`, `app/` |

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv fetches Python 3.11 if you don't have it.

```bash
make setup                          # .venv with core + dev
source .venv/bin/activate
make test
```

Add the extras a task needs:

```bash
bash scripts/setup_env.sh dev,vision
bash scripts/setup_env.sh dev,nlp
```

Or do it by hand:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
```

Run everything from the repo root, because config paths are relative to it.

**Platform note:** vLLM and bitsandbytes install only on Linux, and QLoRA training needs a CUDA GPU. On Intel Macs the newest available PyTorch is 2.2, which may need `numpy<2`, so use a Linux GPU machine for training.

## After `git init`

DVC was initialised with `--no-scm` because the repo had no git yet. Once you have run `git init`:

```bash
.venv/bin/dvc config --unset core.no_scm
.venv/bin/pre-commit install
```

Then set a remote for the data: `dvc remote add -d storage <remote-url>`.

If `.dvc/` is missing, install the dev extra and run `dvc init --no-scm` (or `dvc init` once git exists).

## Repo map

| Path | Contents |
|---|---|
| `configs/` | YAML configs for vision, NLP, KB and the app; `paths.yaml` is merged into all of them |
| `data/` | DVC-tracked data, see [`data/README.md`](data/README.md) |
| `kb/source_register.csv` | Register of every KB source and its licence |
| `src/zarai/common/` | Config, logging, seeds, run metadata, file I/O |
| `src/zarai/data/` | Schemas, validation, splits, inter-annotator agreement |
| `src/zarai/kb/` | Ingest, clean, chunk, index |
| `src/zarai/vision/` | Dataset, models, training, calibration, export |
| `src/zarai/nlp/` | Retrievers, generator, fine-tuning, translate baseline, safety, script detection |
| `src/zarai/fusion/` | Image + question + chunks pipeline |
| `src/zarai/eval/` | Retrieval, vision, expert rating and SUS metrics |
| `app/` | FastAPI backend and Vite + React frontend |
| `eval/` | Expert rating sheet and pilot templates |
| `experiments/` | Experiment table; run outputs are gitignored |
| `docs/` | Annotation guidelines, field protocol, consent template, architecture |

See [`docs/architecture.md`](docs/architecture.md) for the pipeline.

## Running each module

Every config-driven command accepts `key=value` overrides, such as `train.lr=0.001`. Training and eval runs record config, seed and git commit in `experiments/runs/<time>_<name>/`.

### Data

```bash
python -m zarai.data.validate --schema qa_record data/raw/qa/qa.jsonl
python -m zarai.data.validate --schema image_meta data/raw/images_field/meta.jsonl
python -m zarai.data.split_qa --input data/raw/qa/qa.jsonl
python -m zarai.data.split_images --input data/raw/images_field/meta.jsonl
python -m zarai.data.agreement --input double.csv --rater-a a --rater-b b --weights quadratic
```

### Knowledge base (needs `nlp`)

```bash
make kb-build
```

### Vision (needs `vision`)

```bash
make vision-train VISION_CONFIG=configs/vision/convnext_t.yaml
make vision-eval  VISION_CONFIG=configs/vision/convnext_t.yaml
python -m zarai.vision.calibrate --config configs/vision/finetune_local.yaml logits_file=val_logits.npz
```

Pass `logits_file=x.npz` (arrays `logits`, `labels`) to `vision.eval` to score saved logits.

### RAG

```bash
make rag-eval      # retriever mode is set in configs/nlp/retriever.yaml
python -m zarai.nlp.generator.bakeoff --config configs/nlp/generator_bakeoff.yaml
python -m zarai.nlp.finetune.build_sft_data --config configs/nlp/qlora.yaml
python -m zarai.nlp.finetune.qlora_train --config configs/nlp/qlora.yaml
python -m zarai.nlp.baseline_translate.pipeline --config configs/nlp/translate_baseline.yaml
```

### Scoring

```bash
python -m zarai.eval.expert_ratings --input ratings.csv --compare S1 S2
python -m zarai.eval.sus --input pilot_log.csv
```

### App

```bash
make app-dev     # API on :8000, frontend on :5173
```

`/health` works now. `/chat`, `/classify` and `/ask` return 501 until the models and KB exist. API docs are at `http://localhost:8000/docs`. To build the API image: `docker build -f app/backend/Dockerfile -t zarai-api .`

## Rules

- No data, weights or secrets in git.
- Do not touch `test` or `field_test` until the config is frozen.
- Report no number without a run directory behind it.
