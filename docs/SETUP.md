# Development environment setup

How to recreate the ZarAI environment on a fresh Ubuntu x86_64 machine. Run everything from the repo root.

## Detected on the reference machine (2026-11-05)

| Item | Value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-42-generic |
| System Python | 3.14.4 (not used) |
| Environment | `.venv` (prompt `zarai`), CPython 3.11.18 fetched by uv 0.13.7 |
| GPU hardware | NVIDIA RTX 4000 Ada Generation (AD104GL), found with `lspci` |
| VRAM | 20475 MiB (`nvidia-smi`) |
| NVIDIA driver | 595.99.02 |
| Highest CUDA the driver supports | 13.2 |
| PyTorch | `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `torchaudio 2.11.0+cu130` |
| `torch.cuda.is_available()` | `True`, device name `NVIDIA RTX 4000 Ada Generation`, PyTorch's CUDA runtime 13.0 |
| timm forward and backward pass | Passed: `efficientnetv2_rw_s`, batch of 8 at 224x224, peak VRAM 1349 MiB |
| bitsandbytes 4-bit load | Passed: `bitsandbytes 0.50.2`, tiny randomly initialised LLaMA-architecture model built locally, NF4, 28 `Linear4bit` layers on `cuda:0`, `generate()` ran |

These are the results of the checks in [Verify](#5-verify) on this machine. The 4-bit load used a tiny test model, so it shows the GPU stack works but says nothing about VRAM needs for a real model.

## 1. NVIDIA driver (needs sudo and a reboot)

Skip this step if `nvidia-smi` already works.

```bash
sudo ubuntu-drivers install nvidia:595-open
sudo reboot
nvidia-smi    # note the driver version and "CUDA Version"
```

`595-open` is what `ubuntu-drivers devices` recommends for this GPU. Ubuntu has prebuilt modules for it on the current kernel (`linux-modules-nvidia-595-open-generic`), which should work with Secure Boot on. If a blue "MOK management" screen appears at reboot, follow it to enrol the key.

## 2. Install uv and create the environment

The machine has no `python3.11` package. uv fetches its own Python 3.11, so neither step needs sudo or touches the system Python.

```bash
mkdir -p ~/.local/bin
V=0.13.7
BASE=https://github.com/astral-sh/uv/releases/download/$V
TMP=$(mktemp -d)
(
  cd "$TMP"
  curl -fsSLO "$BASE/uv-x86_64-unknown-linux-gnu.tar.gz"
  curl -fsSLO "$BASE/uv-x86_64-unknown-linux-gnu.tar.gz.sha256"
  sha256sum -c uv-x86_64-unknown-linux-gnu.tar.gz.sha256
  tar xzf uv-x86_64-unknown-linux-gnu.tar.gz
  install -m755 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx ~/.local/bin/
)
rm -rf "$TMP"
export PATH="$HOME/.local/bin:$PATH"    # if ~/.local/bin is not already on PATH

uv venv --python 3.11 --seed --prompt zarai .venv
```

The environment is in `.venv` rather than a directory called `zarai` because the `Makefile` and `.gitignore` expect that path. The `zarai` name appears as the shell prompt.

## 3. Install the packages

From the lock file (exact versions, recommended):

```bash
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements/lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

Or re-resolve from `requirements/*.txt` (how the lock file was made):

```bash
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cu130 --index-strategy first-index \
  "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" "torchaudio==2.11.0+cu130"
uv pip install --python .venv/bin/python -e ".[core,vision,nlp,app,dev]"
.venv/bin/pip freeze --exclude-editable    # add to requirements/lock.txt, keeping its header lines
```

Why these versions:

- **PyTorch:** `vllm 0.33.0` pins `torch==2.14.0`, `torchvision==0.29.0` and `torchaudio==2.11.0`, and uses CUDA 13 packages.
- **CUDA build:** `cu130` is the lowest CUDA 13 build. If `nvidia-smi` reports a CUDA version below 13.0, install a newer driver.
- **FAISS:** `faiss-cpu`, because `faiss-gpu` has no Python 3.11 wheel.

## 4. Set up the tooling

```bash
.venv/bin/pre-commit install
[ -f .env ] || cp .env.example .env
```

DVC is already initialised in this repo, so don't run `dvc init` again. If `.dvc/` is missing, run `.venv/bin/dvc init`. No DVC remote is configured yet.

## 5. Verify

```bash
source .venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# One forward and backward pass on the GPU. Fails if CUDA is unavailable.
python - <<'PY'
import timm, torch
assert torch.cuda.is_available(), "CUDA unavailable: check nvidia-smi"
model = timm.create_model("efficientnetv2_rw_s", pretrained=False, num_classes=15).cuda()
x = torch.randn(8, 3, 224, 224).cuda()
y = torch.randint(0, 15, (8,)).cuda()
torch.nn.functional.cross_entropy(model(x), y).backward()
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
PY

# Real 4-bit load and generate() with bitsandbytes on a tiny local model. Downloads nothing.
python scripts/check_gpu.py

pytest -q
ruff check .
```

## Activate

```bash
source .venv/bin/activate    # deactivate with: deactivate
```

## Optional: `make`

The `Makefile` targets need `make`. If it's missing, install it with `sudo apt install make`, or call the tools directly as shown above.

## wheat_v2: `dataloader.py` and `train.py`

Two thin scripts at the repo root; all the logic is in `src/zarai`. Run them with `.venv/bin/python` (plain `python3` is the system Python, which has none of the packages), or after `source .venv/bin/activate`.

### `dataloader.py`: build the dataset and check the loaders

```bash
.venv/bin/python dataloader.py --protocol loso                    # build if needed, check every fold
.venv/bin/python dataloader.py --protocol loso --fold cgiar --aug strong --skip-preprocess
```

1. Runs `dvc repro wheat_v2` (`zarai.data.wheat_v2`), which only rebuilds if an input changed. `--skip-preprocess` skips this.
2. Asserts the leakage checks on `data/splits/wheat_v2.csv`, then again on the rows each loader holds.
3. Per fold, prints class and source counts per split (train, val, test, nwrd), saves 32-sample grids per split to `data/processed/wheat_v2/samples/` (train augmented, the others with the eval transform), and times 200 training batches.

| flag | values | default |
| --- | --- | --- |
| `--protocol` | `pooled` (70/15/15, development only) or `loso` | `loso` |
| `--fold` | a held-out source, or any unique part of its name (`cgiar`, `iari`, `ethiopia`, `kaggle`) | every LOSO fold |
| `--aug` | `basic` (wheat_v1's) or `strong` | `basic` |
| `--background` | `none` or `remove` (black out everything outside the leaf mask) | `none` |
| `--skip-preprocess` | don't run `dvc repro` | off |

The dataset report is `data/processed/wheat_v2/REPORT.md`. Images with conflicting labels are quarantined and listed in `data/interim/label_review_v2.csv`, with contact sheets in `data/interim/wheat_v2/review/`. To apply an expert's labels, fill in `final_label` (a v2 class, or `exclude`), set `apply_label_review: data/interim/label_review_v2.csv` in `configs/data/wheat_v2.yaml`, and run `.venv/bin/dvc repro --force wheat_v2`. DVC never deletes that CSV, and a rebuild keeps its `final_label` values.

### `train.py`: train, evaluate, and update the results

```bash
.venv/bin/python train.py --dry-run --backbone dinov2_vit_l14 --mode full    # fit check: 1 epoch, 2 batches
.venv/bin/python train.py --backbone dinov2_vit_b14 --mode lora --aug strong --fold all --seeds 42,0,1
```

Each (fold, seed) is one MLflow run in the experiment `wheat_v2`, tagged with `fold`, `source`, `backbone`, `mode`, `aug`, `background`, `seed` and `dry_run`. After training, the run fits temperature and the abstention threshold on val, and scores the held-out source and the NWRD tiles: macro-F1 and per-class recall with bootstrap 95% CIs, ECE, the coverage-accuracy curve and the share of confidently wrong predictions. Finally it prints the run IDs and scores, and rebuilds `experiments/wheat_v2_results.md`, where dry runs are ignored.

| flag | values | default (from `configs/vision/wheat_v2.yaml`) |
| --- | --- | --- |
| `--config` | YAML | `configs/vision/wheat_v2.yaml` |
| `--backbone` | `efficientnetv2_s`, `dinov2_vit_s14`, `dinov2_vit_b14`, `dinov2_vit_l14`, `siglip2_vit_b16` | `efficientnetv2_s` |
| `--mode` | `linear`, `lora` (ViTs only), `full` | `full` |
| `--aug` | `basic`, `strong` | `basic` |
| `--protocol` | `pooled`, `loso` | `loso` |
| `--fold` | a held-out source, a unique part of one, or `all` | `mendeley_wheat_leaf_ethiopia` |
| `--seeds` | comma-separated | `42` |
| `--background` | `none`, `remove` | `none` |
| `--dry-run` | 1 epoch of 2 optimizer steps, evaluated on 2 batches; tagged `dry_run=true` | off |
| `--no-results` | don't rebuild the results file | off |

Any `key=value` argument overrides the config, and wins over `--dry-run`, for example `train.epochs=5`.

### `make` targets and monitoring

```bash
make data ARGS="--protocol loso --skip-preprocess"
make train ARGS="--backbone dinov2_vit_b14 --mode lora --fold all --seeds 42,0,1"
make results                                   # rebuild experiments/wheat_v2_results.md only
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000   # http://127.0.0.1:5000
```

MLflow 3 refuses a local `./mlruns` store unless `MLFLOW_ALLOW_FILE_STORE=true` is set; the training code sets it itself. In the UI, filter with `tags.dry_run = 'false'`, and compare runs by `tags.fold`, `tags.backbone`, `tags.mode` and `tags.aug`.

### Training the whole grid: `scripts/run_wheat_v2_grid.sh`

```bash
tmux new -s grid                              # so closing the terminal doesn't stop it
bash scripts/run_wheat_v2_grid.sh             # first grid: 9 configurations x 4 folds x 3 seeds
bash scripts/run_wheat_v2_grid.sh full        # all 28 configurations (first-grid runs are skipped)
```

It checks the dataset and loaders first, then trains each configuration on every fold with `--skip-done`. Rerunning the same command after a crash only runs what's missing. One failing configuration doesn't stop the others. Per-configuration logs go to `experiments/runs/grid_logs/`, and the results table is rebuilt at the end. `SEEDS=42` limits it to one seed, and extra `key=value` arguments go to every run. `make grid` (or `make grid GRID=full`) does the same.

## Trying the image model in the chatbot

`configs/app.yaml` points the API at `configs/vision/serving.yaml`, which names one wheat_v2 MLflow
run. The run supplies the checkpoint and the temperature fitted on its val split. To serve another
run, change `run_id` there.

```bash
make app-dev   # API on :8000, frontend on :5173 (needs Node.js 18+ for the frontend)
```

Open http://localhost:5173, attach a leaf photo and ask a question. The reply depends on the photo:

- not a plant (CLIP filter, `plant_filter` in serving.yaml): "Please send a photo of a wheat leaf";
- top calibrated probability >= 0.90: the diagnosis;
- 0.60 to 0.90: "probably <disease>", plus a question to confirm the symptoms;
- below 0.60: a request for a clearer photo, with the model's guesses still shown.

Until the knowledge base and generator exist, the reply is the image model's diagnosis on its own
(status `diagnosis`), with all 5 classes shown as bars. A question without a photo still returns
"not ready". experiments/serving_bands.md has the share and accuracy of each band, and
experiments/plant_filter.md has the filter's calibration. To test without the UI:

```bash
curl -F "image=@leaf.jpg;type=image/jpeg" localhost:8000/classify
curl -F "question=ye kya bimari hai?" -F "image=@leaf.jpg;type=image/jpeg" localhost:8000/ask
```

The served model has not been adapted to field photos: it misses most NWRD stripe rust (tile
recall 0.06), and most of those misses land in the confident band. The plant filter only asks
"is this a plant?", so a photo of another crop still gets a wheat disease label.
