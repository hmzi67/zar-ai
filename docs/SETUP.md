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
