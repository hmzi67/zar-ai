#!/usr/bin/env bash
# Usage: bash scripts/setup_env.sh [extras]   e.g. dev (default), dev,vision,nlp
set -euo pipefail
cd "$(dirname "$0")/.."

extras="${1:-dev}"
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[${extras}]"
[ -f .env ] || cp .env.example .env
if [ -d .git ]; then .venv/bin/python -m pre_commit install; fi
echo "Done. Activate with: source .venv/bin/activate"
