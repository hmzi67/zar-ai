#!/usr/bin/env bash
# Train, evaluate and calibrate one model, and print its MLflow run ID as the last line.
# Usage: bash scripts/run_wheat_v1.sh <config name> [key=value ...]
#   e.g. bash scripts/run_wheat_v1.sh wheat_v1_convnext_t seed=0
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:?usage: run_wheat_v1.sh <config name> [key=value ...]}"
shift
config="configs/vision/${name}.yaml"
python=.venv/bin/python

run_id="$("$python" -m zarai.vision.train --config "$config" "$@" | tail -n 1)"
"$python" -m zarai.vision.eval --config "$config" "run_id=$run_id" "$@"
"$python" -m zarai.vision.calibrate --config "$config" "run_id=$run_id" "$@"
echo "$run_id"
