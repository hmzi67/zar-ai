#!/usr/bin/env bash
# Train EfficientNetV2-S without the watermarked source on seeds 42, 0, 1, 2, score that source as
# unseen data, and fill experiments/wheat_v1_leave_source_out.md.
# Needs data/interim/wheat_v1_watermarks.csv, from python -m zarai.vision.watermark_check.
# Usage: bash scripts/run_wheat_v1_leave_source_out.sh [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."

name=wheat_v1_efficientnetv2_lso
config="configs/vision/${name}.yaml"
python=.venv/bin/python

for seed in 42 0 1 2; do
  run_id="$(bash scripts/run_wheat_v1.sh "$name" "seed=$seed" "$@" | tail -n 1)"
  "$python" -m zarai.vision.leave_source_out --config "$config" "run_id=$run_id" "seed=$seed" "$@"
done
"$python" -m zarai.vision.leave_source_out --config "$config" report=true "$@"
