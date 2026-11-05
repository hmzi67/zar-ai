#!/usr/bin/env bash
# Run the three seed 42 baselines, export the one selected on val macro-F1 to ONNX, and fill
# experiments/wheat_v1_baselines.md. key=value overrides are passed to every step.
# Usage: bash scripts/run_wheat_v1_baselines.sh [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python=.venv/bin/python
results_args=(
  --tracking-uri "${RESULTS_TRACKING_URI:-file:./mlruns}"
  --doc "${RESULTS_DOC:-experiments/wheat_v1_baselines.md}"
)

for name in wheat_v1_efficientnetv2 wheat_v1_convnext_t wheat_v1_dinov2_linear; do
  bash scripts/run_wheat_v1.sh "$name" "$@"
done

read -r best_name best_run_id < <("$python" -m zarai.vision.results "${results_args[@]}" | tail -n 1)
"$python" -m zarai.vision.export_onnx --config "configs/vision/${best_name}.yaml" "run_id=$best_run_id" "$@"
"$python" -m zarai.vision.results "${results_args[@]}" >/dev/null
echo "Selected on val macro-F1: $best_name ($best_run_id)"
