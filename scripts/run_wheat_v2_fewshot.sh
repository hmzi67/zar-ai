#!/usr/bin/env bash
# Few-shot adaptation to NWRD (experiments/wheat_v2_fewshot_nwrd.md):
#   1. train the pooled models it starts from (3 configurations x 3 seeds; finished runs are skipped)
#   2. adapt each configuration's best pooled run (by val macro-F1) for every k and draw
# Usage: bash scripts/run_wheat_v2_fewshot.sh       (safe to rerun; run inside tmux or with nohup)
set -euo pipefail
cd "$(dirname "$0")/.."

python=.venv/bin/python
seeds=${SEEDS:-42,0,1}
log_dir=experiments/runs/grid_logs
mkdir -p "$log_dir"

for combo in "efficientnetv2_s full strong" "dinov2_vit_b14 full strong" "dinov2_vit_l14 linear basic"; do
  read -r backbone mode aug <<<"$combo"
  log="$log_dir/pooled_${backbone}_${mode}_${aug}.log"
  echo "[$(date '+%F %T')] pooled $backbone / $mode / $aug, seeds $seeds -> $log"
  "$python" train.py --protocol pooled --backbone "$backbone" --mode "$mode" --aug "$aug" \
    --seeds "$seeds" --skip-done --no-results >>"$log" 2>&1
done

echo "[$(date '+%F %T')] few-shot adaptation -> $log_dir/fewshot_nwrd.log"
"$python" -m zarai.vision.fewshot_nwrd >>"$log_dir/fewshot_nwrd.log" 2>&1
"$python" -m zarai.vision.results_v2 >/dev/null
echo "Done: experiments/wheat_v2_fewshot_nwrd.md and experiments/wheat_v2_fewshot_nwrd_recall.png"
