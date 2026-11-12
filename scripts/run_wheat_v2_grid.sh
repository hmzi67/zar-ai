#!/usr/bin/env bash
# Train the wheat_v2 grid on every LOSO fold, then rebuild experiments/wheat_v2_results.md.
#
# Usage:
#   bash scripts/run_wheat_v2_grid.sh            # first grid, about 11-15 GPU hours
#   bash scripts/run_wheat_v2_grid.sh full       # whole grid, about 40-55 GPU hours
#   SEEDS=42 bash scripts/run_wheat_v2_grid.sh   # one seed only
#   bash scripts/run_wheat_v2_grid.sh first train.log_checkpoint=false   # extra key=value overrides
#
# Safe to rerun: a (fold, seed) that already finished on this dataset version is skipped, so after
# a crash or reboot just run the same command again. One failed configuration doesn't stop the rest.
# Run it inside tmux or with nohup so it survives closing the terminal. Logs: experiments/runs/grid_logs/
set -uo pipefail
cd "$(dirname "$0")/.."

python=.venv/bin/python
grid=${1:-first}
shift || true
seeds=${SEEDS:-42,0,1}
log_dir=experiments/runs/grid_logs
mkdir -p "$log_dir"

# backbone mode aug
first=(
  "efficientnetv2_s full basic"
  "efficientnetv2_s full strong"
  "dinov2_vit_b14 lora basic"
  "dinov2_vit_b14 lora strong"
  "dinov2_vit_b14 full basic"
  "dinov2_vit_b14 full strong"
  "siglip2_vit_b16 full basic"
  "siglip2_vit_b16 full strong"
  "dinov2_vit_l14 linear basic"
)
full=()
for backbone in efficientnetv2_s dinov2_vit_s14 dinov2_vit_b14 dinov2_vit_l14 siglip2_vit_b16; do
  for mode in linear lora full; do
    [[ $backbone == efficientnetv2_s && $mode == lora ]] && continue # no attention layers
    for aug in basic strong; do
      full+=("$backbone $mode $aug")
    done
  done
done

case $grid in
  first) combos=("${first[@]}") ;;
  full) combos=("${full[@]}") ;;
  *) echo "usage: $0 [first|full] [key=value ...]" >&2; exit 2 ;;
esac

echo "Checking the dataset and loaders (leakage asserts) before training"
if ! "$python" dataloader.py --protocol loso --skip-preprocess --batches 5 >"$log_dir/dataloader_check.log" 2>&1; then
  echo "dataloader check failed: see $log_dir/dataloader_check.log" >&2
  exit 1
fi

failed=()
n=0
started=$(date +%s)
for combo in "${combos[@]}"; do
  read -r backbone mode aug <<<"$combo"
  n=$((n + 1))
  log="$log_dir/${backbone}_${mode}_${aug}.log"
  echo "[$(date '+%F %T')] ($n/${#combos[@]}) $backbone / $mode / $aug, every fold, seeds $seeds -> $log"
  "$python" train.py --backbone "$backbone" --mode "$mode" --aug "$aug" --fold all \
    --seeds "$seeds" --skip-done --no-results "$@" >>"$log" 2>&1
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "  FAILED (exit $status): tail -50 $log"
    failed+=("$combo")
  else
    grep -E "held-out macro-F1|skipping" "$log" | tail -12 | sed 's/^/  /'
  fi
done

echo "Rebuilding the results table"
"$python" -m zarai.vision.results_v2
echo "Done in $((($(date +%s) - started) / 60)) min. Results: experiments/wheat_v2_results.md"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed configurations (rerun the same command to retry only these):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
