#!/usr/bin/env bash
# Re-run the frozen JS-easy suite after the multicam5k continuation reaches step 5000.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
TRAIN=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
OUT=/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step5000_20261012
TRAIN_SESSION=fastvideowm_multicam5k_to5000_20261012
CHECKPOINT="$TRAIN/runs/train/checkpoints/checkpoint-5000"

on_exit() {
  local status=$?
  if (( status != 0 )); then
    touch "$OUT/AUTO_EVAL_NEEDS_ATTENTION"
    echo "$(date --iso-8601=seconds) step-5000 auto-evaluation failed; inspect $OUT/auto_eval.log" >&2
  fi
}
mkdir -p "$OUT"
trap on_exit EXIT

if [[ -f "$OUT/AUTO_EVAL_CORE_COMPLETE" ]]; then
  echo "$(date --iso-8601=seconds) step-5000 evaluation already complete"
  exit 0
fi

echo "$(date --iso-8601=seconds) waiting for step-5000 training session"
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 300
done
if [[ ! -s "$CHECKPOINT/dcp/.metadata" ]] || \
   ! rg -q 'checkpoint-5000 complete' "$TRAIN/extend_to_5000.log"; then
  echo "Step 5000 did not fully complete; refusing inference" >&2
  exit 1
fi

echo "$(date --iso-8601=seconds) step 5000 complete; waiting for idle GPU0–7"
while ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; do
  sleep 300
done

for attempt in 1 2; do
  echo "$(date --iso-8601=seconds) evaluation attempt=$attempt"
  if systemd-run --user --scope --quiet \
      -p MemoryMax=650G -p MemorySwapMax=0 \
      env CHECKPOINT="$CHECKPOINT" OUT="$OUT" \
      bash "$REPO/examples/train/evaluate_minimax_h3_js_easy16_step1200_20261012.sh"; then
    touch "$OUT/AUTO_EVAL_CORE_COMPLETE"
    echo "$(date --iso-8601=seconds) step-5000 inference and core metrics complete"
    exit 0
  fi
  echo "$(date --iso-8601=seconds) evaluation attempt=$attempt failed" >&2
  sleep 120
  while ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; do
    sleep 300
  done
done
exit 1
