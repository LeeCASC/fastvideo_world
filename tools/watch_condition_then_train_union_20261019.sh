#!/usr/bin/env bash
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019
MARKER=$ROOT/new_conditioned/CONDITIONING_COMPLETE
LOG=$ROOT/union_train_watchdog.log
ATTENTION=$ROOT/TRAINING_NEEDS_ATTENTION

echo "$(date --iso-8601=seconds) waiting for conditioned 50,599-row cache" | tee -a "$LOG"
while [[ ! -s "$MARKER" ]]; do
  sleep 300
done
echo "$(date --iso-8601=seconds) conditioning complete; auditing union" | tee -a "$LOG"
if ! "$PY" "$REPO/tools/prepare_fastvideowm_conditioned_union_20261019.py" >> "$LOG" 2>&1; then
  echo "$(date --iso-8601=seconds) conditioned union audit failed" | tee -a "$LOG" >&2
  touch "$ATTENTION"
  exit 1
fi
echo "$(date --iso-8601=seconds) union audit passed; waiting for idle GPUs" | tee -a "$LOG"
while ! "$PY" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; do
  echo "$(date --iso-8601=seconds) GPUs occupied; waiting 300 s" | tee -a "$LOG"
  sleep 300
done
echo "$(date --iso-8601=seconds) launching FastVideoWM union continuation" | tee -a "$LOG"
exec bash "$REPO/tools/launch_fastvideowm_union_115988_20261019.sh" >> "$LOG" 2>&1
