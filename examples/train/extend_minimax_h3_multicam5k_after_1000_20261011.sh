#!/usr/bin/env bash
# Let the existing eight-GPU run finish, then resume the same run to step 1200.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
OUTPUT="$ROOT/runs/train/checkpoints"
CURRENT_GUARD=fastvideowm_multicam5k_guard_20261008
NEEDS_ATTENTION="$ROOT/EXTENSION_TO_1200_NEEDS_ATTENTION"

on_exit() {
  local status=$?
  if (( status != 0 )); then
    touch "$NEEDS_ATTENTION"
    echo "$(date --iso-8601=seconds) extension failed with status=$status; review required" >&2
  fi
}
trap on_exit EXIT

echo "$(date --iso-8601=seconds) waiting for $CURRENT_GUARD to finish step 1000"
while tmux has-session -t "$CURRENT_GUARD" 2>/dev/null; do
  sleep 60
done

if [[ -e "$ROOT/GUARD_NEEDS_ATTENTION" ]]; then
  echo "The step-1000 guard needs attention; refusing automatic continuation" >&2
  exit 1
fi
if [[ ! -s "$OUTPUT/checkpoint-1000/dcp/.metadata" ]]; then
  echo "Complete step-1000 checkpoint missing; refusing automatic continuation" >&2
  exit 1
fi

echo "$(date --iso-8601=seconds) step 1000 complete; resuming to step 1200 with BS=32"
GUARD_LOG_PREFIX=extend1200 TARGET_STEPS=1200 \
  bash "$REPO/examples/train/guard_minimax_h3_multicam5k_20261008.sh"
if [[ ! -s "$OUTPUT/checkpoint-1200/dcp/.metadata" ]]; then
  echo "Complete step-1200 checkpoint missing after the guard returned" >&2
  exit 1
fi
echo "$(date --iso-8601=seconds) step 1200 checkpoint complete"
