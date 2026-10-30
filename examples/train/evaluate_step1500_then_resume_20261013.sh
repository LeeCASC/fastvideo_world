#!/usr/bin/env bash
# Pause the existing guarded eight-GPU trainer, run the frozen JS-easy suite,
# then release the guard so it resumes from the newest complete checkpoint.
set -Eeuo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
TRAIN_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
CHECKPOINT="$TRAIN_ROOT/runs/train/checkpoints/checkpoint-1500"
OUT=/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013
TRAIN_SESSION=fastvideowm_multicam5k_to5000_20261012
GUARD_PID="${GUARD_PID:?Pass the verified trainer guard PID}"
TORCHRUN_PID="${TORCHRUN_PID:?Pass the verified torchrun PID}"
guard_paused=0

resume_guard() {
  local status=$?
  trap - EXIT
  if (( guard_paused )); then
    if kill -0 "$GUARD_PID" 2>/dev/null; then
      kill -CONT "$GUARD_PID"
      echo "$(date --iso-8601=seconds) released training guard PID=$GUARD_PID"
    else
      echo "$(date --iso-8601=seconds) training guard disappeared; manual restart required" >&2
      touch "$OUT/TRAIN_RESUME_NEEDS_ATTENTION"
    fi
  fi
  if (( status != 0 )); then
    touch "$OUT/EVAL_NEEDS_ATTENTION"
    echo "$(date --iso-8601=seconds) step-1500 evaluation failed with status=$status" >&2
  fi
  exit "$status"
}
trap resume_guard EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT"
test -s "$CHECKPOINT/dcp/.metadata"
for rank in {0..7}; do test -s "$CHECKPOINT/rng_state_rank${rank}.pt"; done
tmux has-session -t "$TRAIN_SESSION"
[[ "$(ps -o user= -p "$GUARD_PID" | xargs)" == "$(id -un)" ]]
[[ "$(ps -o args= -p "$GUARD_PID")" == *guard_minimax_h3_multicam5k_20261008.sh* ]]
[[ "$(ps -o user= -p "$TORCHRUN_PID" | xargs)" == "$(id -un)" ]]
[[ "$(ps -o args= -p "$TORCHRUN_PID")" == *torch.distributed.run*--master_port\ 29984* ]]

echo "$(date --iso-8601=seconds) pausing trainer at checkpoint-1500"
kill -STOP "$GUARD_PID"
guard_paused=1
kill -TERM "$TORCHRUN_PID"

idle=0
for _ in {1..90}; do
  if "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; then
    idle=1
    break
  fi
  sleep 10
done
if (( ! idle )); then
  echo "GPU0-7 did not become idle within 15 minutes" >&2
  exit 1
fi

echo "$(date --iso-8601=seconds) starting step-1500 JS-easy evaluation"
systemd-run --user --scope --quiet \
  -p MemoryMax=650G -p MemorySwapMax=0 \
  env CHECKPOINT="$CHECKPOINT" OUT="$OUT" \
  bash "$REPO/examples/train/evaluate_minimax_h3_js_easy16_step1200_20261012.sh"
test -f "$OUT/CORE_METRICS_COMPLETE"
echo "$(date --iso-8601=seconds) step-1500 evaluation complete: $OUT"
