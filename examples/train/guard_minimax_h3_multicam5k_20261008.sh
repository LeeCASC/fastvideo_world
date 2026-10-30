#!/usr/bin/env bash
# Resume the same 15,719-row run after a process failure; never overlap a GPU owner.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
OUTPUT="$ROOT/runs/train/checkpoints"
TARGET="${TARGET_STEPS:-1000}"
LOG="$ROOT/guard.log"
NEEDS_ATTENTION="$ROOT/GUARD_NEEDS_ATTENTION"

latest_complete() {
  local checkpoint number best=0
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    [[ -d "$checkpoint" && -s "$checkpoint/dcp/.metadata" ]] || continue
    number="${checkpoint##*/checkpoint-}"
    [[ "$number" =~ ^[0-9]+$ ]] || continue
    if (( number > best )); then best="$number"; fi
  done
  printf '%s' "$best"
}

mkdir -p "$ROOT"
for ((attempt=1; attempt<=5; attempt++)); do
  step="$(latest_complete)"
  if (( step >= TARGET )); then
    echo "$(date --iso-8601=seconds) checkpoint-$step complete" | tee -a "$LOG"
    exit 0
  fi
  while ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; do
    echo "$(date --iso-8601=seconds) GPUs occupied; waiting 300 s" | tee -a "$LOG"
    sleep 300
  done
  echo "$(date --iso-8601=seconds) attempt=$attempt resume=$step target=$TARGET BS=32" | tee -a "$LOG"
  run_log="$ROOT/runs/train/logs/${GUARD_LOG_PREFIX:-guard}_attempt_${attempt}.log"
  mkdir -p "$(dirname "$run_log")"
  if (( step > 0 )); then
    if RESUME_CHECKPOINT="$OUTPUT/checkpoint-$step" TARGET_STEPS="$TARGET" \
      bash "$REPO/examples/train/run_minimax_h3_multicam5k_20261008.sh" train > "$run_log" 2>&1; then
      continue
    fi
  elif TARGET_STEPS="$TARGET" bash "$REPO/examples/train/run_minimax_h3_multicam5k_20261008.sh" \
      train > "$run_log" 2>&1; then
    continue
  fi
  echo "$(date --iso-8601=seconds) attempt=$attempt failed; see $run_log" | tee -a "$LOG" >&2
  sleep 60
done
touch "$NEEDS_ATTENTION"
echo "Five attempts failed; manual review required" | tee -a "$LOG" >&2
exit 1
