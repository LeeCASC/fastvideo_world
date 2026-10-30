#!/usr/bin/env bash
# Continue the existing step-1300 run only after the JS-easy sweep succeeds.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003
EVAL="$ROOT/evaluation/js_easy_benchmark_step1300_20261008"
CHECKPOINTS="$ROOT/runs/train/checkpoints"
LOG="$ROOT/resume_after_js_benchmark_20261008.log"
NEEDS_ATTENTION="$ROOT/RESUME_AFTER_JS_BENCHMARK_NEEDS_ATTENTION"
TARGET=3000
EVAL_SESSION=fastvideowm_js_easy_benchmark_step1300_20261008

mkdir -p "$ROOT/runs/train/logs"

latest_complete() {
  local checkpoint number best=0
  for checkpoint in "$CHECKPOINTS"/checkpoint-*; do
    [[ -d "$checkpoint" && -s "$checkpoint/dcp/.metadata" ]] || continue
    number="${checkpoint##*/checkpoint-}"
    [[ "$number" =~ ^[0-9]+$ ]] || continue
    if (( number > best )); then best="$number"; fi
  done
  printf '%s' "$best"
}

while [[ ! -f "$EVAL/EVALUATION_COMPLETE" ]]; do
  if ! tmux has-session -t "$EVAL_SESSION" 2>/dev/null; then
    echo "$(date --iso-8601=seconds) inference ended without completion marker; not starting training" | tee -a "$LOG" >&2
    touch "$NEEDS_ATTENTION"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) waiting for JS-easy inference; next check in 300 s" | tee -a "$LOG"
  sleep 300
done
echo "$(date --iso-8601=seconds) JS-easy inference complete; eligible to resume training" | tee -a "$LOG"

for ((attempt=1; attempt<=5; attempt++)); do
  step="$(latest_complete)"
  if (( step >= TARGET )); then
    echo "$(date --iso-8601=seconds) checkpoint-$step complete" | tee -a "$LOG"
    exit 0
  fi
  if (( step < 1300 )); then
    echo "$(date --iso-8601=seconds) checkpoint regressed to step-$step; not starting training" | tee -a "$LOG" >&2
    touch "$NEEDS_ATTENTION"
    exit 1
  fi
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    echo "$(date --iso-8601=seconds) GPUs occupied; next check in 300 s" | tee -a "$LOG"
    sleep 300
  done
  run_log="$ROOT/runs/train/logs/after_js_benchmark_20261008_attempt_${attempt}.log"
  echo "$(date --iso-8601=seconds) attempt=$attempt resume=checkpoint-$step target=$TARGET BS=32" | tee -a "$LOG"
  if RESUME_CHECKPOINT="$CHECKPOINTS/checkpoint-$step" TARGET_STEPS="$TARGET" \
    bash "$REPO/examples/train/run_minimax_h3_full_residual_prope_20261003.sh" train > "$run_log" 2>&1; then
    continue
  fi
  echo "$(date --iso-8601=seconds) attempt=$attempt failed; see $run_log" | tee -a "$LOG" >&2
  sleep 60
done
touch "$NEEDS_ATTENTION"
echo "Five resume attempts failed; manual review required" | tee -a "$LOG" >&2
exit 1
