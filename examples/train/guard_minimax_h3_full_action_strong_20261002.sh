#!/usr/bin/env bash
# Failover for the active schedule: keep training if a staged evaluation fails.
# Never overlap the primary scheduler or another GPU owner's work.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002
OUTPUT="$ROOT/runs/train/checkpoints"
PRIMARY=fastvideowm_strong_action_full_schedule_20261002
LOG="$ROOT/guard.log"
NEEDS_ATTENTION="$ROOT/GUARD_NEEDS_ATTENTION"
trap 'status=$?; if (( status != 0 )); then touch "$NEEDS_ATTENTION"; fi' EXIT

latest_complete_before() {
  local target="$1" checkpoint step latest=0
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    [[ -d "$checkpoint" ]] || continue
    step="${checkpoint##*/checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step < target && step > latest )) && [[ -s "$checkpoint/dcp/.metadata" ]]; then
      latest="$step"
    fi
  done
  printf '%s' "$latest"
}

wait_for_free_gpus() {
  local attempt
  for ((attempt=1; attempt<=288; attempt++)); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
      return 0
    fi
    echo "$(date --iso-8601=seconds) GPUs occupied; waiting ($attempt/288)" | tee -a "$LOG"
    sleep 300
  done
  echo "GPUs occupied for 24 hours; manual review required" | tee -a "$LOG" >&2
  return 1
}

resume_to() {
  local target="$1" attempt prior phase_log
  [[ -s "$OUTPUT/checkpoint-$target/dcp/.metadata" ]] && return 0
  for ((attempt=1; attempt<=5; attempt++)); do
    wait_for_free_gpus
    prior="$(latest_complete_before "$target")"
    phase_log="$ROOT/logs/guard_to_${target}_attempt_${attempt}.log"
    echo "$(date --iso-8601=seconds) fallback target=$target resume=$prior BS=32 attempt=$attempt" | tee -a "$LOG"
    if (( prior > 0 )); then
      if RESUME_CHECKPOINT="$OUTPUT/checkpoint-$prior" TARGET_STEPS="$target" \
        bash "$REPO/examples/train/run_minimax_h3_full_action_strong_20261002.sh" train \
          --training.loop.gradient_accumulation_steps 8 > "$phase_log" 2>&1; then
        [[ -s "$OUTPUT/checkpoint-$target/dcp/.metadata" ]] && return 0
      fi
    else
      if TARGET_STEPS="$target" \
        bash "$REPO/examples/train/run_minimax_h3_full_action_strong_20261002.sh" train \
          --training.loop.gradient_accumulation_steps 8 > "$phase_log" 2>&1; then
        [[ -s "$OUTPUT/checkpoint-$target/dcp/.metadata" ]] && return 0
      fi
    fi
    echo "$(date --iso-8601=seconds) attempt failed; see $phase_log" | tee -a "$LOG" >&2
    sleep 300
  done
  echo "$(date --iso-8601=seconds) five attempts failed for step $target" | tee -a "$LOG" >&2
  touch "$NEEDS_ATTENTION"
  return 1
}

mkdir -p "$ROOT/logs"
echo "$(date --iso-8601=seconds) guard active; primary=$PRIMARY" | tee -a "$LOG"
while tmux has-session -t "$PRIMARY" 2>/dev/null; do
  sleep 300
done
echo "$(date --iso-8601=seconds) primary schedule exited; checking final checkpoint" | tee -a "$LOG"
if [[ -s "$OUTPUT/checkpoint-2200/dcp/.metadata" ]]; then
  echo "$(date --iso-8601=seconds) checkpoint-2200 complete; no fallback needed" | tee -a "$LOG"
  exit 0
fi
resume_to 1000
resume_to 2200
echo "$(date --iso-8601=seconds) fallback reached checkpoint-2200" | tee -a "$LOG"
