#!/usr/bin/env bash
# Sequential training/inference checkpoints; restart only from complete DCP state.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002
OUTPUT="$ROOT/runs/train/checkpoints"
mkdir -p "$ROOT/logs"

wait_for_free_gpus() {
  for ((attempt=1; attempt<=144; attempt++)); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
      return 0
    fi
    echo "$(date --iso-8601=seconds) waiting for all eight GPUs to be free ($attempt/144)"
    sleep 300
  done
  echo "GPUs remained occupied for 12 hours; not touching another process" >&2
  return 1
}

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

train_to() {
  local target="$1" accumulation="$2" prior attempt checkpoint log
  checkpoint="$OUTPUT/checkpoint-$target"
  if [[ -s "$checkpoint/dcp/.metadata" ]]; then
    echo "$(date --iso-8601=seconds) checkpoint-$target already complete"
    return 0
  fi
  for ((attempt=1; attempt<=3; attempt++)); do
    prior="$(latest_complete_before "$target")"
    log="$ROOT/logs/train_to_${target}_attempt_${attempt}.log"
    wait_for_free_gpus
    echo "$(date --iso-8601=seconds) train to $target, resume=$prior, total_bs=$((4 * accumulation))"
    if (( prior > 0 )); then
      if RESUME_CHECKPOINT="$OUTPUT/checkpoint-$prior" TARGET_STEPS="$target" \
          bash "$REPO/examples/train/run_minimax_h3_full_action_strong_20261002.sh" train \
          --training.loop.gradient_accumulation_steps "$accumulation" > "$log" 2>&1; then
        test -s "$checkpoint/dcp/.metadata" && return 0
      fi
    else
      if TARGET_STEPS="$target" \
          bash "$REPO/examples/train/run_minimax_h3_full_action_strong_20261002.sh" train \
          --training.loop.gradient_accumulation_steps "$accumulation" > "$log" 2>&1; then
        test -s "$checkpoint/dcp/.metadata" && return 0
      fi
    fi
    echo "$(date --iso-8601=seconds) phase $target attempt $attempt failed; see $log" >&2
  done
  echo "training to $target failed after three attempts" >&2
  return 1
}

evaluate_at() {
  local step="$1" marker="$ROOT/evaluation/step${step}_heldout_50steps/EVALUATION_COMPLETE"
  if [[ -f "$marker" ]]; then
    echo "$(date --iso-8601=seconds) step-$step evaluation already complete"
    return 0
  fi
  wait_for_free_gpus
  bash "$REPO/examples/train/evaluate_minimax_h3_full_action_strong_20261002.sh" "$step" \
    > "$ROOT/logs/evaluate_step_${step}.log" 2>&1
  test -f "$marker"
}

# Fast feedback at 200 updates, followed by almost one full 65,389-row epoch.
train_to 200 2
evaluate_at 200
train_to 1000 8
evaluate_at 1000
train_to 2200 8
evaluate_at 2200
echo "$(date --iso-8601=seconds) all training and evaluation phases complete"
