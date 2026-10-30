#!/usr/bin/env bash
# Check the eight-GPU run every five minutes; recover once from a complete DCP.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002
OUTPUT="$ROOT/runs/postadaln_8gpu_train/checkpoints"
EVAL="$ROOT/evaluation/step300_heldout_50steps/EVALUATION_COMPLETE"
SESSION=fastvideowm_multiaction_train_eval_20261002
LOG="$ROOT/logs/watchdog.log"
attempt=0
eval_attempt=0
while true; do
  if test -f "$EVAL"; then
    printf '%s evaluation complete\n' "$(date --iso-8601=seconds)" >> "$LOG"
    exit 0
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    latest=0
    for checkpoint in "$OUTPUT"/checkpoint-*; do
      if test -s "$checkpoint/dcp/.metadata"; then
        step="${checkpoint##*-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest )); then latest="$step"; fi
      fi
    done
    printf '%s active; latest_complete_step=%s\n' "$(date --iso-8601=seconds)" "$latest" >> "$LOG"
    sleep 300
    continue
  fi
  # Never overlap another GPU owner; the first launch was already checked.
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    printf '%s run absent; GPUs occupied; waiting\n' "$(date --iso-8601=seconds)" >> "$LOG"
    sleep 300
    continue
  fi
  latest=0
  latest_path=''
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    if test -s "$checkpoint/dcp/.metadata"; then
      step="${checkpoint##*-}"
      if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest )); then
        latest="$step"
        latest_path="$checkpoint"
      fi
    fi
  done
  if (( latest >= 300 )); then
    if (( eval_attempt >= 2 )); then
      printf '%s evaluation failed twice; manual inspection required\n' "$(date --iso-8601=seconds)" >> "$LOG"
      touch "$ROOT/WATCHDOG_NEEDS_ATTENTION"
      exit 1
    fi
    eval_attempt=$((eval_attempt + 1))
    printf '%s trainer exited after step=%s; relaunching evaluation\n' "$(date --iso-8601=seconds)" "$latest" >> "$LOG"
    tmux new-session -d -s "$SESSION" \
      "bash '$REPO/examples/train/evaluate_minimax_h3_multiaction_20261002.sh' >> '$ROOT/logs/evaluation_launcher.log' 2>&1"
  elif (( latest > 0 && attempt < 2 )); then
    attempt=$((attempt + 1))
    printf '%s trainer exited early; resuming checkpoint-%s attempt=%s\n' \
      "$(date --iso-8601=seconds)" "$latest" "$attempt" >> "$LOG"
    tmux new-session -d -s "$SESSION" \
      "RESUME_CHECKPOINT='$latest_path' bash '$REPO/examples/train/run_minimax_h3_multiaction_20261002.sh' train > '$ROOT/logs/train_resume_${attempt}.log' 2>&1 && bash '$REPO/examples/train/evaluate_minimax_h3_multiaction_20261002.sh' >> '$ROOT/logs/evaluation_launcher.log' 2>&1"
  else
    printf '%s unresolved failure; latest_complete_step=%s attempts=%s\n' \
      "$(date --iso-8601=seconds)" "$latest" "$attempt" >> "$LOG"
    touch "$ROOT/WATCHDOG_NEEDS_ATTENTION"
    exit 1
  fi
  sleep 300
done
