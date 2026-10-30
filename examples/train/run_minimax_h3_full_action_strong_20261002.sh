#!/usr/bin/env bash
# Eight-GPU, full-cache action training. Smoke and train artifacts never overlap.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
VAE=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
MODE="${1:-train}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  smoke) STEPS=2; SAVE_EVERY=1; PORT=29978; OUTPUT="$ROOT/runs/smoke/checkpoints" ;;
  train) STEPS="${TARGET_STEPS:-2000}"; SAVE_EVERY=100; PORT=29979; OUTPUT="$ROOT/runs/train/checkpoints" ;;
  *) echo "Expected smoke or train, got $MODE" >&2; exit 2 ;;
esac
LOG_DIR="$ROOT/runs/$MODE/logs"

test -f "$FULL/CONDITIONED_VERIFIED" || { echo "conditioned cache not verified" >&2; exit 1; }
test -f "$VAE/VAE_FULL_AUDIT_COMPLETE" || { echo "full VAE audit missing" >&2; exit 1; }
test -f "$VAE/VAE_INTRINSICS_AUDIT_COMPLETE" || { echo "intrinsics audit missing" >&2; exit 1; }
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    test ! -e "$checkpoint" || { echo "existing output; refusing fresh restart" >&2; exit 1; }
  done
else
  test -s "$RESUME_CHECKPOINT/dcp/.metadata" || { echo "resume checkpoint incomplete" >&2; exit 1; }
  [[ "$RESUME_CHECKPOINT" == "$OUTPUT"/checkpoint-* ]] || {
    echo "resume checkpoint must belong to this run" >&2; exit 1;
  }
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "GPU process already present; refusing overlapping eight-GPU launch" >&2
  exit 1
fi
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--training.checkpoint.resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
smoke_args=()
if [[ "$MODE" == smoke ]]; then
  smoke_args=(--callbacks.worldmodel_validation.num_batches 2)
fi
mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_full_action_strong_20261002.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT="$PORT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --training.loop.max_train_steps "$STEPS" \
  --training.checkpoint.output_dir "$OUTPUT" \
  --training.checkpoint.training_state_checkpointing_steps "$SAVE_EVERY" \
  --training.tracker.run_name "h3_full_action_strong_20261002_$MODE" \
  "${resume_args[@]}" "${smoke_args[@]}" "$@"
