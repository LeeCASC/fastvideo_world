#!/usr/bin/env bash
# Eight-GPU short camera-control curriculum from a frozen 512-row index.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
VAE=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
MODE="${1:-train}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  smoke) STEPS=2; SAVE_EVERY=1; PORT=29975 ;;
  train) STEPS=200; SAVE_EVERY=25; PORT=29976 ;;
  *) echo "Expected smoke or train, got $MODE" >&2; exit 2 ;;
esac
OUTPUT="$ROOT/runs/stage1_8gpu_$MODE/checkpoints"
LOG_DIR="$ROOT/runs/stage1_8gpu_$MODE/logs"

test -f "$FULL/CONDITIONED_VERIFIED" || { echo "conditioned cache not verified" >&2; exit 1; }
test -f "$VAE/VAE_FULL_AUDIT_COMPLETE" || { echo "full VAE audit missing" >&2; exit 1; }
test -f "$VAE/VAE_INTRINSICS_AUDIT_COMPLETE" || { echo "intrinsics audit missing" >&2; exit 1; }
test -s "$ROOT/data_plan_v1/train_indices.json" || { echo "simple-camera index missing" >&2; exit 1; }
"$REPO/.venv/bin/python" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert len(p)==512 and len(set(p))==512' \
  "$ROOT/data_plan_v1/train_indices.json"
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    test ! -e "$checkpoint" || { echo "existing output; refusing fresh restart" >&2; exit 1; }
  done
else
  test -s "$RESUME_CHECKPOINT/dcp/.metadata" || { echo "resume checkpoint incomplete" >&2; exit 1; }
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "GPU process already present; refusing overlapping eight-GPU launch" >&2
  exit 1
fi
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--training.checkpoint.resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_simple_camera_20261002.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT="$PORT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --training.loop.max_train_steps "$STEPS" \
  --training.checkpoint.output_dir "$OUTPUT" \
  --training.checkpoint.training_state_checkpointing_steps "$SAVE_EVERY" \
  --training.tracker.run_name "h3_simple_camera_20261002_$MODE" \
  "${resume_args[@]}" \
  "$@"
