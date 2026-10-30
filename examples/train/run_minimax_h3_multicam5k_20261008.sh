#!/usr/bin/env bash
# Train the frozen 15,719-row action-only subset without changing H3 caches.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
VAE=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/runs/train/checkpoints/checkpoint-1300
MODE="${1:-train}"
case "$MODE" in
  smoke) STEPS=1; SAVE_EVERY=1; PORT=29985 ;;
  train) STEPS="${TARGET_STEPS:-1000}"; SAVE_EVERY=100; PORT=29984 ;;
  *) echo "Expected smoke or train, got $MODE" >&2; exit 2 ;;
esac
OUTPUT="$ROOT/runs/$MODE/checkpoints"
LOG_DIR="$ROOT/runs/$MODE/logs"

test -s "$PARENT/dcp/.metadata" || { echo "Parent checkpoint incomplete" >&2; exit 1; }
test -f "$FULL/CONDITIONED_VERIFIED" || { echo "Conditioned cache not verified" >&2; exit 1; }
test -f "$VAE/VAE_FULL_AUDIT_COMPLETE" || { echo "Full VAE audit missing" >&2; exit 1; }
test -f "$VAE/VAE_INTRINSICS_AUDIT_COMPLETE" || { echo "Intrinsics audit missing" >&2; exit 1; }
test -s "$PLAN/sampling_indices.json" || { echo "Frozen sampling plan missing" >&2; exit 1; }
test -s "$PLAN/report.json" || { echo "Frozen sampling report missing" >&2; exit 1; }
[[ "$(jq -r '.selected_unique' "$PLAN/report.json")" == 15719 ]] || {
  echo "Unexpected subset size" >&2; exit 1;
}
[[ "$(jq -r '.selected_multicam' "$PLAN/report.json")" == 5000 ]] || {
  echo "Unexpected multicam count" >&2; exit 1;
}
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  for checkpoint in "$OUTPUT"/checkpoint-*; do
    test ! -e "$checkpoint" || { echo "Existing output; refusing fresh restart" >&2; exit 1; }
  done
else
  test -s "$RESUME_CHECKPOINT/dcp/.metadata" || { echo "Resume checkpoint incomplete" >&2; exit 1; }
  [[ "$RESUME_CHECKPOINT" == "$OUTPUT"/checkpoint-* ]] || {
    echo "Resume checkpoint must belong to this run" >&2; exit 1;
  }
fi
if ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; then
  echo "GPUs are occupied or CUDA is unavailable; refusing overlapping eight-GPU launch" >&2
  exit 1
fi

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--training.checkpoint.resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
validation_args=()
if [[ "$MODE" == smoke ]]; then
  validation_args=(--callbacks.worldmodel_validation.num_batches 2)
fi
mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT="$PORT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --models.student.sampling_index_path "$PLAN/sampling_indices.json" \
  --training.loop.max_train_steps "$STEPS" \
  --training.checkpoint.output_dir "$OUTPUT" \
  --training.checkpoint.training_state_checkpointing_steps "$SAVE_EVERY" \
  --training.tracker.run_name "h3_multicam5k_20261008_$MODE" \
  "${resume_args[@]}" "${validation_args[@]}"
