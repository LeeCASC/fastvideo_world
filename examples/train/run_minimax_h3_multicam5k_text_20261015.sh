#!/usr/bin/env bash
# Fresh step counter: initialize from checkpoint-2100 weights, enable the
# precomputed VLM caption/text embeddings, and train a new 3,000-step run.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-2100

test -s "$PARENT/dcp/.metadata" || { echo "Incomplete checkpoint-2100" >&2; exit 1; }
test -f "$FULL/CONDITIONED_VERIFIED" || { echo "Conditioned cache not verified" >&2; exit 1; }
test -s "$PLAN/sampling_indices.json" || { echo "Frozen sampling plan missing" >&2; exit 1; }
test -s "$PLAN/report.json" || { echo "Sampling report missing" >&2; exit 1; }
[[ "$(jq -r '.selected_unique' "$PLAN/report.json")" == 15719 ]] || { echo "Unexpected subset size" >&2; exit 1; }

if ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; then
  echo "GPUs are occupied or CUDA is unavailable; refusing overlapping launch" >&2
  exit 1
fi

OUTPUT="$ROOT/runs/train/checkpoints"
LOG_DIR="$ROOT/runs/train/logs"
mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29986
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# init_from_training_checkpoint loads model weights only. No resume_from_checkpoint
# is supplied, so optimizer/scheduler/global_step start from zero.
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --models.student.sampling_index_path "$PLAN/sampling_indices.json" \
  --models.student.vision_only_conditioning false \
  --training.loop.max_train_steps 3000 \
  --training.checkpoint.output_dir "$OUTPUT" \
  --training.checkpoint.training_state_checkpointing_steps 100 \
  --training.tracker.run_name h3_multicam5k_text_20261015 \
  --training.data.data_path "$TRAIN_DATA" \
  --callbacks.worldmodel_validation.data_path "$VALIDATION_DATA"
