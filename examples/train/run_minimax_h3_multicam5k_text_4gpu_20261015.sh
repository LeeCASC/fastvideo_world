#!/usr/bin/env bash
# Resume the text-conditioned run from a complete 8-GPU checkpoint on GPUs 0-3.
# Four GPUs use gradient accumulation 16 so effective BS remains 64.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
OUTPUT_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015_4gpu
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-2100
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

[[ -n "$RESUME_CHECKPOINT" && -s "$RESUME_CHECKPOINT/dcp/.metadata" ]] || {
  echo "A complete RESUME_CHECKPOINT is required" >&2
  exit 1
}
test -s "$PARENT/dcp/.metadata" || { echo "Incomplete model parent checkpoint" >&2; exit 1; }
test -f "$FULL/CONDITIONED_VERIFIED" || { echo "Conditioned cache not verified" >&2; exit 1; }
test -s "$PLAN/sampling_indices.json" || { echo "Frozen sampling plan missing" >&2; exit 1; }
test -s "$PLAN/report.json" || { echo "Sampling report missing" >&2; exit 1; }
[[ "$(jq -r '.selected_unique' "$PLAN/report.json")" == 15719 ]] || { echo "Unexpected subset size" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT/runs/train/logs"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR="$OUTPUT_ROOT/runs/train/logs" WANDB_MODE=offline NUM_GPUS=4 MASTER_PORT=29987
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --models.student.sampling_index_path "$PLAN/sampling_indices.json" \
  --models.student.vision_only_conditioning false \
  --training.distributed.num_gpus 4 \
  --training.distributed.sp_size 2 \
  --training.distributed.tp_size 1 \
  --training.distributed.hsdp_replicate_dim 2 \
  --training.distributed.hsdp_shard_dim 2 \
  --training.data.data_path "$TRAIN_DATA" \
  --training.loop.gradient_accumulation_steps 16 \
  --training.loop.max_train_steps 3000 \
  --training.checkpoint.resume_from_checkpoint "$RESUME_CHECKPOINT" \
  --training.checkpoint.output_dir "$OUTPUT_ROOT/runs/train/checkpoints" \
  --training.checkpoint.training_state_checkpointing_steps 100 \
  --training.tracker.run_name h3_multicam5k_text_20261015_4gpu \
  --callbacks.worldmodel_validation.data_path "$VALIDATION_DATA"
