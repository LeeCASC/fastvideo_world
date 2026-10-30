#!/usr/bin/env bash
# Continue the text-conditioned multicam5k run from complete checkpoint-300.
# Eight GPUs, SP=2 -> four data-parallel replicas; micro-batch 1 and
# accumulation 8 therefore keep effective total batch size at 32.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
FULL=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-2100
RESUME=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015_4gpu/runs/train/checkpoints/checkpoint-300
OUTPUT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261016_8gpu

test -s "$PARENT/dcp/.metadata" || { echo "Incomplete model parent checkpoint" >&2; exit 1; }
test -s "$RESUME/dcp/.metadata" || { echo "Incomplete resume checkpoint: $RESUME" >&2; exit 1; }
test -f "$FULL/CONDITIONED_VERIFIED" || { echo "Conditioned cache not verified" >&2; exit 1; }
test -s "$PLAN/sampling_indices.json" || { echo "Frozen sampling plan missing" >&2; exit 1; }
test -s "$PLAN/report.json" || { echo "Sampling report missing" >&2; exit 1; }
[[ "$(jq -r '.selected_unique' "$PLAN/report.json")" == 15719 ]] || {
  echo "Unexpected subset size" >&2
  exit 1
}

if ! "$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; then
  echo "GPUs are occupied or CUDA is unavailable; refusing overlapping launch" >&2
  exit 1
fi

mkdir -p "$OUTPUT/runs/train/logs"
cd "$REPO"
export CONFIG_FILE="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
export TRAIN_DATA="$FULL/preprocessed/all"
export VALIDATION_DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
export LOG_DIR="$OUTPUT/runs/train/logs" WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29986
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --models.student.sampling_index_path "$PLAN/sampling_indices.json" \
  --models.student.vision_only_conditioning false \
  --training.distributed.num_gpus 8 \
  --training.distributed.sp_size 2 \
  --training.distributed.tp_size 1 \
  --training.distributed.hsdp_replicate_dim 2 \
  --training.distributed.hsdp_shard_dim 4 \
  --training.data.data_path "$TRAIN_DATA" \
  --training.loop.gradient_accumulation_steps 8 \
  --training.loop.max_train_steps 3000 \
  --training.checkpoint.resume_from_checkpoint "$RESUME" \
  --training.checkpoint.output_dir "$OUTPUT/runs/train/checkpoints" \
  --training.checkpoint.training_state_checkpointing_steps 100 \
  --training.tracker.run_name h3_multicam5k_text_20261016_8gpu \
  --callbacks.worldmodel_validation.data_path "$VALIDATION_DATA"
