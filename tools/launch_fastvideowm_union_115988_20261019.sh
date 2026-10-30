#!/usr/bin/env bash
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019
CONFIG=$REPO/examples/train/configs/minimax_h3_union_20261019.yaml
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-2100
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$ROOT/runs/train/checkpoints/checkpoint-200}
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
LOG_DIR=$ROOT/runs/train/logs

[[ -s "$PARENT/dcp/.metadata" ]] || { echo "incomplete parent checkpoint: $PARENT" >&2; exit 1; }
[[ -s "$RESUME_CHECKPOINT/dcp/.metadata" ]] || { echo "incomplete resume checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }
[[ -s "$ROOT/data_summary.json" ]] || { echo "conditioned union audit missing: $ROOT/data_summary.json" >&2; exit 1; }
[[ -d "$ROOT/preprocessed/all" ]] || { echo "union train data missing" >&2; exit 1; }
[[ -d "$VALIDATION" ]] || { echo "validation data missing" >&2; exit 1; }
"$PY" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"
mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$CONFIG"
export MODEL_PATH=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
export TRAIN_DATA="$ROOT/preprocessed/all"
export VALIDATION_DATA="$VALIDATION"
export LOG_DIR
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export MASTER_PORT=29970

bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --training.checkpoint.resume_from_checkpoint "$RESUME_CHECKPOINT" \
  --training.loop.max_train_steps 3000 \
  --training.checkpoint.output_dir "$ROOT/runs/train/checkpoints" \
  --training.checkpoint.training_state_checkpointing_steps 100 \
  --training.tracker.run_name h3_union_115988_20261019
