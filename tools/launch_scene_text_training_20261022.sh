#!/usr/bin/env bash
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022
trap 'touch "$ROOT/TRAINING_NEEDS_ATTENTION"' ERR
exec 8>"$ROOT/training.lock"
flock -n 8 || { echo "Scene-text training launcher is already running" >&2; exit 1; }
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022/checkpoint/checkpoint-800
[[ -s "$PARENT/dcp/.metadata" ]] || { echo "Incomplete parent checkpoint" >&2; exit 1; }
[[ -f "$ROOT/TEXT_DATA_VERIFIED" ]] || { echo "Text data audit missing" >&2; exit 1; }
cd "$REPO"
# Read at launch time: the dependency shell is already running.
# CPU-only HUD completion; never feed presentation pixels to metrics/models.
HUD_ROOTS=(
  /datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022
  /datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022
  /datacc05/lizihao/outputs/h3_world/step2000_subject_text_on_off_portable_20261022
)
"$REPO/.venv/bin/python" tools/visualize_h3_controls.py --roots "${HUD_ROOTS[@]}" --workers 4 \
  > "$ROOT/logs/pretraining_visualization_check.log" 2>&1
for hud_root in "${HUD_ROOTS[@]}"; do
  [[ -f "$hud_root/VISUALIZATION_COMPLETE" ]] || { echo "HUD incomplete: $hud_root" >&2; exit 1; }
done
"$REPO/.venv/bin/python" examples/train/check_minimax_h3_gpus_idle.py
export CONFIG_FILE=$ROOT/training_config.yaml
export TRAIN_DATA=$ROOT/preprocessed/train
export VALIDATION_DATA=$ROOT/preprocessed/validation
export LOG_DIR=$ROOT/runs/train/logs
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=30530 WANDB_MODE=offline
# Variable scene-text lengths exhausted Dynamo's graph cache after step-0
# validation; activation-checkpoint backward then replayed a different graph.
# Keep full activation checkpointing and its determinism checks, but run the
# compiled-decorated LoRA/model functions eagerly for this training recipe.
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
echo "Runtime: eager, full activation checkpointing, SP2, micro1, accumulation8, effective BS32"
mkdir -p "$LOG_DIR"
# The parent contains LoRA and the trained residual-PRoPE branch. Load model
# weights only, not its optimizer, scheduler, RNG, or old step/dataloader state.
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --training.checkpoint.resume_from_checkpoint '' \
  --callbacks.worldmodel_validation.run_at_start "${RUN_INITIAL_VALIDATION:-true}"
touch "$ROOT/TRAINING_COMPLETE"
