#!/usr/bin/env bash
# Finish the frozen action-only pilot and start 100 training steps after VAE.
set -euo pipefail

REPO_ROOT=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925
DATA_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
VAE_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
PILOT_ROOT="$PLAN_ROOT/pilot"
MODEL_ROOT=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
LOG_ROOT="$PLAN_ROOT/logs"
START_AT_LOCAL="${START_AT_LOCAL:-2026-09-27 03:00:00}"
mkdir -p "$LOG_ROOT"

status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$LOG_ROOT/action_only_pilot.log"
}

start_epoch="$(TZ=Asia/Shanghai date -d "$START_AT_LOCAL" +%s)"
now_epoch="$(date +%s)"
if [ "$now_epoch" -lt "$start_epoch" ]; then
    status "scheduled_for_${START_AT_LOCAL// /T}+08:00"
    sleep "$((start_epoch - now_epoch))"
fi

status waiting_for_vae_completion
while [ ! -f "$VAE_ROOT/VAE_LATENT_COMPLETE" ]; do
    if ! pgrep -f '^bash code/vwm/schedule_full_latent_after_gpu_qc_1900.sh$' >/dev/null \
        && ! pgrep -f 'fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess vae' >/dev/null; then
        status vae_worker_absent_before_completion
        exit 1
    fi
    sleep 300
done
status vae_complete

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$REPO_ROOT/.venv/bin/python" examples/train/stage_minimax_h3_action_only_pilot.py \
    --plan "$PLAN_ROOT" \
    --source-mapping "$DATA_ROOT/source_mapping.jsonl" \
    --vae-input "$VAE_ROOT/intermediate/all" \
    --output "$PILOT_ROOT" \
    > "$LOG_ROOT/stage_pilot.log" 2>&1
status pilot_vae_rows_staged

export NUM_WORKERS=8 GPU_IDS=0,1,2,3,4,5,6,7 SAMPLES_PER_SHARD=16
export MODEL_PATH="$MODEL_ROOT" MANIFEST_PATH="$DATA_ROOT/manifests/approved_uncached.jsonl"
export WORK_ROOT="$PILOT_ROOT" CONDITIONING=prope

for split in train validation; do
    bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition "$split" \
        > "$LOG_ROOT/condition_${split}.log" 2>&1
    expected=2048
    [ "$split" = validation ] && expected=64
    bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify "$split" "$expected" \
        > "$LOG_ROOT/verify_${split}.log" 2>&1
    status "conditioned_${split}_${expected}"
done

"$REPO_ROOT/.venv/bin/python" examples/train/verify_minimax_h3_action_only_pilot.py \
    --plan "$PLAN_ROOT" \
    --mapping "$DATA_ROOT/source_mapping.jsonl" \
    --data-root "$PILOT_ROOT" \
    > "$LOG_ROOT/verify_action_only_pilot.log" 2>&1
status pilot_data_verified

for gpu in 0 1 2 3 4 5 6 7; do
    while [ "$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 1024 ]; do
        sleep 300
    done
done

status launching_action_only_pilot_training
export CONFIG_FILE="$REPO_ROOT/examples/train/configs/minimax_h3_worldmodel_action_only_v1_pilot.yaml"
export TRAIN_DATA="$PILOT_ROOT/preprocessed/train"
export VALIDATION_DATA="$PILOT_ROOT/preprocessed/validation"
export LOG_DIR="$LOG_ROOT/train"
export WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29531
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    > "$LOG_ROOT/train_pilot.log" 2>&1
status action_only_pilot_training_complete
