#!/usr/bin/env bash
# Condition and validate already encoded videos, then run smoke + 100-step pilot.
set -euo pipefail

REPO_ROOT=/home/lizihao/projects/military_world_model_agent/FastVideoWM
EARLY_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
SOURCE_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
DATA_ROOT="$EARLY_ROOT/data"
LOG_ROOT="$EARLY_ROOT/logs"
export CONFIG_FILE="$REPO_ROOT/examples/train/configs/minimax_h3_worldmodel_action_only_v1_pilot.yaml"
mkdir -p "$LOG_ROOT"

status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$LOG_ROOT/early_pilot_status.log"
}
trap 'status "failed_at_line_${LINENO}"' ERR

wait_for_idle_gpus() {
    for gpu in 0 1 2 3 4 5 6 7; do
        while [ "$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 1024 ]; do
            status "waiting_for_gpu_${gpu}_to_be_idle"
            sleep 30
        done
    done
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -s "$EARLY_ROOT/report.json"
status waiting_for_vae_row_staging
while [ ! -s "$DATA_ROOT/staging_report.json" ]; do
    if ! pgrep -f '^\.venv/bin/python examples/train/stage_minimax_h3_action_only_pilot.py ' >/dev/null; then
        status staging_process_absent_before_completion
        exit 1
    fi
    sleep 30
done
test -s "$DATA_ROOT/staging_report.json"
status vae_rows_staged

export NUM_WORKERS=8 GPU_IDS=0,1,2,3,4,5,6,7 SAMPLES_PER_SHARD=16
export MODEL_PATH=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
export MANIFEST_PATH="$SOURCE_ROOT/manifests/approved_uncached.jsonl"
export WORK_ROOT="$DATA_ROOT" CONDITIONING=prope

for split in train validation; do
    if [ "${SKIP_CONDITION:-0}" != 1 ]; then
        status "conditioning_${split}"
        bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition "$split" \
            > "$LOG_ROOT/condition_${split}.log" 2>&1
    fi
    expected=1024
    [ "$split" = validation ] && expected=48
    bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify "$split" "$expected" \
        > "$LOG_ROOT/verify_${split}.log" 2>&1
    status "conditioned_${split}_${expected}"
done

"$REPO_ROOT/.venv/bin/python" examples/train/verify_minimax_h3_action_only_pilot.py \
    --plan "$EARLY_ROOT" --mapping "$SOURCE_ROOT/source_mapping.jsonl" \
    --data-root "$DATA_ROOT" > "$LOG_ROOT/verify_action_only.log" 2>&1
status data_verified

export TRAIN_DATA="$DATA_ROOT/preprocessed/train"
export VALIDATION_DATA="$DATA_ROOT/preprocessed/validation"
export LOG_DIR="$LOG_ROOT/train" WANDB_MODE=offline NUM_GPUS=8
export MASTER_PORT=29541
wait_for_idle_gpus
status launching_one_step_smoke
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    --training.loop.max_train_steps 1 \
    --training.checkpoint.training_state_checkpointing_steps 1 \
    --training.checkpoint.output_dir "$EARLY_ROOT/runs/smoke/checkpoints" \
    --training.tracker.run_name minimax_h3_action_only_early_smoke \
    --callbacks.worldmodel_validation.num_batches 2 \
    > "$LOG_ROOT/smoke_stdout.log" 2>&1
test -d "$EARLY_ROOT/runs/smoke/checkpoints/checkpoint-1"
status one_step_smoke_complete

export MASTER_PORT=29542
wait_for_idle_gpus
status launching_100_step_pilot
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    --training.checkpoint.output_dir "$EARLY_ROOT/runs/pilot/checkpoints" \
    --training.tracker.run_name minimax_h3_action_only_early_pilot \
    > "$LOG_ROOT/pilot_stdout.log" 2>&1
status 100_step_pilot_complete
