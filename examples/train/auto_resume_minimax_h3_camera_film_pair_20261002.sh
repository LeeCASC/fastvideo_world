#!/usr/bin/env bash
# Resume the paired camera-control run with additional FSDP sharding and a
# fragmentation-resistant allocator. Stop near 08:00 CST, then infer J/L.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
HIGH_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_high_20261001.yaml"
MIXED_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_mixed_20261001.yaml"
AUTOMATION_DIR="$ROOT/automation_20261002"
CUTOFF=$(TZ=Asia/Shanghai date -d '2026-10-02 07:45:00' +%s)
EVALUATE_AT=$(TZ=Asia/Shanghai date -d '2026-10-02 08:00:00' +%s)
mkdir -p "$AUTOMATION_DIR"
exec 9>"$AUTOMATION_DIR/launcher.lock"
flock -n 9 || { echo 'An overnight camera FiLM launcher already holds the lock' >&2; exit 1; }

log() { echo "$(date --iso-8601=seconds) $*"; }

gpus_idle() {
    local usage
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
    awk 'BEGIN { ok=1; count=0 } { count++; if ($1 !~ /^[0-9]+$/ || $1 >= 2048) ok=0 }
         END { if (count != 8 || !ok) exit 1 }' <<< "$usage"
}

wait_for_idle_gpus() {
    local deadline="$1"
    while ! gpus_idle; do
        if (( $(date +%s) >= deadline )); then return 1; fi
        log 'Waiting for eight healthy, idle GPUs; checking again in 60 seconds'
        sleep 60
    done
}

latest_checkpoint() {
    local directory="$1"
    [[ -d "$directory" ]] || return 0
    find "$directory" -mindepth 3 -maxdepth 3 -path '*/dcp/.metadata' -printf '%h\n' \
        | sed 's@/dcp$@@' | sort -V | tail -n 1
}

run_stage() {
    local name="$1" config="$2" output="$3" attempt checkpoint remaining status
    for attempt in 1 2; do
        remaining=$((CUTOFF - $(date +%s)))
        if (( remaining <= 0 )); then return 124; fi
        wait_for_idle_gpus "$CUTOFF" || return 124
        checkpoint=$(latest_checkpoint "$output")
        log "Starting $name attempt=$attempt from ${checkpoint:-baseline}; deadline=07:45 CST"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 MASTER_PORT=29980
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline CONFIG_FILE="$config" TRAIN_DATA="$ROOT/data"
        export VALIDATION_DATA="$VALIDATION" LOG_DIR="$AUTOMATION_DIR/$name/attempt_$attempt"
        mkdir -p "$LOG_DIR"
        local -a overrides=(
            --training.distributed.hsdp_replicate_dim 1
            --training.distributed.hsdp_shard_dim 8
            --training.checkpoint.training_state_checkpointing_steps 5
        )
        if [[ -n "$checkpoint" ]]; then
            overrides+=(--training.checkpoint.resume_from_checkpoint "$checkpoint")
        fi
        set +e
        timeout --signal=INT --kill-after=180s "${remaining}s" \
            bash "$REPO/examples/train/launch_minimax_h3_worldmodel_stage0.sh" \
            "${overrides[@]}" > "$LOG_DIR/training.log" 2>&1
        status=$?
        set -e
        log "$name attempt=$attempt exit=$status latest=$(latest_checkpoint "$output")"
        if [[ "$status" -eq 0 || "$status" -eq 124 ]]; then return "$status"; fi
        if (( attempt == 1 )); then
            log "$name failed; retrying once from the latest complete checkpoint"
        fi
    done
    return "$status"
}

test -s "$ROOT/pair_report.json"
test -s "$ROOT/runs/high_noise/checkpoints/checkpoint-30/dcp/.metadata"
test -f "$HIGH_CONFIG"
test -f "$MIXED_CONFIG"
test -f /datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3/transformer/config.json

high_output="$ROOT/runs/high_noise/checkpoints"
mixed_output="$ROOT/runs/mixed_noise/checkpoints"
train_failed=0
if [[ ! -s "$high_output/checkpoint-200/dcp/.metadata" ]] && (( $(date +%s) < CUTOFF )); then
    run_stage high_noise "$HIGH_CONFIG" "$high_output" || train_failed=$?
fi
if [[ "$train_failed" -eq 0 && -s "$high_output/checkpoint-200/dcp/.metadata" ]] &&
   (( $(date +%s) < CUTOFF )); then
    run_stage mixed_noise "$MIXED_CONFIG" "$mixed_output" || train_failed=$?
fi
if [[ "$train_failed" -ne 0 && "$train_failed" -ne 124 ]]; then
    log "Training stopped early with status $train_failed; preserving checkpoints and scheduling inference"
fi

checkpoint=$(latest_checkpoint "$mixed_output")
config="$MIXED_CONFIG"
if [[ -z "$checkpoint" ]]; then
    checkpoint=$(latest_checkpoint "$high_output")
    config="$HIGH_CONFIG"
fi
if [[ -z "$checkpoint" ]]; then
    log 'No complete main-run checkpoint available for inference'
    exit 1
fi
log "Selected $checkpoint for J/L inference"
while (( $(date +%s) < EVALUATE_AT )); do sleep 30; done
if ! wait_for_idle_gpus "$((EVALUATE_AT + 3600))"; then
    log 'Inference could not start: GPUs unavailable by 09:00 CST'
    exit 1
fi
export EVALUATION_ROOT="$ROOT/evaluation_20261002"
if bash "$REPO/examples/train/evaluate_minimax_h3_camera_film_pair_20261001.sh" \
    "$config" "$checkpoint" > "$AUTOMATION_DIR/evaluation.log" 2>&1; then
    log "J/L inference completed: $EVALUATION_ROOT"
else
    log "J/L inference failed; inspect $AUTOMATION_DIR/evaluation.log"
    exit 1
fi
