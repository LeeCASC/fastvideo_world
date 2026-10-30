#!/usr/bin/env bash
# Continue from the old step-30 *weights*, not its shard-4 optimizer state.
# Each attempt has a separate output directory; no existing DCP is overwritten.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
HIGH_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_high_20261001.yaml"
MIXED_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_mixed_20261001.yaml"
LEGACY_CHECKPOINT="$ROOT/runs/high_noise/checkpoints/checkpoint-30"
AUTOMATION_DIR="$ROOT/automation_20261002"
CUTOFF=$(TZ=Asia/Shanghai date -d '2026-10-02 07:45:00' +%s)
EVALUATE_AT=$(TZ=Asia/Shanghai date -d '2026-10-02 08:00:00' +%s)
mkdir -p "$AUTOMATION_DIR"
exec 9>"$AUTOMATION_DIR/weights_only_launcher.lock"
flock -n 9 || { echo 'Another weights-only camera FiLM launcher holds the lock' >&2; exit 1; }

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

checkpoint_step() {
    local checkpoint="$1"
    local step="${checkpoint##*/checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || { log "Invalid checkpoint step: $checkpoint"; return 1; }
    echo "$step"
}

# Return 0 on phase completion, 124 on scheduled cutoff, or a real error.
# STAGE_LATEST always names the most recent complete checkpoint, including
# the incoming checkpoint when an attempt fails before its first save.
run_stage() {
    local name="$1" config="$2" init_path="$3" target_steps="$4"
    local attempt completed=0 remaining output_dir status candidate gained stage_log_dir seconds
    STAGE_LATEST="$init_path"
    for attempt in 1 2 3; do
        remaining=$((target_steps - completed))
        if (( remaining <= 0 )); then return 0; fi
        seconds=$((CUTOFF - $(date +%s)))
        if (( seconds <= 0 )); then return 124; fi
        wait_for_idle_gpus "$CUTOFF" || return 124
        output_dir="$ROOT/runs/${name}_shard8_20261002_attempt${attempt}/checkpoints"
        if [[ -e "$output_dir" ]]; then
            log "Refusing to overwrite existing attempt output: $output_dir"
            return 1
        fi
        stage_log_dir="$AUTOMATION_DIR/$name/attempt_$attempt"
        mkdir -p "$stage_log_dir"
        log "Starting $name attempt=$attempt from weights $STAGE_LATEST for up to $remaining steps"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 MASTER_PORT=29980
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export WANDB_MODE=offline CONFIG_FILE="$config" TRAIN_DATA="$ROOT/data"
        export VALIDATION_DATA="$VALIDATION" LOG_DIR="$stage_log_dir"
        set +e
        timeout --signal=INT --kill-after=180s "${seconds}s" \
            bash "$REPO/examples/train/launch_minimax_h3_worldmodel_stage0.sh" \
            --models.student.init_from_training_checkpoint "$STAGE_LATEST" \
            --training.distributed.hsdp_replicate_dim 1 \
            --training.distributed.hsdp_shard_dim 8 \
            --training.loop.max_train_steps "$remaining" \
            --training.checkpoint.output_dir "$output_dir" \
            --training.checkpoint.training_state_checkpointing_steps 5 \
            > "$stage_log_dir/training.log" 2>&1
        status=$?
        set -e
        candidate=$(latest_checkpoint "$output_dir")
        if [[ -n "$candidate" ]]; then
            gained=$(checkpoint_step "$candidate")
            completed=$((completed + gained))
            STAGE_LATEST="$candidate"
        fi
        log "$name attempt=$attempt exit=$status completed_in_phase=$completed/$target_steps latest=$STAGE_LATEST"
        if [[ "$status" -eq 0 ]]; then
            if (( completed == target_steps )); then return 0; fi
            log "$name exited zero without the expected complete checkpoint"
            status=1
        fi
        if [[ "$status" -eq 124 ]]; then return 124; fi
        if (( attempt < 3 )); then
            log "$name failed; the next attempt will initialize weights from the last complete checkpoint"
        fi
    done
    return "$status"
}

test -s "$ROOT/pair_report.json"
test -s "$LEGACY_CHECKPOINT/dcp/.metadata"
test -f "$HIGH_CONFIG"
test -f "$MIXED_CONFIG"
test -f /datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3/transformer/config.json

train_status=0
STAGE_LATEST="$LEGACY_CHECKPOINT"
if (( $(date +%s) < CUTOFF )); then
    run_stage high_noise "$HIGH_CONFIG" "$LEGACY_CHECKPOINT" 170 || train_status=$?
fi
if [[ "$train_status" -eq 0 ]] && (( $(date +%s) < CUTOFF )); then
    high_final="$STAGE_LATEST"
    run_stage mixed_noise "$MIXED_CONFIG" "$high_final" 1000 || train_status=$?
fi
if [[ "$train_status" -ne 0 && "$train_status" -ne 124 ]]; then
    log "Training stopped early with status $train_status; preserving checkpoints and scheduling inference"
fi

checkpoint="$STAGE_LATEST"
log "Selected $checkpoint for J/L inference"
while (( $(date +%s) < EVALUATE_AT )); do sleep 30; done
if ! wait_for_idle_gpus "$((EVALUATE_AT + 3600))"; then
    log 'Inference could not start: GPUs unavailable by 09:00 CST'
    exit 1
fi
export EVALUATION_ROOT="$ROOT/evaluation_20261002"
# The high and mixed phases use the same H3 architecture. The high config has
# no obsolete weight-initialization path, so it can load either DCP for eval.
if bash "$REPO/examples/train/evaluate_minimax_h3_camera_film_pair_20261001.sh" \
    "$HIGH_CONFIG" "$checkpoint" > "$AUTOMATION_DIR/evaluation.log" 2>&1; then
    log "J/L inference completed: $EVALUATION_ROOT"
else
    log "J/L inference failed; inspect $AUTOMATION_DIR/evaluation.log"
    exit 1
fi
