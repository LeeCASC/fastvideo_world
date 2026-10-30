#!/usr/bin/env bash
# Wait for a healthy, idle eight-GPU node; train until 18:50 CST, then use
# the latest complete checkpoint for an automatic 19:00 J/L evaluation.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
HIGH_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_high_20261001.yaml"
MIXED_CONFIG="$REPO/examples/train/configs/minimax_h3_camera_film_pair_mixed_20261001.yaml"
CUTOFF=$(TZ=Asia/Shanghai date -d '2026-10-01 18:50:00' +%s)
EVALUATE_AT=$(TZ=Asia/Shanghai date -d '2026-10-01 19:00:00' +%s)
AUTOMATION_DIR="$ROOT/automation_20261001"
mkdir -p "$AUTOMATION_DIR"
exec 9>"$AUTOMATION_DIR/launcher.lock"
flock -n 9 || { echo 'Another camera FiLM launcher already holds the lock' >&2; exit 1; }

log() { echo "$(date --iso-8601=seconds) $*"; }

gpus_idle() {
    local usage
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
    awk 'BEGIN { ok=1; count=0 } { count++; if ($1 !~ /^[0-9]+$/ || $1 >= 2048) ok=0 }
         END { if (count != 8 || !ok) exit 1 }' <<< "$usage"
}

latest_checkpoint() {
    local directory="$1"
    [[ -d "$directory" ]] || return 0
    find "$directory" -mindepth 3 -maxdepth 3 -path '*/dcp/.metadata' -printf '%h\n' \
        | sed 's@/dcp$@@' | sort -V | tail -n 1
}

run_stage() {
    local name="$1" config="$2" output="$3" remaining status stage_log_dir
    shift 3
    remaining=$((CUTOFF - $(date +%s)))
    if (( remaining <= 0 )); then return 0; fi
    stage_log_dir="$AUTOMATION_DIR/$name"
    mkdir -p "$stage_log_dir"
    log "starting $name; at most $remaining seconds before training cutoff"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 MASTER_PORT=29980
    export WANDB_MODE=offline CONFIG_FILE="$config" TRAIN_DATA="$ROOT/data"
    export VALIDATION_DATA="$VALIDATION" LOG_DIR="$stage_log_dir"
    set +e
    timeout --signal=INT --kill-after=120s "${remaining}s" \
        bash "$REPO/examples/train/launch_minimax_h3_worldmodel_stage0.sh" "$@" \
        > "$stage_log_dir/training.log" 2>&1
    status=$?
    set -e
    if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
        log "$name exited with status $status; inspect $stage_log_dir/training.log"
        return "$status"
    fi
    log "$name ended with status $status; latest checkpoint: $(latest_checkpoint "$output")"
}

test -s "$ROOT/pair_report.json"
test -f "$HIGH_CONFIG"
test -f "$MIXED_CONFIG"
test -f /datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3/transformer/config.json
if (( $(date +%s) >= CUTOFF )); then
    log 'Training cutoff already passed; not starting a late GPU job'
    exit 1
fi

while ! gpus_idle; do
    if (( $(date +%s) >= CUTOFF )); then
        log 'Eight healthy, idle GPUs did not become available before 18:50 CST'
        exit 1
    fi
    log 'Waiting for NVIDIA driver and eight idle GPUs; checking again in 5 minutes'
    sleep 300
done

high_output="$ROOT/runs/high_noise/checkpoints"
mixed_output="$ROOT/runs/mixed_noise/checkpoints"
if [[ ! -s "$high_output/checkpoint-200/dcp/.metadata" ]]; then
    high_resume=$(latest_checkpoint "$high_output")
    if [[ -n "$high_resume" ]]; then
        log "Resuming high-noise optimizer from $high_resume"
        run_stage high_noise "$HIGH_CONFIG" "$high_output" \
            --training.checkpoint.resume_from_checkpoint "$high_resume"
    else
        log 'Starting high-noise training from official MiniMax-H3 baseline'
        run_stage high_noise "$HIGH_CONFIG" "$high_output"
    fi
fi

if [[ -s "$high_output/checkpoint-200/dcp/.metadata" ]] && (( $(date +%s) < CUTOFF )); then
    mixed_resume=$(latest_checkpoint "$mixed_output")
    if [[ -n "$mixed_resume" ]]; then
        log "Resuming mixed-noise optimizer from $mixed_resume"
        run_stage mixed_noise "$MIXED_CONFIG" "$mixed_output" \
            --training.checkpoint.resume_from_checkpoint "$mixed_resume"
    else
        log 'Starting mixed-noise stage from high-noise checkpoint-200'
        run_stage mixed_noise "$MIXED_CONFIG" "$mixed_output"
    fi
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
if ! gpus_idle; then
    log 'Inference cannot start: NVIDIA driver unavailable or GPUs occupied'
    exit 1
fi
bash "$REPO/examples/train/evaluate_minimax_h3_camera_film_pair_20261001.sh" \
    "$config" "$checkpoint" > "$AUTOMATION_DIR/evaluation.log" 2>&1
log "J/L inference finished; inspect $AUTOMATION_DIR/evaluation.log"
