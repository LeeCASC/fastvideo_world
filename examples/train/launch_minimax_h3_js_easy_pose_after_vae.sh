#!/usr/bin/env bash
# Dependency-gated launch: full VAE completion -> independent full CPU audit
# and strict 80-row audit -> conditioning -> 200-step J/L diagnostic.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
DATA=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
VAE=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
EARLY=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
PYTHON="$REPO/.venv/bin/python"
LOG="$PLAN/auto_launch.log"
mkdir -p "$PLAN"
exec 9>"$PLAN/auto_launch.lock"
flock -n 9 || { echo "Another JS-easy launcher already holds the lock" >&2; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline

status() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$LOG"; }
wait_for_free_gpus() {
    local used gpu
    while true; do
        for gpu in 0 1 2 3 4 5 6 7; do
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
            if (( used >= 4096 )); then
                status "waiting_for_gpu_${gpu}_${used}MiB"
                sleep 300
                continue 2
            fi
        done
        return 0
    done
}

status waiting_for_all_eight_vae_workers
while [[ ! -f "$VAE/VAE_LATENT_COMPLETE" ]]; do
    if [[ -f "$VAE/VAE_WORKERS_0_3_COMPLETE" && -f "$VAE/VAE_WORKERS_4_7_COMPLETE" ]]; then
        sleep 30
        [[ -f "$VAE/VAE_LATENT_COMPLETE" ]] || {
            status error_both_worker_markers_exist_but_global_completion_missing
            exit 1
        }
        break
    fi
    if ! pgrep -f '[f]astvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess vae' >/dev/null; then
        status error_vae_workers_stopped_before_completion
        exit 1
    fi
    sleep 300
done
status vae_workers_complete_starting_background_full_cpu_audit
if [[ ! -f "$PLAN/VAE_FULL_AUDIT_COMPLETE" ]] && ! tmux has-session -t fastvideowm_vae_full_audit_20260927 2>/dev/null; then
    tmux new-session -d -s fastvideowm_vae_full_audit_20260927 \
        "bash $REPO/examples/train/run_minimax_h3_vae_full_audit.sh"
fi

"$PYTHON" examples/train/prepare_minimax_h3_js_easy_overfit.py \
    --source-mapping "$DATA/source_mapping.jsonl" \
    --train-ids /datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/train_ids.jsonl \
    --output-dir "$PLAN" > "$PLAN/frozen_plan.log" 2>&1
"$PYTHON" examples/train/verify_minimax_h3_js_easy_split.py \
    --plan "$PLAN/stage3_all8_ids.jsonl" \
    --registry /datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925 \
    --mapping "$DATA/source_mapping.jsonl" \
    --validation "$EARLY/data/preprocessed/validation" \
    > "$PLAN/verify_scene_split.log" 2>&1
"$PYTHON" examples/train/stage_minimax_h3_js_easy_overfit.py \
    --plan "$PLAN/stage3_all8_ids.jsonl" \
    --vae-input "$VAE/intermediate/all" \
    --output "$PLAN" > "$PLAN/stage_vae_80.log" 2>&1
status staged_80_vae_rows
"$PYTHON" -u examples/train/verify_minimax_h3_vae_cache.py \
    --input-dir "$PLAN/intermediate/train" \
    --manifest "$DATA/manifests/approved_uncached.jsonl" \
    --selected-ids-file "$PLAN/stage3_all8_ids.jsonl" \
    --expected-rows 80 \
    --report "$PLAN/vae_stage80_audit.json" \
    > "$PLAN/vae_stage80_audit.log" 2>&1
date --iso-8601=seconds > "$PLAN/VAE_STAGE80_AUDIT_COMPLETE"
status vae_stage80_audit_complete_full_audit_continues_in_background

wait_for_free_gpus
export NUM_WORKERS=8 GPU_IDS=0,1,2,3,4,5,6,7
export MANIFEST_PATH="$DATA/manifests/approved_uncached.jsonl"
export WORK_ROOT="$PLAN" CONDITIONING=prope
bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition train \
    > "$PLAN/condition_80.log" 2>&1
bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify train 80 \
    > "$PLAN/verify_conditioned_80.log" 2>&1
# The held-out validation lives in the earlier pilot's data root.
CONDITIONING=prope WORK_ROOT="$EARLY/data" \
    bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify validation 48 \
    > "$PLAN/verify_heldout_48.log" 2>&1
"$PYTHON" examples/train/prepare_minimax_h3_js_easy_overfit.py \
    --source-mapping "$DATA/source_mapping.jsonl" \
    --train-ids /datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/train_ids.jsonl \
    --output-dir "$PLAN" --parquet-data "$PLAN/preprocessed/train" \
    > "$PLAN/frozen_indices.log" 2>&1
date --iso-8601=seconds > "$PLAN/CONDITIONED_80_VERIFIED"
status conditioned_80_and_heldout_validation_verified

wait_for_free_gpus
status starting_stage1_jl_200_steps
bash examples/train/run_minimax_h3_js_easy_pose_stage.sh stage1_jl \
    > "$PLAN/runs_stage1_launcher.log" 2>&1
status stage1_jl_200_steps_finished_starting_heldout_JL_evaluation
bash examples/train/evaluate_minimax_h3_js_easy_pose_stage.sh stage1_jl \
    > "$PLAN/evaluate_stage1_jl.log" 2>&1
status stage1_jl_evaluation_finished_review_required_before_stage2
