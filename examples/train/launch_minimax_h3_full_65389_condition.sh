#!/usr/bin/env bash
# Resumable eight-GPU first-frame conditioning for the audited 65,389 VAE rows.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PYTHON="$REPO/.venv/bin/python"
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
DATA=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925/intermediate/all
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
OUTPUT="$ROOT/preprocessed/all"
LOGS="$ROOT/logs/condition"
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925
MAP=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925/source_mapping.jsonl

on_exit() {
    status=$?
    if (( status != 0 )); then
        printf 'exit_status=%s\n' "$status" > "$ROOT/CONDITION_OR_TRAIN_FAILED"
    fi
}
trap on_exit EXIT

test -d "$DATA" || { echo "Missing audited VAE data: $DATA" >&2; exit 1; }
test -f "$PLAN/js_easy_pose_overfit/VAE_INTRINSICS_AUDIT_COMPLETE" || {
    echo "Full VAE first-frame/intrinsics audit has not passed" >&2; exit 1;
}
test ! -f "$ROOT/CONDITIONED_VERIFIED" || { echo "Already conditioned and verified"; exit 0; }
mkdir -p "$LOGS"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"

pids=()
for worker in {0..7}; do
    CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" \
        -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess condition \
        --input-dir "$DATA" --model-path "$MODEL" --output-dir "$OUTPUT" \
        --worker-index "$worker" --num-workers 8 --device-index 0 \
        --master-port "$((29940 + worker))" --conditioning prope \
        > "$LOGS/worker_$worker.log" 2>&1 &
    pids+=("$!")
    echo "Started first-frame conditioning worker $worker on GPU$worker"
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "At least one conditioning worker failed; see $LOGS" >&2; exit 1; }

"$PYTHON" -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess verify \
    --input-dir "$OUTPUT" --conditioning prope --expected-rows 65389 \
    > "$ROOT/verify_conditioned.log" 2>&1
"$PYTHON" examples/train/prepare_minimax_h3_full_vae_split.py \
    --conditioned-dir "$OUTPUT" --source-mapping "$MAP" \
    --train-ids "$PLAN/train_ids.jsonl" --validation-ids "$PLAN/validation_ids.jsonl" \
    --output-dir "$ROOT" --expected-rows 65389 \
    > "$ROOT/split_report.log" 2>&1
touch "$ROOT/CONDITIONED_VERIFIED"
echo "All 65,389 conditioned rows verified; 62,820 train and 2,569 held out."
bash examples/train/launch_minimax_h3_full_65389_bs64.sh \
    > "$ROOT/train_bs64_1000.log" 2>&1
