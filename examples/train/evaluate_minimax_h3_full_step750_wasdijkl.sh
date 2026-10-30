#!/usr/bin/env bash
# Held-out single-action camera sweeps on four distinct source first frames.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
CHECKPOINT="$ROOT/runs/baseline_bs64_1000/checkpoints/checkpoint-750"
DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
OUT="$ROOT/inference_step750_single_action_20260930"
CONFIG="$REPO/examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml"

test -s "$CHECKPOINT/dcp/.metadata" || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 1; }
test -d "$DATA" || { echo "Missing validation data: $DATA" >&2; exit 1; }
mkdir -p "$OUT/logs"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false

# Distinct held-out sources: multicam, UE-easy, MiniMax records, H3 JS-easy.
for row in 0 2 3 11; do
    case_dir="$OUT/row_$(printf '%02d' "$row")"
    for action_group in WASD IJKL; do
        actions=("${action_group:0:1}" "${action_group:1:1}" "${action_group:2:1}" "${action_group:3:1}")
        complete=1
        for action in "${actions[@]}"; do
            test -s "$case_dir/checkpoint-750/$action.mp4" || complete=0
        done
        if (( complete == 1 )); then
            echo "Reusing completed row=$row actions=$action_group"
            continue
        fi
        mkdir -p "$case_dir"
        echo "Starting row=$row actions=$action_group at $(date --iso-8601=seconds)"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$REPO/.venv/bin/torchrun" \
            --nnodes=1 --nproc_per_node=8 --master_port=29961 \
            scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
            --config "$CONFIG" --checkpoint "$CHECKPOINT" \
            --data-path "$DATA" --row-index "$row" --output-dir "$case_dir" \
            --seed "$((20260930 + row))" --num-inference-steps 50 \
            --trajectories "${actions[@]}" --trajectory-scale 1.0 \
            --freeze-intrinsics-first-frame \
            > "$OUT/logs/row_$(printf '%02d' "$row")_$action_group.log" 2>&1
        for action in "${actions[@]}"; do
            test -s "$case_dir/checkpoint-750/$action.mp4" || {
                echo "Missing result: $case_dir/checkpoint-750/$action.mp4" >&2; exit 1;
            }
        done
        echo "Completed row=$row actions=$action_group at $(date --iso-8601=seconds)"
    done
done
date --iso-8601=seconds > "$OUT/INFERENCE_COMPLETE"
