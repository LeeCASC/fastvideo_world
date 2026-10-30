#!/usr/bin/env bash
# Fixed-noise, real/static/reverse/shuffled pose counterfactuals for trained and held-out scenes.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930
CONFIG="$REPO/examples/train/configs/minimax_h3_camera_curriculum_20260930.yaml"
CHECKPOINT="$ROOT/runs/overfit200/checkpoints/checkpoint-200"
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
OUTPUT="$ROOT/runs/overfit200/prope_real_pose_diagnostic_20260930"

test -s "$CHECKPOINT/dcp/.metadata"
test -d "$ROOT/data"
test -d "$VALIDATION"
mkdir -p "$OUTPUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for spec in train_A:84 train_D:86 heldout_js:11; do
    name=${spec%%:*}
    row=${spec##*:}
    data="$ROOT/data"
    if [[ "$name" == heldout_js ]]; then
        data="$VALIDATION"
    fi
    case_dir="$OUTPUT/$name"
    video_dir="$case_dir/checkpoint-200"
    mkdir -p "$case_dir"
    ready=true
    for variant in correct static reverse shuffled; do
        if [[ ! -s "$video_dir/real_${variant}.mp4" ]]; then
            ready=false
        fi
    done
    if [[ "$ready" == false ]]; then
        echo "$(date --iso-8601=seconds) inference start $name row=$row"
        "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29981 \
            scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
            --config "$CONFIG" --checkpoint "$CHECKPOINT" \
            --data-path "$data" --row-index "$row" --output-dir "$case_dir" \
            --seed 20260930 --num-inference-steps 50 \
            --real-trajectory-variants correct static reverse shuffled \
            > "$case_dir/inference.log" 2>&1
    fi
    test -s "$video_dir/ground_truth.mp4"
    for variant in correct static reverse shuffled; do
        test -s "$video_dir/real_${variant}.mp4"
    done
    set +e
    "$REPO/.venv/bin/python" scripts/evaluation/analyze_minimax_h3_camera_control.py \
        "$video_dir" > "$case_dir/analysis.log" 2>&1
    result=$?
    set -e
    if [[ "$result" -ne 0 && "$result" -ne 2 ]]; then
        echo "Analysis failed for $name (exit=$result)" >&2
        exit "$result"
    fi
    echo "$(date --iso-8601=seconds) complete $name action_gate_exit=$result"
done
date --iso-8601=seconds > "$OUTPUT/DIAGNOSTIC_COMPLETE"
