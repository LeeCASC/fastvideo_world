#!/usr/bin/env bash
# Held-out first-frame counterfactuals; inspection is required before writing
# the next stage's *_EVAL_APPROVED gate file.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
EARLY=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
STAGE="${1:?Usage: $0 stage1_jl|stage2_jlik|stage3_all8}"
case "$STAGE" in
    stage1_jl) STEP=200; PAIRS=("J L") ;;
    stage2_jlik) STEP=200; PAIRS=("J L" "I K") ;;
    stage3_all8) STEP=400; PAIRS=("J L" "I K" "A D" "W S") ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
CHECKPOINT="$PLAN/runs/$STAGE/checkpoints/checkpoint-$STEP"
test -s "$CHECKPOINT/dcp/.metadata" || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 1; }
OUT="$PLAN/runs/$STAGE/heldout_action_eval"
mkdir -p "$OUT/logs"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
for scope in train heldout; do
    if [[ "$scope" == train ]]; then DATA="$PLAN/preprocessed/train"; else DATA="$EARLY/data/preprocessed/validation"; fi
    for row in 0 2 3; do
        for pair in "${PAIRS[@]}"; do
            read -r a b <<< "$pair"
            case_dir="$OUT/$scope/row_$row/$a$b"
            if [[ -s "$case_dir/checkpoint-$STEP/$a.mp4" && -s "$case_dir/checkpoint-$STEP/$b.mp4" ]]; then
                continue
            fi
            mkdir -p "$case_dir"
            CUDA_VISIBLE_DEVICES=0,1,2,3 "$REPO/.venv/bin/torchrun" \
                --nnodes=1 --nproc_per_node=4 --master_port=29922 \
                scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
                --config examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml \
                --checkpoint "$CHECKPOINT" \
                --data-path "$DATA" \
                --output-dir "$case_dir" --row-index "$row" \
                --seed "$((20260926 + row))" --num-inference-steps 50 \
                --trajectories "$a" "$b" --trajectory-scale 1.0 \
                > "$OUT/logs/${scope}_row_${row}_${a}${b}.log" 2>&1
            test -s "$case_dir/checkpoint-$STEP/$a.mp4"
            test -s "$case_dir/checkpoint-$STEP/$b.mp4"
        done
    done
done
date --iso-8601=seconds > "$OUT/INFERENCE_COMPLETE"
