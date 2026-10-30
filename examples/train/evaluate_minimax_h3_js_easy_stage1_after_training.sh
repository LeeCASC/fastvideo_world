#!/usr/bin/env bash
# Run the planned J/L counterfactual evaluation only after a complete 200-step checkpoint.
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
TRAIN_SESSION=fastvideowm_js_easy_stage1_jl_intrinsics_fixed_20260927
OUT="$PLAN/runs/stage1_jl/heldout_action_eval"
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
    sleep 300
done
test -s "$PLAN/runs/stage1_jl/checkpoints/checkpoint-200/dcp/.metadata" || {
    echo "Stage-1 training ended without a complete checkpoint-200; no evaluation launched" >&2
    exit 1
}
if [[ -f "$OUT/INFERENCE_COMPLETE" ]]; then
    echo "Stage-1 inference already completed"
    exit 0
fi
while true; do
    busy=0
    for gpu in 0 1 2 3; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        if (( used >= 4096 )); then busy=1; fi
    done
    if (( busy == 0 )); then break; fi
    sleep 300
done
cd "$REPO"
bash examples/train/evaluate_minimax_h3_js_easy_pose_stage.sh stage1_jl \
    > "$PLAN/evaluate_stage1_jl_intrinsics_fixed.log" 2>&1
