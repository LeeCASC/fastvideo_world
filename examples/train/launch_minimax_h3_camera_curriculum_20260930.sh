#!/usr/bin/env bash
# One 200-update hybrid-camera diagnostic, then same-seed eight-way inference.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930
CONFIG="$REPO/examples/train/configs/minimax_h3_camera_curriculum_20260930.yaml"
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
OUTPUT="$ROOT/runs/overfit200/checkpoints"
LOGS="$ROOT/runs/overfit200/logs"
INFERENCE="$ROOT/runs/overfit200/inference"

test -s "$ROOT/report.json" || { echo "No completed frozen data report" >&2; exit 1; }
test -s "$ROOT/train_indices.json" || { echo "No balanced sampling plan" >&2; exit 1; }
test ! -e "$OUTPUT/checkpoint-50" || { echo "Existing run; refusing overwrite" >&2; exit 1; }
"$REPO/.venv/bin/python" - "$ROOT/report.json" "$ROOT/train_indices.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
indices=json.load(open(sys.argv[2]))
assert report['unique_samples']==156,report
assert len(indices)==report['balanced_draws_total']==216,report
assert len(set(indices))==156
PY

mkdir -p "$LOGS" "$INFERENCE"
cd "$REPO"
export CONFIG_FILE="$CONFIG" TRAIN_DATA="$ROOT/data" VALIDATION_DATA="$VALIDATION"
export LOG_DIR="$LOGS" WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29973
echo "Training start: $(date --iso-8601=seconds)"
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    --training.checkpoint.output_dir "$OUTPUT" \
    --training.tracker.run_name minimax_h3_camera_curriculum_156_hybrid200
echo "Training exit: $(date --iso-8601=seconds)"
test -s "$OUTPUT/checkpoint-200/dcp/.metadata" || {
    echo "No complete checkpoint-200; inference not started" >&2; exit 1;
}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export TOKENIZERS_PARALLELISM=false
# Two first frames, eight single-action trajectories each, equal seeds across
# directions. This is a diagnostic, not a blind production validation set.
for row in 3 11; do
    case_dir="$INFERENCE/row_$(printf '%02d' "$row")"
    mkdir -p "$case_dir"
    for actions in WASD IJKL; do
        group=("${actions:0:1}" "${actions:1:1}" "${actions:2:1}" "${actions:3:1}")
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$REPO/.venv/bin/torchrun" \
            --nnodes=1 --nproc_per_node=8 --master_port=29974 \
            scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
            --config "$CONFIG" --checkpoint "$OUTPUT/checkpoint-200" \
            --data-path "$VALIDATION" --row-index "$row" --output-dir "$case_dir" \
            --seed "$((20260930 + row))" --num-inference-steps 50 \
            --trajectories "${group[@]}" --trajectory-scale 1.0 \
            --freeze-intrinsics-first-frame \
            > "$LOGS/infer_row_${row}_${actions}.log" 2>&1
        for action in "${group[@]}"; do
            test -s "$case_dir/checkpoint-200/$action.mp4" || {
                echo "Missing inferred video row=$row action=$action" >&2; exit 1;
            }
        done
    done
done
date --iso-8601=seconds > "$INFERENCE/INFERENCE_COMPLETE"
echo "Eight-way inference complete: $(date --iso-8601=seconds)"
