#!/usr/bin/env bash
# Evaluate one train and one held-out first frame with one constant J or L
# camera command per generated video. The two four-GPU lanes run concurrently.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
CONFIG="${1:?Usage: $0 CONFIG CHECKPOINT}"
CHECKPOINT="${2:?Usage: $0 CONFIG CHECKPOINT}"
OUTPUT="${EVALUATION_ROOT:-$ROOT/evaluation_20261001}/$(basename "$(dirname "$(dirname "$CHECKPOINT")")")"

test -s "$CHECKPOINT/dcp/.metadata"
mkdir -p "$OUTPUT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd "$REPO"

run_lane() {
    local name="$1" devices="$2" port="$3" data="$4" row="$5"
    local lane="$OUTPUT/$name" video_dir="$OUTPUT/$name/$(basename "$CHECKPOINT")"
    mkdir -p "$lane"
    if [[ -s "$video_dir/J.mp4" && -s "$video_dir/L.mp4" ]]; then
        echo "$(date --iso-8601=seconds) $name J/L already complete"
        return
    fi
    CUDA_VISIBLE_DEVICES="$devices" "$REPO/.venv/bin/torchrun" \
        --nnodes=1 --nproc_per_node=4 --master_port="$port" \
        scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --data-path "$data" --row-index "$row" --output-dir "$lane" \
        --seed 20261001 --num-inference-steps 50 --trajectories J L \
        --freeze-intrinsics-first-frame \
        > "$lane/inference.log" 2>&1
    test -s "$video_dir/J.mp4"
    test -s "$video_dir/L.mp4"
    echo "$(date --iso-8601=seconds) $name J/L complete: $video_dir"
}

run_lane train_scene 0,1,2,3 29991 "$ROOT/data" 81 &
train_pid=$!
run_lane heldout_scene 4,5,6,7 29992 "$VALIDATION" 11 &
heldout_pid=$!
status=0
wait "$train_pid" || status=1
wait "$heldout_pid" || status=1
if (( status != 0 )); then
    echo "One or both inference lanes failed; inspect $OUTPUT/*/inference.log" >&2
    exit 1
fi
date --iso-8601=seconds > "$OUTPUT/EVALUATION_COMPLETE"
echo "Evaluation complete: $OUTPUT"
