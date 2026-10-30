#!/usr/bin/env bash
# Matched-seed eight-direction evaluation at a completed full-data checkpoint.
set -euo pipefail

STEP="${1:?Pass a completed checkpoint step (for example 500)}"
[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || { echo "step must be a positive integer" >&2; exit 2; }
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002
CHECKPOINT="$ROOT/runs/train/checkpoints/checkpoint-$STEP"
CONFIG="$REPO/examples/train/configs/minimax_h3_full_action_strong_20261002.yaml"
DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
OUTPUT="$ROOT/evaluation/step${STEP}_heldout_50steps"
test -s "$CHECKPOINT/dcp/.metadata" || { echo "checkpoint-$STEP not complete" >&2; exit 1; }
mkdir -p "$OUTPUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for row in 1 11; do
  for group in wasd ijkl; do
    case "$group" in
      wasd) directions=(W A S D) ;;
      ijkl) directions=(I J K L) ;;
    esac
    lane="$OUTPUT/validation_row_${row}/$group"
    videos="$lane/checkpoint-$STEP"
    mkdir -p "$lane"
    complete=true
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || complete=false
    done
    if [[ "$complete" == true ]]; then
      echo "$(date --iso-8601=seconds) already complete: row=$row group=$group"
      continue
    fi
    echo "$(date --iso-8601=seconds) inference start: step=$STEP row=$row group=$group"
    "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29980 \
      scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA" \
      --row-index "$row" --output-dir "$lane" --seed 20261002 \
      --num-inference-steps 50 --trajectories "${directions[@]}" \
      --freeze-intrinsics-first-frame > "$lane/inference.log" 2>&1
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || { echo "missing $direction video for row $row" >&2; exit 1; }
    done
    echo "$(date --iso-8601=seconds) inference complete: step=$STEP row=$row group=$group"
  done
done

FFMPEG="$($REPO/.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
for row in 1 11; do
  base="$OUTPUT/validation_row_${row}"
  "$FFMPEG" -hide_banner -loglevel error -y \
    -i "$base/wasd/checkpoint-$STEP/W.mp4" -i "$base/wasd/checkpoint-$STEP/A.mp4" \
    -i "$base/wasd/checkpoint-$STEP/S.mp4" -i "$base/wasd/checkpoint-$STEP/D.mp4" \
    -i "$base/ijkl/checkpoint-$STEP/I.mp4" -i "$base/ijkl/checkpoint-$STEP/J.mp4" \
    -i "$base/ijkl/checkpoint-$STEP/K.mp4" -i "$base/ijkl/checkpoint-$STEP/L.mp4" \
    -filter_complex '[0:v]scale=320:184[v0];[1:v]scale=320:184[v1];[2:v]scale=320:184[v2];[3:v]scale=320:184[v3];[4:v]scale=320:184[v4];[5:v]scale=320:184[v5];[6:v]scale=320:184[v6];[7:v]scale=320:184[v7];[v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|320_0|640_0|960_0|0_184|320_184|640_184|960_184[v]' \
    -map '[v]' -an -c:v libx264 -preset veryfast -crf 20 "$base/WASD_IJKL_montage.mp4"
  test -s "$base/WASD_IJKL_montage.mp4"
done
"$REPO/.venv/bin/python" scripts/evaluation/analyze_minimax_h3_multiaction_20261002.py \
  --root "$OUTPUT" --step "$STEP" --rows 1 11 > "$OUTPUT/analysis.log" 2>&1
test -s "$OUTPUT/analysis.md"
touch "$OUTPUT/EVALUATION_COMPLETE"
echo "Evaluation complete: $OUTPUT"
