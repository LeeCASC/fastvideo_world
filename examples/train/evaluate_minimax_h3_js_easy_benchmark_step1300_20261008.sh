#!/usr/bin/env bash
# Four exact JS-easy benchmark first frames, each with eight single-action paths.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003
OUT="$ROOT/evaluation/js_easy_benchmark_step1300_20261008"
CHECKPOINT="$ROOT/runs/train/checkpoints/checkpoint-1300"
CONFIG="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
BENCHMARK=/home/lizihao/projects/worldmodel/test_data/js_easy_benchmark
CACHE=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all

test -s "$CHECKPOINT/dcp/.metadata" || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 1; }
test -d "$BENCHMARK" && test -d "$CACHE"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU compute processes already present; refusing overlapping inference" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"$REPO/.venv/bin/python" scripts/evaluation/prepare_minimax_h3_js_easy_benchmark_20261008.py \
  --benchmark-root "$BENCHMARK" --cache-root "$CACHE" --model-path "$MODEL" \
  --output-root "$OUT/conditioning" --device-index 0 > "$OUT/conditioning.log" 2>&1

for case in js_00004 js_00007 js_00009 js_00012; do
  test -s "$OUT/conditioning/$case/data_000000.parquet"
  for group in wasd ijkl; do
    case "$group" in
      wasd) directions=(W A S D) ;;
      ijkl) directions=(I J K L) ;;
    esac
    lane="$OUT/videos/$case/$group"
    videos="$lane/checkpoint-1300"
    if [[ -f "$lane/COMPLETE" ]]; then
      echo "$(date --iso-8601=seconds) already complete: $case $group"
      continue
    fi
    mkdir -p "$lane"
    echo "$(date --iso-8601=seconds) starting: $case $group"
    "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29984 \
      scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" \
      --data-path "$OUT/conditioning/$case" --row-index 0 --output-dir "$lane" \
      --seed 20261008 --num-inference-steps 50 --trajectories "${directions[@]}" \
      --trajectory-scale 1.0 --freeze-intrinsics-first-frame --omit-reference-video \
      > "$lane/inference.log" 2>&1
    video_paths=()
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || { echo "Missing $case $direction video" >&2; exit 1; }
      video_paths+=("$videos/$direction.mp4")
    done
    "$REPO/.venv/bin/python" -c 'import cv2,sys; from pathlib import Path; paths=[Path(p) for p in sys.argv[1:]]; counts=[int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT)) for p in paths]; assert counts == [124] * len(paths), list(zip(paths, counts))' \
      "${video_paths[@]}" >/dev/null
    touch "$lane/COMPLETE"
    echo "$(date --iso-8601=seconds) completed: $case $group"
  done
done

FFMPEG="$($REPO/.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p "$OUT/montages"
for case in js_00004 js_00007 js_00009 js_00012; do
  base="$OUT/videos/$case"
  montage="$OUT/montages/${case}_WASD_IJKL.mp4"
  if [[ ! -s "$montage" ]]; then
    "$FFMPEG" -hide_banner -loglevel error -y \
      -i "$base/wasd/checkpoint-1300/W.mp4" -i "$base/wasd/checkpoint-1300/A.mp4" \
      -i "$base/wasd/checkpoint-1300/S.mp4" -i "$base/wasd/checkpoint-1300/D.mp4" \
      -i "$base/ijkl/checkpoint-1300/I.mp4" -i "$base/ijkl/checkpoint-1300/J.mp4" \
      -i "$base/ijkl/checkpoint-1300/K.mp4" -i "$base/ijkl/checkpoint-1300/L.mp4" \
      -filter_complex '[0:v]scale=320:184[v0];[1:v]scale=320:184[v1];[2:v]scale=320:184[v2];[3:v]scale=320:184[v3];[4:v]scale=320:184[v4];[5:v]scale=320:184[v5];[6:v]scale=320:184[v6];[7:v]scale=320:184[v7];[v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|320_0|640_0|960_0|0_184|320_184|640_184|960_184[v]' \
      -map '[v]' -an -c:v libx264 -preset veryfast -crf 20 "$montage"
  fi
  test -s "$montage"
done
touch "$OUT/EVALUATION_COMPLETE"
echo "Evaluation complete: $OUT"
