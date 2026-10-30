#!/usr/bin/env bash
# Held-out single-command camera sweeps for the full-data residual-PRoPE run.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003
OLD=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/inference_step750_single_action_20260930
CHECKPOINT="$ROOT/runs/train/checkpoints/checkpoint-1300"
CONFIG="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
DATA=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
OUT="$ROOT/evaluation/step1300_wasdijkl_20261008"

test -s "$CHECKPOINT/dcp/.metadata" || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 1; }
test -d "$DATA" || { echo "Missing validation data: $DATA" >&2; exit 1; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU compute processes already present; refusing overlapping inference" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# These rows are held out and match the old step-750 test: multicam, UE-easy, JS-easy.
for row in 0 2 11; do
  for group in wasd ijkl; do
    case "$group" in
      wasd) directions=(W A S D) ;;
      ijkl) directions=(I J K L) ;;
    esac
    lane="$OUT/validation_row_${row}/$group"
    videos="$lane/checkpoint-1300"
    complete=true
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || complete=false
    done
    if [[ "$complete" == true ]]; then
      echo "$(date --iso-8601=seconds) already complete: row=$row group=$group"
      continue
    fi
    mkdir -p "$lane"
    echo "$(date --iso-8601=seconds) starting: row=$row group=$group"
    "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29984 \
      scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA" \
      --row-index "$row" --output-dir "$lane" --seed "$((20260930 + row))" \
      --num-inference-steps 50 --trajectories "${directions[@]}" \
      --trajectory-scale 1.0 --freeze-intrinsics-first-frame \
      > "$lane/inference.log" 2>&1
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || { echo "Missing $direction video for row $row" >&2; exit 1; }
    done
    echo "$(date --iso-8601=seconds) completed: row=$row group=$group"
  done
done

FFMPEG="$($REPO/.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
for row in 0 2 11; do
  base="$OUT/validation_row_${row}"
  montage="$base/WASD_IJKL_montage.mp4"
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

  old_row="$OLD/row_$(printf '%02d' "$row")/checkpoint-750"
  comparison="$base/step750_vs_step1300.mp4"
  if [[ -d "$old_row" && ! -s "$comparison" ]]; then
    inputs=()
    for direction in W A S D I J K L; do
      test -s "$old_row/$direction.mp4" || { echo "Missing old $direction video for row $row" >&2; exit 1; }
      inputs+=(-i "$old_row/$direction.mp4")
    done
    for direction in W A S D I J K L; do
      group=wasd
      [[ "$direction" == [IJKL] ]] && group=ijkl
      inputs+=(-i "$base/$group/checkpoint-1300/$direction.mp4")
    done
    filters=''
    labels=''
    layout=''
    for index in {0..15}; do
      filters+="[$index:v]scale=320:184[v$index];"
      labels+="[v$index]"
      layout+="$((index % 4 * 320))_$((index / 4 * 184))|"
    done
    "$FFMPEG" -hide_banner -loglevel error -y "${inputs[@]}" \
      -filter_complex "${filters}${labels}xstack=inputs=16:layout=${layout%|}[v]" \
      -map '[v]' -an -c:v libx264 -preset veryfast -crf 20 "$comparison"
    test -s "$comparison"
  fi
done

"$REPO/.venv/bin/python" scripts/evaluation/analyze_minimax_h3_multiaction_20261002.py \
  --root "$OUT" --step 1300 --rows 0 2 11 > "$OUT/analysis.log" 2>&1
test -s "$OUT/analysis.md"
touch "$OUT/EVALUATION_COMPLETE"
echo "Evaluation complete: $OUT"
