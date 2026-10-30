#!/usr/bin/env bash
# Exact JS-easy first frames, eight single-camera commands per image, one DCP.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
BENCHMARK=/home/lizihao/projects/worldmodel/test_data/js_easy_benchmark
CACHE=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
CONFIG="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
CHECKPOINT="${CHECKPOINT:?Set the frozen complete FastVideoWM checkpoint}"
OUT="${OUT:?Set the output directory}"

# The orchestrator may already be running an older copy of its shell script.
# Raise this scope's cap before allocating any inference tensors in that case.
scope_name="$(basename "$(cut -d: -f3 /proc/self/cgroup)")"
if [[ "$scope_name" == run-*.scope ]]; then
  systemctl --user set-property --runtime "$scope_name" MemoryMax=650G MemorySwapMax=0
  [[ "$(systemctl --user show "$scope_name" -p MemoryMax --value)" -eq $((650 * 1024 * 1024 * 1024)) ]] || {
    echo "Failed to set FastVideoWM inference memory cap to 650 GiB" >&2
    exit 1
  }
fi

test -s "$CHECKPOINT/dcp/.metadata"
test -d "$BENCHMARK" && test -d "$CACHE"
[[ "$(find "$BENCHMARK" -mindepth 2 -maxdepth 2 -name sample_001_first_frame.png | wc -l)" -eq 17 ]]
"$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py" || {
  echo "GPUs are occupied or CUDA is unavailable; refusing overlapping inference" >&2
  exit 1
}

mkdir -p "$OUT/conditioning" "$OUT/videos"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

conditioned=false
for attempt in 1 2; do
  if "$REPO/.venv/bin/python" scripts/evaluation/prepare_minimax_h3_js_easy_benchmark_20261008.py \
    --benchmark-root "$BENCHMARK" --cache-root "$CACHE" --model-path "$MODEL" \
    --output-root "$OUT/conditioning" --device-index 0 --all-benchmark-images \
    >> "$OUT/conditioning.log" 2>&1; then
    conditioned=true
    break
  fi
  echo "$(date --iso-8601=seconds) conditioning attempt $attempt failed" >&2
  sleep 15
done
[[ "$conditioned" == true ]] || exit 1

checkpoint_label="$(basename "$CHECKPOINT")"
for case_number in $(seq -w 1 17); do
  case_id="js_$(printf '%05d' "$((10#$case_number))")"
  test -s "$OUT/conditioning/$case_id/data_000000.parquet"
  for group in wasd ijkl; do
    case "$group" in
      wasd) directions=(W A S D) ;;
      ijkl) directions=(I J K L) ;;
    esac
    lane="$OUT/videos/$case_id/$group"
    videos="$lane/$checkpoint_label"
    if [[ -f "$lane/COMPLETE" ]]; then
      echo "$(date --iso-8601=seconds) already complete: $case_id $group"
      continue
    fi
    mkdir -p "$lane"
    echo "$(date --iso-8601=seconds) starting: $case_id $group"
    generated=false
    for attempt in 1 2 3; do
      if "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29984 \
        scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --data-path "$OUT/conditioning/$case_id" --row-index 0 --output-dir "$lane" \
        --seed 811 --num-inference-steps 50 --trajectories "${directions[@]}" \
        --trajectory-scale 1.0 --freeze-intrinsics-first-frame --omit-reference-video \
        >> "$lane/inference.log" 2>&1; then
        generated=true
        break
      fi
      echo "$(date --iso-8601=seconds) $case_id $group attempt $attempt failed" >&2
      sleep 15
    done
    [[ "$generated" == true ]] || exit 1
    video_paths=()
    for direction in "${directions[@]}"; do
      test -s "$videos/$direction.mp4" || { echo "Missing $case_id $direction" >&2; exit 1; }
      video_paths+=("$videos/$direction.mp4")
    done
    "$REPO/.venv/bin/python" -c 'import cv2,sys; from pathlib import Path; paths=[Path(p) for p in sys.argv[1:]]; counts=[int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT)) for p in paths]; assert counts == [124] * len(paths), list(zip(paths, counts))' \
      "${video_paths[@]}" >/dev/null
    touch "$lane/COMPLETE"
    echo "$(date --iso-8601=seconds) completed: $case_id $group"
  done
done

count="$(find "$OUT/videos" -path "*/$checkpoint_label/*.mp4" -type f | wc -l)"
[[ "$count" -eq 136 ]] || { echo "Expected 136 MP4s; found $count" >&2; exit 1; }
touch "$OUT/EVALUATION_COMPLETE"
echo "Evaluation complete: $OUT (136 videos)"
