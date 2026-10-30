#!/usr/bin/env bash
# Freeze two complete checkpoints, infer 17 x 8 for each model, then resume the paused training.
set -euo pipefail

H3_REPO=/home/lizihao/projects/worldmodel/H3-World
FW_REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009
H3_CHECKPOINT=/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/from_step2450_legacy_plus_captioned_v2_bs35_gpu0_6_to5000_20260924/step-1000.safetensors
FW_CHECKPOINT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-400
GUARD=fastvideowm_multicam5k_guard_20261008
GUARD_SCRIPT="$FW_REPO/examples/train/guard_minimax_h3_multicam5k_20261008.sh"

test -s "$H3_CHECKPOINT"
test -s "$H3_CHECKPOINT" && test -s "${H3_CHECKPOINT%/*}/state-001000/complete.json"
test -s "$FW_CHECKPOINT/dcp/.metadata"
mkdir -p "$ROOT"

restart_training() {
  result=$?
  trap - EXIT
  if [[ "$result" -ne 0 ]]; then
    touch "$ROOT/INFERENCE_FAILED"
    echo "$(date --iso-8601=seconds) inference failed, exit=$result; training stays paused for diagnosis" >&2
    exit "$result"
  fi
  test -f "$ROOT/h3_world/status.json"
  test -f "$ROOT/fastvideowm/EVALUATION_COMPLETE"
  if [[ -f "$ROOT/INFERENCE_FAILED" ]]; then
    mv "$ROOT/INFERENCE_FAILED" "$ROOT/INFERENCE_FAILED.previous"
  fi
  touch "$ROOT/INFERENCE_COMPLETE"
  echo "$(date --iso-8601=seconds) both models complete; restoring training guard"
  if ! tmux has-session -t "$GUARD" 2>/dev/null; then
    tmux new-session -d -s "$GUARD" \
      "TARGET_STEPS=1000 bash '$GUARD_SCRIPT'"
    echo "$(date --iso-8601=seconds) FastVideoWM resume guard launched: $GUARD"
  fi
  exit "$result"
}
trap restart_training EXIT

run_h3_shards() {
  local memory_max="$1" vram_limit="$2" stagger="$3"
  shift 3
  # A killed scope can leave GPU allocations visible for a few seconds.
  # Wait for the devices to clear before falling back to a smaller shard count.
  for ((wait_seconds=0; wait_seconds<300; wait_seconds+=5)); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
      break
    fi
    sleep 5
  done
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    echo "$(date --iso-8601=seconds) GPUs did not become idle after 300 seconds" >&2
    return 1
  fi
  echo "$(date --iso-8601=seconds) H3-World attempt: GPUs=$* MemoryMax=${memory_max}G VRAM=${vram_limit}G stagger=${stagger}s"
  systemd-run --user --scope --quiet \
    -p "MemoryMax=${memory_max}G" -p MemorySwapMax=0 \
    /home/lizihao/anaconda3/envs/minimax_h3/bin/python -u \
    "$H3_REPO/code/vwm/run_js17_latest_single_checkpoint.py" \
    --checkpoint "$H3_CHECKPOINT" --output "$ROOT/h3_world" \
    --group-oom-kill --vram-limit-gb "$vram_limit" --stagger-seconds "$stagger" --gpu-ids "$@"
}

if [[ ! -f "$ROOT/h3_world/status.json" ]] || \
   [[ "$(jq -r '.status' "$ROOT/h3_world/status.json")" != complete ]]; then
  echo "$(date --iso-8601=seconds) H3-World inference starts"
  # Try all available GPUs with more weights resident on device. Every attempt
  # has a host-memory cap and cleanly falls back to fewer resumable shards.
  if run_h3_shards 650 68 60 0 1 2 3 4 5 6 7; then
    echo "$(date --iso-8601=seconds) eight-GPU H3-World inference complete"
  elif run_h3_shards 540 64 60 0 1 2 3 4 5 6; then
    echo "$(date --iso-8601=seconds) seven-GPU H3-World fallback complete"
  elif run_h3_shards 500 60 60 0 1 2 3 4 5; then
    echo "$(date --iso-8601=seconds) six-GPU H3-World fallback complete"
  elif run_h3_shards 460 56 60 0 1 2 3 4; then
    echo "$(date --iso-8601=seconds) five-GPU H3-World fallback complete"
  elif run_h3_shards 420 56 60 0 1 2 3; then
    echo "$(date --iso-8601=seconds) four-GPU H3-World fallback complete"
  elif run_h3_shards 380 56 60 0 1; then
    echo "$(date --iso-8601=seconds) two-GPU H3-World fallback complete"
  else
    echo "$(date --iso-8601=seconds) falling back to one H3-World GPU" >&2
    run_h3_shards 300 56 0 0
  fi
fi

echo "$(date --iso-8601=seconds) FastVideoWM inference starts"
systemd-run --user --scope --quiet \
  -p MemoryMax=650G -p MemorySwapMax=0 \
  env CHECKPOINT="$FW_CHECKPOINT" OUT="$ROOT/fastvideowm" \
  bash "$FW_REPO/examples/train/evaluate_minimax_h3_js_easy17_latest_20261009.sh"

[[ "$(jq -r '.validated_videos' "$ROOT/h3_world/status.json")" -eq 136 ]]
test -f "$ROOT/fastvideowm/EVALUATION_COMPLETE"
echo "$(date --iso-8601=seconds) 272 videos verified"
