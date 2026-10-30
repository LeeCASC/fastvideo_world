#!/usr/bin/env bash
# GPU0-3: corrected 124f/243f advisory GPU-QC followed by Qwen3-VL captions.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
H3_ROOT=/home/lizihao/projects/worldmodel/H3-World
H3_PYTHON=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016
MODEL=/datacc05/lizihao/models/worldarena/qwenvl3

run_qc() {
  local name="$1" frames="$2"
  local base="$ROOT/$name" sel="$ROOT/$name/gpu_qc/selections"
  local out="$ROOT/$name/gpu_qc/results" logroot="$ROOT/$name/gpu_qc/logs"
  local expected
  expected=$(awk '{n += 1} END {print n + 0}' "$sel"/selection_shard_*.jsonl)
  mkdir -p "$out" "$logroot"
  for gpu in 0 1 2 3; do
    local shard=$((gpu))
    local selection="$sel/selection_shard_$(printf '%02d' "$shard")_of_04.jsonl"
    local output="$out/results_shard_$(printf '%02d' "$shard")_of_04.jsonl"
    local session="vwm_${name}_gpu_qc_gpu${gpu}_20261017"
    if [[ ! -s "$selection" ]]; then
      echo "missing selection: $selection" >&2
      exit 1
    fi
    local shard_expected
    shard_expected=$(wc -l < "$selection")
    if [[ -s "$output" && "$(wc -l < "$output")" -ge "$shard_expected" ]]; then
      continue
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      continue
    fi
    tmux new-session -d -s "$session" \
      "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u code/vwm/run_fastvideowm_gpu_quality_pilot.py --selection '$selection' --output '$output' --model '$MODEL' --batch-size 4 --decode-workers 4 > '$logroot/gpu${gpu}.log' 2>&1"
  done
  while :; do
    local done_rows=0
    for output in "$out"/results_shard_*.jsonl; do
      [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
    done
    echo "[$(date '+%F %T %Z')] $name GPU-QC progress=$done_rows/$expected"
    if (( done_rows >= expected )); then
      printf 'completed_at=%s\nrows=%s\n' "$(date '+%F %T %Z')" "$done_rows" > "$base/gpu_qc/GPU_QC_COMPLETE"
      break
    fi
    sleep 300
  done
}

run_caption() {
  local name="$1" frames="$2"
  local base="$ROOT/$name" sel="$ROOT/$name/gpu_qc/selections"
  local out="$ROOT/$name/vlm_qwen3vl"
  local expected
  expected=$(awk '{n += 1} END {print n + 0}' "$sel"/selection_shard_*.jsonl)
  mkdir -p "$out"
  for gpu in 0 1 2 3; do
    local selection="$sel/selection_shard_$(printf '%02d' "$gpu")_of_04.jsonl"
    local session="vwm_${name}_vlm_gpu${gpu}_20261017"
    if tmux has-session -t "$session" 2>/dev/null; then
      continue
    fi
    mkdir -p "$out/gpu${gpu}/logs"
    tmux new-session -d -s "$session" \
      "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u '$REPO/tools/caption_vwm_61f_qwen3vl.py' --registry '$selection' --output-root '$out/gpu${gpu}' --shard 0 --num-shards 1 --device cuda:0 --model '$MODEL' > '$out/gpu${gpu}/logs/caption.log' 2>&1"
  done
  while :; do
    local done_rows=0
    for output in "$out"/gpu*/captions_shard_*.jsonl; do
      [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
    done
    echo "[$(date '+%F %T %Z')] $name VLM progress=$done_rows/$expected"
    if (( done_rows >= expected )); then
      printf 'completed_at=%s\nrows=%s\n' "$(date '+%F %T %Z')" "$done_rows" > "$base/VLM_CAPTIONS_COMPLETE"
      break
    fi
    sleep 300
  done
}

run_qc 124f 124
run_caption 124f 124
run_qc 243f 243
run_caption 243f 243
echo "[$(date '+%F %T %Z')] 124f/243f GPU-QC and VLM complete"
