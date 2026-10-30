#!/usr/bin/env bash
# GPU4-7 quality/VLM pass for the user-prioritized source families only.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
H3_ROOT=/home/lizihao/projects/worldmodel/H3-World
H3_PYTHON=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
DATA_ROOT=/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015
READY="$DATA_ROOT/registry/windows_61_ready_for_vlm.jsonl"
PRIORITY_ROOT="${PRIORITY_ROOT:-$DATA_ROOT/priority_sources_20261016}"
REGISTRY="$PRIORITY_ROOT/registry/windows_61_priority_ready_for_vlm.jsonl"
QC_ROOT="$PRIORITY_ROOT/gpu_qc"
CAP_ROOT="$PRIORITY_ROOT/prompts/qwen3vl"
LOG="$PRIORITY_ROOT/priority_qc_vlm.log"
mkdir -p "$PRIORITY_ROOT" "$QC_ROOT/logs" "$CAP_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

test -s "$READY" || { echo "ready registry missing: $READY" >&2; exit 1; }
FILTER_SCRIPT="${FILTER_SCRIPT:-$REPO/tools/filter_vwm_61f_priority_sources.py}"
"$REPO/.venv/bin/python" "$FILTER_SCRIPT" \
  --input "$READY" --output "$REGISTRY"
"$REPO/.venv/bin/python" "$REPO/tools/prepare_vwm_61f_gpu_qc_inputs.py" \
  --registry "$REGISTRY" --output-dir "$QC_ROOT/selections" --shards 4
expected="$(wc -l < "$REGISTRY")"
echo "[$(date --iso-8601=seconds)] priority rows=$expected; GPU4-7 only"

for gpu in 4 5 6 7; do
  shard=$((gpu - 4))
  session="vwm_61f_priority_qc_gpu${gpu}_20261016"
  selection="$QC_ROOT/selections/selection_shard_$(printf '%02d' "$shard")_of_04.jsonl"
  output="$QC_ROOT/results_shard_$(printf '%02d' "$shard")_of_04.jsonl"
  log="$QC_ROOT/logs/gpu${gpu}.log"
  tmux new-session -d -s "$session" \
    "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u code/vwm/run_fastvideowm_gpu_quality_pilot.py --selection '$selection' --output '$output' --batch-size 1 --decode-workers 2 > '$log' 2>&1"
done
echo "[$(date --iso-8601=seconds)] launched priority GPU-QC"
while :; do
  done_rows=0
  for output in "$QC_ROOT"/results_shard_*.jsonl; do
    [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
  done
  echo "[$(date --iso-8601=seconds)] priority GPU-QC progress=$done_rows/$expected"
  if (( done_rows >= expected )); then
    break
  fi
  sleep 300
done
touch "$QC_ROOT/GPU_QC_COMPLETE"

for gpu in 4 5 6 7; do
  shard=$((gpu - 4))
  session="vwm_61f_priority_caption_gpu${gpu}_20261016"
  log="$CAP_ROOT/logs/gpu${gpu}.log"
  tmux new-session -d -s "$session" \
    "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u '$REPO/tools/caption_vwm_61f_qwen3vl.py' --registry '$REGISTRY' --output-root '$CAP_ROOT' --shard $shard --num-shards 4 --device cuda:0 > '$log' 2>&1"
done
echo "[$(date --iso-8601=seconds)] launched priority Qwen3-VL captions"

while :; do
  done_rows=0
  for output in "$CAP_ROOT"/captions_shard_*.jsonl; do
    [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
  done
  echo "[$(date --iso-8601=seconds)] priority VLM progress=$done_rows/$expected"
  if (( done_rows >= expected )); then
    break
  fi
  sleep 300
done
touch "$CAP_ROOT/VLM_CAPTIONS_COMPLETE"
echo "[$(date --iso-8601=seconds)] priority GPU-QC/VLM complete"
