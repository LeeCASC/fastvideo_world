#!/usr/bin/env bash
set -euo pipefail

cd /home/lizihao/projects/worldmodel/H3-World
PY=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
CAP=/home/lizihao/projects/military_world_model_agent/FastVideoWM/tools/caption_vwm_61f_qwen3vl.py
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/vlm_qwen3vl_strict_50599
REG=$ROOT/registry_strict_50599.jsonl
MODEL=/datacc05/lizihao/models/worldarena/qwenvl3

mkdir -p "$ROOT/logs"
for shard in 0 1 2 3; do
  gpu=$shard
  session="vwm_124f_strict_vlm_gpu${gpu}_20261018"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  tmux new-session -d -s "$session" \
    "cd /home/lizihao/projects/worldmodel/H3-World && CUDA_VISIBLE_DEVICES=${gpu} '$PY' -u '$CAP' --registry '$REG' --output-root '$ROOT/gpu${gpu}' --shard ${shard} --num-shards 4 --device cuda:0 --model '$MODEL' > '$ROOT/logs/gpu${gpu}.log' 2>&1"
done
echo "launched strict 124f VLM on GPU0-3"
