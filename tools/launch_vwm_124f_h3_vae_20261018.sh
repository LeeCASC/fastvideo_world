#!/usr/bin/env bash
set -euo pipefail

cd /home/lizihao/projects/worldmodel/H3-World
PY=/home/lizihao/projects/military_world_model_agent/FastVideoWM/.venv/bin/python
ENC=/home/lizihao/projects/military_world_model_agent/FastVideoWM/tools/encode_vwm_registry_h3_20261018.py
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016
OUT=$ROOT/124f/vae_latents/minimax_h3_prope
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
OLD=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925

mkdir -p "$OUT/logs"
for worker in 0 1 2 3; do
  gpu=$((worker + 4))
  session="vwm_124f_h3_vae_gpu${gpu}_20261018"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  tmux new-session -d -s "$session" \
    "cd /home/lizihao/projects/worldmodel/H3-World && CUDA_VISIBLE_DEVICES=${gpu} '$PY' -u '$ENC' --root '$ROOT' --output '$OUT' --model '$MODEL' --worker-index ${worker} --num-workers 4 --device-index 0 --shard-size 8 --existing-cache '$OLD' > '$OUT/logs/gpu${gpu}.log' 2>&1"
done
echo "launched 124f MiniMax-H3 VAE encoding on GPU4-7"
