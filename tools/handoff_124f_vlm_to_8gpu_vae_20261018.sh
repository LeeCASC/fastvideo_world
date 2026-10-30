#!/usr/bin/env bash
set -euo pipefail

ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016
STRICT="$ROOT/124f/vlm_qwen3vl_strict_50599"
PHASE1="$ROOT/124f/vae_latents/minimax_h3_prope"
PHASE2="$ROOT/124f/vae_latents/minimax_h3_prope_phase2_8gpu"
PY=/home/lizihao/projects/military_world_model_agent/FastVideoWM/.venv/bin/python
ENC=/home/lizihao/projects/military_world_model_agent/FastVideoWM/tools/encode_vwm_registry_h3_20261018.py
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
OLD=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925

echo "waiting for strict VLM (50599 successful captions)"
while true; do
  vlm_done=$($PY - <<'PY'
from pathlib import Path
import json
p = Path('/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/vlm_qwen3vl_strict_50599')
n = 0
for f in p.glob('gpu*/captions_shard_*.jsonl'):
    with f.open(encoding='utf-8') as h:
        for line in h:
            try:
                x = json.loads(line)
                n += int(x.get('error') in (None, '') and bool(x.get('scene_description')))
            except Exception:
                pass
print(n)
PY
  )
  running=$(tmux ls 2>/dev/null | grep -c 'vwm_124f_strict_vlm_gpu' || true)
  echo "vlm_success=$vlm_done running_vlm_sessions=$running"
  if [ "$vlm_done" -ge 50599 ] && [ "$running" -eq 0 ]; then
    break
  fi
  sleep 300
done

echo "strict VLM complete; stopping phase-1 VAE workers at safe resumable boundaries"
for gpu in 4 5 6 7; do
  tmux kill-session -t "vwm_124f_h3_vae_gpu${gpu}_20261018" 2>/dev/null || true
done
sleep 10
mkdir -p "$PHASE2/logs"

for worker in 0 1 2 3 4 5 6 7; do
  session="vwm_124f_h3_vae_8gpu_w${worker}_20261018"
  if tmux has-session -t "$session" 2>/dev/null; then
    continue
  fi
  tmux new-session -d -s "$session" \
    "cd /home/lizihao/projects/worldmodel/H3-World && CUDA_VISIBLE_DEVICES=${worker} '$PY' -u '$ENC' --root '$ROOT' --output '$PHASE2' --model '$MODEL' --worker-index ${worker} --num-workers 8 --device-index 0 --shard-size 8 --existing-cache '$OLD' --existing-cache '$PHASE1' > '$PHASE2/logs/gpu${worker}.log' 2>&1"
done

date -Iseconds > "$PHASE2/STARTED_AFTER_VLM_8GPU"
echo "launched phase-2 VAE on GPU0-7"
