#!/usr/bin/env bash
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f
INPUT=$ROOT/vae_latents/minimax_h3_prope_union_input
OUTPUT=/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/new_conditioned
LOG=$OUTPUT/logs
mkdir -p "$LOG"

[[ -x "$PY" ]] || { echo "missing python: $PY" >&2; exit 1; }
[[ -f "$MODEL/model_index.json" ]] || { echo "missing model index: $MODEL" >&2; exit 1; }
[[ -d "$INPUT" ]] || { echo "missing VAE input: $INPUT" >&2; exit 1; }

pids=()
for worker in 0 1 2 3 4 5 6 7; do
  gpu=$worker
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess condition \
    --input-dir "$INPUT" \
    --model-path "$MODEL" \
    --output-dir "$OUTPUT" \
    --worker-index "$worker" \
    --num-workers 8 \
    --device-index 0 \
    --master-port "$((29700 + worker))" \
    --conditioning prope >"$LOG/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed != 0 )); then
  echo "condition failed; inspect $LOG" >&2
  exit 1
fi

date -Iseconds > "$OUTPUT/CONDITIONING_COMPLETE"
echo "conditioned 124f new VAE cache successfully"
