#!/usr/bin/env bash
# At 23:00, switch the text-conditioned trainer at a complete checkpoint.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
H3_ROOT=/home/lizihao/projects/worldmodel/H3-World
H3_PYTHON=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
DATA_ROOT=/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015
READY="$DATA_ROOT/registry/windows_61_ready_for_vlm.jsonl"
TRAIN_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015
TRAIN_SESSION=fastvideowm_multicam5k_text_20261015
TRAIN4_SESSION=fastvideowm_multicam5k_text_20261015_4gpu
RUN_LOG="$DATA_ROOT/schedule_4gpu_qc_vlm.log"
QC_ROOT="$DATA_ROOT/gpu_qc"
CAP_ROOT="$DATA_ROOT/prompts/qwen3vl_61f"
mkdir -p "$DATA_ROOT" "$QC_ROOT/logs" "$CAP_ROOT/logs"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] scheduler started"
target_epoch="$(date -d 'today 23:00:00' +%s)"
now_epoch="$(date +%s)"
if (( now_epoch < target_epoch )); then
  sleep "$((target_epoch - now_epoch))"
fi
echo "[$(date --iso-8601=seconds)] reached scheduled switch time"

checkpoint_dir() {
  local best=0 path number
  for path in "$TRAIN_ROOT/runs/train/checkpoints"/checkpoint-*; do
    [[ -d "$path" && -s "$path/dcp/.metadata" ]] || continue
    number="${path##*/checkpoint-}"
    [[ "$number" =~ ^[0-9]+$ ]] || continue
    (( number > best )) && best="$number"
  done
  if (( best > 0 )); then
    printf '%s/checkpoint-%s' "$TRAIN_ROOT/runs/train/checkpoints" "$best"
  fi
}

while :; do
  resume="$(checkpoint_dir || true)"
  if [[ -n "$resume" && "${resume##*/checkpoint-}" -ge 200 ]]; then
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting for complete checkpoint-200; current=${resume:-none}"
  sleep 30
done
echo "[$(date --iso-8601=seconds)] selected resume checkpoint=$resume"

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] requesting graceful stop of $TRAIN_SESSION"
  tmux send-keys -t "$TRAIN_SESSION:0" C-c
fi
for _ in $(seq 1 120); do
  if ! pgrep -f 'fastvideowm_multicam5k_text_20261015.*torch.distributed.run|fastvideo.train.entrypoint.train.*multicam5k_text_20261015' >/dev/null; then
    break
  fi
  sleep 5
done
if pgrep -f 'fastvideowm_multicam5k_text_20261015.*torch.distributed.run|fastvideo.train.entrypoint.train.*multicam5k_text_20261015' >/dev/null; then
  echo "[$(date --iso-8601=seconds)] trainer did not stop cleanly; refusing GPU overlap" >&2
  touch "$DATA_ROOT/SWITCH_NEEDS_ATTENTION"
  exit 1
fi

if tmux has-session -t "$TRAIN4_SESSION" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] 4-GPU session already exists; refusing duplicate launch" >&2
  exit 1
fi
tmux new-session -d -s "$TRAIN4_SESSION" \
  "cd '$REPO' && RESUME_CHECKPOINT='$resume' bash examples/train/run_minimax_h3_multicam5k_text_4gpu_20261015.sh"
echo "[$(date --iso-8601=seconds)] launched 4-GPU training on CUDA 0-3, grad_accumulation=16, total_bs=64"

while [[ ! -s "$READY" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for 61f materialization: $READY"
  sleep 300
done
echo "[$(date --iso-8601=seconds)] materialized registry found"

"$REPO/.venv/bin/python" "$REPO/tools/prepare_vwm_61f_gpu_qc_inputs.py" \
  --registry "$READY" --output-dir "$QC_ROOT/selections" --shards 4
expected="$(wc -l < "$READY")"
for gpu in 4 5 6 7; do
  shard=$((gpu - 4))
  session="vwm_61f_gpu_qc_gpu${gpu}_20261015"
  selection="$QC_ROOT/selections/selection_shard_$(printf '%02d' "$shard")_of_04.jsonl"
  output="$QC_ROOT/results_shard_$(printf '%02d' "$shard")_of_04.jsonl"
  log="$QC_ROOT/logs/gpu${gpu}.log"
  tmux new-session -d -s "$session" \
    "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u code/vwm/run_fastvideowm_gpu_quality_pilot.py --selection '$selection' --output '$output' --batch-size 1 --decode-workers 2 > '$log' 2>&1"
done
echo "[$(date --iso-8601=seconds)] launched four GPU-QC shards; expected rows=$expected"

while :; do
  done_rows=0
  for output in "$QC_ROOT"/results_shard_*.jsonl; do
    [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
  done
  echo "[$(date --iso-8601=seconds)] GPU-QC progress=$done_rows/$expected"
  if (( done_rows >= expected )); then
    break
  fi
  sleep 300
done
touch "$QC_ROOT/GPU_QC_COMPLETE"

for gpu in 4 5 6 7; do
  shard=$((gpu - 4))
  session="vwm_61f_caption_gpu${gpu}_20261015"
  log="$CAP_ROOT/logs/gpu${gpu}.log"
  tmux new-session -d -s "$session" \
    "cd '$H3_ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$H3_PYTHON' -u '$REPO/tools/caption_vwm_61f_qwen3vl.py' --registry '$READY' --output-root '$CAP_ROOT' --shard $shard --num-shards 4 --device cuda:0 > '$log' 2>&1"
done
echo "[$(date --iso-8601=seconds)] launched four Qwen3-VL caption shards"

while :; do
  done_rows=0
  for output in "$CAP_ROOT"/captions_shard_*.jsonl; do
    [[ -f "$output" ]] && done_rows=$((done_rows + $(wc -l < "$output")))
  done
  echo "[$(date --iso-8601=seconds)] caption progress=$done_rows/$expected"
  if (( done_rows >= expected )); then
    break
  fi
  sleep 300
done
touch "$CAP_ROOT/VLM_CAPTIONS_COMPLETE"
echo "[$(date --iso-8601=seconds)] GPU QC and VLM captions complete"
