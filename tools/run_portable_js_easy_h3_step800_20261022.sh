#!/usr/bin/env bash
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
BUNDLE=/datacc05/lizihao/js_easy_benchmark_portable_20261021
OUT=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022
CHECKPOINT=/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/runs/train/checkpoints/checkpoint-800
CONFIG=$REPO/examples/train/configs/minimax_h3_union_20261019.yaml
cd "$REPO"
mkdir -p "$OUT/logs"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
args=(--bundle "$BUNDLE" --output "$OUT" --checkpoint "$CHECKPOINT" --config "$CONFIG")
trap 'touch "$OUT/NEEDS_ATTENTION"' ERR
"$PY" scripts/evaluation/portable_js_easy_h3.py freeze "${args[@]}" > "$OUT/logs/freeze.log" 2>&1
# Four independent VAE/condition workers keep host RAM below the old
# eight-full-model inference loading peak. Each worker prepares four scenes.
pids=()
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((worker * 2)) "$PY" -u scripts/evaluation/portable_js_easy_h3.py prepare \
    "${args[@]}" --worker "$worker" --workers 4 > "$OUT/logs/prepare_$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then touch "$OUT/NEEDS_ATTENTION"; exit 1; fi
touch "$OUT/INPUTS_COMPLETE"
# One distributed model load, four simultaneous SP lanes and 32 waves.
# Native HSDP shares the 34B backbone; no eight independent full-model loads.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/evaluation/portable_js_easy_h3.py infer "${args[@]}" > "$OUT/logs/infer.log" 2>&1
pids=()
for worker in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$worker "$PY" -u scripts/evaluation/portable_js_easy_h3.py decode \
    "${args[@]}" --worker "$worker" --workers 8 > "$OUT/logs/decode_$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then touch "$OUT/NEEDS_ATTENTION"; exit 1; fi
touch "$OUT/ALL_VIDEOS_COMPLETE"
"$PY" tools/visualize_h3_controls.py --roots "$OUT" --workers 4 > "$OUT/logs/visualization.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluation/score_portable_js_easy_h3.py --output "$OUT" \
  > "$OUT/logs/metrics.log" 2>&1
# Continuation is owned by tools/after_text_evaluations_train_20261022.sh.
