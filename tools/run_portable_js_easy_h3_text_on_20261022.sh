#!/usr/bin/env bash
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
OUT=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022
PREVIOUS=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022
BUNDLE=/datacc05/lizihao/js_easy_benchmark_portable_20261021
CHECKPOINT=$PREVIOUS/checkpoint/checkpoint-800
CONFIG=$REPO/examples/train/configs/minimax_h3_union_20261019.yaml
cd "$REPO"
mkdir -p "$OUT/logs"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
trap 'touch "$OUT/NEEDS_ATTENTION"' ERR
"$PY" examples/train/check_minimax_h3_gpus_idle.py
text_args=(--output "$OUT" --previous "$PREVIOUS" --bundle "$BUNDLE")
"$PY" scripts/evaluation/portable_js_easy_h3_subject_text.py setup "${text_args[@]}" > "$OUT/logs/setup.log" 2>&1
pids=()
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((worker * 2)) "$PY" -u scripts/evaluation/portable_js_easy_h3_subject_text.py condition \
    "${text_args[@]}" --worker "$worker" --workers 4 > "$OUT/logs/condition_$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then touch "$OUT/NEEDS_ATTENTION"; exit 1; fi
"$PY" scripts/evaluation/portable_js_easy_h3_subject_text.py audit "${text_args[@]}" > "$OUT/logs/input_audit.log" 2>&1
args=(--bundle "$BUNDLE" --output "$OUT" --checkpoint "$CHECKPOINT" --config "$CONFIG")
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
"$PY" scripts/evaluation/portable_js_easy_h3_subject_text.py summarize "${text_args[@]}" > "$OUT/logs/summary.log" 2>&1
"$PY" tools/visualize_h3_controls.py --roots "$OUT" --workers 4 > "$OUT/logs/visualization.log" 2>&1
