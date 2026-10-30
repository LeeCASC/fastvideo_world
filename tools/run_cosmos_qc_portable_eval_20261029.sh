#!/usr/bin/env bash
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
OUT=/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_step1700_portable_eval_20261029
BUNDLE=/datacc05/lizihao/js_easy_benchmark_portable_20261021
TRAIN=/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_108439_resume1100_20261027/subject_motion
CHECKPOINT=$TRAIN/runs/train/checkpoints/checkpoint-1700
CONFIG=$TRAIN/training_config.yaml
cd "$REPO"
mkdir -p "$OUT/logs"
exec 9>"$OUT/job.lock"
flock -n 9 || { echo 'Evaluation already active' >&2; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
trap 'touch "$OUT/NEEDS_ATTENTION"' ERR
"$PY" tools/prepare_cosmos_qc_portable_eval_20261029.py idle --output "$OUT"
args=(--bundle "$BUNDLE" --output "$OUT" --checkpoint "$CHECKPOINT" --config "$CONFIG")
if [[ ! -f "$OUT/checkpoints.json" ]]; then
  # The shared runner imports Triton CUDA kernels even for CPU-only hashing.
  # Expose one idle device for import; freeze does not load model weights.
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluation/portable_js_easy_h3.py freeze \
    "${args[@]}" --preserve-init-checkpoint --inference-num-gpus 4 > "$OUT/logs/freeze.log" 2>&1
fi
CUDA_VISIBLE_DEVICES= "$PY" -u tools/prepare_cosmos_qc_portable_eval_20261029.py setup \
  --output "$OUT" > "$OUT/logs/setup.log" 2>&1
pids=()
for worker in 0 1; do
  CUDA_VISIBLE_DEVICES=$((worker * 2)) "$PY" -u tools/prepare_cosmos_qc_portable_eval_20261029.py condition \
    --output "$OUT" --worker "$worker" --workers 2 > "$OUT/logs/condition_$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then touch "$OUT/NEEDS_ATTENTION"; exit 1; fi
CUDA_VISIBLE_DEVICES= "$PY" -u tools/prepare_cosmos_qc_portable_eval_20261029.py audit \
  --output "$OUT" > "$OUT/logs/input_audit.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/evaluation/portable_js_easy_h3.py infer "${args[@]}" > "$OUT/logs/infer.log" 2>&1
pids=()
for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$worker "$PY" -u scripts/evaluation/portable_js_easy_h3.py decode \
    "${args[@]}" --worker "$worker" --workers 4 > "$OUT/logs/decode_$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then touch "$OUT/NEEDS_ATTENTION"; exit 1; fi
CUDA_VISIBLE_DEVICES= "$PY" tools/visualize_h3_controls.py --roots "$OUT" --workers 4 \
  > "$OUT/logs/visualization.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluation/score_portable_js_easy_h3.py --output "$OUT" \
  > "$OUT/logs/metrics.log" 2>&1
printf '%s\n' 'Evaluation complete; training remains paused. No automatic training restart.'
