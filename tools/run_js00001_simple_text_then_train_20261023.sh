#!/usr/bin/env bash
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PY=$REPO/.venv/bin/python
OUT=/datacc05/lizihao/worldmodel/fastvideowm_step800_js00001_simple_text_20261023
TRAIN=/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022
PREVIOUS=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022
BUNDLE=/datacc05/lizihao/js_easy_benchmark_portable_20261021
CHECKPOINT=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022/checkpoint/checkpoint-800
cd "$REPO"
mkdir -p "$OUT/logs"
exec 9>"$OUT/job.lock"
flock -n 9 || { echo 'Two-video inference chain already active' >&2; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
wait_idle() {
  for attempt in $(seq 1 60); do
    if "$PY" examples/train/check_minimax_h3_gpus_idle.py; then return 0; fi
    sleep 10
  done
  echo 'GPUs did not release; refusing overlap' >&2
  return 1
}
run_eval() {
  wait_idle
  "$PY" tools/js00001_simple_text_20261023.py setup --output "$OUT"
  if [[ -f "$OUT/INFERENCE_COMPLETE" ]]; then
    "$PY" tools/js00001_simple_text_20261023.py finish --output "$OUT"
    return
  fi
  text_args=(--output "$OUT" --previous "$PREVIOUS" --bundle "$BUNDLE")
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluation/portable_js_easy_h3_subject_text.py condition \
    "${text_args[@]}" --worker 0 --workers 1 > "$OUT/logs/condition.log" 2>&1
  "$PY" tools/js00001_simple_text_20261023.py audit --output "$OUT" > "$OUT/logs/input_audit.log" 2>&1
  args=(--bundle "$BUNDLE" --output "$OUT" --checkpoint "$CHECKPOINT" --config "$OUT/inference_config.yaml")
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/evaluation/portable_js_easy_h3.py infer "${args[@]}" > "$OUT/logs/infer.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/evaluation/portable_js_easy_h3.py decode \
    "${args[@]}" --worker 0 --workers 1 > "$OUT/logs/decode.log" 2>&1
  "$PY" tools/js00001_simple_text_20261023.py finish --output "$OUT" > "$OUT/logs/finish.log" 2>&1
}

# A failed diagnostic must not silently leave the already-approved trainer idle.
# GPU workers are synchronous; still confirm release before any resumption.
set +e
( set -e; run_eval )
eval_status=$?
set -e
if [[ "$eval_status" -ne 0 ]]; then
  touch "$OUT/NEEDS_ATTENTION"
  echo "Two-video diagnostic failed ($eval_status); retain logs and resume independent training"
fi
wait_idle || { touch "$OUT/RESUME_NEEDS_ATTENTION"; exit 1; }
if [[ -f "$TRAIN/TRAINING_NEEDS_ATTENTION" ]]; then
  mv "$TRAIN/TRAINING_NEEDS_ATTENTION" "$OUT/logs/TRAINING_NEEDS_ATTENTION_from_user_pause"
fi
touch "$OUT/TRAINING_RESTART_REQUESTED"
echo "[$(date -Is)] restarting eight-GPU BS32 scene-text training from last complete checkpoint-800"
export RUN_INITIAL_VALIDATION=false
exec bash tools/launch_scene_text_training_20261022.sh
