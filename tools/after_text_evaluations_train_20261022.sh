#!/usr/bin/env bash
# Dependency chain, not a fixed-time guess. Never overlap active GPU jobs.
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
H3=/home/lizihao/projects/worldmodel/H3-World
PY=$REPO/.venv/bin/python
H3PY=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022
FAST=/datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022
H3OUT=/datacc05/lizihao/outputs/h3_world/step2000_subject_text_on_off_portable_20261022
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/chain.lock"
flock -n 9 || { echo "Another dependency chain is already running" >&2; exit 1; }
trap 'touch "$ROOT/NEEDS_ATTENTION"' ERR
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
phase() { date -Is; echo "PHASE=$1"; }
wait_idle() {
  until "$PY" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"; do
    echo "GPUs still occupied; wait five minutes; never kill another job"
    sleep 300
  done
}
phase waiting_for_fastvideo_text_evaluation
while [[ ! -f "$FAST/INFERENCE_COMPLETE" ]]; do
  if [[ -f "$FAST/NEEDS_ATTENTION" ]]; then echo "FastVideo evaluation needs attention" >&2; exit 1; fi
  sleep 300
done
wait_idle
phase h3_world_step2000_text_on_off_evaluation
cd "$H3"
if [[ ! -f "$H3OUT/INFERENCE_COMPLETE" ]]; then
  # This limit is verified before queueing. It protects the host from native
  # DiffSynth's CPU offload; it is not a licence to consume all physical RAM.
  systemd-run --user --scope --quiet -p MemoryMax=600G -p MemorySwapMax=0 \
    "$H3PY" -u code/vwm/run_portable_subject_text.py run --output "$H3OUT" --workers 4 \
    --compile-cache /datacc05/lizihao/outputs/h3_world/step2000_subject_text_portable_20261022/compile \
    > "$ROOT/logs/h3_subject_text_on_off.log" 2>&1
fi
[[ -f "$H3OUT/INFERENCE_COMPLETE" ]] || { echo "H3 evaluation incomplete" >&2; exit 1; }
wait_idle
phase encoding_verified_scene_text
cd "$REPO"
pids=()
for worker in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$worker "$PY" -u tools/prepare_text_training_20261022.py encode \
    --root "$ROOT" --worker "$worker" --workers 8 > "$ROOT/logs/text_encoder_gpu$worker.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "Text encoding failed; refusing training" >&2; exit 1; }
phase auditing_new_training_inputs
"$PY" tools/prepare_text_training_20261022.py audit --root "$ROOT" > "$ROOT/logs/text_data_audit.log" 2>&1
wait_idle
phase starting_eight_gpu_text_and_camera_training
exec bash tools/launch_scene_text_training_20261022.sh
