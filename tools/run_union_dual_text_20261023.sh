#!/usr/bin/env bash
# Completion-gated preparation; never train on an incomplete caption subset.
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=${UNION_DUAL_ROOT:-/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023}
PY=$REPO/.venv/bin/python
VLM_PY=/home/lizihao/anaconda3/envs/minimax_h3/bin/python
SCRIPT=$REPO/tools/prepare_union_dual_captions_20261023.py
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/chain.lock"
flock -n 9 || { echo 'Dual-caption chain already running'; exit 1; }
on_exit() {
  local code=$?
  if ((code != 0)); then
    echo "CHAIN_FAILED exit=$code at $(TZ=Asia/Tokyo date -Is)"
    touch "$ROOT/NEEDS_ATTENTION"
  fi
}
trap on_exit EXIT
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
RUN_TAG=$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%S)
echo "CHAIN_RECOVERY_START $RUN_TAG JST"

wait_for_gpus() {
  local waited=0
  until "$PY" examples/train/check_minimax_h3_gpus_idle.py; do
    if ((waited >= 86400)); then echo 'GPU wait exceeded 24h; training blocked'; return 1; fi
    echo 'GPUs busy; no processes will be stopped; retry in 300 seconds'
    sleep 300
    waited=$((waited + 300))
  done
}

[[ -s "$ROOT/inventory_summary.json" ]] || { echo 'Inventory must finish first'; exit 1; }
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["usable"]==115988 and r["errors"]==0' "$ROOT/inventory_summary.json"
if [[ -d "$ROOT/subject_motion/runs/train/checkpoints" ]] &&
  compgen -G "$ROOT/subject_motion/runs/train/checkpoints/checkpoint-*" > /dev/null; then
  echo 'New-run checkpoints exist; use explicit full-state resume, not parent initialization'
  exit 1
fi

# Eight independent Qwen models, microbatch4 (32 windows / 192 frames globally).
# Failures are retried by exact window ID; no generic fallback is permitted.
if "$PY" "$SCRIPT" verify_captions --root "$ROOT" > "$ROOT/logs/caption_gate_${RUN_TAG}.log" 2>&1; then
  echo 'Frozen caption gate verified; skip raw-caption rescans and GPU caption loading'
else
[[ ${REQUIRE_VERIFIED_CAPTIONS:-0} == 0 ]] || { echo 'Frozen caption gate failed; VLM retry is disabled'; exit 1; }
for pass in 1 2 3; do
  "$PY" "$SCRIPT" plan --root "$ROOT" --workers 8 > "$ROOT/logs/caption_plan_${RUN_TAG}_${pass}.log" 2>&1
  if "$PY" "$SCRIPT" finalize --root "$ROOT" > "$ROOT/logs/caption_finalize_${RUN_TAG}_${pass}_before.log" 2>&1; then
    echo 'All exact-window captions already pass; skipping caption GPU loading'
    break
  fi
  wait_for_gpus
  pids=()
  for worker in {0..7}; do
    CUDA_VISIBLE_DEVICES="$worker" "$VLM_PY" "$SCRIPT" caption --root "$ROOT" \
      --worker "$worker" --workers 8 --batch-size 4 --attempt "$pass" \
      > "$ROOT/logs/caption_${worker}_${RUN_TAG}_pass_${pass}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then echo 'A caption worker failed; inspect logs before resuming'; exit 1; fi
  if "$PY" "$SCRIPT" finalize --root "$ROOT" > "$ROOT/logs/caption_finalize_${RUN_TAG}_${pass}.log" 2>&1; then
    break
  fi
  echo "Caption pass $pass left invalid windows; retrying only unsuccessful IDs"
done
fi
[[ -s "$ROOT/CAPTIONS_VERIFIED" ]] || { echo 'Dual captions incomplete; training blocked'; exit 1; }
echo "CAPTION_STAGE_COMPLETE $(TZ=Asia/Tokyo date -Is)"

wait_for_gpus
echo "ENCODING_START $(TZ=Asia/Tokyo date -Is)"
pids=()
for worker in {0..7}; do
  CUDA_VISIBLE_DEVICES="$worker" "$PY" "$SCRIPT" encode --root "$ROOT" \
    --worker "$worker" --workers 8 > "$ROOT/logs/encode_${worker}_${RUN_TAG}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'A text encoder failed; training blocked'; exit 1; }
echo "FULL_AUDIT_START $(TZ=Asia/Tokyo date -Is)"
"$PY" "$SCRIPT" audit --root "$ROOT" > "$ROOT/logs/persisted_tensor_audit_${RUN_TAG}.log" 2>&1
[[ -f "$ROOT/subject_motion/TEXT_DATA_VERIFIED" ]] || exit 1

# New dataset needs a fresh loader/optimizer; retain the latest text-trained
# model weights (100), not the action-only parent800. This is a new run counter.
"$PY" tools/configure_union_dual_training_20261023.py --root "$ROOT"
wait_for_gpus
export CONFIG_FILE=$ROOT/subject_motion/training_config.yaml
export TRAIN_DATA=$ROOT/subject_motion/preprocessed/train
export VALIDATION_DATA=$ROOT/subject_motion/preprocessed/validation
export LOG_DIR=$ROOT/subject_motion/runs/train/logs
export NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=30630 WANDB_MODE=offline
PARENT=/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/runs/train/checkpoints/checkpoint-100
[[ -s "$PARENT/dcp/.metadata" ]] || { echo 'Missing preserved text checkpoint100'; exit 1; }
echo 'All data gates passed; subject-motion text training, eight GPUs/SP2, total BS32, new 3000-step run'
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
  --models.student.init_from_training_checkpoint "$PARENT" \
  --training.checkpoint.resume_from_checkpoint ''
touch "$ROOT/TRAINING_COMPLETE"
