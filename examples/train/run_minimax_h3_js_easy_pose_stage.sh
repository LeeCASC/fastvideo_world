#!/usr/bin/env bash
# Start one frozen measured-pose curriculum stage; later stages require an
# explicit evaluation gate and initialize weights without restoring optimizer.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
EARLY=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
CONFIG="$REPO/examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml"
STAGE="${1:?Usage: $0 stage1_jl|stage2_jlik|stage3_all8}"
case "$STAGE" in
    stage1_jl) STEPS=200; PREVIOUS="" ; EXPECTED=20 ;;
    stage2_jlik) STEPS=200; PREVIOUS="$PLAN/runs/stage1_jl/checkpoints/checkpoint-200" ; EXPECTED=40 ;;
    stage3_all8) STEPS=400; PREVIOUS="$PLAN/runs/stage2_jlik/checkpoints/checkpoint-200" ; EXPECTED=80 ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

test -f "$PLAN/VAE_STAGE80_AUDIT_COMPLETE" || { echo "Frozen 80-row VAE audit has not passed" >&2; exit 1; }
if [[ "$STAGE" != stage1_jl ]]; then
    test -f "$PLAN/VAE_FULL_AUDIT_COMPLETE" || { echo "Full VAE audit has not passed for expansion" >&2; exit 1; }
    test -f "$PLAN/VAE_INTRINSICS_AUDIT_COMPLETE" || { echo "Full VAE intrinsics audit has not passed for expansion" >&2; exit 1; }
fi
test -f "$PLAN/CONDITIONED_80_VERIFIED" || { echo "Conditioned 80-row dataset has not passed verification" >&2; exit 1; }
test -s "$PLAN/${STAGE}_indices.json" || { echo "Frozen sampling index missing" >&2; exit 1; }
if [[ -n "$PREVIOUS" ]]; then
    test -s "$PREVIOUS/dcp/.metadata" || { echo "Previous-stage checkpoint is incomplete: $PREVIOUS" >&2; exit 1; }
    test -f "$PLAN/${STAGE}_EVAL_APPROVED" || { echo "Previous-stage action evaluation has not approved $STAGE" >&2; exit 1; }
fi

PYTHON="$REPO/.venv/bin/python"
"$PYTHON" - "$PLAN/${STAGE}_indices.json" "$EXPECTED" <<'PY'
import json, sys
indices = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
if len(indices) != expected or len(set(indices)) != expected or any(type(i) is not int or i < 0 for i in indices):
    raise SystemExit(f"Frozen indices must contain {expected} distinct nonnegative parquet rows")
PY

OUTPUT="$PLAN/runs/$STAGE/checkpoints"
if [[ -e "$OUTPUT/checkpoint-50" || -e "$OUTPUT/checkpoint-$STEPS" ]]; then
    echo "Existing $STAGE checkpoint: refuse accidental restart/overwrite" >&2
    exit 1
fi
mkdir -p "$PLAN/runs/$STAGE/logs"
cd "$REPO"
export CONFIG_FILE="$CONFIG"
export TRAIN_DATA="$PLAN/preprocessed/train"
export VALIDATION_DATA="$EARLY/data/preprocessed/validation"
export LOG_DIR="$PLAN/runs/$STAGE/logs"
export WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29921
extra=(
    --models.student.sampling_index_path "$PLAN/${STAGE}_indices.json"
    --training.loop.max_train_steps "$STEPS"
    --training.checkpoint.output_dir "$OUTPUT"
    --training.tracker.run_name "minimax_h3_js_easy_pose_${STAGE}"
)
if [[ -n "$PREVIOUS" ]]; then
    extra+=(--models.student.init_from_training_checkpoint "$PREVIOUS")
fi
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh "${extra[@]}"
