#!/usr/bin/env bash
# Train on all scene-disjoint approved train rows from the 65,389-row VAE set.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
DATA="$ROOT/preprocessed/all"
PLAN="$ROOT/train_indices.json"
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
CONFIG="$REPO/examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml"
OUTPUT="$ROOT/runs/baseline_bs64_1000/checkpoints"
LOG_DIR="$ROOT/runs/baseline_bs64_1000/logs"

test -f "$ROOT/CONDITIONED_VERIFIED" || { echo "Full first-frame conditioning has not passed verification" >&2; exit 1; }
test -s "$PLAN" || { echo "Missing scene-disjoint training index" >&2; exit 1; }
test ! -e "$OUTPUT/checkpoint-50" || { echo "Existing checkpoint found; refuse accidental restart" >&2; exit 1; }
"$REPO/.venv/bin/python" - "$ROOT/split_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if (report["total_rows"], report["train_rows"], report["validation_rows"]) != (65389, 62820, 2569):
    raise SystemExit(f"Unexpected frozen data split: {report}")
PY

mkdir -p "$LOG_DIR"
cd "$REPO"
export CONFIG_FILE="$CONFIG" TRAIN_DATA="$DATA" VALIDATION_DATA="$VALIDATION"
export LOG_DIR WANDB_MODE=offline NUM_GPUS=8 MASTER_PORT=29931
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    --models.student.sampling_index_path "$PLAN" \
    --training.loop.gradient_accumulation_steps 16 \
    --training.loop.max_train_steps 1000 \
    --training.checkpoint.output_dir "$OUTPUT" \
    --training.checkpoint.training_state_checkpointing_steps 50 \
    --training.tracker.run_name minimax_h3_full_62820_baseline_bs64 \
    "$@"
