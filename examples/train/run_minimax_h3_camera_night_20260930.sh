#!/usr/bin/env bash
# One independent four-GPU branch; launched twice in separate tmux sessions.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_pair_night_20260930
BASELINE=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
BRANCH="${1:?Usage: $0 prope|hybrid}"
case "$BRANCH" in
    prope) export CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29981 ;;
    hybrid) export CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29982 ;;
    *) echo "Unknown branch: $BRANCH" >&2; exit 2 ;;
esac
export NUM_GPUS=4 WANDB_MODE=offline
export CONFIG_FILE="$ROOT/configs/${BRANCH}_high_noise.yaml"
export TRAIN_DATA="$ROOT/data" VALIDATION_DATA="$VALIDATION"
export LOG_DIR="$ROOT/runs/$BRANCH/high_noise/logs"

test -s "$ROOT/report.json" || { echo "Paired data not frozen" >&2; exit 1; }
test -s "$ROOT/train_indices.json" || { echo "Balanced index not frozen" >&2; exit 1; }
test -f "$ROOT/configs/${BRANCH}_mixed_noise.yaml" || { echo "Missing mixed-noise config" >&2; exit 1; }
test -f "$BASELINE/transformer/config.json" || { echo "Missing H3 baseline" >&2; exit 1; }
test ! -e "$ROOT/runs/$BRANCH/high_noise/checkpoints/checkpoint-25" || {
    echo "Existing checkpoint: refusing an accidental restart" >&2; exit 1;
}
mkdir -p "$LOG_DIR"
cd "$REPO"
start=$(date +%s)
printf '%s\n' "$(date --iso-8601=seconds)" > "$ROOT/runs/$BRANCH/high_noise/STARTED"
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh
end=$(date +%s)
test -s "$ROOT/runs/$BRANCH/high_noise/checkpoints/checkpoint-200/dcp/.metadata" || {
    echo "High-noise checkpoint-200 incomplete; stopping branch" >&2; exit 1;
}
printf '%s\n' "$(date --iso-8601=seconds)" > "$ROOT/runs/$BRANCH/high_noise/COMPLETE"

# Set second-stage length from the measured first-stage wall time. Reserve
# 15 minutes for final checkpoint/shutdown and round to checkpoint boundaries.
steps=$("$REPO/.venv/bin/python" - "$start" "$end" <<'PY'
import datetime, sys
from zoneinfo import ZoneInfo
start, end = map(int, sys.argv[1:])
target = int(datetime.datetime(2026, 10, 1, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
remaining = max(0, target - end - 900)
per_step = max(1.0, (end - start) / 200)
print(min(1200, max(25, int(remaining / per_step) // 25 * 25)))
PY
)
export CONFIG_FILE="$ROOT/configs/${BRANCH}_mixed_noise.yaml"
export LOG_DIR="$ROOT/runs/$BRANCH/mixed_noise/logs"
mkdir -p "$LOG_DIR"
printf '%s\n' "$steps" > "$ROOT/runs/$BRANCH/mixed_noise/PLANNED_STEPS"
printf '%s\n' "$(date --iso-8601=seconds)" > "$ROOT/runs/$BRANCH/mixed_noise/STARTED"
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh \
    --training.loop.max_train_steps "$steps"
test -s "$ROOT/runs/$BRANCH/mixed_noise/checkpoints/checkpoint-$steps/dcp/.metadata" || {
    echo "Mixed-noise final checkpoint incomplete" >&2; exit 1;
}
printf '%s\n' "$(date --iso-8601=seconds)" > "$ROOT/runs/$BRANCH/mixed_noise/COMPLETE"
