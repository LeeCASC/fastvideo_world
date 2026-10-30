#!/usr/bin/env bash
# Repair completed derived VAE shards on CPU without interrupting GPU training.
set -euo pipefail

REPO_ROOT=/home/lizihao/projects/military_world_model_agent/FastVideoWM
VAE_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
INPUT_DIR="$VAE_ROOT/intermediate/all"
LOG_DIR="$VAE_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if pgrep -f 'fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess vae' >/dev/null; then
    echo "VAE workers are running; refusing to rewrite their output" >&2
    exit 1
fi
printf '%s camera_rebase_started\n' "$(date --iso-8601=seconds)" > "$LOG_DIR/camera_rebase_status.log"
nice -n 19 ionice -c3 "$REPO_ROOT/.venv/bin/python" -u \
    examples/train/rebase_minimax_h3_camera_parquet.py \
    --input-dir "$INPUT_DIR" --expected-rows 34304 \
    > "$LOG_DIR/camera_rebase_20260926.log" 2>&1

nice -n 19 ionice -c3 "$REPO_ROOT/.venv/bin/python" -u \
    examples/train/rebase_minimax_h3_camera_parquet.py \
    --input-dir "$INPUT_DIR" --expected-rows 34304 \
    > "$LOG_DIR/camera_rebase_verify_20260926.log" 2>&1
grep -q '^complete: rows=34304 rebased=0$' "$LOG_DIR/camera_rebase_verify_20260926.log"
date --iso-8601=seconds > "$VAE_ROOT/CAMERA_REBASE_COMPLETE"
printf '%s camera_rebase_complete\n' "$(date --iso-8601=seconds)" >> "$LOG_DIR/camera_rebase_status.log"
