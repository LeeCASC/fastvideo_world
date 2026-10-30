#!/usr/bin/env bash
# Safe handoff trigger: stop only after checkpoint-600 has a complete DCP.
set -euo pipefail

SESSION=${FASTVIDEOWM_SESSION:-fastvideowm_multicam5k_text_20261016_8gpu}
CKPT=${FASTVIDEOWM_CHECKPOINT:-/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261016_8gpu/runs/train/checkpoints/checkpoint-600}
MARKER=${HANDOFF_MARKER:-/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261016_8gpu/CHECKPOINT_600_TRAINING_STOPPED}
LOG=${HANDOFF_LOG:-/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261016_8gpu/checkpoint600_handoff.log}

mkdir -p "$(dirname "$MARKER")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T %Z')] waiting for complete $CKPT"
while [[ ! -s "$CKPT/dcp/.metadata" ]]; do
  sleep 30
done
echo "[$(date '+%F %T %Z')] checkpoint-600 DCP is complete"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "[$(date '+%F %T %Z')] stopped $SESSION"
else
  echo "[$(date '+%F %T %Z')] session already stopped: $SESSION"
fi
printf 'stopped_at=%s\ncheckpoint=%s\n' "$(date '+%F %T %Z')" "$CKPT" > "$MARKER"
echo "[$(date '+%F %T %Z')] handoff marker written: $MARKER"
