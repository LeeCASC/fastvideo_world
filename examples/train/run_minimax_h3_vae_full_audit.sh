#!/usr/bin/env bash
# CPU-only complete cache audit. Runs independently of the small-set trainer.
set -euo pipefail
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
DATA=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
VAE=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
test -f "$VAE/VAE_LATENT_COMPLETE" || { echo "VAE encoding is incomplete" >&2; exit 1; }
exec 9>"$PLAN/vae_full_audit.lock"
flock -n 9 || { echo "Full VAE audit is already running" >&2; exit 1; }
if [[ -f "$PLAN/VAE_FULL_AUDIT_COMPLETE" ]]; then
    echo "Full VAE audit already passed"
    exit 0
fi
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
nice -n 19 ionice -c3 "$REPO/.venv/bin/python" -u examples/train/verify_minimax_h3_vae_cache.py \
    --input-dir "$VAE/intermediate/all" \
    --manifest "$DATA/manifests/approved_uncached.jsonl" \
    --expected-rows 65389 \
    --report "$PLAN/vae_full_audit.json" \
    > "$PLAN/vae_full_audit.log" 2>&1
date --iso-8601=seconds > "$PLAN/VAE_FULL_AUDIT_COMPLETE"
