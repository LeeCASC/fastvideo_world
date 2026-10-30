#!/usr/bin/env bash
# Wait for the in-flight read-only VAE audit, then repair the 829 JS-easy K rows
# and re-audit all 65,389 VAE rows with normalized-intrinsics checks enabled.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
DATA=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
VAE=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
PYTHON="$REPO/.venv/bin/python"
exec 9>"$PLAN/vae_intrinsics_repair.lock"
flock -n 9 || { echo "Intrinsics repair is already running" >&2; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

while [[ ! -f "$PLAN/VAE_FULL_AUDIT_COMPLETE" && ! -f "$PLAN/VAE_FULL_AUDIT_COMPLETE.pre_intrinsics_fix" ]]; do
    if ! tmux has-session -t fastvideowm_vae_full_audit_20260927 2>/dev/null; then
        echo "Initial full audit exited without a completion marker" >&2
        exit 1
    fi
    sleep 300
done

# The first audit was started before the normalized-intrinsics guard existed.
# Its success marker cannot gate dataset expansion until this second pass.
if [[ -f "$PLAN/VAE_FULL_AUDIT_COMPLETE" ]]; then
    mv "$PLAN/VAE_FULL_AUDIT_COMPLETE" "$PLAN/VAE_FULL_AUDIT_COMPLETE.pre_intrinsics_fix"
fi
nice -n 10 ionice -c2 -n7 "$PYTHON" -u examples/train/repair_minimax_h3_js_easy_intrinsics.py \
    --input-dir "$VAE/intermediate/all" \
    --manifest "$DATA/manifests/approved_uncached.jsonl" \
    --expected-rows 829 \
    > "$PLAN/vae_intrinsics_repair.log" 2>&1
nice -n 19 ionice -c3 "$PYTHON" -u examples/train/verify_minimax_h3_vae_cache.py \
    --input-dir "$VAE/intermediate/all" \
    --manifest "$DATA/manifests/approved_uncached.jsonl" \
    --expected-rows 65389 \
    --report "$PLAN/vae_full_intrinsics_audit.json" \
    > "$PLAN/vae_full_intrinsics_audit.log" 2>&1
date --iso-8601=seconds > "$PLAN/VAE_INTRINSICS_AUDIT_COMPLETE"
date --iso-8601=seconds > "$PLAN/VAE_FULL_AUDIT_COMPLETE"
