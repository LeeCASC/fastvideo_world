#!/usr/bin/env bash
# Build-independent waiter for the corrected 124/243-frame CPU quality pass.
# It never touches source videos or pose archives and does not use GPUs.
set -euo pipefail

ROOT=/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
WORKERS="${CPU_WORKERS:-32}"

for spec in 124f:124 243f:243; do
  name="${spec%%:*}"
  frames="${spec##*:}"
  dir="$ROOT/$name"
  candidate="$dir/registry/windows_${frames}_candidates.jsonl"
  summary="$dir/reports/registry_summary.json"
  while [[ ! -s "$candidate" || ! -s "$summary" ]]; do
    sleep 60
  done
  report="$dir/reports/cpu_quality.jsonl"
  marker="$dir/CPU_QUALITY_COMPLETE"
  if [[ ! -s "$marker" ]]; then
    "$REPO/.venv/bin/python" "$REPO/tools/filter_vwm_61f_cpu.py" \
      --input "$candidate" --output "$report" --workers "$WORKERS"
    printf 'completed_at=%s\nframes=%s\ncandidate_rows=%s\n' \
      "$(date '+%F %T %Z')" "$frames" "$(wc -l < "$candidate")" > "$marker"
  fi
done

date '+%F %T %Z CPU_QUALITY_124_243_COMPLETE'
