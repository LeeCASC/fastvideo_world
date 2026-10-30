#!/usr/bin/env bash
set -euo pipefail
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PYTHON="$REPO/.venv/bin/python"
AUDIT="${INPUT_AUDIT:-$ROOT/reports/cpu_quality_61f_spatial50k.jsonl}"
REGISTRY="${INPUT_REGISTRY:-$ROOT/registry/windows_61_candidates_spatial50k.jsonl}"
while [[ ! -s "$AUDIT" ]] || pgrep -f 'filter_vwm_61f_cpu.py' >/dev/null; do
  echo "waiting for CPU quality completion" >&2
  sleep 300
done
exec "$PYTHON" "$REPO/tools/materialize_vwm_61f_cpu.py" \
  --audit "$AUDIT" --registry "$REGISTRY" --root "$ROOT"
