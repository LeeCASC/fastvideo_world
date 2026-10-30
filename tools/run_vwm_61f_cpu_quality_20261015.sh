#!/usr/bin/env bash
set -euo pipefail
ROOT=/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015
REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PYTHON="$REPO/.venv/bin/python"
INPUT="${INPUT_REGISTRY:-$ROOT/registry/windows_61_candidates_spatial50k.jsonl}"
OUTPUT="${OUTPUT_AUDIT:-$ROOT/reports/cpu_quality_61f.jsonl}"
mkdir -p "$ROOT/reports"
while [[ ! -s "$INPUT" ]]; do
  echo "waiting for candidate registry: $INPUT" >&2
  sleep 300
done
exec "$PYTHON" "$REPO/tools/filter_vwm_61f_cpu.py" --input "$INPUT" --output "$OUTPUT" --workers "${CPU_WORKERS:-32}"
