#!/usr/bin/env bash
# Launch local A100x8 MiniMax-H3 Stage-0 world-model training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/train/configs/minimax_h3_worldmodel_stage0.yaml}"
MODEL_PATH="${MODEL_PATH:-/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3}"
TRAIN_DATA="${TRAIN_DATA:-/datacc05/lizihao_wm/worldmodel/preprocessed/train}"
VALIDATION_DATA="${VALIDATION_DATA:-/datacc05/lizihao_wm/worldmodel/preprocessed/validation}"
LOG_DIR="${LOG_DIR:-/datacc05/lizihao_wm/worldmodel/logs}"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || { echo "Missing environment under ${REPO_ROOT}/.venv" >&2; exit 1; }
[[ -f "${MODEL_PATH}/transformer/config.json" ]] || { echo "Incomplete H3 transformer checkpoint: ${MODEL_PATH}" >&2; exit 1; }
find "${TRAIN_DATA}" -name '*.parquet' -print -quit | grep -q . || { echo "No train parquet under ${TRAIN_DATA}" >&2; exit 1; }
find "${VALIDATION_DATA}" -name '*.parquet' -print -quit | grep -q . || { echo "No validation parquet under ${VALIDATION_DATA}" >&2; exit 1; }

mkdir -p "${LOG_DIR}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CPATH="${REPO_ROOT}/.venv/sysroot/usr/include:${REPO_ROOT}/.venv/sysroot/usr/include/python3.12${CPATH:+:${CPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NNODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29501}"
export LOG_DIR

cd "${REPO_ROOT}"
bash examples/train/run.sh "${CONFIG_FILE}" \
    --models.student.init_from "${MODEL_PATH}" \
    --training.data.data_path "${TRAIN_DATA}" \
    --callbacks.worldmodel_validation.data_path "${VALIDATION_DATA}" \
    "$@"
