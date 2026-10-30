#!/usr/bin/env bash
# Build and preprocess the requested train_v2 subset on GPUs 4-7.
# This script deliberately stops after data verification; it never starts training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
SOURCE_DATA_ROOT="/datacc05/datasets/vwm_datasets/train_v2"
SOURCE_MANIFEST="${SOURCE_DATA_ROOT}/manifests/train.jsonl"
SOURCE_RECORDS="${SOURCE_DATA_ROOT}/records"
DATA_ROOT="/datacc05/lizihao_wm/worldmodel_prope/datasets/train_v2_34380_v5"
SUBSET_MANIFEST="${DATA_ROOT}/manifests/train.jsonl"
EXPECTED_MANIFEST_SHA256="b57a081db89e9dbdc7774c7c9b7b6954161a8f665d67e23d0e8934fe120a407d"
PREPROCESS_GPU_IDS="${GPU_IDS:-4,5,6,7}"
PREPROCESS_NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${DATA_ROOT}"
if [[ -L "${DATA_ROOT}/records" ]]; then
    if [[ "$(readlink -f "${DATA_ROOT}/records")" != "$(readlink -f "${SOURCE_RECORDS}")" ]]; then
        echo "Existing records symlink points to the wrong dataset: ${DATA_ROOT}/records" >&2
        exit 1
    fi
elif [[ -e "${DATA_ROOT}/records" ]]; then
    echo "Refusing to replace a non-symlink records path: ${DATA_ROOT}/records" >&2
    exit 1
else
    ln -s "${SOURCE_RECORDS}" "${DATA_ROOT}/records"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_vwm_source_subset.py" \
    --manifest "${SOURCE_MANIFEST}" \
    --output "${SUBSET_MANIFEST}" \
    --source-limit multicam_video=10000 \
    --source-limit syncam_video=5000 \
    --seed 20260909
printf '%s  %s\n' "${EXPECTED_MANIFEST_SHA256}" "${SUBSET_MANIFEST}" | sha256sum --check --status

GPU_IDS="${PREPROCESS_GPU_IDS}" \
NUM_WORKERS="${PREPROCESS_NUM_WORKERS}" \
CONDITIONING=prope \
MIN_SOURCE_DURATION_SECONDS=0 \
MANIFEST_PATH="${SUBSET_MANIFEST}" \
WORK_ROOT="${DATA_ROOT}" \
    bash "${SCRIPT_DIR}/launch_minimax_h3_worldmodel_preprocess.sh" vae all 34380

GPU_IDS="${PREPROCESS_GPU_IDS}" \
NUM_WORKERS="${PREPROCESS_NUM_WORKERS}" \
CONDITIONING=prope \
MANIFEST_PATH="${SUBSET_MANIFEST}" \
WORK_ROOT="${DATA_ROOT}" \
    bash "${SCRIPT_DIR}/launch_minimax_h3_worldmodel_preprocess.sh" condition all

CONDITIONING=prope \
MANIFEST_PATH="${SUBSET_MANIFEST}" \
WORK_ROOT="${DATA_ROOT}" \
    bash "${SCRIPT_DIR}/launch_minimax_h3_worldmodel_preprocess.sh" verify all 34380

echo "Prepared and verified 34380 train_v2 samples at ${DATA_ROOT}."
echo "Training was not started."
