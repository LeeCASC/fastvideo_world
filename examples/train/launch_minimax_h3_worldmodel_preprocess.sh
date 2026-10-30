#!/usr/bin/env bash
# Run one resumable H3 world-model preprocessing phase on local GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3}"
MANIFEST_PATH="${MANIFEST_PATH:-/datacc05/datasets/vwm_datasets/train_v2/manifests/train.jsonl}"
LEGACY_METADATA_PATH="${LEGACY_METADATA_PATH:-}"
LEGACY_CACHE_DIR="${LEGACY_CACHE_DIR:-}"
WORK_ROOT="${WORK_ROOT:-/datacc05/lizihao_wm/worldmodel}"
ADDITIVE_ROOT="${ADDITIVE_ROOT:-/datacc05/lizihao_wm/worldmodel}"
CONDITIONING="${CONDITIONING:-additive}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-16}"
MIN_SOURCE_DURATION_SECONDS="${MIN_SOURCE_DURATION_SECONDS:-4.5}"
PHASE="${1:?Usage: $0 <vae|condition|augment|legacy-cache|verify> [train|validation|all] [max_samples]}"
SPLIT="${2:-train}"
MAX_SAMPLES="${3:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing environment: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${MANIFEST_PATH}" ]] || { echo "Missing VWM manifest: ${MANIFEST_PATH}" >&2; exit 1; }
if [[ "${PHASE}" == "vae" || "${PHASE}" == "condition" ]]; then
    [[ -f "${MODEL_PATH}/model_index.json" ]] || {
        echo "Missing H3 checkpoint: ${MODEL_PATH}" >&2; exit 1;
    }
fi
if [[ "${PHASE}" == "legacy-cache" ]]; then
    [[ -f "${LEGACY_METADATA_PATH}" ]] || {
        echo "Missing legacy H3 metadata: ${LEGACY_METADATA_PATH}" >&2; exit 1;
    }
    [[ -d "${LEGACY_CACHE_DIR}" ]] || {
        echo "Missing legacy H3 cache: ${LEGACY_CACHE_DIR}" >&2; exit 1;
    }
fi
[[ "${SPLIT}" == "train" || "${SPLIT}" == "validation" || "${SPLIT}" == "all" ]] || {
    echo "Invalid split: ${SPLIT}" >&2; exit 1;
}
(( NUM_WORKERS >= 1 && NUM_WORKERS <= 32 )) || { echo "NUM_WORKERS must be in [1,32]" >&2; exit 1; }
if [[ "${PHASE}" != "legacy-cache" ]]; then
    (( NUM_WORKERS <= 8 )) || { echo "GPU preprocessing supports at most 8 workers" >&2; exit 1; }
fi
[[ "${CONDITIONING}" == "additive" || "${CONDITIONING}" == "prope" ]] || {
    echo "CONDITIONING must be additive or prope" >&2; exit 1;
}

GPU_ID_LIST=()
if [[ -n "${GPU_IDS}" ]]; then
    IFS="," read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
    (( ${#GPU_ID_LIST[@]} == NUM_WORKERS )) || {
        echo "GPU_IDS must contain exactly NUM_WORKERS comma-separated devices" >&2; exit 1;
    }
elif [[ "${PHASE}" != "legacy-cache" ]]; then
    for ((worker = 0; worker < NUM_WORKERS; worker++)); do
        GPU_ID_LIST+=("${worker}")
    done
fi
for gpu_id in "${GPU_ID_LIST[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: ${gpu_id}" >&2; exit 1; }
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CPATH="${REPO_ROOT}/.venv/sysroot/usr/include:${REPO_ROOT}/.venv/sysroot/usr/include/python3.12${CPATH:+:${CPATH}}"
export TOKENIZERS_PARALLELISM=false

INTERMEDIATE_DIR="${WORK_ROOT}/intermediate/${SPLIT}"
OUTPUT_DIR="${WORK_ROOT}/preprocessed/${SPLIT}"
LOG_DIR="${WORK_ROOT}/logs/preprocess_${PHASE}_${SPLIT}"
mkdir -p "${LOG_DIR}"

if [[ "${PHASE}" == "verify" ]]; then
    expected_args=()
    [[ -n "${MAX_SAMPLES}" ]] && expected_args+=(--expected-rows "${MAX_SAMPLES}")
    "${PYTHON_BIN}" -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess verify \
        --input-dir "${OUTPUT_DIR}" --conditioning "${CONDITIONING}" "${expected_args[@]}"
    exit 0
fi

[[ "${PHASE}" == "vae" || "${PHASE}" == "condition" || "${PHASE}" == "augment" || "${PHASE}" == "legacy-cache" ]] || {
    echo "Invalid phase: ${PHASE}" >&2; exit 1;
}

pids=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    log_file="${LOG_DIR}/worker_${worker}.log"
    if [[ "${PHASE}" == "vae" ]]; then
        gpu_id="${GPU_ID_LIST[worker]}"
        max_args=()
        [[ -n "${MAX_SAMPLES}" ]] && max_args+=(--max-samples "${MAX_SAMPLES}")
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" \
            -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess vae \
            --manifest "${MANIFEST_PATH}" \
            --model-path "${MODEL_PATH}" \
            --output-dir "${INTERMEDIATE_DIR}" \
            --split "${SPLIT}" \
            --worker-index "${worker}" \
            --num-workers "${NUM_WORKERS}" \
            --device-index 0 \
            --master-port "$((29600 + worker))" \
            --samples-per-shard "${SAMPLES_PER_SHARD}" \
            --conditioning "${CONDITIONING}" \
            --min-source-duration-seconds "${MIN_SOURCE_DURATION_SECONDS}" \
            "${max_args[@]}" >"${log_file}" 2>&1 &
    elif [[ "${PHASE}" == "condition" ]]; then
        gpu_id="${GPU_ID_LIST[worker]}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" \
            -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess condition \
            --input-dir "${INTERMEDIATE_DIR}" \
            --model-path "${MODEL_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --worker-index "${worker}" \
            --num-workers "${NUM_WORKERS}" \
            --device-index 0 \
            --master-port "$((29700 + worker))" \
            --conditioning "${CONDITIONING}" >"${log_file}" 2>&1 &
    elif [[ "${PHASE}" == "augment" ]]; then
        "${PYTHON_BIN}" -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess augment \
            --input-dir "${ADDITIVE_ROOT}/preprocessed/${SPLIT}" \
            --manifest "${MANIFEST_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --worker-index "${worker}" \
            --num-workers "${NUM_WORKERS}" >"${log_file}" 2>&1 &
    else
        max_args=()
        [[ -n "${MAX_SAMPLES}" ]] && max_args+=(--max-samples "${MAX_SAMPLES}")
        "${PYTHON_BIN}" -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess legacy-cache \
            --metadata "${LEGACY_METADATA_PATH}" \
            --cache-dir "${LEGACY_CACHE_DIR}" \
            --vwm-manifest "${MANIFEST_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --worker-index "${worker}" \
            --num-workers "${NUM_WORKERS}" \
            --samples-per-shard "${SAMPLES_PER_SHARD}" \
            "${max_args[@]}" >"${log_file}" 2>&1 &
    fi
    pids+=("$!")
    echo "started ${PHASE} worker ${worker}; log=${log_file}"
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
(( failed == 0 )) || { echo "One or more preprocessing workers failed; inspect ${LOG_DIR}" >&2; exit 1; }
echo "Completed ${PHASE} preprocessing for ${SPLIT}."
