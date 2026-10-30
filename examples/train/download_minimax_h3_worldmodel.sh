#!/usr/bin/env bash
# Download only the audio-free MiniMax-H3 world-model components and verify every weight shard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REVISION="${REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
CHECKSUM_FILE="${SCRIPT_DIR}/checksums/minimax_h3_worldmodel.sha256"

if ! command -v aria2c >/dev/null 2>&1; then
    LOCAL_ARIA_ROOT="${LOCAL_ARIA_ROOT:-/datacc05/lizihao_wm/tools/aria2}"
    if [[ ! -x "${LOCAL_ARIA_ROOT}/usr/bin/aria2c" ]]; then
        echo "aria2c is required; install aria2 or set LOCAL_ARIA_ROOT" >&2
        exit 1
    fi
    export PATH="${LOCAL_ARIA_ROOT}/usr/bin:${PATH}"
    export LD_LIBRARY_PATH="${LOCAL_ARIA_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${MODEL_PATH}"
HFD_SCRIPT="$(mktemp)"
trap 'rm -f "${HFD_SCRIPT}"' EXIT
curl --http1.1 -L --fail --silent --show-error \
    --retry 10 --retry-all-errors --retry-delay 2 \
    --output "${HFD_SCRIPT}" "${HF_ENDPOINT}/hfd/hfd.sh"
HF_ENDPOINT="${HF_ENDPOINT}" bash "${HFD_SCRIPT}" MiniMaxAI/MiniMax-H3 \
    --include \
        model_index.json \
        'processor/*' \
        'scheduler/*' \
        'text_encoder/*' \
        'tokenizer/*' \
        'transformer/*' \
        'vae/*' \
    --exclude \
        'FL2VA/*' \
        'Ref2VA/*' \
        'transformer_ref/*' \
        'audio_*' \
        'assets/*' \
        'scripts/*' \
    --tool aria2c \
    -x "${H3_DOWNLOAD_THREADS:-4}" \
    -j "${H3_DOWNLOAD_JOBS:-5}" \
    --local-dir "${MODEL_PATH}" \
    --revision "${REVISION}"

(
    cd "${MODEL_PATH}"
    sha256sum --check "${CHECKSUM_FILE}"
)

echo "Verified audio-free MiniMax-H3 checkpoint at ${MODEL_PATH}"
