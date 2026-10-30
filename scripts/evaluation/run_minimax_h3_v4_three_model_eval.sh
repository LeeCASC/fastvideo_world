#!/usr/bin/env bash
# Evaluate the three V4 exact-PRoPE overfit checkpoints under identical controls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
DATA_PATH="${DATA_PATH:-/datacc05/lizihao_wm/worldmodel_prope/overfit/camera_action_diagnostic_v4_4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/datacc05/lizihao_wm/worldmodel_prope/evaluation/action_overfit_v4_three_model_row0}"
ROW_INDEX="${ROW_INDEX:-0}"
SEED="${SEED:-20260909}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
[[ ${#gpu_array[@]} -eq 2 || ${#gpu_array[@]} -eq 4 || ${#gpu_array[@]} -eq 8 ]] || {
  echo "This evaluator requires 2, 4, or 8 GPUs, got GPU_IDS=${GPU_IDS}" >&2
  exit 1
}
num_sp_groups=$((${#gpu_array[@]} / 2))
[[ -x "${PYTHON}" && -x "${TORCHRUN}" ]] || {
  echo "Missing FastVideoWM virtual environment under ${REPO_ROOT}/.venv" >&2
  exit 1
}

labels=(
  model_a_attention_lora
  model_b_boundary
  model_c_last_two_blocks
)
configs=(
  examples/train/configs/minimax_h3_worldmodel_action_overfit_v4_lora_gpu2_7.yaml
  examples/train/configs/minimax_h3_worldmodel_action_overfit_v4_boundary_gpu4_7_resume100.yaml
  examples/train/configs/minimax_h3_worldmodel_action_overfit_v4_deep_gpu4_7.yaml
)
checkpoints=(
  /datacc05/lizihao_wm/worldmodel_prope/runs/minimax_h3_action_overfit_v4_lora_gpu2_7/checkpoints/checkpoint-300
  /datacc05/lizihao_wm/worldmodel_prope/runs/minimax_h3_action_overfit_v4_boundary_gpu2_7/checkpoints/checkpoint-300
  /datacc05/lizihao_wm/worldmodel_prope/runs/minimax_h3_action_overfit_v4_deep_gpu4_7/checkpoints/checkpoint-300
)
variants=(correct static reverse shuffled)

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for index in "${!labels[@]}"; do
  label="${labels[index]}"
  config="${configs[index]}"
  checkpoint="${checkpoints[index]}"
  model_output="${OUTPUT_ROOT}/${label}"
  log_path="${OUTPUT_ROOT}/${label}.log"
  mkdir -p "${model_output}"

  for ((offset = 0; offset < ${#variants[@]}; offset += num_sp_groups)); do
    variant_batch=("${variants[@]:offset:num_sp_groups}")
    echo "[$(date --iso-8601=seconds)] ${label}: ${variant_batch[*]}" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TORCHRUN}" \
      --standalone \
      --nproc_per_node="${#gpu_array[@]}" \
      scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --data-path "${DATA_PATH}" \
      --output-dir "${model_output}" \
      --row-index "${ROW_INDEX}" \
      --seed "${SEED}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --real-trajectory-variants "${variant_batch[@]}" \
      >> "${log_path}" 2>&1
  done

  result_dir="${model_output}/checkpoint-300"
  "${PYTHON}" scripts/evaluation/analyze_minimax_h3_camera_control.py "${result_dir}" \
    > "${model_output}/analysis_stdout.json" 2>&1 || [[ $? -eq 2 ]]
done

echo "[$(date --iso-8601=seconds)] Three-model evaluation complete: ${OUTPUT_ROOT}"
