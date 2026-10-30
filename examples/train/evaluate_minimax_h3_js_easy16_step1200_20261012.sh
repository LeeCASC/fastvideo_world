#!/usr/bin/env bash
# Exact 16-case GT-path and paired single-action inference after step 1200.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
BENCHMARK=/home/lizihao/projects/worldmodel/test_data/js_easy_benchmark
CACHE=/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all
PREVIOUS_FIRST_FRAMES=/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm/conditioning
MODEL=/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008
CHECKPOINT="${CHECKPOINT:-/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train/checkpoints/checkpoint-1200}"
CONFIG="$REPO/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml"
OUT="${OUT:-/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012}"
CHECKPOINT_LABEL="$(basename "$CHECKPOINT")"

test -s "$CHECKPOINT/dcp/.metadata"
test -d "$BENCHMARK" && test -d "$CACHE"
test -s "$PLAN/train_ids.jsonl"
"$REPO/.venv/bin/python" "$REPO/examples/train/check_minimax_h3_gpus_idle.py"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [[ ! -f "$OUT/FIRST_FRAMES_COMPLETE" ]]; then
  # Exact benchmark PNG conditions were already encoded for an earlier DCP.
  # Copy them into this run and let the original preparer verify all hashes.
  if [[ ! -e "$OUT/first_frame" && -d "$PREVIOUS_FIRST_FRAMES" ]]; then
    cp -a "$PREVIOUS_FIRST_FRAMES" "$OUT/first_frame"
  fi
  "$REPO/.venv/bin/python" scripts/evaluation/prepare_minimax_h3_js_easy_benchmark_20261008.py \
    --benchmark-root "$BENCHMARK" --cache-root "$CACHE" --model-path "$MODEL" \
    --output-root "$OUT/first_frame" --device-index 0 --all-benchmark-images \
    > "$OUT/first_frame.log" 2>&1
  for number in $(seq 1 16); do
    case_id="js_$(printf '%05d' "$number")"
    test -s "$OUT/first_frame/$case_id/data_000000.parquet"
  done
  touch "$OUT/FIRST_FRAMES_COMPLETE"
fi

if [[ ! -f "$OUT/GT_CONDITIONS_COMPLETE" || ! -s "$OUT/simple_condition/js_00016/data_000000.parquet" ]]; then
  "$REPO/.venv/bin/python" scripts/evaluation/prepare_minimax_h3_js_easy16_gt.py \
    --benchmark-root "$BENCHMARK" --first-frame-root "$OUT/first_frame" \
    --model-path "$MODEL" --output-root "$OUT/gt_condition" \
    --simple-output-root "$OUT/simple_condition" --device-index 0 \
    > "$OUT/gt_condition.log" 2>&1
  for number in $(seq 1 16); do
    case_id="js_$(printf '%05d' "$number")"
    test -s "$OUT/gt_condition/$case_id/data_000000.parquet"
    test -s "$OUT/gt_condition/$case_id/alignment.json"
    test -s "$OUT/simple_condition/$case_id/data_000000.parquet"
  done
  touch "$OUT/GT_CONDITIONS_COMPLETE"
fi

"$REPO/.venv/bin/python" scripts/evaluation/build_minimax_h3_js_easy16_manifests.py \
  --benchmark-root "$BENCHMARK" --first-frame-root "$OUT/simple_condition" \
  --gt-root "$OUT/gt_condition" --train-ids "$PLAN/train_ids.jsonl" \
  --checkpoint "$CHECKPOINT" --output-root "$OUT" --seed 811 \
  > "$OUT/manifests.log" 2>&1

for mode in gt single; do
  for group in 0 1 2 3; do
    lane="$OUT/videos/$mode/group_$group"
    if [[ -f "$lane/COMPLETE" ]]; then
      echo "$(date --iso-8601=seconds) reusing $mode group $group"
      continue
    fi
    mkdir -p "$lane"
    extra_args=(--omit-reference-video)
    if [[ "$mode" == gt ]]; then
      extra_args+=(--evaluate-flow-loss)
    else
      extra_args+=(--freeze-intrinsics-first-frame)
    fi
    echo "$(date --iso-8601=seconds) inference $mode group $group"
    "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29984 \
      scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" \
      --case-manifest "$OUT/manifests/${mode}_group_${group}.json" \
      --output-dir "$OUT/videos/$mode" --seed 811 --num-inference-steps 50 \
      --trajectory-scale 1.0 "${extra_args[@]}" \
      > "$lane/inference.log" 2>&1
    "$REPO/.venv/bin/python" scripts/evaluation/verify_minimax_h3_js_easy16_lane.py \
      --manifest "$OUT/manifests/${mode}_group_${group}.json" \
      --video-root "$OUT/videos/$mode/$CHECKPOINT_LABEL" \
      --require-flow-loss "$([[ "$mode" == gt ]] && echo yes || echo no)" \
      > "$lane/verify.log" 2>&1
    touch "$lane/COMPLETE"
    echo "$(date --iso-8601=seconds) complete $mode group $group"
  done
done

"$REPO/.venv/bin/python" scripts/evaluation/verify_minimax_h3_js_easy16_lane.py \
  --suite-root "$OUT" --checkpoint-label "$CHECKPOINT_LABEL" > "$OUT/verify_all.log" 2>&1
touch "$OUT/INFERENCE_COMPLETE"
"$REPO/.venv/bin/python" scripts/evaluation/score_minimax_h3_js_easy16.py \
  --benchmark-root "$BENCHMARK" --output-root "$OUT" \
  --checkpoint-label "$CHECKPOINT_LABEL" --require-lpips \
  > "$OUT/metrics.log" 2>&1
touch "$OUT/CORE_METRICS_COMPLETE"
echo "$(date --iso-8601=seconds) 32 videos and core metrics complete: $OUT"
