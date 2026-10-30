#!/usr/bin/env bash
# Four-GPU same-seed real/static/reverse/shuffled rollouts for one branch.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_pair_night_20260930
VALIDATION=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926/data/preprocessed/validation
BRANCH="${1:?Usage: $0 prope|hybrid}"
case "$BRANCH" in
    prope) export CUDA_VISIBLE_DEVICES=0,1,2,3; port=29991 ;;
    hybrid) export CUDA_VISIBLE_DEVICES=4,5,6,7; port=29992 ;;
    *) echo "Unknown branch: $BRANCH" >&2; exit 2 ;;
esac
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
OUT="$ROOT/evaluation_20261001/$BRANCH"
mkdir -p "$OUT"
cd "$REPO"

# Rows 73 (A) and 81 (J) share an identical first-frame latent and vision
# tokens in scene 00010. Heldout row 11 is scene-disjoint validation data.
for phase in high_noise mixed_noise; do
    if [[ "$phase" == high_noise ]]; then step=200; else step=500; fi
    config="$ROOT/configs/${BRANCH}_${phase}.yaml"
    checkpoint="$ROOT/runs/$BRANCH/$phase/checkpoints/checkpoint-$step"
    test -s "$checkpoint/dcp/.metadata"
    for spec in train_A:73 train_J:81 heldout:11; do
        name=${spec%%:*}
        row=${spec##*:}
        data="$ROOT/data"
        if [[ "$name" == heldout ]]; then data="$VALIDATION"; fi
        case_dir="$OUT/$phase/$name"
        video_dir="$case_dir/checkpoint-$step"
        mkdir -p "$case_dir"
        for pair in correct_static reverse_shuffled; do
            if [[ "$pair" == correct_static ]]; then
                variants=(correct static)
            else
                variants=(reverse shuffled)
            fi
            if [[ -s "$video_dir/real_${variants[0]}.mp4" && -s "$video_dir/real_${variants[1]}.mp4" ]]; then
                echo "$(date --iso-8601=seconds) skip completed $BRANCH $phase $name $pair"
                continue
            fi
            echo "$(date --iso-8601=seconds) start $BRANCH $phase $name $pair"
            "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=4 --master_port="$port" \
                scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
                --config "$config" --checkpoint "$checkpoint" \
                --data-path "$data" --row-index "$row" --output-dir "$case_dir" \
                --seed 20261001 --num-inference-steps 50 \
                --real-trajectory-variants "${variants[@]}" \
                > "$case_dir/${pair}.log" 2>&1
            for variant in "${variants[@]}"; do
                test -s "$video_dir/real_${variant}.mp4"
            done
            echo "$(date --iso-8601=seconds) complete $BRANCH $phase $name $pair"
        done
        test -s "$video_dir/ground_truth.mp4"
        set +e
        "$REPO/.venv/bin/python" scripts/evaluation/analyze_minimax_h3_camera_control.py \
            "$video_dir" > "$case_dir/analysis.log" 2>&1
        result=$?
        set -e
        if [[ "$result" -ne 0 && "$result" -ne 2 ]]; then
            echo "Analysis failed for $BRANCH/$phase/$name ($result)" >&2
            exit "$result"
        fi
        test -s "$video_dir/analysis_metrics.json"
        echo "$(date --iso-8601=seconds) analyzed $BRANCH $phase $name gate=$result"
    done
done
date --iso-8601=seconds > "$OUT/EVALUATION_COMPLETE"
