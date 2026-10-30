#!/usr/bin/env bash
# One-step fixed-noise prediction ablation, with the identical checkpoint/data/seed.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930
CONFIG="$REPO/examples/train/configs/minimax_h3_camera_curriculum_20260930.yaml"
CHECKPOINT="$ROOT/runs/overfit200/checkpoints/checkpoint-200"
OUTPUT="$ROOT/runs/overfit200/prope_branch_ablation_20260930"

test -s "$CHECKPOINT/dcp/.metadata"
mkdir -p "$OUTPUT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for ablation in none adapter prope both; do
    case_dir="$OUTPUT/$ablation"
    video_dir="$case_dir/checkpoint-200"
    mkdir -p "$case_dir"
    if [[ ! -s "$video_dir/real_correct.pt" || ! -s "$video_dir/real_static.pt" || ! -s "$video_dir/real_reverse.pt" ]]; then
        echo "$(date --iso-8601=seconds) branch=$ablation start"
        "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc_per_node=8 --master_port=29982 \
            scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
            --config "$CONFIG" --checkpoint "$CHECKPOINT" \
            --data-path "$ROOT/data" --row-index 84 --output-dir "$case_dir" \
            --seed 20260930 --num-inference-steps 2 --skip-decode \
            --real-trajectory-variants correct static reverse shuffled \
            --camera-ablation "$ablation" > "$case_dir/inference.log" 2>&1
    fi
    for variant in correct static reverse shuffled; do
        test -s "$video_dir/real_${variant}.pt"
    done
    echo "$(date --iso-8601=seconds) branch=$ablation complete"
done

"$REPO/.venv/bin/python" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1])
result = {}
for ablation in ("none", "adapter", "prope", "both"):
    directory = root / ablation / "checkpoint-200"
    latents = {name: torch.load(directory / f"real_{name}.pt", map_location="cpu", weights_only=True).float()
               for name in ("correct", "static", "reverse", "shuffled")}
    reference = latents["correct"]
    result[ablation] = {
        "correct_rms": float(reference.square().mean().sqrt()),
        **{f"correct_vs_{name}_rms": float((reference - latents[name]).square().mean().sqrt())
           for name in ("static", "reverse", "shuffled")},
    }
(root / "branch_response.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
PY
date --iso-8601=seconds > "$OUTPUT/ABLATION_COMPLETE"
