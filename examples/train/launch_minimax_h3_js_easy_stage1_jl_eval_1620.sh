#!/usr/bin/env bash
# At/after 2026-09-27 16:20 CST, use eight GPUs for distinct one-action first frames.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
CHECKPOINT="$PLAN/runs/stage1_jl/checkpoints/checkpoint-200"
OUT="$PLAN/runs/stage1_jl/single_action_eval_20260927"
TRAIN_SESSION=fastvideowm_js_easy_stage1_jl_intrinsics_fixed_20260927
TARGET_EPOCH=$(TZ=Asia/Shanghai date -d '2026-09-27 16:20:00' +%s)
mkdir -p "$OUT/logs"
exec >> "$OUT/scheduler.log" 2>&1
exec 9>"$OUT/scheduler.lock"
flock -n 9 || { echo 'Evaluation scheduler is already active' >&2; exit 1; }
echo "$(date --iso-8601=seconds) scheduled J/L inference for 16:20 CST after checkpoint-200"

while (( $(date +%s) < TARGET_EPOCH )); do sleep 30; done
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do sleep 30; done
test -s "$CHECKPOINT/dcp/.metadata" || {
    echo "No complete checkpoint-200 at $CHECKPOINT; refusing inference" >&2
    exit 1
}
while true; do
    busy=0
    for gpu in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        if (( used >= 4096 )); then busy=1; fi
    done
    if (( busy == 0 )); then break; fi
    sleep 30
done
echo "$(date --iso-8601=seconds) all eight GPUs free; starting inference"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
for scope in train heldout; do
    if [[ "$scope" == train ]]; then
        manifest=examples/train/js_easy_stage1_jl_train_cases.json
    else
        manifest=examples/train/js_easy_stage1_jl_heldout_cases.json
    fi
    if [[ -f "$OUT/${scope^^}_COMPLETE" ]]; then continue; fi
    date --iso-8601=seconds > "$OUT/${scope^^}_STARTED"
    echo "$(date --iso-8601=seconds) start $scope four-case inference"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$REPO/.venv/bin/torchrun" \
        --nnodes=1 --nproc_per_node=8 --master_port=29927 \
        scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
        --config examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml \
        --checkpoint "$CHECKPOINT" --case-manifest "$manifest" \
        --output-dir "$OUT/$scope" --num-inference-steps 50 \
        --trajectory-scale 1.0 --freeze-intrinsics-first-frame \
        > "$OUT/logs/${scope}.log" 2>&1
    "$REPO/.venv/bin/python" - "$manifest" "$OUT/$scope/checkpoint-200" <<'PY'
import json
import pathlib
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
cases = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
videos = []
for case in cases:
    video = root / case["case_id"] / (case["trajectory"] + ".mp4")
    metadata = root / case["case_id"] / "metadata.json"
    if not video.is_file() or video.stat().st_size == 0 or not metadata.is_file():
        raise SystemExit(f"Missing or empty case output: {video}")
    details = json.loads(metadata.read_text(encoding="utf-8"))
    if (details.get("case_id") != case["case_id"]
            or details.get("trajectories") != [case["trajectory"]]
            or details.get("freeze_intrinsics_first_frame") is not True):
        raise SystemExit(f"Wrong first-frame/action metadata: {metadata}")
    with imageio.get_reader(video, format="ffmpeg") as reader:
        if reader.count_frames() != 124:
            raise SystemExit(f"Expected 124 frames: {video}")
    videos.append(video)
readers = [imageio.get_reader(video, format="ffmpeg") for video in videos]
try:
    with imageio.get_writer(root.parent / "contact_sheet.mp4", fps=24, format="ffmpeg", macro_block_size=None) as writer:
        for frame_index in range(124):
            canvas = Image.new("RGB", (832, 480))
            for case_index, (case, reader) in enumerate(zip(cases, readers)):
                tile = Image.fromarray(reader.get_data(frame_index)).resize((416, 240))
                x, y = (case_index % 2) * 416, (case_index // 2) * 240
                canvas.paste(tile, (x, y))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, y, x + 250, y + 20), fill="black")
                draw.text((x + 4, y + 3), f"{case['case_id']} : {case['trajectory']}", fill="white")
            writer.append_data(np.asarray(canvas))
finally:
    for reader in readers:
        reader.close()
print(f"Verified {len(cases)} distinct first-frame/action videos under {root}")
PY
    "$REPO/.venv/bin/python" scripts/evaluation/analyze_minimax_h3_jl_single_action.py \
        --case-manifest "$manifest" --output-root "$OUT/$scope/checkpoint-200" \
        > "$OUT/logs/${scope}_motion_proxy.log" 2>&1
    date --iso-8601=seconds > "$OUT/${scope^^}_COMPLETE"
    echo "$(date --iso-8601=seconds) complete $scope four-case inference"
done
date --iso-8601=seconds > "$OUT/INFERENCE_COMPLETE"
echo "$(date --iso-8601=seconds) all eight J/L videos and diagnostics complete"
