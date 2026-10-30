#!/usr/bin/env bash
# After the 100-step pilot: evaluate 5 held-out first frames x WASDIJKL while
# resuming the original 8-way VAE cache layout. Safe to rerun after interruption.
set -euo pipefail

REPO=/home/lizihao/projects/military_world_model_agent/FastVideoWM
EARLY=/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
VAE_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
SOURCE=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
EVAL_ROOT=/datacc05/lizihao/worldmodel/fastvideowm_action_only_step100_eval_20260926
CHECKPOINT="$EARLY/runs/pilot/checkpoints/checkpoint-100"
CONFIG="$REPO/examples/train/configs/minimax_h3_worldmodel_action_only_v1_pilot.yaml"
DATA="$EARLY/data/preprocessed/validation"
PYTHON="$REPO/.venv/bin/python"
START_AT="${START_AT:-2026-09-26 18:00:00}"
MODE="${1:-schedule}"
mkdir -p "$EVAL_ROOT/logs" "$VAE_ROOT/logs/resume_20260926"

status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$EVAL_ROOT/logs/schedule.log"
}

wait_for_gpus() {
    local gpu used
    for gpu in "$@"; do
        while true; do
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
            (( used < 4096 )) && break
            status "waiting_for_gpu_${gpu}_used_${used}MiB"
            sleep 60
        done
    done
}

run_vae_half() {
    local first="$1" last="$2" gpu failed=0 pid
    local -a pids=()
    wait_for_gpus $(seq "$first" "$last")
    cd "$REPO"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
    export TOKENIZERS_PARALLELISM=false
    for gpu in $(seq "$first" "$last"); do
        status "starting_vae_worker_${gpu}_of_8"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
            -m fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess vae \
            --manifest "$SOURCE/manifests/approved_uncached.jsonl" \
            --model-path /datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3 \
            --output-dir "$VAE_ROOT/intermediate/all" \
            --split all --worker-index "$gpu" --num-workers 8 \
            --device-index 0 --master-port "$((29820 + gpu))" \
            --samples-per-shard 16 --conditioning prope \
            --min-source-duration-seconds 4.5 \
            > "$VAE_ROOT/logs/resume_20260926/worker_${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        status "vae_workers_${first}_${last}_failed"
        return 1
    fi
    date --iso-8601=seconds > "$VAE_ROOT/VAE_WORKERS_${first}_${last}_COMPLETE"
    status "vae_workers_${first}_${last}_complete"
    if [[ -f "$VAE_ROOT/VAE_WORKERS_0_3_COMPLETE" && \
          -f "$VAE_ROOT/VAE_WORKERS_4_7_COMPLETE" ]]; then
        "$PYTHON" - "$VAE_ROOT/intermediate/all" \
            "$SOURCE/manifests/approved_uncached.jsonl" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
files = sorted(Path(sys.argv[1]).rglob("*.parquet"))
actual = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
expected = sum(1 for _ in open(sys.argv[2], encoding="utf-8"))
if actual != expected:
    raise RuntimeError(f"VAE completion count mismatch: {actual} != {expected}")
print(f"VAE latent complete: {actual} rows in {len(files)} shards", flush=True)
PY
        date --iso-8601=seconds > "$VAE_ROOT/VAE_LATENT_COMPLETE"
        status all_vae_latent_complete
    fi
}

video_complete() {
    [[ -s "$1" ]] || return 1
    "$PYTHON" - "$1" <<'PY'
import cv2
import sys
capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 124:
    sys.exit(1)
capture.set(cv2.CAP_PROP_POS_FRAMES, 123)
ok, frame = capture.read()
capture.release()
sys.exit(0 if ok and frame is not None else 1)
PY
}

run_inference() {
    local scene index pair a b out log
    local -a rows=(0 2 3 11 23)
    local -a pairs=('W S' 'A D' 'I K' 'J L')
    wait_for_gpus 0 1 2 3
    cd "$REPO"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export CPATH="$REPO/.venv/sysroot/usr/include:$REPO/.venv/sysroot/usr/include/python3.12${CPATH:+:$CPATH}"
    export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
    for scene in "${!rows[@]}"; do
        index="${rows[$scene]}"
        out="$EVAL_ROOT/scene_$(printf '%02d' "$scene")_row_$(printf '%02d' "$index")"
        mkdir -p "$out"
        for pair in "${pairs[@]}"; do
            read -r a b <<< "$pair"
            if video_complete "$out/checkpoint-100/${a}.mp4" && \
               video_complete "$out/checkpoint-100/${b}.mp4"; then
                status "reuse_scene_${scene}_${a}_${b}"
                continue
            fi
            log="$EVAL_ROOT/logs/scene_${scene}_${a}_${b}.log"
            status "infer_scene_${scene}_row_${index}_${a}_${b}_start"
            CUDA_VISIBLE_DEVICES=0,1,2,3 "$REPO/.venv/bin/torchrun" \
                --nnodes=1 --nproc_per_node=4 --master_port=29811 \
                scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py \
                --config "$CONFIG" --checkpoint "$CHECKPOINT" \
                --data-path "$DATA" --output-dir "$out" --row-index "$index" \
                --seed "$((20260926 + scene))" --num-inference-steps 50 \
                --trajectories "$a" "$b" --trajectory-scale 1.0 \
                > "$log" 2>&1 || return 1
            video_complete "$out/checkpoint-100/${a}.mp4" || return 1
            video_complete "$out/checkpoint-100/${b}.mp4" || return 1
            status "infer_scene_${scene}_${a}_${b}_complete"
        done
    done
    "$PYTHON" scripts/evaluation/summarize_minimax_h3_action_sweep.py \
        --root "$EVAL_ROOT" --rows 0 2 3 11 23 \
        > "$EVAL_ROOT/logs/summary.log" 2>&1 || return 1
    date --iso-8601=seconds > "$EVAL_ROOT/INFERENCE_COMPLETE"
    status inference_complete
}

case "$MODE" in
    schedule)
        start_epoch=$(TZ=Asia/Shanghai date -d "$START_AT" +%s)
        status "scheduled_for_${START_AT// /T}+08:00"
        status waiting_for_100_step_training_completion
        while ! rg -q '100_step_pilot_complete' "$EARLY/logs/early_pilot_status.log"; do
            if ! pgrep -f 'fastvideo\.train|examples/train/launch_minimax_h3_worldmodel_stage0.sh' >/dev/null; then
                status training_absent_before_100_step_completion
                exit 1
            fi
            sleep 60
        done
        test -s "$CHECKPOINT/dcp/.metadata"
        status training_checkpoint_100_ready
        tmux has-session -t fastvideowm_vae_resume_4_7_20260926 2>/dev/null || \
            tmux new-session -d -s fastvideowm_vae_resume_4_7_20260926 \
                "bash '$REPO/examples/train/launch_minimax_h3_postpilot_eval_and_vae_20260926.sh' vae-half 4 7"
        # Resume useful cache work immediately; only inference has an 18:00 floor.
        while (( $(date +%s) < start_epoch )); do sleep 60; done
        set +e
        run_inference
        infer_rc=$?
        set -e
        tmux has-session -t fastvideowm_vae_resume_0_3_20260926 2>/dev/null || \
            tmux new-session -d -s fastvideowm_vae_resume_0_3_20260926 \
                "bash '$REPO/examples/train/launch_minimax_h3_postpilot_eval_and_vae_20260926.sh' vae-half 0 3"
        (( infer_rc == 0 )) || { status "inference_failed_rc_${infer_rc}"; exit "$infer_rc"; }
        ;;
    vae-half) run_vae_half "${2:?first worker}" "${3:?last worker}" ;;
    infer) run_inference ;;
    *) echo "Usage: $0 [schedule|infer|vae-half FIRST LAST]" >&2; exit 2 ;;
esac
