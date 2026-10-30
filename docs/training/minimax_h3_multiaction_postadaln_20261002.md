# MiniMax-H3 multi-action post-AdaLN run (2026-10-02)

This is the current FastVideoWM experiment record, not a replacement for the
dated `fastvideowm-h3-handoff` skill snapshot. Source videos and original pose
archives remain read-only. All new data plans, checkpoints, logs, and videos
are under `/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/`.

## Data and control

- Existing 65,389-row conditioned cache is reused; the complete VAE,
  intrinsics, and first-frame audits are required by the launcher.
- `examples/train/prepare_minimax_h3_balanced_multiaction_20261002.py`
  freezes 12,000 unique scene-disjoint training row indices in `data_plan_v4`.
  It selects multi-label camera-motion windows, not synthetic pure-action
  videos. The first 2,400 rows cover all eight directions; 35% are from
  `multicam_video`, down from about 83% in the full cache. Source and
  direction/speed counts are in `data_plan_v4/report.json`.
- GPU visual QC is an automatic strict pass. The original QC report still
  says human source-stratified spot checks are pending; this experiment does
  not relabel those rows as human approved.
- The model reads 37 latent-frame first-frame-relative C2W matrices and
  normalized K. I means camera pitch up, K means camera pitch down. Qwen
  first-frame vision tokens are retained; scene-text and audio tokens are
  absent. No source media or VAE latent is rewritten.

## Model and objective

Configuration: `examples/train/configs/minimax_h3_multiaction_postadaln_20261002.yaml`.
It starts from official MiniMax-H3 weights, not a prior unsuccessful action
checkpoint. Numeric 12-channel cumulative+incremental camera features use an
`asinh` transform instead of saturating hard clipping. The per-frame FiLM
can now be applied after block norm/AdaLN, alongside exact PRoPE. The old
pre-block path remains the default for existing configs/checkpoints.

Trainable: rank-64 Q/K/V/out LoRA, action encoder/adapter/gates, selected
video boundary layers, and the full attention output projection of the last
four DiT blocks. The new camera modules use a faster optimizer group. Training
uses correct-pose video flow MSE, motion-weighted frames, temporal-difference
loss, same-noisy-input correct-vs-counterfactual ranking, and a bounded
camera-response loss. The auxiliary action losses are not a promise that the
generated camera direction is correct; video evaluation remains mandatory.

Eight GPUs use SP=2, four data-parallel groups, microbatch=1,
accumulation=2, so **total batch size is 8**. The run is scheduled for 300
steps, checkpoint every 25, fixed held-out loss validation every 50. A two-step
eight-GPU smoke run completed with finite gradients, checkpoint save/load
metadata, and nonzero camera response before the main run was launched.

## Operation and results

- Training session: `fastvideowm_multiaction_train_eval_20261002`.
- Five-minute watcher: `fastvideowm_multiaction_watchdog_20261002`; it will
  wait for all GPUs to be free before a bounded checkpoint resume and will
  mark `WATCHDOG_NEEDS_ATTENTION` after repeated failures.
- Training launcher: `examples/train/run_minimax_h3_multiaction_20261002.sh`.
- Automatic evaluation: `examples/train/evaluate_minimax_h3_multiaction_20261002.sh`.
  It runs two held-out first frames through W/A/S/D/I/J/K/L, one command per
  124-frame video, at 50 inference steps and matched seeds. It writes a
  WASD-over-IJKL montage per first frame and runs
  `scripts/evaluation/analyze_minimax_h3_multiaction_20261002.py`.
- Training log: `logs/train_launcher.log`; smoke log:
  `logs/smoke_launcher_retry3.log`; watcher log: `logs/watchdog.log`.
- Expected final checkpoint: `runs/postadaln_8gpu_train/checkpoints/checkpoint-300`.
- Expected evaluation root: `evaluation/step300_heldout_50steps/`.
  `EVALUATION_COMPLETE` is written only after all videos, montages, and the
  proxy report exist. `motion_proxy.json` and `analysis.md` are explicitly
  *not* a substitute for human review of camera control and visual quality.
