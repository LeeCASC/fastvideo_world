# MiniMax-H3 stronger-action full-cache experiment (2026-10-02)

This experiment responds to the full-data Cosmos/Wan result: the previous
FastVideoWM 200-step pilot improved video flow MSE while correct camera pose
still lost to zero pose on all five fixed validations. Its 16 held-out videos
did not reliably follow opposite commands. The old checkpoint is preserved as
a diagnostic baseline; this run initializes from official MiniMax-H3 weights.

## Data and alignment

- Training source: the previously verified 65,389-row action-only conditioned
  cache at `/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all`.
  Source media and VAE latents are read-only. The full VAE, intrinsics, and
  conditioned-cache verification markers are launcher prerequisites.
- Motion-quantile sampling interleaves 65,388 of the 65,389 rows; the one-row
  drop is due to four-way data-parallel divisibility, not quality filtering.
  This balances motion magnitude, **not** direction or source, so it does not
  imply WASDIJKL balance. Automatic quality pass is not human sign-off.
- The model gets the audited 37-frame first-frame-relative C2W path and
  normalized intrinsics. Qwen first-frame vision tokens and H3 keyframe
  conditioning remain; scene text and audio tokens do not enter the video
  forward. I is camera pitch up and K pitch down.

## Model and supervision

Config: `examples/train/configs/minimax_h3_full_action_strong_20261002.yaml`.

- Exact PRoPE spans `[96,120)` (24 rather than 12 channels per head), outside
  H3's native 96-channel RoPE. The 50-block camera FiLM now supports separate
  trainable per-channel scale and shift strengths; existing scalar-gate
  checkpoints retain their default architecture.
- The action adapter/encoder/gates, LoRA across Q/K/V/out, boundary layers,
  and full Q/K/V/out attention projections in blocks 38–49 are trainable.
  The load-time audit reports about 2.05B trainable parameters, up from
  0.352B in the previous pilot. This is **partial**, not Cosmos-style full
  33B-parameter fine-tuning.
- In addition to video rectified-flow MSE, training uses temporal latent
  difference, correct-versus-reverse/static camera ranking on matched noise,
  and differentiable visual-latent translation magnitude/direction matching.
  The visual motion observer is a global proxy; it is not a proven camera
  estimator in clips with moving subjects. Do not declare instruction control
  from its loss alone.

## Execution and evaluation

- Runner: `examples/train/run_minimax_h3_full_action_strong_20261002.sh`.
  `smoke` runs two real steps on all eight GPUs and saves a separate DCP.
- Scheduler: `examples/train/schedule_minimax_h3_full_action_strong_20261002.sh`.
  Phase 1: 200 steps, total BS=8 for quick feedback. Phase 2: resume the full
  optimizer/dataloader state to step 1,000 with BS=32. Phase 3: resume to
  step 2,200 with BS=32. The total draw count is approximately 65,600, just
  over one 65,388-row motion-balanced pass. Checkpoints save every 100 steps.
- Each phase ends with matched-seed W/A/S/D/I/J/K/L inference on two fixed
  held-out first frames, 124 frames and 50 inference steps per video. Output
  root: `/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002`.
  Automatic flow proxies are not a substitute for reviewing the videos.
- The scheduler waits for free GPUs every five minutes and never kills another
  process. Training or evaluation failures stop that primary schedule instead
  of silently claiming completion. The independent
  `examples/train/guard_minimax_h3_full_action_strong_20261002.sh` session
  waits for the primary schedule to exit and, if checkpoint-2200 is not yet
  complete, resumes from the newest complete DCP to step 1,000 and then 2,200
  at BS=32. This deliberately prioritizes continued training over retrying a
  failed intermediate inference. It leaves `GUARD_NEEDS_ATTENTION` after five
  unsuccessful restart attempts. Neither process overlaps a GPU owner.

The two-step smoke test finished with finite gradients and a complete DCP
before the long schedule started at 2026-10-02 20:11 CST. The 200-step phase
will pause for its scheduled inference before the BS=32 continuation. Report
actual speed and ETA from the logs rather than extrapolating from the previous
352M-parameter run.
