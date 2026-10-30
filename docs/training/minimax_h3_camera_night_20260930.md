# Camera-control overnight diagnostic (2026-09-30)

This is an isolated FastVideoWM MiniMax-H3 experiment, initialized from the
official MiniMax-H3 baseline, not the unrelated H3-World DiffSynth checkpoint.
The existing VAE cache and source videos are never rewritten.

## Frozen data and corrections

- Derived data: `/datacc05/lizihao/worldmodel/fastvideowm_camera_pair_night_20260930/data`.
- IDs, rejected IDs, matched groups, balanced draw list, and scale audit live in
  the same parent directory (`ids.jsonl`, `rejected_ids.jsonl`,
  `pair_groups.json`, `train_indices.json`, `report.json`).
- 106 measured-camera videos in 26 groups. Within each scene/speed/prompt group,
  a real anchor first frame is accepted only when the source pixel MAE at
  832×480 is at most 3/255. Accepted members share the identical first-frame
  latent and text-vision tokens. The original target video latent and
  intrinsics are bit-identical to the source cache. This is a close-first-frame
  approximation, not a claim that the source videos have mathematically
  identical first frames.
- The old per-axis camera translation multipliers had already been removed in
  the 156-row diagnostic. This derived data reapplies the recorded fast/slow
  factor *isotropically* to W/A/S/D translations only; measured rotations are
  unchanged. These are calibrated relative scales, not metric metres.
- The hybrid adapter uses cumulative relative poses. Its six channel scales
  are derived from the 99th percentile of the frozen paths, with headroom
  below the clipping threshold; observed clipped-frame rate is 0.46%.
- The sampling list repeats sparse directions up to 18 draws each (144 draws
  total). Unique direction counts are in `report.json`. Fast/slow classes are
  **not** exactly balanced; this run is a control-function diagnostic, not a
  final speed-regression model.

## Two concurrent, controlled comparisons

| Branch | GPUs | Conditioning | Trainable parts |
| --- | --- | --- | --- |
| `prope` | 0–3 | exact PRoPE | rank-64 Q/K/V/out LoRA + input/output boundary |
| `hybrid` | 4–7 | exact PRoPE + cumulative-pose adapter | same LoRA/boundary + adapter |

Each branch has SP=2, HSDP replicate=2/shard=2, local batch=1, gradient
accumulation=2: **total batch 4 per branch**, 8 concurrent across both jobs.
Both use the same 106 videos and sampling list. Phase 1 has 200 updates with
sigma=1, making noisy video pixels uninformative; flow MSE plus temporal loss,
same-input reversed/static camera ranking, and bounded camera-response loss
must learn from first frame and pose. Phase 2 initializes *weights only* from
phase 1, uses sigma≥0.6, and gets a fresh optimizer. Its step count is
calculated after phase 1 from actual speed to finish near 2026-10-01 09:00
Asia/Shanghai. A checkpoint is saved every 25 updates; held-out flow and
counterfactual camera metrics run every 50 updates. W&B is offline.

Launcher: `examples/train/run_minimax_h3_camera_night_20260930.sh prope|hybrid`.
Sessions: `h3cam_night_prope` and `h3cam_night_hybrid`. Each branch has separate
`runs/<branch>/<phase>/checkpoints` and `logs`; `STARTED`, `COMPLETE`, and
`PLANNED_STEPS` markers indicate transitions. An incomplete first phase stops
that branch; it does not silently start phase 2.

After training, run the same-first-frame, same-noise correct/static/reverse
diagnostic on *train and held-out* scenes, then equal-seed J/L and A/D video
inference. Do not claim camera control from decreasing aggregate flow loss
alone. If the pure branch fails but hybrid succeeds, adapter capacity matters;
if both fail on train scenes, revisit the projection/gradient path and
conditioning supervision before scaling data; if only held-out fails, improve
paired data diversity and speed balance before a full-dataset run.
