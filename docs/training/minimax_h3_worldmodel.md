# MiniMax-H3 camera world-model Stage 0

This recipe adapts the released MiniMax-H3 bidirectional video generator to
camera-conditioned world-model training. It deliberately keeps the H3 video
backbone and native first-frame/Qwen3-VL conditioning, removes audio from the
dataset, forward target, and loss, and trains only attention LoRA, a camera
action adapter, and selected input/output layers.

## Fixed experiment contract

- Dataset: VWM train_v2 manifest at
  `/datacc05/datasets/vwm_datasets/train_v2/manifests/train.jsonl`.
- Geometry: 832×480 RGB, 124 frames at 24 FPS, encoded to 37 H3 video latent
  frames of shape `[24,37,30,52]`.
- Conditioning: the first frame is encoded both by the native H3 keyframe VAE
  path and by the native Qwen3-VL FL2VA presentation.
- Control: camera motion only. Object actions and audio are absent.
- Compute: one node with eight 80GB A100 GPUs, sequence parallel size 2,
  four data-parallel replicas, and HSDP shard size 2.
- Validation: a deterministic 1% manifest split, normally preprocessed with a
  small `max_samples` limit for the initial run.
- Ordering: train_v2 records are hash-shuffled within each source and then
  round-robin interleaved. Finite preprocessing prefixes therefore cover every
  available source instead of inheriting the source-grouped manifest order;
  full-dataset membership and frequencies remain unchanged.

The camera coordinate system follows canonical VWM: X points right, Y points
up, and Z points forward. Each latent-frame action contains local forward,
right, and up translation followed by pitch, yaw, and roll increments. Because
VWM sources use incompatible world units, translation is made relative to the
first camera and divided by the trajectory's maximum displacement norm before
local increments are computed. Rotation increments are wrapped to
`[-180°,180°]` and divided by 180. The resulting six values are the only action
features passed to H3.

## Environment and checkpoint

The project environment is an editable `.venv` install with the optional
`worldmodel` dependency group. On this host, Python development headers live in
`.venv/sysroot`, so the supplied launch scripts export `CPATH` automatically.

The checkpoint is pinned to Hugging Face revision
`42ed227ee7df40d41602854ae760620d6eb651fe` and downloaded through
`https://hf-mirror.com` to:

```text
/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3
```

Use the resumable downloader below. It fetches only the processor, tokenizer,
text encoder, video transformer, video VAE, and scheduler, then verifies all 31
weight shards against the revision's Git-LFS SHA256 object IDs:

```bash
tmux new-session -d -s fastvideowm_h3_download \
  "cd /home/lizihao/projects/military_world_model_agent/FastVideoWM && \
   bash examples/train/download_minimax_h3_worldmodel.sh \
   > /datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3.download.log 2>&1"
```

World-model preprocessing and training never load `audio_vae`. The dormant H3
audio projections remain in the transformer only so the released checkpoint
can be loaded without altering unrelated H3 inference behavior.

## Two-phase preprocessing

VAE and Qwen3-VL are intentionally loaded in separate phases. Each worker owns
fixed parquet shards and writes by atomic rename. A restarted phase verifies
the exact schema and sample IDs before reusing a completed shard.

First run a one-sample pilot in a separate root. A one-row shard cannot later
be reused by the default 16-row full-data plan, which is why the pilot does not
share the production directory.

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM

tmux new-session -d -s h3_vwm_vae_pilot \
  "NUM_WORKERS=1 WORK_ROOT=/datacc05/lizihao_wm/worldmodel/pilot \
   bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh vae train 1"

tmux new-session -d -s h3_vwm_condition_pilot \
  "NUM_WORKERS=1 WORK_ROOT=/datacc05/lizihao_wm/worldmodel/pilot \
   bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition train"

WORK_ROOT=/datacc05/lizihao_wm/worldmodel/pilot \
  bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify train 1
```

For the initial production run, 256 training rows align exactly with the
8-worker × 16-row shard plan and can later be extended without rewriting them.
The action-sensitive v2 run uses 2048 training rows so one effective batch of
64 no longer revisits the same subset every four optimizer steps. Thirty-two
fixed validation rows cover all sources in the small validation gate.

```bash
tmux new-session -d -s h3_vwm_vae_train \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh vae train 256"
tmux new-session -d -s h3_vwm_condition_train \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition train"

tmux new-session -d -s h3_vwm_vae_validation \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh vae validation 32"
tmux new-session -d -s h3_vwm_condition_validation \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition validation"
```

To extend the existing training cache to 2048 rows, rerun only the missing VAE
shards and then condition them. Completed shards are verified and reused:

```bash
tmux new-session -d -s h3_vwm_vae_train_2048 \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh vae train 2048"

# Run after the VAE session completes successfully.
tmux new-session -d -s h3_vwm_condition_train_2048 \
  "bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition train"

bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify train 2048
```

Do not run a condition phase until its corresponding VAE phase has completed.
Worker logs are under `/datacc05/lizihao_wm/worldmodel/logs`.

## Reusing a DiffSynth H3 PTH cache

The `legacy-cache` phase converts an existing native MiniMax-H3 DiffSynth cache
without loading either the video VAE or Qwen3-VL. It accepts only the fixed H3
contract `[1,24,37,30,52]`, verifies a complete numeric PTH index, and retains:

- the normalized video VAE latent;
- the native first-frame VAE condition (the legacy BF16 value `0.999` rounds
  to `1.0`, so its stored anchor is already the clean condition);
- the original first-frame visual/text embedding head and its semantic tags.

It explicitly discards `audio_input_latents` and removes the appended legacy
action-text spans. Camera C2W and intrinsics are reloaded from the canonical
VWM manifest, aligned to the legacy 124-frame sampling plan, transformed for
the 832×480 center crop, and serialized through the current PRoPE/action record
creator. This prevents two control implementations from being mixed.

The current 20,668-row cache contains 10,000 `multicam_video`, 5,000
`syncam_video`, 656 `sekai_real_drone`, 74 `sekai_real_walking_hq`, 4,420
`ue_easy`, and 518 `ue_game` rows. It was selected with seed 42 and excludes
the 13,712 clips that the older pipeline rejected as shorter than 124 frames at
24 FPS; it is therefore a different membership contract from the later
34,380-row seed-20260909 manifest.

Convert it with CPU workers and then verify the exact row count and audio-free
PRoPE schema:

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM

LEGACY_METADATA_PATH=/datacc05/lizihao/outputs/h3_world/vwm_train_v2/manifests/train_20668.jsonl \
LEGACY_CACHE_DIR=/datacc05/lizihao/datasets/H3-World/vwm_train_v2_h3_cache_480x832_124f \
MANIFEST_PATH=/datacc05/datasets/vwm_datasets/train_v2/manifests/train.jsonl \
WORK_ROOT=/datacc05/lizihao_wm/worldmodel_prope/datasets/h3_legacy_20668_v5 \
NUM_WORKERS=8 CONDITIONING=prope \
  bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh legacy-cache all 20668

WORK_ROOT=/datacc05/lizihao_wm/worldmodel_prope/datasets/h3_legacy_20668_v5 \
CONDITIONING=prope \
  bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify all 20668
```

## Training

The run configuration is
`examples/train/configs/minimax_h3_worldmodel_stage0.yaml`. It enables rank-8
LoRA on every H3 `to_q`, `to_k`, `to_v`, and `to_out` layer and additionally
trains `action_adapter`, `proj_in`, `norm_out`, and `proj_out`. All other
pretrained video-generation parameters remain frozen.

After both train and validation parquet directories pass verification, launch
training in tmux:

```bash
tmux new-session -d -s h3_vwm_stage0 \
  "cd /home/lizihao/projects/military_world_model_agent/FastVideoWM && \
   bash examples/train/launch_minimax_h3_worldmodel_stage0.sh"
```

The default recipe uses one sample per sequence-parallel group and four
data-parallel groups. With gradient accumulation 16, its effective total batch
is `1 × 4 × 16 = 64`. It uses learning rate `2e-5` with 100 warmup steps,
gradient clipping at 1.0, and a distributed checkpoint every 100 steps. W&B
defaults to offline mode and writes under the checkpoint run directory.

Every 100 steps, validation uses fixed rows and fixed noise to report:

- `validation_worldmodel/flow_loss`;
- `validation_worldmodel/zero_action_flow_loss`;
- `validation_worldmodel/action_loss_gain`, which should become positive when
  camera control helps prediction;
- `validation_worldmodel/action_response_rms`, which is zero at initialization
  because the adapter is zero-initialized and should grow during training.

Before scaling beyond the initial subset, require a finite one-step loss,
nonzero gradients on LoRA/action parameters, unchanged frozen parameters, a
successful checkpoint resume, and improving action-sensitive validation.

## Action-sensitive v2 recipe

The v2 configuration is
`examples/train/configs/minimax_h3_worldmodel_stage0_v2.yaml`. It retains the
same H3 weights, LoRA targets, selected important layers, 832×480 geometry, and
effective batch 64. It changes only the camera-learning path:

- scale each of the six action channels by dataset-derived robust values, clip
  normalized outliers to 5, and multiply the adapter input by 16 so the action
  residual survives the H3 BF16 residual-add boundary;
- remove `action_adapter` bias so zero action cannot learn an unconditional
  offset;
- use learning rate `1e-4` with no weight decay for the newly initialized
  adapter while keeping LoRA and pretrained selected layers at `2e-5`;
- apply mean-one per-frame loss weights that emphasize camera-motion frames
  without changing fully static samples or the global loss scale;
- validate every 50 steps on all 32 fixed rows and compare true actions against
  both zero actions and a deterministic temporal permutation.

The decisive metrics are `action_loss_gain` and
`shuffled_action_loss_gain`. Both should become positive: the true action
sequence should predict the target better than absent or temporally incorrect
actions. `action_response_rms` alone is not sufficient because a model can
respond to actions without responding correctly.

Launch v2 explicitly so the original configuration remains reproducible:

```bash
CONFIG_FILE=examples/train/configs/minimax_h3_worldmodel_stage0_v2.yaml \
  MASTER_PORT=29502 \
  bash examples/train/launch_minimax_h3_worldmodel_stage0.sh
```

The v2 run writes to a new checkpoint directory and never resumes or overwrites
the v1 checkpoints.

## Paired-static multi-sigma control diagnostic

The V5 diagnostic configuration is
`examples/train/configs/minimax_h3_worldmodel_action_overfit_v5_paired_static_multisigma_gpu4_7.yaml`.
It keeps the trainable capacity of V4 diagnostic B (action adapter, H3
input/output boundary layers, and rank-16 Q/K/V/out LoRA) and resumes its
`checkpoint-300`. The new run has a separate output directory, so the V4
checkpoint is read but never overwritten.

V5 deliberately gives the three counterfactuals different supervision:

- `static` repeats the first clean latent frame across time, applies the same
  sampled noise at the same sigma, and uses a constant camera trajectory. Its
  direct flow, temporal-delta, and low-frequency motion targets are therefore
  all genuinely static.
- `reverse` and `shuffled` keep the original moving latent, noise, sigma, and
  timestep. Only the camera trajectory changes, so they remain valid negatives
  for ranking, bounded sensitivity, and reverse-direction supervision.
- Sigma is sampled over the rollout range and clamped only at 0.05. This
  removes the V4 pure-noise-only shortcut and teaches control at later denoising
  states.

The four-row diagnostic dataset is unchanged. On four GPUs with sequence
parallel size 2 and gradient accumulation 2, the effective global batch is 4,
so each optimizer step covers the full overfit set. Static-only losses are
weighted by 3 because the static branch is visited once per three
counterfactual microbatches.

Validation additionally logs
`validation_worldmodel/paired_static_hybrid_flow_loss`. This metric tests
the matched static target; the existing zero/shuffled loss gains continue to
measure whether camera conditioning changes the prediction on the same noisy
moving input. A later rollout gate must still compare correct/static/reverse
videos and camera-motion estimates—loss separation alone is not sufficient.

## Audited camera data and speed calibration (2026-09-24)

The newer multi-source audit lives under
`/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/`. It includes
`train_v2/records`, `js_refine_data`, `h3_js_easy`, and the separate MiniMax-H3
`records/minimax_h3/mi` source. The `cpu_candidate_ids.jsonl` files are **not**
training approval: sampled-frame blur/trajectory checks cannot certify subject
deformation, texture collapse, or that estimated camera motion matches visible
parallax. Those checks require a separate GPU/human gate and a frozen manifest
with `quality_approved: true` for every row. Keep evaluation-sensitive groups
out of training and split by scene/group, not by individual window.

VGGT source poses in JS refine and H3 JS easy are OpenCV Y-down. They must be
rebased to frame zero and converted to canonical Y-up on both sides of the
transform. MiniMax-H3 `mi` records and canonical VWM poses are already Y-up.
VGGT intrinsics are pixel-valued on the estimator's processed image plane;
normalize by its recorded `processed_hw`, then apply the 832×480 resize/crop.
Never treat those matrices as the normalized VWM intrinsics directly.

The existing Stage-0 path divides each trajectory by its own maximum
translation, which removes across-video speed. For a new **audited** dataset,
calibrate source-relative speed factors (clipped to `[0.25, 4]`) in the camera
poses, and set `models.student.normalize_camera_translation: false`. This
passes the calibrated displacement through both relative-pose action features
and PRoPE. Leave the option at its default `true` when loading an old Stage-0
checkpoint or dataset; changing it silently would change the conditioning
contract. Raw VGGT translation is scale-ambiguous, so the factors are only
relative within a source/version, not physical meters per second.

`fastvideo.pipelines.preprocess.minimax_h3_worldmodel.indexed_legacy_cache`
converts a frozen approved JSONL manifest of existing DiffSynth H3 PTH caches
into the audio-free `h3_worldmodel_prope` Parquet schema without running VAE or
Qwen again. Its required per-row fields are `global_id`, `source_name`,
`prompt`, `cache_path`, `camera_path`, `camera_convention`,
`translation_scale_xyz`, `intrinsics_path`, `intrinsics_key`,
`intrinsics_alignment`, `intrinsics_units`, `intrinsics_source_hw`,
`source_frame_indices`, `source_fps`, and `quality_approved`. The converter
rejects unapproved rows, missing cache/geometry, misaligned camera tensors,
and schema drift. Uncached videos still require GPU VAE and text encoding.

For direction/speed balancing, construct an audited parquet-row index plan
that includes every approved row at least once and repeats minority buckets.
Set `models.student.sampling_index_path` to its JSON array of parquet indices.
This repeat-aware plan is exclusive with `motion_balanced_sampling`; the latter
only reorders or subsamples unique rows and cannot oversample rare directions.
Before any new training run, use a **scene-held-out** validation parquet,
multiple noise sigmas, and matched static/reverse/true-action rollouts. The V7
overfit configuration reuses training rows for validation and fixes sigma to
1, so it is not a generalization or full-denoising quality gate.
