# MiniMax-H3 world-model data audit and preprocessing

The first vision-and-action-only training recipe and its frozen scene split are
documented in `minimax_h3_action_only_v1_20260925.md`.

This document records the data preparation performed on 2026-09-24 and
2026-09-25 for the MiniMax-H3/FastVideoWM camera-controlled world model.  It is
an operational provenance document: every filter has a reproducible script and
every exclusion/review decision is stored as an ID file.  None of the source
dataset directories is modified.

## Source scope

The audit covers 205,433 original videos from four source families:

| Source | Root | Original videos |
|---|---|---:|
| VWM train-v2 records | `/datacc05/datasets/vwm_datasets/train_v2/records` | 189,380 |
| JS refine | `/home/lizihao/projects/worldmodel/train_data/js_refine_data` | 6,480 canonical unique videos |
| H3 JS easy | `/home/lizihao/projects/worldmodel/vggt-omega/h3_js_easy` | 2,809 |
| MiniMax-H3 `mi` records | `/home/lizihao/projects/worldmodel/train_data/minimax-h3/records/minimax_h3/mi` | 6,764 |

All derived audit files live under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/
```

The combined CPU result lives under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/combined_cpu_screen_complete/
```

The full GPU-QC run lives under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/
```

## Filter pipeline

### 1. Inventory and camera-pose validation

Script:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_source_inventory.py
```

The inventory resolves each source video, pose archive, prompt/record metadata,
source family, subset, and scene/group hint.  Pose arrays must be finite
homogeneous `[T,4,4]` camera-to-world matrices and align with the source video.
JS refine and H3 JS easy VGGT poses are marked as OpenCV Y-down; train-v2 and
MiniMax-H3 `mi` records are canonical X-right/Y-up/Z-forward.

Results:

- 205,433 videos have readable pose data;
- zero videos are in the missing/invalid-pose list;
- 47,737 videos are classified as static camera;
- 501 videos are near-static or have conflicting motion evidence and are sent
  to review;
- 157,195 moving-camera videos continue to action/quality screening.

ID files:

```text
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/pose_rejected_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/camera_static_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/camera_motion_review_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/minimax_records/camera_motion_review_ids.jsonl
```

All 34,000 `syncam_video` records are in the static-camera exclusion.  Static
samples are deliberately excluded from the moving-camera action dataset rather
than used as implicit `steady` controls.

### 2. Pose-derived WASDIJKL action audit

Script:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_actions.py
```

Every trajectory is rebased to its first camera.  OpenCV Y-down poses are
converted to canonical Y-up before action classification.  Translation is
resolved into camera-local forward/right/up components and rotation into
pitch/yaw/roll increments.  The key semantics are:

| Key | Camera control |
|---|---|
| W/S | forward/backward translation |
| A/D | left/right translation |
| I/K | pitch up/down |
| J/L | yaw left/right |

Actions require a stable directional component, sufficient axis dominance,
and consistent sign over the trajectory.  A video may contain several actions,
so direction totals are video-direction occurrences rather than mutually
exclusive video counts.

Before GPU visual QC, the 148,492 CPU candidates have these occurrences:

| W | A | S | D | I | J | K | L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47,073 | 45,109 | 38,310 | 44,411 | 26,810 | 49,840 | 49,853 | 49,514 |

Unclassifiable/unstable trajectories are excluded or reviewed.  Direction ID
files are named `action_direction_<KEY>_ids.txt`; the corresponding slow,
medium, and fast files add `_<speed>_ids.txt`.  They are stored in both the
main audit root and its `minimax_records` subdirectory.

### 3. Motion magnitude and learning scale

Translation from VGGT is scale-ambiguous, so it is not interpreted as physical
metres per second.  Magnitudes are calibrated independently for each
source/subset and action.  The 33.3% and 66.7% quantiles define slow, medium,
and fast.  The numeric learning factor is:

```text
clip(sample magnitude per second / source-action median, 0.25, 4.0)
```

Per-sample actions, bins, and factors are recorded in:

```text
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/action_scale_provisional.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/minimax_records/action_scale_provisional.jsonl
```

For new FastVideoWM data, the trajectory is rebased, canonicalized, normalized
once, and the audited XYZ scale is multiplied into its translation.  Training
must set `models.student.normalize_camera_translation: false`; otherwise the
model would normalize each trajectory again and erase the learned speed
difference.  The calibrated camera is consumed by both the relative-pose
action adapter and PRoPE in hybrid conditioning.

Relevant implementation:

```text
/home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/pipelines/preprocess/minimax_h3_worldmodel/indexed_legacy_cache.py
/home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py
/home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3_prope.py
```

### 4. CPU media and trajectory screening

Scripts:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_cpu_quality.py
/home/lizihao/projects/worldmodel/H3-World/code/vwm/assemble_fastvideowm_cpu_candidates.py
/home/lizihao/projects/worldmodel/H3-World/code/vwm/merge_fastvideowm_cpu_screen.py
```

Seven frames are sampled from each video.  The screen checks decode success,
duration, extreme brightness, contrast, Laplacian sharpness, nearly frozen
frames, translation spikes, and abrupt rotation.  It also excludes unstable
eight-direction actions, evaluation-sensitive cases, JS command/pose
mismatches, and clips shorter than 4.5 seconds.

Combined result:

| Bucket | Videos |
|---|---:|
| CPU candidate | 148,492 |
| CPU review | 5,985 |
| CPU rejected | 2,718 |

The complete per-ID outputs are:

```text
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/combined_cpu_screen_complete/cpu_candidate_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/combined_cpu_screen_complete/cpu_review_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/combined_cpu_screen_complete/cpu_rejected_ids.jsonl
```

`cpu_rejected_ids.jsonl` is the authoritative sample-ID list filtered at this
stage.  `cpu_review_ids.jsonl` is not silently included in training.

CPU heuristics cannot reliably certify texture collapse, subject deformation,
or agreement between estimated pose and visible parallax.  Those checks are
therefore a separate GPU gate.

### 5. MiniMax-H3/Qwen3-VL GPU visual QC

Input preparation and inference scripts:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/prepare_fastvideowm_full_gpu_qc.py
/home/lizihao/projects/worldmodel/H3-World/code/vwm/run_fastvideowm_gpu_quality_pilot.py
/home/lizihao/projects/worldmodel/H3-World/code/vwm/finalize_fastvideowm_full_gpu_qc.py
```

The 148,492 CPU candidates are SHA-256 partitioned into eight disjoint shards.
Six frames are sampled across the deterministic 124-frame, 24-fps target
window.  Qwen3-VL assesses:

- blur;
- texture breakdown;
- subject deformation;
- translation jitter;
- visible camera-motion agreement with the pose-derived action;
- uncertainty.

The eight selection files and raw result files are:

```text
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/selection_shard_00_of_08.jsonl
...
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/selection_shard_07_of_08.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/results_shard_00_of_08.jsonl
...
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/results_shard_07_of_08.jsonl
```

The finalizer verifies the exact expected count of every shard and the global
ID uniqueness before publishing:

```text
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/gpu_qc_pass_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/gpu_qc_review_ids.jsonl
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/report.json
```

A sample enters `gpu_qc_pass_ids.jsonl` only when all four artifact flags are
false, visible camera-motion agreement is `yes`, and uncertainty is false.
Every `no`, `uncertain`, parse failure, or artifact flag enters
`gpu_qc_review_ids.jsonl`.  That review file is the authoritative ID list
filtered by the GPU gate.  GPU pass is still marked `quality_approved=false`
until source-stratified human spot checks are complete.

The full run started on eight GPUs at 2026-09-25 06:40 CST.  Initial measured
throughput was 2.0–2.18 seconds per video per GPU.  The ID counts in
`report.json` supersede this document once the run finishes.

## H3 latent preprocessing

### Pilot

Pilot scripts and results:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/prepare_fastvideowm_gpu_pilot.py
/home/lizihao/projects/worldmodel/H3-World/code/vwm/run_fastvideowm_single_gpu_pilot.sh
/datacc05/lizihao/worldmodel/fastvideowm_gpu7_pilot_20260924/
```

The pilot used 16 uncached videos, with two examples for every WASDIJKL
direction, plus four CPU warning cases for QC calibration.  It validated 16
audio-free PRoPE Parquet rows.  Measured single-GPU steady-state throughput was
2.19 seconds/video for visual QC and about 16.2 seconds/video for MiniMax-H3
VAE encoding.  The native H3 condition encoder loaded and completed without
OOM.  Final pilot records are under `preprocess/preprocessed/all`.

### Existing-cache reuse

Of the 148,492 CPU candidates, 18,302 have a direct existing H3 cache pointer;
130,190 do not.  Existing native H3 PTH caches are converted on CPU by:

```text
/home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/pipelines/preprocess/minimax_h3_worldmodel/indexed_legacy_cache.py
```

The converter rejects rows without explicit final approval, removes audio and
legacy action-text spans, retains the native video/keyframe/text tensors, and
writes the FastVideoWM PRoPE schema.  Reusing these caches avoids unnecessary
VAE and Qwen encoding.

### Missing-latent generation

After GPU QC, the following script joins GPU-pass IDs to CPU candidates,
excludes rows with an existing cache, and creates a non-destructive canonical
VWM view:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/build_fastvideowm_approved_uncached_dataset.py
```

It writes symlinks to source videos, separate calibrated camera archives, a
source mapping, and a derived manifest under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925/
```

Source videos and source pose archives remain unchanged.  At the VAE-only
stage a generic scene prompt is stored; final scene captions can be inserted
before condition encoding without repeating VAE encoding.

The scheduler starts immediately, waits for exact completion of every GPU-QC
shard, and then launches latent generation without a fixed wall-clock delay:

```text
/home/lizihao/projects/worldmodel/H3-World/code/vwm/schedule_full_latent_after_gpu_qc_1900.sh
```

It refuses to proceed until all eight QC shard counts are exact.  It then
freezes pass/review IDs, builds the derived uncached dataset, waits for all
GPUs to be free, and launches eight native MiniMax-H3 VAE workers.  Latents are
written under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925/
```

The expected native target shape is `[24,37,30,52]` for 124 RGB frames at
480×832.  Audio is not encoded and is absent from the FastVideoWM schema.

## FastVideoWM code review and tests

The training path uses H3 first-frame conditioning, LoRA, the camera action
adapter, and PRoPE.  The model supports preserving calibrated translation with
`normalize_camera_translation=false` and repeat-aware balancing through
`sampling_index_path`.  The audited cache converter enforces pose/intrinsic
alignment and exact schema.  Relevant unit and training-method tests passed 92
tests, with only PyTorch deprecation warnings; pre-commit and diff checks also
passed at the time of review.

Before a production training run, create a scene/group-held-out validation
Parquet, test several noise levels, verify checkpoint resume, and compare
true/static/reverse camera rollouts.  The older V7 overfit configuration uses
training rows for validation and a fixed noise sigma, so it must not be reused
unchanged as the production configuration.

## Reproducibility and source safety

- Original dataset directories are read-only inputs to this workflow.
- Every stage publishes complete sample-ID lists rather than relying on an
  implicit directory scan.
- GPU results are resumable JSONL files, one writer per shard.
- Finalization checks exact counts and duplicate IDs before publishing.
- Uncertain samples are routed to review rather than silently accepted.
- Existing cache reuse and new VAE output are kept in derived `/datacc05`
  directories.
- Audio is deleted from the FastVideoWM data contract rather than represented
  by synthetic silence.
