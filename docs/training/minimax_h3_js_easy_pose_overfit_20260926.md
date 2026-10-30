# JS-easy measured-pose action overfit diagnostic

This is a **diagnostic**, not a production train split. Do not start it while
the eight-GPU H3 VAE cache is still being written. The video target and its
camera matrices must describe the same observed motion; the analytic
`js_easy_multitraj_8/sample_*_pose.npz` paths are not interchangeable with
VGGT estimates from the actual generated videos. This recipe uses the
`h3_i2v_js_easy_124` source records from the FastVideoWM audit, whose
`camera_c2w.npz` derives from the source-video pose estimate. The source
videos and original benchmark files are never changed.

## Why this differs from the first action-only pilot

The validated Cosmos-Predict2 PRoPE recipe
(`cosmos_predict2/configs/base/wan_exps/wan_causal_t2v_framewise.py`,
`cosmos_predict2/models/wan_warped_causal_dense_t2v_framewise_se3_prope.py`)
uses 12 homogeneous channels in attention, the camera intrinsic matrix, a
whole video-flow MSE target, and a trainable DiT. Its training loader supplies
C2W trajectories, which the model converts to W2C once. The attention path
uses the projective transform on Q/K/V and maps the output back. The Cosmos
run's 2,000-iteration result does **not** establish that H3 will converge in
2,000 iterations: its backbone, teacher-forcing layout, and optimization
capacity differ.

The first H3 diagnostic therefore isolates the analogous geometry and loss:

- pure `camera_conditioning: prope`, `prope_mode: exact`, 12 channels
  `[96,108)` outside H3's native rotary channels, on every main attention
  block;
- measured, first-frame-rebased C2W and crop-adjusted intrinsics; no
  second per-video translation normalization that would erase speed scale;
- first-frame Qwen vision and native H3 keyframe-VAE condition, with no scene
  text and no audio;
- only H3 video rectified-flow MSE for optimization. Static/reverse/shuffled
  trajectories are used in validation, not as synthetic training targets;
- rank-32 Q/K/V/output LoRA plus `proj_in`, `norm_out`, `proj_out`, initialized
  from the released H3 model. This is less capacity than Cosmos's full-DiT
  fine-tune. If a *verified* tiny set cannot be memorized, raise LoRA capacity
  or unfreeze selected H3 blocks before blaming PRoPE or dataset scale.

The model's `require_rebased_camera` guard rejects a non-identity first latent
camera, non-finite pose/K, non-rigid rotations, or nonpositive focal lengths.
It cannot prove that a camera estimate matches visible video motion: that
still requires the existing GPU-QC and manual side-by-side review.

The first 2026-09-27 launch failed before completing step 1: its JS-easy
VGGT K used **672×384 pixel coordinates** although source videos are
1344×768, and the old crop transform assumed normalized K. PRoPE therefore
received focal values around 400–1900 and produced NaN gradients. The
preprocessor now converts this source-specific K before crop; PRoPE and
the VAE/final Parquet verifiers reject pixel-valued K. The exact 80 frozen
train rows and 2 affected held-out validation rows were repaired in derived
Parquet from original source K, leaving latents and source files unchanged.
The repaired stage 1 was relaunched in tmux
`fastvideowm_js_easy_stage1_jl_intrinsics_fixed_20260927`; check live logs
before quoting progress. Its 80-row VAE report is `vae_stage80_audit.json`
and its training log is `runs_stage1_intrinsics_fixed.log` in the plan root.
Each repaired shard has a `.pre_intrinsics_fix.bak` backup. The initial full
65,389-row audit began before this discovery, so its marker alone is not
enough for stage 2/3. The separate
`fastvideowm_intrinsics_repair_after_audit_20260927` session waited for that
audit, then repaired all 829 JS-easy VAE K rows across four sources (770 on
the final pass, 59 already repaired by an interrupted prior pass). The
read-only per-source K audit found no pixel-valued K among the other 64,560
cached rows from `train_v2` and `minimax_h3_records`. A second full K-aware
audit is required before declaring the bulk cache ready. Known JS-easy K
uses 672x384 VGGT pixel coordinates regardless of its 1344x768 stored video;
the source files and VAE latents are unchanged. Some valid telephoto K has
normalized focal length above 10, so the unit guard permits focal lengths up
to 32 while also checking the normalized principal point.
Stage 2/3 require `VAE_INTRINSICS_AUDIT_COMPLETE` in addition to their
action-evaluation approval.

## Frozen directional curriculum

The CPU-only selector
`examples/train/prepare_minimax_h3_js_easy_overfit.py` reads the existing
FastVideoWM `source_mapping.jsonl` and `train_ids.jsonl`. It requires a
passing GPU-QC flag, a nominal command detected in the *measured* path, and
both axis fraction and direction consistency at least 0.8. It groups videos
by scene, speed, and camera-only/scene-camera variant, and covers distinct
scenes before repeating variants. A frozen ID file and report are written for
each stage; a rerun refuses to alter them. The audit's `quality_approved=false`
is preserved in the files—automatic QC is not manual sign-off.

At the 2026-09-26 source-mapping snapshot, 156 individual records pass this
strict directional screen. Available opposite-direction pairs are J/L 10,
I/K 13, A/D 11, and W/S 12. The default balanced diagnostic uses 10 pairs
per axis:

| Stage | Training examples | Purpose |
|---|---:|---|
| J/L | 20 videos | Same-scene opposite yaw, find a minimal camera signal |
| J/L + I/K | 40 videos | Add pitch without dropping earlier yaw examples |
| All eight directions | 80 videos | Add A/D and W/S, 10 examples per direction |

All eight should **not** be judged from a J/L-only checkpoint. Nor should
these 80 videos be called generalization data: this is intentional
memorization. Keep scenes outside the frozen plan for a separate inference
test. The original 24 J/L pairs were only a preliminary scan of all 544 source
videos. The stricter 10-pair set is a memorization diagnostic, not the final
scale-up set. The canonical key convention is **I = camera up, K = camera
down**. Frozen H3 text caches with the opposite legacy mapping are not
silently relabelled or mixed into this numeric-pose experiment.

Use 200 steps and effective batch 8 for the first gate; save and inspect at
steps 50/100/200. Stage 2 adds I/K for 200 *new* steps and stage 3 adds
A/D/W/S for 400 new steps: 20/40/80 videos and 200+200+400=800 updates.
Stage 1 automatically generates same-seed J/L counterfactuals from three
training and three held-out first frames. Stage 2 and stage 3 need explicit evaluation approval;
flow loss alone cannot promote a stage. Only after eight-way control works do
we prepare a larger, independently approved multi-source set for about
1,000–2,000 further updates. That scale-up is not scheduled yet.
For the first gate, require opposite J/L background-motion signs on at least
two of three training first frames, visible scene stability, and a nonzero
camera response in fixed-noise validation. Held-out cases are reported
separately; a tiny-set overfit is not called generalization. At the next gate,
I/K must pass the same pitch-direction check while J/L remains correct.
At the all-eight gate, inspect each direction and slow/medium/fast amplitudes
before admitting additional sources. If a gate fails, hold the next stage and
compare camera/video alignment, LoRA capacity, and loss variants on the same
frozen examples.
When moving to the next stage, load the preceding stage's model weights with
`models.student.init_from_training_checkpoint`, set its new frozen
`sampling_index_path` and a new output directory, and initialize a fresh
optimizer. Do **not** use full `resume_from_checkpoint` across stages: the
trainer checkpoint includes a stateful dataloader whose prior sampling plan
must not be restored onto the expanded dataset. Keep earlier pairs in the
plan to limit forgetting. Do not mix these test scenes back into an independent
benchmark. For eventual scale-up, draw many more verified pairs from other
sources and rebalance camera amplitude/speed as well as direction.

The J/L selector labels each complete 124-frame clip by its *dominant*
measured yaw. It does not make all 124 frames pure J or L: 18 of the 20
selected clips also have an auxiliary detected direction, most often W,
A, or D. Their nominal yaw axis fractions range from 0.812 to 0.999
(mean 0.964). The exact PRoPE input is the measured per-latent-frame 6-DoF
camera matrix, including these residual components, and the video flow MSE
uses every latent frame uniformly. This preserves temporal continuity but
does not establish control of the other directions from this tiny set.
The stage is an isolated yaw-control diagnostic; if residual motions obscure
the sign, tighten the purity screen or add verified paired data before
interpreting it as eight-way learning.

The current stage sets `prope_use_camera_intrinsics: true`. Crop-adjusted K
and the measured C2W are fixed conditioning inputs, not trainable parameters.
For each latent frame, PRoPE builds `P = K @ W2C` and applies it in attention;
the intrinsics are not encoded as text and are not updated by the optimizer.
This matches the verified Cosmos-Predict2 recipe's `prope_use_ks=True` in the
sense of *using* a frozen per-example K, not learning K. The additive
`action_continuous` branch is disabled by `camera_conditioning: prope`.

For a quick visual readout after stage 1, the 2026-09-27 scheduler waits until
16:20 Asia/Shanghai **and** checkpoint-200/training completion, then uses
GPU0-7 in four two-rank sequence-parallel groups. Each group gets a distinct
first frame and exactly one synthetic J or L path (20 degrees over 124
frames, 50 denoising steps). Inference fixes K to the first-frame value
throughout each clip so the synthetic camera condition changes only in
extrinsics; training itself still used the measured per-latent-frame K.
Four train first frames are generated first;
then two JS-easy and two multicam held-out first frames. Case manifests are
`examples/train/js_easy_stage1_jl_{train,heldout}_cases.json`, and outputs
are under `runs/stage1_jl/single_action_eval_20260927/`. The evaluator
reads the complete DCP checkpoint rather than silently dropping the trained
LoRA/PRoPE state. The old same-first-frame J/L evaluator was disabled for
this scheduled run. Each scope writes four individual 124-frame videos, a
`contact_sheet.mp4`, and a `motion_proxy.json` with a coarse optical-flow
estimate. Inspect motion direction and temporal stability before
declaring action control; a 20-video memorization test cannot establish
generalization.

## Completion-gated preparation and launch

The bulk job produces VAE intermediates, **not** the fully conditioned H3
training Parquet. The dependency-gated launcher waits five minutes between
completion checks. After all eight workers finish, it starts the 65,389-row
full VAE audit in a separate low-priority CPU-only tmux session. In parallel,
it stages and audits the **exact frozen 80 IDs**: source frame indices, finite
video/keyframe latents, decodable first-frame PNG, intrinsics, and first-latent
C2W identity within 1e-4. Only this 80-row pass permits eight-GPU Qwen
first-frame conditioning, verification of 80 train rows and 48 existing
held-out validation rows, frozen parquet indices, then stage-1 J/L training.
Stage 1 may therefore run while the full audit continues. A full-audit
failure blocks expansion beyond the verified 80 rows; stage 2/3 require its
complete pass. Failed 80-row or conditioned-data gates prevent all training.
Source videos and historical H3 caches are not rewritten.
The PNG and keyframe latent are generated from the same decoded video tensor;
the audit does not claim a VAE can be inverted to prove pixel identity.

```bash
tmux new-session -d -s fastvideowm_js_easy_after_vae_20260926 \
  'bash /home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/launch_minimax_h3_js_easy_pose_after_vae.sh'
```

The 200-step J/L checkpoint and held-out J/L videos are automatic. After
review, run `examples/train/run_minimax_h3_js_easy_pose_stage.sh stage2_jlik`
or `stage3_all8`; each requires an explicit `*_EVAL_APPROVED` file. Use
`examples/train/evaluate_minimax_h3_js_easy_pose_stage.sh` for follow-up
held-out trajectory sweeps.

Manual equivalents of the data preparation steps:

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM
PLAN=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/js_easy_pose_overfit
MAP=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925/source_mapping.jsonl
TRAIN_IDS=/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/train_ids.jsonl
VAE=/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925/intermediate/all

.venv/bin/python examples/train/prepare_minimax_h3_js_easy_overfit.py \
  --source-mapping "$MAP" --train-ids "$TRAIN_IDS" --output-dir "$PLAN"
.venv/bin/python examples/train/stage_minimax_h3_js_easy_overfit.py \
  --plan "$PLAN/stage3_all8_ids.jsonl" --vae-input "$VAE" --output "$PLAN"

NUM_WORKERS=8 GPU_IDS=0,1,2,3,4,5,6,7 WORK_ROOT="$PLAN" \
  MANIFEST_PATH=/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925/manifests/approved_uncached.jsonl \
  CONDITIONING=prope \
  bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh condition train
WORK_ROOT="$PLAN" CONDITIONING=prope \
  bash examples/train/launch_minimax_h3_worldmodel_preprocess.sh verify train 80

.venv/bin/python examples/train/prepare_minimax_h3_js_easy_overfit.py \
  --source-mapping "$MAP" --train-ids "$TRAIN_IDS" --output-dir "$PLAN" \
  --parquet-data "$PLAN/preprocessed/train"
```

These 80 examples passed automatic GPU quality checks but are not a
human-approved production set. Inspect video/pose pairs and same-scene first
frames before claiming action control. The validation path uses the already
conditioned early-pilot held-out split; no training starts from merely
producing the frozen plan.
