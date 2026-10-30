# MiniMax-H3 action-only first training run

The first run trains camera control from the released MiniMax-H3 checkpoint.
It uses the native H3 keyframe VAE condition and Qwen3-VL **first-frame image**
tokens, while removing Qwen **text** tokens from the DiT input. The per-video
scene captions and the generic placeholder prompt therefore do not participate
in training. Audio remains absent. At inference, use the same YAML config so
the identical first-frame vision-only policy is applied.

## Frozen data and sampling

The strict automatic GPU-QC list contains 76,780 windows. The scene-held-out
plan is under:

```text
/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/
```

`train_ids.jsonl` contains 73,671 windows and `validation_ids.jsonl` contains
3,109. This split uses a stable hash of the source scene/recording group. It
holds out 542 groups and checks that no group occurs in both splits. Multi
action windows are retained in this full plan, but the first pilot uses only
single-action windows without a prior cache: 256 training and 8 validation
windows for each of W/A/S/D/I/J/K/L, totaling 2,048 + 64. The exact IDs are
in `pilot_train_ids.jsonl` and `pilot_validation_ids.jsonl`. The pilot is a
diagnostic run; a source-stratified manual spot check is still needed before
calling the entire 76,780-row set quality approved.

The source videos, source camera files, and old H3 caches are never edited.
The ongoing eight-GPU VAE job writes 65,389 previously uncached strict-pass
windows. The other 11,391 strict-pass windows have old H3 caches and are
planned for a separate audited conversion before full-scale training. The
first pilot deliberately uses one encoding path and balanced single actions.

## Action amplitude

Each source camera path is rebased, transformed to Y-up if required,
normalized once by its maximum translation norm, then multiplied by its
audited X/Y/Z factor in `[0.25, 4]`. These factors are relative to their
source and direction, not physical meters per second. Rotation remains the
measured pose angle. Each final PRoPE record includes latent-frame C2W and
intrinsics.

The pilot YAML sets `normalize_camera_translation: false`. The hybrid model
derives both cumulative six-DoF action features and PRoPE matrices from those
calibrated C2W matrices. The action adapter uses channel scales
`[1, 1, 0.25, 0.1, 0.1, 0.05]` and clips normalized values at 4. Those are
model input normalization constants, not the per-video speed labels. To
request a different speed at inference, scale the translation increments of a
camera trajectory before building the C2W path. Do not rescale an old
checkpoint's camera inputs without its corresponding training config.

## Prepared run

The following scripts are ready:

- `examples/train/prepare_minimax_h3_action_only_v1.py`: freeze the GPU pass
  IDs, scene split, action/speed statistics, and balanced pilot IDs.
- `examples/train/stage_minimax_h3_action_only_pilot.py`: select the pilot's
  exact VAE rows from completed 65,389-row intermediate Parquet.
- `examples/train/verify_minimax_h3_action_only_pilot.py`: verify exact IDs,
  schema, latent shapes/finiteness, first-frame vision tokens, calibrated
  cameras, and no audio fields.
- `examples/train/launch_minimax_h3_action_only_v1_pilot_after_vae.sh`: wait
  for the VAE completion marker, stage the pilot, encode Qwen first-frame
  conditions on eight GPUs, verify both splits, then start the 100-step run.
- `examples/train/configs/minimax_h3_worldmodel_action_only_v1_pilot.yaml`:
  official H3 initialization, eight A100 GPUs, sequence parallel size 2,
  data parallel size 4, accumulation 8, total batch size 32.

The model trains rank-16 Q/K/V/out LoRA, the action adapter, selected video
input/output layers, and the configured camera transform path. It saves and
validates at steps 25, 50, 75, and 100. Validation reports flow loss,
correct-versus-static/shuffled action loss, and response size. Those metrics
test conditioning sensitivity; they do not establish visible instruction
following. After the pilot, evaluate generated videos from held-out first
frames with matched seeds, static/reverse paths, all eight directions, and
slow/medium/fast trajectory scales before increasing the dataset.

The checkpoint evaluator now accepts `--trajectories` and
`--trajectory-scale`. With eight GPUs, run four directions per invocation,
for example `--trajectories W A S D --trajectory-scale 0.5`, then
`--trajectories I J K L --trajectory-scale 1.0`. Test scales 0.5, 1, and 2
with the same held-out first frame and seed. The C2W trajectory, not a text
string such as "move fast", carries the scale.

The 100-step pilot is the automatic training stage. Subsequent steps are
deliberately gated on the above held-out videos and loss curves. If it works,
condition the rest of the VAE data, convert the 11,391 old caches using the
audited indexed converter, construct a repeat-aware row index for minority
direction/speed buckets, and continue from the pilot checkpoint on the full
scene-held-out split. Keep the same vision-only and scale-preserving settings.
