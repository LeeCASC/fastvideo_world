# H3-World 与 FastVideoWM 数据、训练和推理总报告

更新时间：2026-10-21（Asia/Shanghai）。本报告按磁盘上的注册表、完成标记、Parquet 审计、训练日志和输出目录整理。自动 GPU 质检表示程序通过，不等同于人工视觉验收。

## 当前运行状态

服务器断开后，FastVideoWM 已在 tmux `fastvideowm_union_train_20261019` 重新启动，使用 GPU 0–7，训练数据为 115,988 条 union 数据，目标 step 为 3,000，已从完整的 `checkpoint-200` 恢复，目前约 step 438，完整的 `checkpoint-400` 已保存。配置为 micro-batch 1、gradient accumulation 8、8 卡，total/effective batch size = 32。最近一次启动日志为：

`/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/runs/train/logs/minimax_h3_union_20261019_20261020_185513.log`

恢复入口脚本为 [`launch_fastvideowm_union_115988_20261019.sh`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/tools/launch_fastvideowm_union_115988_20261019.sh)。旧训练在 step 208 左右因 358 条新缓存记录中存在超过 512 token 的文本条件而触发长度一致性异常；已在 [`worldmodel.py`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py) 增加安全裁剪：embedding 被 H3 512 token 上限裁剪时同步裁剪 tags，tags 更短则继续报错。旧 checkpoint-200 保留，未覆盖。

此前实测约 209 秒/step；恢复后稳定速度约 208 秒/step，按约 2,562 个剩余 optimizer steps 粗估还需约 148 小时，另加每 100 step 验证时间。每 100 step 保存一次状态 checkpoint，验证 callback 每 100 step 使用 12 个 validation batches。

## 数据版本和目录

### 当前 FastVideoWM union（实际用于训练）

训练视图：[`/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/preprocessed/all`](/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/preprocessed/all)

|组成|数量|说明|
|---|---:|---|
|旧 H3 条件缓存|65,389|来自 [`fastvideowm_full_65389_20260927`](/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927)，已含 H3 VAE、首帧条件、相机数据和条件 embedding|
|本轮新增 124f 条件缓存|50,599|来自 124f strict GPU-QC、Qwen3-VL、MiniMax-H3 VAE 后的 conditioned Parquet|
|合计|115,988|ID 唯一 115,988 条，旧、新缓存通过 symlink 组成训练视图|

注册表和审计：[`data_summary.json`](/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/data_summary.json)、[`training_files.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/training_files.jsonl)。union 的来源统计为：multicam_video 56,402、spatialvid 48,041、context 7,407、mi_main 1,378、ue_easy 556、mi_v6 447、fasth3_t2v_js_1000_long 376、h3_i2v_js_easy_124 294、mi_v3 175、minmax 148、ue_stage3 133、ue_stage1 125、mi_v5 118、mi_v4 116、fasth3_t2v_js_1000_124 97、game_factory 80、h3_i2v_js_easy_long 62、ue_game 20、ltx 7、sekai_real_drone 6。`data_summary.json` 是来源归一化后的权威统计。

### 124 帧统一数据

构建根目录：[`vwm_unified_124_243_20261016/124f`](/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f)。目标 124 RGB 帧、24 fps，H3 latent 时间轴 37 帧。完整注册表的候选窗口为 359,165，结构失败 98,646；CPU provisional pass 335,066。随后 GPU-QC 结果文件位于 `124f/gpu_qc/results/`，选择分片位于 `124f/gpu_qc/selections/`，并写有 `GPU_QC_COMPLETE`。本轮实际进入 VAE/条件化 union 的 strict 子集为 50,599 条。

124f 的源分布（注册表候选）为：spatialvid 294,669、multicam_video 54,462、ue_easy 1,620、ue_game_video 518、minmax 3,306、mi_main 1,378、mi_v6 447、mi_v3 175、mi_v5 118、mi_v4 116、ue_stage1 404、ue_stage3 399、ltx 201、sekai_real_drone 453、sekai_real_walking_hq 70，以及 H3 JS-easy 四个版本共 829。注册表摘要：[`registry_summary.json`](/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/reports/registry_summary.json)。

### 243 帧统一数据

构建根目录：[`vwm_unified_124_243_20261016/243f`](/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/243f)。目标 243 RGB 帧、约 10.1 秒（24 fps），长视频由完整 manifest 重新切窗得到。候选窗口 224,670，结构失败 233,141，CPU provisional pass 211,654；GPU-QC 四个选择分片约为 52,914、52,914、52,913、52,913。其 registry summary、CPU quality 和 GPU selection 文件已落盘，但当前 FastVideoWM union 训练只使用 124f；243f 尚未进入这个训练视图，也没有被错误地当作 124f 使用。243f 的 VLM 和 MiniMax-H3 VAE 应在单独完成对应标记后再进入长视频训练。

历史的 `vwm_unified_121_241_20261016` 是错误规格（121/241 帧）的中间结果，只保留作历史审计，不能作为最终 124/243 数据计数。

### 61 帧数据

61f 根目录：[`vwm_unified_61f_20261015`](/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015)，ready registry 为 `registry/windows_61_ready_for_vlm.jsonl`，约 130,236 条。它用于短窗口试验和另一个 VAE 方案，和当前 H3 124f union 训练不是同一训练视图；完整 61f MiniMax-H3 latent 在早期快照中尚未作为当前 union 的输入。61f 的首帧、poses、VLM 和 latent 均应按各自完成标记核对后再训练。

## 过滤和数据处理流程

所有原始视频和 pose 目录保持只读，派生 registry、pose 修复、QC、VLM、latent 和 Parquet 均写在 `/datacc05/lizihao/worldmodel`。

1. **全集登记与窗口构建。** 合并 train_v2、Cosmos `vwm_unified/records`、H3 JS-easy、MiniMax-H3 mi、ltx 等来源；为 124f 和 243f 分别建立独立窗口、首帧、37/对应 latent 对齐的相机轨迹记录。长视频按窗口取帧，未把 121/241 的历史规格混入最终版本。

2. **CPU 结构和媒体质量。** 检查可读视频、帧数、FPS、目标窗口完整性、pose/K 是否存在、首帧与窗口帧索引是否对齐，删除无法解码或结构不完整的窗口。随后检查亮度、对比度、清晰度、冻结帧、异常跳帧、全局抖动和相机动作一致性。相机不动或近似静止的窗口进入 rejected/review；历史全量审计中约 34,000 条 syncam_video 因静止相机被排除。CPU 结果保存在 `124f/reports/cpu_quality.jsonl`、`243f/reports/cpu_quality.jsonl` 及各自 `CPU_QUALITY_COMPLETE`。

3. **GPU-QC。** 对候选窗口抽取多帧，检查模糊、纹理崩坏、主体变形、平移抖动、相邻帧异常和画面运动与 pose 是否一致。GPU-QC 是自动筛选与 advisory 标记，`pass/review/reject` 仍需结合 registry 和人工抽查。124f/243f 的结果分别位于 `*/gpu_qc/results/` 与 `*/gpu_qc/selections/`；124f 已完成并写入 `GPU_QC_COMPLETE`，243f 的选择结果已生成，是否完成最终严格入选以对应报告和标记为准。

4. **相机几何修复。** 每个窗口的 C2W 以窗口首个 latent 帧重设为单位矩阵，旋转转换为右手坐标并检查正交性，平移保留审计后的尺度。JS-easy 的 VGGT K 原本是 672×384 像素坐标，而视频是 1344×768；处理时先归一化 K，再进行 resize/crop。K 保存为数值矩阵；当前 FastVideoWM active config 保存并审计 K，但 `prope_use_camera_intrinsics=false`，实际控制主要使用 first-frame-relative C2W。

5. **Qwen3-VL 场景描述。** 对 124f strict 50,599 条生成了 `vlm_qwen3vl_strict_50599/` 下的 registry 和四个 captions shard；描述主体外观、场景和可见主体状态，删除相机运动措辞，将相机轨迹保存在数值 pose 字段。VLM 描述被保存在 Parquet schema 的 `caption`/embedding 字段中，但当前训练配置 `vision_only_conditioning=true`，DiT 前向只使用首帧视觉 token，不把场景文本作为控制信号。

6. **MiniMax-H3 VAE。** 124f strict 的新 latent 分两批保存：

   - `124f/vae_latents/minimax_h3_prope/`：第一阶段 7,408 条；
   - `124f/vae_latents/minimax_h3_prope_phase2_8gpu/`：第二阶段 43,191 条；
   - 合计 50,599 条，并进一步经过 H3 conditioner 形成 `fastvideowm_union_115988_20261019/new_conditioned/` 的 6,326 个 Parquet 文件。

   每条训练记录含 `[24,37,30,52]` 视频 latent、`[24,1,30,52]` 首帧 latent、首帧 PNG/视觉条件、37 个 latent 对齐 C2W、K、action continuous 和 valid mask。音频模块仍存在于预训练模型，但当前 forward 传入零长度 audio token，未用音频监督。

## H3-World 与 FastVideoWM 的差异

H3-World 位于 [`/home/lizihao/projects/worldmodel/H3-World`](/home/lizihao/projects/worldmodel/H3-World)，历史训练以 H3 DiffSynth 的 latent/text action prompt 为主，checkpoint 常见为 `state-xxxxxx` 或 safetensors；相机指令通常通过文本描述注入。FastVideoWM 位于 [`/home/lizihao/projects/military_world_model_agent/FastVideoWM`](/home/lizihao/projects/military_world_model_agent/FastVideoWM)，当前使用 MiniMax-H3 的 numeric PRoPE：37 帧 C2W/相机动作逐 latent 注入，首帧视觉条件进入 H3，scene text 仅保留在 schema 中。两者不能直接混用缓存或把 H3 文本标签当作 FastVideoWM 的 numeric pose。

FastVideoWM 当前方向约定为 W/S 前后、A/D 左右、I 上仰、K 下俯、J 左转、L 右转；历史 H3 文本缓存曾有 I/K 相反的遗留约定，评估时使用对应实验的约定。

## FastVideoWM 实验、checkpoint 和日志

当前 union 实验：

`/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/`

配置为 [`minimax_h3_union_20261019.yaml`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_union_20261019.yaml)，checkpoint 在 `runs/train/checkpoints/checkpoint-100`、`checkpoint-200`，训练日志在 `runs/train/logs/`，W&B 离线记录在 `runs/train/checkpoints/tracker/wandb/`。

主要历史实验根目录如下，目录中的 `runs/train/checkpoints`、`runs/.../logs`、`status.json`、`metadata.json`、TensorBoard/W&B 文件分别记录权重、状态、loss 和配置：

```text
/datacc05/lizihao/worldmodel/fastvideowm_action_only_early_20260926
/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925
/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927
/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925
/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003
/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008
/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002
/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015
/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261015_4gpu
/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_text_20261016_8gpu
/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002
/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002
/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930
/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001
/datacc05/lizihao/worldmodel/fastvideowm_camera_pair_night_20260930
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step5000_20261012
/datacc05/lizihao/worldmodel/fastvideowm_gpu7_pilot_20260924
/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924
/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925
/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925
```

动作诊断试验的 100-step checkpoint 在 `fastvideowm_action_only_early_20260926/runs/pilot/checkpoints/`，对应推理和分析在 [`fastvideowm_action_only_step100_eval_20260926`](/datacc05/lizihao/worldmodel/fastvideowm_action_only_step100_eval_20260926)。JS-easy 固定评估的 1200/1500/5000 目录分别为上述 `fastvideowm_js_easy16_*`。

## H3-World 实验、checkpoint 和日志

H3-World 的 baseline 权重为 [`step-10000.safetensors`](/home/lizihao/projects/worldmodel/H3-World/checkpoints/H3-World/step-10000.safetensors)，主要输出根目录为 [`/datacc05/lizihao/outputs/h3_world`](/datacc05/lizihao/outputs/h3_world)。训练实验包括：

```text
vwm_train_v2/lora_from_step_10000
vwm_train_v2/lora_from_step_10000_bs32_2000iters
vwm_train_v2/lora_from_step_10000_bs32_2000iters_pre_instruction_probe_attempt
vwm_filtered_v1/from_step2000_plus1000_bs32_gpu4_7_20260917
vwm_filtered_v1/from_step2000_plus1000_bs32_gpu0_7_20260918
vwm_filtered_v1/from_step3000_to4000_bs32_gpu0_7_20260920
vwm_filtered_v1/from_step3000_to4000_bs32_gpu4_7_20260920
vwm_filtered_v1/from_internal1800_to3000_bs32_gpu4_7_20260921
vwm_filtered_v1/from_internal1800_to3000_bs32_gpu0_7_final_20260922
vwm_filtered_v1/from_internal1950_to3000_bs30_gpu0_5_20260922
vwm_filtered_v1/from_internal2100_to3000_bs32_gpu0_7_20260922
vwm_filtered_v1/from_step2450_legacy_plus_captioned_v2_bs32_gpu0_7_3000steps_20260923
vwm_filtered_v1/from_step2450_legacy_plus_captioned_v2_bs35_gpu0_6_to5000_20260924
vwm_filtered_v1/train_v2_plus_js_26708_from_step10000_bs32_2000iters
vwm_filtered_v1/train_v2_plus_js_26708_from_step10000_bs35_gpu0_6_2000iters
vwm_qwen_pitch_v2/from_v1_step2000_bs35_gpu0_6_2000iters
simple_action_stage1/from_step10000_no_steady_balanced_bs35_gpu0_6_200iters
jl_pair_curriculum_20260916/stage1_jl_50_bs35_fp32
jl_pair_curriculum_20260916/stage1_jl_extend_to100_bs35_fp32
```

每个训练根目录中应优先查看 `train.log`、`status.json`、`run_configuration.json`、`tensorboard*`/`tensorboard_log`，以及 `state-xxxxxx` 或 checkpoint 子目录。H3 训练数据、过滤 registry 和 prompt 的历史汇总位于 `vwm_train_v2/manifests/`、`vwm_train_v2/logs/` 和 `vwm_filtered_v1/` 对应运行目录。

## H3-World 和 FastVideoWM 推理结果

H3-World 的主要推理结果根目录：

```text
/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_baseline_vs_ft2000_20260917
/datacc05/lizihao/outputs/h3_world/js_easy_first_wasdijkl_baseline_vs_ft2000_20260917
/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_step2450_vs_latest_20260924
/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009
/datacc05/lizihao/outputs/h3_world/js_easy_subset_030407080910_step600_20260924
/datacc05/lizihao/outputs/h3_world/js_easy_subset_0407080910_step600_20260924
/datacc05/lizihao/outputs/h3_world/custom_wasdijkl_selected_20260924
/datacc05/lizihao/outputs/h3_world/step2000_advantage_showcase_20260917
/datacc05/lizihao/outputs/h3_world/step2000_review_v2_20260917
/datacc05/lizihao/outputs/h3_world/vwm_train_v2/inference
/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference
```

其中 `js_easy17_wasdijkl_baseline_vs_ft2000_20260917` 含 `baseline_50steps`、`finetune2000_50steps`、`finetune3000_50steps`、`testset` 和 `comparison`；`js_easy17_h3_latest_vs_fastvideowm_latest_20261009` 含 H3-World 与 FastVideoWM 的同首帧、同 seed 对照视频；`vwm_train_v2/inference` 含 step-350 benchmark、20-step camera compare 和 smoke 结果；`vwm_filtered_v1/inference` 含 step-1650 eval、16-case control validation、action ablation 和多数据集 single-action 结果。

FastVideoWM 的主要推理目录为：

```text
/datacc05/lizihao/worldmodel/fastvideowm_action_only_step100_eval_20260926
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013
/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step5000_20261012
/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm
```

固定评估遵循 16-case JS-easy skill：首帧图像、GT pose 轨迹、单指令 W/A/S/D/I/J/K/L 对照，124 帧、固定 seed，并比较正确/静止/反向轨迹。结果通常包含 `videos/`、`metrics.json`、`conditioning/`、`shards/` 和分析 markdown。H3 baseline 与旧 H3 微调的大规模 WASDIJKL 结果集中在 `js_easy17_*baseline_vs_ft2000*`；FastVideoWM 最新对照集中在 `js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm`。

## 当前可复现入口

FastVideoWM 训练使用 [`minimax_h3_union_20261019.yaml`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_union_20261019.yaml) 与 [`launch_fastvideowm_union_115988_20261019.sh`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/tools/launch_fastvideowm_union_115988_20261019.sh)。数据总表和逐文件清单分别是 `data_summary.json`、`training_files.jsonl`。JS-easy 评估遵循 [`fastvideowm-js-easy-eval/SKILL.md`](/home/lizihao/.codex/skills/fastvideowm-js-easy-eval/SKILL.md) 和仓库中的 `docs/training/js_easy_camera_test_pipeline_zh_20261014.md`。

