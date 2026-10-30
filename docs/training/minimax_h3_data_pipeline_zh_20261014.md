# MiniMax-H3 / FastVideoWM 相机控制数据流水线（中文记录）

本文记录当前相机控制世界模型使用的数据从全集盘点、姿态筛选、动作和媒体质量筛选、GPU 视觉质检，到最终训练抽样的完整过程。源视频、源姿态和原始缓存均保持只读；所有 ID、相机派生文件、latent 和训练索引写入 `/datacc05/lizihao/worldmodel`。

## 1. 数据全集和审计口径

2026-09-24 的源数据盘点共包含 205,433 个可读视频窗口。四类源数据如下。

| 数据源 | 原始数量 | 根目录 |
|---|---:|---|
| `train_v2/records` | 189,380 | `/datacc05/datasets/vwm_datasets/train_v2/records` |
| `js_refine_data`（去重后） | 6,480 | `/home/lizihao/projects/worldmodel/train_data/js_refine_data` |
| `h3_js_easy` | 2,809 | `/home/lizihao/projects/worldmodel/vggt-omega/h3_js_easy` |
| MiniMax-H3 `mi` records | 6,764 | `/home/lizihao/projects/worldmodel/train_data/minimax-h3/records/minimax_h3/mi` |
| **合计** | **205,433** | |

每条记录至少绑定视频、姿态文件、来源子集、场景组和可用帧数。主盘点文件是 [`inventory_all.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/inventory_all.jsonl)，脚本是 [`audit_fastvideowm_source_inventory.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_source_inventory.py)。

## 2. 过滤流水线

### 2.1 姿态文件和相机矩阵检查

检查内容是姿态文件是否存在、矩阵是否为有限值的 `[T,4,4]` 齐次矩阵、帧数是否和视频对齐、旋转是否近似刚体。此关没有发现姿态缺失或非法的样本：`pose_rejected=0`。因此这一条件没有“被过滤掉的两个真实例子”，不能人为编造；对应空清单是 [`pose_rejected_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/pose_rejected_ids.jsonl)。

姿态通过后统一处理坐标系：VGGT 的 OpenCV Y-down 先转成 canonical X-right/Y-up/Z-forward；每个窗口用首帧 C2W 重设原点；内参先按原始处理分辨率归一化，再经过 resize/crop 变换。训练使用 I=俯仰向上、K=俯仰向下，J/L 为左右偏航。

### 2.2 过滤静止相机

对 rebased 后的相机轨迹计算平移和旋转有效幅度。没有足够相机运动的窗口不能提供相机控制监督，因此过滤出 47,737 条静止相机记录；其中约 34,000 条来自 `syncam_video`。静止视频没有作为隐式 `steady` 动作混入移动相机训练。

两个真实例子如下，拼图每行分别是首帧、中间帧、末帧：

![静止相机例子](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/static_camera.jpg)

- `train_v2:multicam_video:f18_aperture10__scene1000__cam10`
- `train_v2:sekai_real_walking_hq:BcoaLf1KF9E_0021750_0023550`

完整 ID 在 [`camera_static_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/camera_static_ids.jsonl)。

### 2.3 近静止或运动证据冲突，送人工复核

有些轨迹幅度接近静止，或者平移、旋转和动作证据互相矛盾。自动筛选不把这类记录直接判为可训练，累计 501 条进入复核清单（主审计目录和 `minimax_records` 子目录合并计算）。

![近静止或冲突例子](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/motion_review.jpg)

- `train_v2:multicam_video:f18_aperture10__scene1038__cam01`
- `train_v2:sekai_real_walking_hq:Tsj9ab1IYXo_0071151_0072951`

清单见 [`camera_motion_review_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/camera_motion_review_ids.jsonl)。

### 2.4 动作方向和幅度审计

对每条轨迹先计算相机局部坐标中的前后、左右、上下平移，以及 pitch/yaw/roll 旋转；只有方向分量稳定、符号一致、主轴占比足够时才生成 WASDIJKL 标签。一个视频可以同时包含多个方向，因此下表是“方向出现次数”，不是互斥的视频数。

在 CPU 候选 148,492 条中，方向出现次数为：

| W | A | S | D | I | J | K | L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47,073 | 45,109 | 38,310 | 44,411 | 26,810 | 49,840 | 49,853 | 49,514 |

平移/旋转的原始尺度不是物理米或米每秒。每个来源和方向分别用 33.3%/66.7% 分位点切分慢、中、快三档，再使用 `clip(样本每秒幅度 / 来源方向中位数, 0.25, 4.0)` 得到学习尺度因子。尺度因子写入 [`action_scale_provisional.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/action_scale_provisional.jsonl)，并折叠进相机平移；训练配置保持 `normalize_camera_translation=false`，以免再次归一化抹掉速度差异。

### 2.5 CPU 视频和轨迹质量筛选

脚本 [`audit_fastvideowm_cpu_quality.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_cpu_quality.py) 从每个视频抽取 7 帧，检查解码、时长、亮度、对比度、Laplace 清晰度、近乎静止帧、平移速度突变、突然旋转和动作标签一致性。窗口少于 4.5 秒时不足以稳定构成 124 帧训练样本，也会被拒绝。

结果是：157,195 条进入 CPU 媒体/轨迹筛选，其中 148,492 条候选、5,985 条复核、2,718 条拒绝。主要拒绝原因包括：无稳定八方向分量 2,216 条、短于 4.5 秒 423 条、极端亮度 47 条、极端亮度且近乎恒定 32 条。主要复核原因包括：突发旋转 3,568 条、命令与姿态不一致 2,127 条、预留验证 136 条、疑似模糊/低纹理 85 条、平移速度突变 40 条，另有组合原因 29 条。

下面每张图都含两个真实例子。

![时长不足](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_reject_duration.jpg)

时长不足：`js_refine_data:js_test/js_00000_tank.png.doubao-seedance-2.0-fast_1`、`train_v2:spatialvid:group_0001__00094653-a9c6-5558-8e2a-4119e7d64f36`。

![动作不稳定](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_reject_action.jpg)

无稳定八方向分量：`h3_js_easy:fasth3_t2v_js_1000_124_0033_W`、`h3_js_easy:fasth3_t2v_js_1000_long_0006_SL`。

![亮度异常](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_reject_brightness.jpg)

极端亮度：`train_v2:multicam_video:f18_aperture10__scene1299__cam09`、`train_v2:multicam_video:f18_aperture10__scene1339__cam05`。

![突发旋转](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_review_rotation.jpg)

突发旋转复核：`h3_js_easy:fasth3_t2v_js_1000_long_0496_DK`、`h3_js_easy:h3_i2v_js_easy_124_js_00002_J_fast_camera_only`。

![命令与姿态不一致](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_review_pose_mismatch.jpg)

命令与姿态不一致：`js_refine_data:ltx2p3_xzz/...rear_seed02`、`js_refine_data:ltx2p3_xzz/...rear_seed04`。完整 ID 和原视频路径以 [`cpu_review_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/combined_cpu_screen_complete/cpu_review_ids.jsonl) 为准。

![模糊或速度突变](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/cpu_review_quality_motion.jpg)

疑似模糊/低纹理：`train_v2:multicam_video:f18_aperture10__scene1599__cam08`；平移速度突变：`h3_js_easy:fasth3_t2v_js_1000_124_0297_W`。

### 2.6 GPU-QC：模糊、纹理崩坏、主体变形、抖动和姿态可见性

对 CPU 候选按 SHA-256 分成 8 个 shard，每条确定性 124 帧窗口抽取 6 帧，使用 Qwen3-VL 检查：模糊、纹理崩坏、主体变形、平移抖动，以及画面中可见的相机运动是否与姿态方向相符。任一项为真、`no`、`uncertain` 或解析失败都进入复核。

148,492 条中，GPU-QC 通过 76,780 条，复核 71,712 条。复核原因计数为：可见运动不确定 55,777、GPU 不确定 55,777、可见运动为否 13,770、模糊 3,994、平移抖动 223、主体变形 5、纹理崩坏 4、解析失败或未解析 217。GPU 通过仍标记 `quality_approved=false`，它是自动严格筛选结果，不能替代人工抽查。

![GPU 模糊](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/gpu_review_blur.jpg)

模糊：`h3_js_easy:fasth3_t2v_js_1000_124_0233_W`、`h3_js_easy:fasth3_t2v_js_1000_124_0234_S`。

![GPU 主体变形](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/gpu_review_deformation.jpg)

主体变形：`train_v2:multicam_video:f18_aperture10__scene1060__cam09`、`train_v2:multicam_video:f24_aperture5__scene1059__cam09`。

![GPU 纹理或抖动](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/gpu_review_texture_jitter.jpg)

纹理崩坏/平移抖动：`train_v2:multicam_video:f35_aperture2.4__scene784__cam08`、`h3_js_easy:fasth3_t2v_js_1000_124_0557_J`。

![GPU 可见运动不匹配](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/gpu_review_motion.jpg)

可见运动为否或不确定：`h3_js_easy:fasth3_t2v_js_1000_124_0002_S`、`h3_js_easy:fasth3_t2v_js_1000_124_0003_A`。逐条结果见 [`gpu_qc_review_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/gpu_qc_review_ids.jsonl)，通过清单见 [`gpu_qc_pass_ids.jsonl`](/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/gpu_qc_pass_ids.jsonl)。

![GPU 通过样例](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/data_pipeline_examples_zh_20261014/sheets/gpu_pass.jpg)

通过样例：`h3_js_easy:fasth3_t2v_js_1000_124_0001_W`、`js_refine_data:ltx2p3_xzz/...m1a2_abrams_berm_climb_direct_front_seed01`。这些样例说明筛选保留了可见运动和画面质量均通过的样本，但不表示所有通过样本已经人工逐条验收。

### 2.7 场景划分和当前训练池

GPU 通过的 76,780 条按场景组划分为 73,671 条训练、3,109 条验证；训练包含 10,604 个场景组，验证保留 542 个场景组，避免同一场景的不同窗口泄漏到两边。完整结果见 [`fastvideowm_action_only_v1_20260925/report.json`](/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925/report.json)。

其中 65,389 条没有可直接复用的旧 H3 latent，进入新的 VAE 编码池；11,391 条有旧缓存，计划单独审计转换。当前 `multicam5k` 训练计划在 65,389 条源缓存上固定抽样：multicam 可用 52,101 条中取 5,000 条，其他来源取 10,719 条，总共 15,719 条唯一记录；为整除 batch 补到 15,744 次抽样。计划报告是 [`multicam5k_plan/report.json`](/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/multicam5k_plan_20261008/report.json)。

当前 15,719 条的来源分布：

| 来源 | 数量 |
|---|---:|
| `train_v2/context` | 7,407 |
| `train_v2/multicam_video` | 5,000 |
| `train_v2/ue_easy` | 359 |
| `train_v2/game_factory` | 76 |
| `train_v2/spatialvid` | 1 |
| H3 JS-easy 四个子集 | 772 |
| MiniMax-H3 `mi` 五个子集 | 2,104 |
| **合计** | **15,719** |

该计划是多标签窗口：方向不是互斥类别。方向出现次数为 W/A/S/D/I/J/K/L = 5,569/4,292/4,345/4,174/2,543/5,420/2,898/5,163；慢、中、快三档已经写入 `direction_speed_occurrences`，但由于原始数据分布差异，I/K 的非 multicam 数量偏少，不能把这张表解释成完全均衡的纯动作数据。

训练记录保留 124 帧、480×832、24 FPS，对应 H3 视频 latent `[24,37,30,52]`，相机条件是 37 个 latent 对齐的 C2W 加归一化内参。当前 action-only 配方使用首帧 H3 keyframe latent 和首帧 Qwen vision tokens，去掉场景文本 token；没有音频输入，H3 预训练音频模块保留但冻结。当前训练不使用“相机控制文本”替代数值轨迹。

## 3. 训练前最后检查

VAE 完成标记不能单独代表可训练。正式开训前需要同时通过完整 VAE、首帧、内参和 Parquet schema 审计；首帧条件和 keyframe latent 必须由同一首帧产生；相机首帧必须接近单位阵；内参必须在 resize/crop 后仍为有效归一化值。GPU-QC 是自动门禁，最终研究结论仍需看相机方向、幅度、响应延迟和视频质量。

主要脚本包括 [`audit_fastvideowm_source_inventory.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_source_inventory.py)、[`audit_fastvideowm_actions.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_actions.py)、[`audit_fastvideowm_cpu_quality.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/audit_fastvideowm_cpu_quality.py)、[`prepare_fastvideowm_full_gpu_qc.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/prepare_fastvideowm_full_gpu_qc.py)、[`finalize_fastvideowm_full_gpu_qc.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/finalize_fastvideowm_full_gpu_qc.py) 和 [`build_fastvideowm_approved_uncached_dataset.py`](/home/lizihao/projects/worldmodel/H3-World/code/vwm/build_fastvideowm_approved_uncached_dataset.py)。
