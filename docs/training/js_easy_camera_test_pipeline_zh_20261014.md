# JS-easy 相机控制测试集和测试流程（中文记录）

测试包位于 [`js_easy_camera_eval_suite_20261013_v2`](/datacc05/lizihao/worldmodel/js_easy_camera_eval_suite_20261013_v2)，其 [`SKILL.md`](/datacc05/lizihao/worldmodel/js_easy_camera_eval_suite_20261013_v2/SKILL.md) 是每次复用测试时的机器可读规范。测试包与模型无关：提供首帧 PNG 和完整的 24 FPS 相机轨迹，H3、FastVideoWM 或其他 AR/世界模型自行选择合法时间窗口。

## 1. 测试集组成

`manifest.jsonl` 共 48 个用例，分成三组：

| 组 | 数量 | 帧数 | 时长 | 是否有匹配 GT | 用途 |
|---|---:|---:|---:|---|---|
| `short_simple` | 16 | 124 | 5.167 秒 | 有原始 benchmark 视频 | H3 固定窗口、可做对齐后的 GT 指标和动作对照 |
| `single_15s` | 16 | 360 | 15 秒 | 无 | 长时间单一动作、漂移和稳定性 |
| `multi_60s` | 16 | 1,440 | 60 秒 | 无 | 四段动作切换、切换延迟、长程漂移和窗口接缝 |

每一类都覆盖 `js_00001` 到 `js_00016`，每个首帧提供一个 PNG。每条轨迹同时保存 `camera_to_world`、`world_to_camera`、24 FPS 时间戳、逐帧 action label、局部控制量和归一化内参。相机坐标是 C2W 列向量约定，X 向右、Y 向上、Z 向前，首帧为单位阵。

短视频的单动作映射为：

| 样本 | 动作 | 样本 | 动作 |
|---|---|---|---|
| `js_00001–02` | W 前进 | `js_00009–10` | I 俯仰向上 |
| `js_00003–04` | A 左移 | `js_00011–12` | J 左转 |
| `js_00005–06` | S 后退 | `js_00013–14` | K 俯仰向下 |
| `js_00007–08` | D 右移 | `js_00015–16` | L 右转 |

15 秒用例使用相同的两样本/方向分配，帧数是 360。60 秒用例每 15 秒切换一次动作，每个方向有两条四段序列：

| 样本 | 四段动作 |
|---|---|
| `js_00001–02` | W → J → S → L |
| `js_00003–04` | A → I → D → K |
| `js_00005–06` | S → L → W → J |
| `js_00007–08` | D → K → A → I |
| `js_00009–10` | I → W → K → S |
| `js_00011–12` | J → A → L → D |
| `js_00013–14` | K → S → I → W |
| `js_00015–16` | L → D → J → A |

## 2. 输入文件和几何约定

目录结构如下：

```text
js_easy_camera_eval_suite_20261013_v2/
├── manifest.jsonl
├── images/js_00001..js_00016/first_frame.png
├── short/cases/<case>/trajectory.npz
├── short/gt/js_00001..js_00016.mp4
└── long/cases/<case>/{metadata.json,trajectory.npz,trajectory.jsonl}
```

模型适配器需要读取 `first_frame.png` 和完整 `trajectory.npz`。若模型只支持局部时间窗口，应从全局轨迹截取窗口，并使用 `inv(C2W[start]) @ C2W[window]` 进行窗口首帧重基；内参必须在每次 resize/crop 后同步变换。测试包本身不要求文本 prompt、音频或预先生成的 latent。

FastVideoWM 的 H3 适配会将 124 RGB 帧对齐到 37 个 latent anchor，视频尺寸为 480×832；H3 的内参来自原始像素标定，先归一化再经过目标 resize/crop。I/K 语义固定为 I 上转、K 下转，不能套用早期 H3 文本缓存中的旧 I/K 标签。

## 3. H3/FastVideoWM 测试流程

第一步固定输入和随机性：每个 checkpoint 使用同一首帧、同一 trajectory、同一 seed、同一采样步数和同一 resize 规则。短测试分成两条 lane：

1. GT lane 使用原始 61 帧 benchmark 视频按等进度映射到 124 帧，并使用对应的 GT 相机轨迹。
2. single-action lane 使用同一首帧和人为构造的单一 W/A/S/D/I/J/K/L 轨迹，测试模型能否响应指定方向；这条 lane 不把原始复杂 GT 当作单一动作真值。

H3/FastVideoWM 的准备和推理脚本是 [`evaluate_minimax_h3_js_easy16_step1200_20261012.sh`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/evaluate_minimax_h3_js_easy16_step1200_20261012.sh)、[`prepare_minimax_h3_js_easy16_gt.py`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/scripts/evaluation/prepare_minimax_h3_js_easy16_gt.py)、[`infer_minimax_h3_worldmodel_checkpoint.py`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py) 和 [`score_minimax_h3_js_easy16.py`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/scripts/evaluation/score_minimax_h3_js_easy16.py)。脚本将 16 个样本分成 4 个八卡 lane，生成 GT 和 single-action 两组视频，并逐 lane 校验帧数、分辨率、文件完整性。

示例启动方式：

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM
CHECKPOINT=/path/to/checkpoint \
OUT=/datacc05/lizihao/worldmodel/eval/js_easy16_<label> \
bash examples/train/evaluate_minimax_h3_js_easy16_step1200_20261012.sh
```

对于 15 秒和 60 秒用例，AR/长时模型直接读取 360/1,440 帧轨迹；H3 适配器可以按自身窗口长度滑窗。每个窗口报告动作方向、响应延迟、幅度、方向反转、漂移、时间一致性和窗口接缝。长测试没有匹配 GT，不能报告有意义的 PSNR、SSIM 或 LPIPS。

## 4. 指标和判定方式

短 GT lane 可以在明确的 61→124 等进度对齐后计算 PSNR、SSIM、LPIPS 和 flow loss；评分代码会排除首帧条件图像，避免首帧复制造成虚高。single-action lane 和长时 lane 的核心是相机控制几何：

- 请求方向与估计相机运动方向的符号是否一致；
- 平移/旋转响应的幅度是否随轨迹尺度变化；
- 响应从第几帧开始，是否出现反向或混入其他轴动作；
- 长时轨迹是否漂移、突然跳变或在窗口拼接处不连续；
- 画面是否出现闪烁、纹理崩坏、主体变形和质量下降。

VBench 的主体一致性、背景一致性、动态程度、运动平滑度和成像质量可作为无 GT 参考，但它们不能单独证明动作方向正确。方向结论必须结合轨迹和生成视频的相机运动审查。

## 5. 已有结果和保存位置

此前的 16 样本测试已经完成：

- step-1200：[`fastvideowm_js_easy16_step1200_20261012`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012)，包含 `INFERENCE_COMPLETE`、`CORE_METRICS_COMPLETE`、`metrics.json`、GT/single 视频和四个 lane 日志；
- step-1500：[`fastvideowm_js_easy16_step1500_20261013`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013)，结构与 step-1200 相同；
- H3 较早的 16 样本结果：[`fastvideowm_full_residual_prope_20261003/evaluation`](/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/evaluation)，其中包含 JS-easy benchmark 的推理和分析产物。

当前 5000-step 训练结束后，自动评测脚本 [`auto_evaluate_minimax_h3_js_easy16_after_5000_20261012.sh`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/auto_evaluate_minimax_h3_js_easy16_after_5000_20261012.sh) 会等待 `checkpoint-5000` 和八卡空闲，再把结果写入 `/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step5000_20261012`。只有全部视频、验证日志和核心指标完成后才写 `AUTO_EVAL_CORE_COMPLETE`。

## 6. 可复现检查清单

运行新模型前应确认：

1. `manifest.jsonl` 的 48 行没有被改写；
2. 首帧 PNG、trajectory、内参和 SHA-256 记录一致；
3. H3 窗口的首帧 C2W 是单位阵，轨迹和内参长度与 37 个 latent anchor 对齐；
4. GT lane 和 single-action lane 使用相同 seed、采样步数和输出尺寸；
5. 短视频报告 GT 指标时写明时间对齐方式，长视频只报告无 GT 的控制和质量指标；
6. 结论同时引用视频、轨迹和指标，不能把 VBench 质量分数当作动作遵循分数。
