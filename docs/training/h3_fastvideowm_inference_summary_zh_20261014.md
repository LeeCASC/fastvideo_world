# H3-World 与 FastVideoWM 推理结果总报告

本文汇总当前工作区中已经完成并保存的 H3-World、FastVideoWM 推理结果。结果按实验协议分组；同一张表只比较协议一致的模型。H3 的 DiffSynth 推理和 FastVideoWM 的 numeric C2W/PRoPE 推理没有被强行合成一个总分。

## 1. 结论先行

当前最可靠的结论是：H3 的 action 注入已经可观察到，FastVideoWM 的相机条件也已经产生响应，但不同实验的“响应强度”和“方向正确率”并不等价。

- H3 的 16-case 标准对照中，`step-2000` 模型在 20 步采样下取得 simple-direction accuracy `1.000`、GT trajectory cosine `0.828`、GT endpoint cosine `0.861`，是当前动作遵循代理指标最强的一组；同一模型改用 50 步采样后，simple-direction accuracy 降到 `0.875`，说明采样步数会改变输出行为，不能只看更长采样是否更好。
- H3 baseline 的 action-on/action-off 消融证明动作输入有因果影响：simple direction rate 从 `0.625` 升到 `0.875`，signed margin 从 `0.031` 升到 `0.207`，GT trajectory cosine 从 `0.260` 升到 `0.584`。这不是“只看视频变动”的证据，而是同一测试集的成对消融结果。
- FastVideoWM 的统一 16-case GT 评测中，step-1200 的 PSNR/SSIM/LPIPS 和 real-camera 对 static-camera 的 flow loss 差值略优于 step-1500：均值分别为 `17.852 / 0.551 / 0.342` 和 `17.416 / 0.539 / 0.358`；step-1200 的 `flow_real - flow_static=-0.00194`，step-1500 为 `-0.00219`。这说明 step-1500 没有在该协议上稳定超过 step-1200。
- FastVideoWM step-100、step-200、step-300、step-750 的结果表明“输入会让画面变化”比“严格按指定轴和方向变化”更容易做到。step-100 的 40 个视频有明显运动响应，但其 flow 符号混杂；step-200/300 的 held-out 8-direction 代理大约只有 7–8/16 个方向符号匹配。
- H3 和 FastVideoWM 的多数旧目录只有视频或代理 flow，没有统一的 GT 指标，不能据此宣称模型排名。下面的排名分为“可量化动作遵循”和“画面质量/稳定性”两条线。

## 2. 模型保存目录

### H3-World

| 模型/阶段 | 保存目录或权重 | 备注 |
|---|---|---|
| 官方 baseline | [`step-10000.safetensors`](/home/lizihao/projects/worldmodel/H3-World/checkpoints/H3-World/step-10000.safetensors) | 所有 H3 baseline 对照的初始化权重 |
| H3 JL 诊断 | [`stage1_jl_50_bs35_fp32`](/datacc05/lizihao/outputs/h3_world/jl_pair_curriculum_20260916/stage1_jl_50_bs35_fp32) | step-1/25/50，动作文本 J/L 诊断 |
| H3 JL 延长 | [`stage1_jl_extend_to100_bs35_fp32`](/datacc05/lizihao/outputs/h3_world/jl_pair_curriculum_20260916/stage1_jl_extend_to100_bs35_fp32) | step-75/100 |
| H3 多源微调 step-2000 | [`train_v2_plus_js_26708.../step-2000.safetensors`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/train_v2_plus_js_26708_from_step10000_bs35_gpu0_6_2000iters/step-2000.safetensors) | 16-case baseline/step-2000 对照使用 |
| H3 step-2450/后续续训 | [`from_internal1950_to3000...`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/from_internal1950_to3000_bs30_gpu0_5_20260922) | step-2000–2450 等多个保存点 |
| H3 最新 H3-World 对照 | [`from_step2450.../step-1000.safetensors`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/from_step2450_legacy_plus_captioned_v2_bs35_gpu0_6_to5000_20260924/step-1000.safetensors) | 17 样本 × 8 方向对照中使用 |

### FastVideoWM / MiniMax-H3

| 模型/阶段 | 保存目录 | 备注 |
|---|---|---|
| action-only step-100 | [`fastvideowm_action_only_step100_eval_20260926`](/datacc05/lizihao/worldmodel/fastvideowm_action_only_step100_eval_20260926) | 5 场景 × 8 方向，含 GT 和 flow 代理 |
| full-cache step-750 | [`fastvideowm_full_65389_20260927/runs/baseline_bs64_1000`](/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/runs/baseline_bs64_1000) | step-750 单动作 sweep |
| camera curriculum step-200 | [`fastvideowm_camera_curriculum_20260930/runs/overfit200`](/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930/runs/overfit200) | hybrid/PRoPE 分支和真实姿态诊断 |
| simple-camera step-200 | [`fastvideowm_simple_camera_20261002/runs/stage1_8gpu_train`](/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/runs/stage1_8gpu_train) | 8 方向 held-out 推理 |
| multi-action post-AdaLN step-300 | [`fastvideowm_multiaction_20261002/runs/postadaln_8gpu_train`](/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/runs/postadaln_8gpu_train) | 8 方向、多动作注入实验 |
| full residual PRoPE step-1300 | [`fastvideowm_full_residual_prope_20261003/runs/train`](/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/runs/train) | JS-easy 和 held-out 8 方向推理 |
| multicam5k step-1200/1500/当前续训 | [`fastvideowm_multicam5k_20261008/runs/train`](/datacc05/lizihao/worldmodel/fastvideowm_multicam5k_20261008/runs/train) | step-1200、step-1500 已评测，当前从 step-1800 续到 step-5000 |

## 3. 推理结果目录索引

### H3-World 推理

- 16-case baseline/action-on/step-2000：[`control_validation_v1_16cases`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/control_validation_v1_16cases)；指标 [`summary.json`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/control_validation_v1_16cases/evaluation/summary.json)，对照报告 [`report.md`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/control_validation_v1_16cases/evaluation/report.md)。
- H3 action on/off 消融：[`baseline_action_ablation_v1_16cases`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/baseline_action_ablation_v1_16cases)。
- H3 step-1650 每数据集 2 样本 × WASDIJKL：[`step-1650_single_action_wasdijkl_2_per_dataset_20steps`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/step-1650_single_action_wasdijkl_2_per_dataset_20steps)。
- H3 17 样本 × 8 方向最新对照：[`js_easy17_h3_latest_vs_fastvideowm_latest_20261009/h3_world`](/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/h3_world)。
- H3 baseline/step-2000/step-3000 17 样本对照：[`js_easy17_wasdijkl_baseline_vs_ft2000_20260917`](/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_baseline_vs_ft2000_20260917)。
- H3 step-2450 与后续模型：[`js_easy17_wasdijkl_step2450_vs_latest_20260924`](/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_step2450_vs_latest_20260924)。
- H3 step-600 子集：[`js_easy_subset_030407080910_step600_20260924`](/datacc05/lizihao/outputs/h3_world/js_easy_subset_030407080910_step600_20260924)。
- H3 早期单独样例/拼接视频：[`custom_wasdijkl_selected_20260924`](/datacc05/lizihao/outputs/h3_world/custom_wasdijkl_selected_20260924)、[`step2000_advantage_showcase_20260917`](/datacc05/lizihao/outputs/h3_world/step2000_advantage_showcase_20260917)、[`step2000_review_v2_20260917`](/datacc05/lizihao/outputs/h3_world/step2000_review_v2_20260917)。
- JL 训练诊断的生成和验证：[`jl_pair_curriculum_20260916`](/datacc05/lizihao/outputs/h3_world/jl_pair_curriculum_20260916)。
- H3 全部旧版推理归档根目录：[`vwm_filtered_v1/inference`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference)。其中还包含 step-1650 每数据集样例、baseline action ablation、16-case control validation 和旧版 20/50-step 结果。

### FastVideoWM 推理

- step-100：[`fastvideowm_action_only_step100_eval_20260926`](/datacc05/lizihao/worldmodel/fastvideowm_action_only_step100_eval_20260926)，含 `metrics.json`、`summary.md` 和 5 个场景 contact sheet。
- step-750：[`inference_step750_single_action_20260930`](/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/inference_step750_single_action_20260930)。
- camera curriculum：[`fastvideowm_camera_curriculum_20260930/runs/overfit200/inference`](/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930/runs/overfit200/inference)，另有 [`prope_real_pose_diagnostic_20260930`](/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930/runs/overfit200/prope_real_pose_diagnostic_20260930)。
- simple-camera step-200：[`step200_heldout_50steps`](/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps)。
- multi-action step-300：[`step300_heldout_50steps`](/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps)。
- full residual PRoPE step-1300：[`step1300_wasdijkl_20261008`](/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/evaluation/step1300_wasdijkl_20261008)；JS-easy 专项目录 [`js_easy_benchmark_step1300_20261008`](/datacc05/lizihao/worldmodel/fastvideowm_full_residual_prope_20261003/evaluation/js_easy_benchmark_step1300_20261008)。
- 统一 16-case GT/single-action：[`fastvideowm_js_easy16_step1200_20261012`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012)、[`fastvideowm_js_easy16_step1500_20261013`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013)。step-5000 目录已预留，但在本报告生成时还没有完成文件。
- 两模型同首帧 17 样本对照：[`js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm`](/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm)。
- 相机成对夜间实验：[`fastvideowm_camera_pair_night_20260930`](/datacc05/lizihao/worldmodel/fastvideowm_camera_pair_night_20260930)；电影式成对数据实验：[`fastvideowm_camera_film_pair_20261001`](/datacc05/lizihao/worldmodel/fastvideowm_camera_film_pair_20261001)。它们保存了 real-correct/reverse/static/shuffled 诊断视频，主要用于分支和反事实检查。
- full-action-strong 目录 [`fastvideowm_strong_action_full_20261002`](/datacc05/lizihao/worldmodel/fastvideowm_strong_action_full_20261002) 保存了训练/守护日志，但没有形成可纳入排名的完整推理结果；不能把未完成任务当作模型失败或成功。

## 4. 量化指标分析

### 4.1 H3 标准 16-case 对照

来源：[`control_validation_v1_16cases/evaluation/summary.json`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/control_validation_v1_16cases/evaluation/summary.json)。16 个用例包括 8 个简单单动作和 8 个 GT-action 用例；代理指标受主体运动、遮挡和生成变形影响。

| 模型 | simple direction accuracy | GT trajectory cosine | GT endpoint cosine | GT RMSE | 首帧 SSIM | sharpness |
|---|---:|---:|---:|---:|---:|---:|
| H3 baseline step-10000，50 steps | 0.875 | 0.584 | 0.604 | 0.0469 | 0.922 | 423.3 |
| H3 fine-tune step-2000，20 steps | **1.000** | **0.828** | **0.861** | 0.0599 | 0.922 | 392.1 |
| H3 fine-tune step-2000，50 steps | 0.875 | 0.664 | 0.667 | 0.0607 | **0.923** | **423.7** |

step-2000 在 20 步采样的方向代理最强，但 RMSE 并没有最低；这说明它可能更强地响应控制方向，同时牺牲了一部分轨迹细节。20 步和 50 步不是同一采样协议，不能据此断言 20 步模型整体优于 50 步。

### 4.2 H3 action on/off 消融

来源：[`baseline_action_ablation_v1_16cases/evaluation/summary.json`](/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/inference/baseline_action_ablation_v1_16cases/evaluation/summary.json)。同一 baseline、同一首帧和同一测试集下，只移除 action 输入。

| 条件 | simple nontrivial rate | signed margin | GT cosine | GT RMSE | sharpness |
|---|---:|---:|---:|---:|---:|
| action on | 0.875 | 0.207 | 0.584 | 0.0469 | 423.3 |
| action off | 0.625 | 0.031 | 0.260 | 0.1199 | 424.6 |

16 对用例中 10 对被判为 action-on 改善。这个实验支持“action 信号确实进入 H3 并改变输出”，但仍不能证明每个方向都正确。

### 4.3 FastVideoWM 统一 16-case GT/single-action

来源：[`step1200/metrics.json`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012/metrics.json) 和 [`step1500/metrics.json`](/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013/metrics.json)。GT 视频做了 61→124 等进度对齐，single-action 视频没有匹配 GT。

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | temporal flicker | flow(real) | flow(static) | real-static |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastVideoWM step-1200 | **17.852** | **0.551** | **0.342** | 0.9859 | 0.1604 | 0.1623 | **-0.00194** |
| FastVideoWM step-1500 | 17.416 | 0.539 | 0.358 | **0.9846** | 0.1632 | 0.1654 | -0.00219 |

step-1200 的重建和控制相关 GT 指标略好；step-1500 的 temporal flicker 代理略好。两者的差距不大，不能把 step-1500 视为稳定退化，也不能把它视为稳定提升。

### 4.4 FastVideoWM 早期和结构实验

- step-100 action-only：5 个场景 × 8 方向，40 个视频。`signed_response_proxy` 非零比例很高，但方向符号在场景之间不稳定，说明 action 分支已产生运动响应，监督还没有把运动锁定到正确轴。
- step-750：4 个 held-out 首帧 × 8 方向。按旧的“非零运动”代理，W/A/S/D/I/J/K/L 分别为 4/3/4/2/4/2/3/2 个非零响应；非零不等于方向正确。
- simple-camera step-200：两个 held-out row 各 8 方向，人工代理符号约 8/16 匹配；其中 row-11 的 A/J/K 与预期相反或不稳定。报告见 [`analysis.md`](/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps/analysis.md)。
- multi-action step-300：两个 held-out row 约 7/16 匹配；多动作和 post-AdaLN 注入提高了响应幅度，但方向仍不稳定。报告见 [`analysis.md`](/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps/analysis.md)。
- camera curriculum 的 branch ablation 显示 adapter、PRoPE 和二者共同分支的 latent 响应不同，但这是噪声空间响应，不是动作正确率；`both` 分支的 counterfactual 差值为 0，不能当作成功证据。原始结果见 [`branch_response.json`](/datacc05/lizihao/worldmodel/fastvideowm_camera_curriculum_20260930/runs/overfit200/prope_branch_ablation_20260930/branch_response.json)。

## 5. 抽帧分析

报告新生成了相同输入条件下的首帧、中帧和末帧拼图，素材目录为 [`inference_review_zh_20261014`](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014)。

### H3 baseline 与 step-2000

![H3 baseline 与 step-2000 的 js_00001 W](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014/h3_baseline_vs_ft2000_js00001_W.jpg)

同一首帧、同一 W 指令下，step-2000 的坦克和道路在中末帧呈现了相机前进趋势，但其背景细节和主体边缘也有更强变化。抽帧支持“微调影响了相机响应”，不能仅凭这一例判断方向泛化。

### H3 与 FastVideoWM 同一批 JS-easy 对照

![H3 与 FastVideoWM 同一首帧 W](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014/cross_model_latest_js00001_W.jpg)

H3 latest step-1000 的前进视角扩张更明显；FastVideoWM step-400 保留背景结构较好，但位移幅度较弱。这个结果来自同一批 JS-easy 首帧和 W 轨迹，适合做定性对照；完整 17×8 视频仍应逐条检查。

### FastVideoWM step-200 和 step-300 的方向稳定性

![FastVideoWM step-200 row-11 抽帧](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014/fastvideowm_step200_row11_all8.jpg)

![FastVideoWM step-300 row-11 抽帧](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014/fastvideowm_step300_row11_all8.jpg)

两组图都能看到明显的运动响应，但 A/J/K 等方向存在与目标轴混合的情况。step-300 的 motion magnitude 更大，并没有同步带来方向一致性提升。

### FastVideoWM step-750/1200/1500 抽帧说明

![FastVideoWM 不同 checkpoint 的代表抽帧](/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/assets/inference_review_zh_20261014/fastvideowm_step750_vs_1200_vs_1500_js00001_W.jpg)

这张图不能做严格 checkpoint 曲线比较：step-750 的 row-00 与 step-1200/1500 的 `js_00001` 不是同一首帧。它只用于检查输出是否能保持场景结构、主体连贯性和明显崩坏；严格数值比较应以统一 16-case 的 step-1200/1500 指标为准。

## 6. 模型排名

排名分成“动作遵循证据”和“画面/GT 重建证据”。单一总榜会把不兼容协议混在一起，因此以下是当前最合理的分层排名。

### 动作遵循排名

| 排名 | 模型/实验 | 证据 | 结论置信度 |
|---:|---|---|---|
| 1 | H3 fine-tune step-2000，20 steps | simple accuracy 1.000，GT cosine 0.828/0.861 | 中；采样协议和 50 steps 不同 |
| 2 | H3 baseline step-10000，action on | 16-case simple 0.875；action-on 明显优于 action-off | 中高；有成对消融支持 |
| 3 | FastVideoWM step-1200 | 统一 16-case 的 GT 指标和 real-static flow 差值最好 | 中；方向仍需逐视频审查 |
| 4 | FastVideoWM step-1500 | 指标接近 step-1200，flicker 略好 | 中；尚未显示稳定动作提升 |
| 5 | FastVideoWM step-750 | 非零运动响应较多，但方向代理只有部分匹配 | 低中 |
| 6 | FastVideoWM step-100 action-only | action 分支明显响应，但符号和轴混杂 | 中；说明注入起效，不说明遵循正确 |
| 7 | FastVideoWM step-200/300 simple-camera、multi-action | held-out 方向代理约 44%–50% | 中；结构实验未解决方向问题 |
| 8 | 早期 H3 JL、step-600/step-2450 等 qualitative-only 结果 | 有视频但缺乏统一可比指标，历史抽帧显示方向不稳定 | 低；不应据此给出数值名次 |

### 画面质量和 GT 重建排名

在统一 16-case GT 协议中，FastVideoWM step-1200 的 PSNR、SSIM、LPIPS 综合优于 step-1500；H3 baseline/step-2000 的指标是首帧/轨迹代理，不与 FastVideoWM 的 PSNR/SSIM/LPIPS 直接等价。定性上 H3 step-2000 的运动幅度更强，FastVideoWM step-1200 的输出更接近其固定 GT 评测协议。

## 7. 结果中的主要问题

1. 方向代理容易被主体运动、遮挡、缩放和生成变形污染；`flow_residual`、`signed_response_proxy`、`motion_proxy` 都不能替代相机姿态估计。
2. H3 的 20 steps 与 50 steps 行为不同，说明采样步数或噪声调度改变了 action 强度，后续比较必须固定步数。
3. FastVideoWM 的 step-1200/1500 已有统一 GT 指标，但 single-action 轨迹和原始 JS-easy GT 并非总是同一条真实轨迹，不能把 PSNR 直接解释成动作准确率。
4. 17×8 H3/FastVideoWM 对照中，H3 曾出现多卡主机内存 OOM，最终 H3 fallback 到 4 卡完成；这影响了推理速度，但不改变已保存视频的数值内容。
5. 部分旧输出使用历史 I/K 文本语义，新的 numeric FastVideoWM 约定是 I 上转、K 下转。旧视频必须依据其 manifest 和生成脚本解释，不能统一重命名。
6. 许多早期结果没有保存完整 GT 指标，只保存视频和拼接片段，因此它们适合人工回看，不适合与新实验做精确排序。

## 8. 下一步建议

第一，等当前 multicam5k 训练到 step-5000 后，严格复用统一 JS-easy 16-case 测试：同一首帧、同一轨迹、同一 seed，分别跑 GT lane 和 single-action lane，并固定 50 steps。把 step-1200、step-1500、step-5000 放在同一 `metrics.json` 表中。

第二，把动作评测从图像 flow 代理升级为“生成视频 VGGT/RAFT 相机运动估计 + 目标 C2W”的几何误差，至少报告每个方向的角度误差、平移方向 cosine、幅度比、响应延迟和反向率。flow 仍可保留为辅助指标。

第三，训练监控使用与推理一致的四类反事实：真实轨迹、静止轨迹、反向轨迹、时间打乱轨迹。要求真实轨迹 loss 同时低于静止、反向和打乱，且这个优势在训练场景和 held-out 场景都成立；单独下降的 flow MSE 不作为放行条件。

第四，保留当前大规模训练，但将动作数据按方向、速度和来源分层采样，降低 multicam_video 的主导比例；同时为 I/K、A/D 等目前较弱的轴补充视觉运动清晰、首帧一致、姿态与视频匹配的配对样本。

第五，继续保留 LoRA 和 PRoPE 的分支审计，同时解冻最后若干 DiT block 的 attention output/QKV 或使用独立高学习率 action adapter。每次只改一个容量因素，并在 20–40 个固定样本上先过拟合验证，确认方向反转有效后再扩大到全量数据。

第六，人工抽查必须覆盖所有方向、三个速度档、四类来源和生成质量异常；自动 GPU-QC 通过不能当作人工质量确认。
