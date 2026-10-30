# FastVideoWM 代码发布说明（2026-10-30）

## 1. 发布目的和内容

目标仓库：`https://github.com/LeeCASC/fastvideo_world`。

这是隔离整理的工作代码快照：保留 FastVideo 上游、当前 MiniMax-H3 世界模型改动、预处理/质量检查/文本条件工具、训练推理脚本、测试和中文实验文档。此次不修改核心 runtime、不运行新的模型实验，也不触碰原服务器的训练进程。

根目录 README 聚焦本项目；上游原 README 保存在 `README_UPSTREAM.md`。保留原 `LICENSE`、源码 SPDX 和第三方出处，不将上游重新归属为本项目原创。模型与数据各自许可不由代码 Apache-2.0 覆盖。

不上传实验模型、训练原视频、完整注册表/pose 数据、VAE latent、条件 embedding、checkpoint、日志缓存和推理结果。上游已有的演示/测试资产不是本项目训练全集。报告中的本机模型/数据链接仅为历史出处，不表示 GitHub 包含相应文件。

## 2. 当前配方，不等于全部历史实验

本项目已经训练的是双向 MiniMax-H3，而非完成了 AR 或少步蒸馏的世界模型。FastVideo 原有其他模型的 causal、Self-Forcing、DMD 等代码被保留，但不能作为 H3 相机控制 AR 已实现的证据。

| 部分 | 当前配方 |
|---|---|
| 图像/文本 | H3 原生首帧视觉条件＋背景环境、主体外观和观察到的行为文本 |
| 相机 | 37 个 latent 时间位置的数值 C2W，相对窗口首帧，保留核验后平移尺度 |
| PRoPE | 视频 token 独立残差注意力，完整 128 维/head，每隔两个 block 接入，`prope_o` 可学习出口 |
| 可训练部分 | rank-64 attention LoRA＋相机分支＋选定主干层/投影，不是仅 LoRA |
| `action_dim=0` | 关闭额外 action adapter，不关闭数值 PRoPE |
| 内参 | 保存/校验 K，当前 `prope_use_camera_intrinsics=false` |
| 音频 | 输入长度 0，预训练模块保留并冻结，不监督音频 |
| 损失 | 视频 flow MSE；额外相机排序/响应/运动 loss 权重 0 |
| 八卡 batch | SP2、DP4、micro1、grad accu8，total BS32 |

实现位置：

- `fastvideo/models/dits/minimax_h3.py`、`minimax_h3_prope.py`：主干及几何注入。
- `fastvideo/train/models/minimax_h3/worldmodel.py`：训练封装、LoRA与可训练模块。
- `fastvideo/train/methods/fine_tuning/worldmodel.py`：flow训练及可选辅助项。
- `fastvideo/train/callbacks/worldmodel_validation.py`：正确/零/乱序相机条件 loss 监控。
- `fastvideo/configs/models/dits/minimax_h3.py`：模型开关。

实际配置以目标实验 YAML 和 checkpoint metadata 为准，不能用早期 stage0 默认 YAML 代替。历史 pure PRoPE、adapter、post-AdaLN 和附加监督代码仍被保留，便于追踪，不代表全部启用或准确控制已收敛。正确 pose loss 更低、输出对条件敏感、语义上精确遵循，是不同层次证据。

## 3. 新环境和固定子模块

先阅读 `AGENTS.md`、`pyproject.toml` 和 `docs/getting_started/installation/gpu.md`。使用干净 Linux/NVIDIA 环境，按驱动、CUDA工具链、PyTorch wheel 和 attention backend 的实际兼容性安装；不要边训练边升级原环境。

```bash
uv venv --python 3.12 --seed
UV_TORCH_BACKEND=cu126 uv pip install -e '.[dev,worldmodel,eval,eval-fast-decode]'
```

`cu126` 是示例，不是任意驱动均可使用的保证。CUDA 13 backend、源码 kernel、FlashAttention/Triton 等需彼此匹配。`nvidia-smi` 的 CUDA 上限不等于已安装 toolkit 版本。H3 data adapter 依赖 `vwm-datasets`；部分编码工具依赖 `decord`，可通过 `eval-fast-decode` 安装。可选 VBench 模型/依赖不一定随主包齐备。

保留以下原始 gitlink 与 `.gitmodules`：

| 路径 | URL | revision |
|---|---|---|
| `fastvideo-kernel/include/cutlass` | `https://github.com/NVIDIA/cutlass.git` | `e67e63c331d6e4b729047c95cf6b92c8454cba89` |
| `fastvideo-kernel/include/tk` | `https://github.com/HazyResearch/ThunderKittens.git` | `6c27e28c8115d1839d9eeeb530913c184a75fc87` |
| `fastvideo/third_party/eval/vbench` | `https://github.com/Vchitect/VBench.git` | `45e79ec14e69a2187202c675d2dbce1a71843d53` |

```bash
git submodule update --init --recursive
```

按需初始化，不能以未记录的 HEAD 替换后仍宣称环境一致。

## 4. 原生数据和缓存契约

典型窗口为 124 RGB 帧、832×480，单条视频 latent `[24,37,30,52]`，batch维由加载器增加。H3 时间采样不是 Wan `4t+1`。相关代码在 `fastvideo/dataset/vwm_h3.py`、`fastvideo/dataset/dataloader/` 和 `fastvideo/pipelines/preprocess/minimax_h3_worldmodel/`。

训练前必须检查：

1. ID、源视频和完整窗口帧索引绑定，不只按同源 ID 复用。
2. 视频 latent、首帧 VAE 和首帧视觉 token 来自同一窗口/首帧。
3. 使用 H3 专用条件编码器产生实际文本/视觉 token 与 tags；VLM captioner只生成文字，不提供可替代的 H3 embedding。Wan/UMT5 缓存不可直接替换。
4. 37 个 C2W 对齐 H3 原生非均匀时间锚点，并重新以窗口首帧为原点。
5. 不混淆 C2W/W2C；由 W2C `[R,t]` 求相机中心用 `-R^T t`，不把 `t` 直接当位置。
6. 像素 K 按实际标定尺寸归一化，再处理 resize/crop。JS-easy 曾有视频尺寸与 K 标定尺寸不同的问题。
7. native Parquet、sidecar 的 split、文本/hash、首帧/窗口和有限张量验收一致。先建立 loader index，避免分布式启动后串行扫描十万级 footer。
8. 当前 `normalize_camera_translation=false`，不能又在外部偷偷把每个视频独立归一化。

动作约定：W/S前进/后退，A/D左移/右移，I/K上转/下转，J/L左转/右转。独立 H3-World 历史文本缓存可能存在旧 I/K 定义，不可静默合并。

源窗口不变可复用已验证视频 VAE；caption改变需重做条件 embedding。VAE 完成标记不等于首帧、pose、文本全部通过。数值 QC/VLM 标记不是相机运动的人工 GT，动态前景污染 pose 仍需视觉验收。不要把未知 pose 用单位矩阵冒充静止；本发布未自动加入未知 pose 门控。

## 5. 通用启动与完整恢复

入口 `examples/train/launch_minimax_h3_worldmodel_stage0.sh` 支持 `MODEL_PATH`、`TRAIN_DATA`、`VALIDATION_DATA`、`CONFIG_FILE`、`LOG_DIR` 等环境变量，并转入 `fastvideo/train/` 模块化训练栈。文件名 stage0 不代表默认 YAML 是当前配方。

先把目标实验配置复制到独立位置，审计路径、mesh、文本开关、可训练模块、loss和输出目录，再传入 `CONFIG_FILE`。不要覆盖旧 checkpoint 目录。

恢复已有实验时：

1. 检查目标 metadata、DCP shards、各 rank RNG 完整且写入已完成。
2. 重建原训练的官方 base → parent model bootstrap。部分 checkpoint 仅含可训练状态，不保证独立保存全部冻结参数。
3. 显式绑定完整 checkpoint，恢复模型、optimizer、scheduler/callback、RNG、loader 和 global step，不能重置成父模型新训练。
4. 数据路径改变时，仅按显式策略重置 loader；其他状态继续恢复。同路径恢复应保留 loader。
5. 八卡/SP2有DP4，因此 `micro1 × DP4 × accu8 = total BS32`，SP不再乘入样本 batch；HSDP mesh也需对应。
6. 保留配方 eager 设置与 activation-checkpoint 一致性检查，不通过关闭检查掩盖错误。

`tools/prepare_cosmos_qc_resume_20261027.py` 是固定旧 checkpoint-1100 的历史准备器，不是任意最新模型的通用恢复器；已有更高步模型时不可直接重复执行。所有 dated launch/watch/stop/resume/auto 脚本可能包含本机路径、PID、tmux、锁和进程停止行为，必须先读实现并审计实际目标，不能凭文件名直接运行。

## 6. 外部依赖与迁移清单

以下内容在历史服务器存在，但不随本仓库自动分发：

| 功能 | 本仓库入口 | 外部依赖 |
|---|---|---|
| Cosmos HUD | `tools/visualize_h3_controls.py` | AST读取独立 Cosmos 项目 `eval/methods/utils.py` 的绘图函数 |
| 数值 pose QC | `tools/filter_current_cosmos_pose_qc_20261027.py` | 独立 `cosmos_pose_qc_reference` 规则目录 |
| 部分 Qwen caption | `tools/caption_vwm_61f_qwen3vl.py` | H3-World `code/vwm/caption_scenes_qwen3vl.py` |
| Portable benchmark | evaluation工具 | 另行传输的同版本测试输入、原生H3缓存与模型 |

这些工具的默认位置是本机绝对路径，需要准备同版本依赖和迁移路径，或另做可移植改造；复制外部源码前必须核查并保留许可证。此代码备份不擅自 vendoring 外部仓库，也不承诺全流程独立复现。

历史 YAML、tools 和报告中的原服务器地址仅是记录，不是新机器上的实时状态。文档本机链接无法在 GitHub 上直接访问；不要将链接存在当成资产已上传。涉及 stop/kill/resume 的脚本尤其需要审核PID、账号、命令行及精确数据范围。

## 7. 最小验证及评估交付

以下CPU回归不加载完整H3模型，不启动实际训练或推理：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q \
  tests/local_tests/test_portable_h3_bootstrap_20261029.py \
  tests/local_tests/test_cosmos_qc_portable_eval_20261029.py \
  tests/local_tests/test_resume_cosmos_qc_after_eval_20261029.py
```

这不替代真实 native H3 单样本持久化验收、forward/backward、条件响应和分布式 smoke test。部分完整模块 import 会触发 CUDA/Triton；CUDA隐藏时“0 active drivers”与专门隔离的CPU测试是不同路径。代码格式/类型等检查按 `AGENTS.md` 使用项目 pre-commit，不擅自绕过有意的 excludes。

评估主体为 `scripts/evaluation/portable_js_easy_h3.py`、`score_portable_js_easy_h3.py`，适配器为 `tools/prepare_cosmos_qc_portable_eval_20261029.py`。参考 `cosmos_poseqc_step1700_portable_eval_20261029.md` 后迁移配置与输入，不直接执行旧launcher。

交付分为 raw视频、visualized输入HUD和指标三部分。HUD显示实际相机/文本输入，不证明生成遵循；指标基于未覆盖raw。可选 VBench维度缺失时明确 unavailable，不标为完整通过。半幅轨迹没有对应半幅GT时，不能把相对原幅GT的PSNR/SSIM/LPIPS当作半幅控制精度。

## 8. 尚待实现的后续工作

逐latent动作文本、相机出口增益消融、方向/速度分层采样、未知pose门控、相机排序监督、背景运动验收、文本解耦及AR/蒸馏路线见 `control_precision_todo_20261024.md` 等计划。需要固定checkpoint、数据和噪声做独立对照，不能因文档存在而宣称均已实现。

此次整理验收目标是代码/文档齐备、外部依赖透明、大文件不混入、验证边界清楚；不新增“精确控制已经成功”的模型结论。
