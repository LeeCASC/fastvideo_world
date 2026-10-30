# FastVideoWM：MiniMax-H3 相机控制世界模型

本仓库整理了基于 FastVideo / MiniMax-H3 的世界模型训练、数据适配、相机条件注入和评估代码。它是现有实验代码的发布备份，不是已完成全部路径迁移的一键运行发行版。

上游介绍保存在 [README_UPSTREAM.md](README_UPSTREAM.md)，原始 [LICENSE](LICENSE) 和源码来源声明保持不变。独立 H3-World / DiffSynth 实验见 [h3world_finetune](https://github.com/LeeCASC/h3world_finetune)，不能与本仓库混用缓存或启动命令。

## 当前模型与边界

当前已经训练的配方是 **MiniMax-H3 双向视频扩散模型**：

- 条件包含首帧图像、实际的背景环境与主体外观/可观察行为文本，以及逐 latent 对齐的数值 C2W。
- 保留 H3 主注意力，增加仅处理视频 token 的残差 PRoPE 相机注意力分支；完整 128 维/head，每隔两个 block 设置分支，经可学习的 `prope_o` 加回主路。
- rank-64 attention LoRA，加上相机分支和部分后层/投影模块解冻；不是仅训练少量 LoRA。精确范围由每次 YAML 决定。
- `action_dim=0` 只关闭额外 action adapter，不关闭 PRoPE。
- 当前 `prope_use_camera_intrinsics=false`：K 存储并验收，但不进入本轮 PRoPE 投影。
- 音频输入长度为 0，相关预训练模块保留并冻结；不是用静音波形训练音频。
- 当前主要监督为视频 flow-matching MSE，额外相机排序/响应/运动 loss 权重均为 0。正确/零/乱序 pose 的 loss 监控不等于启用了额外监督。

上游 FastVideo 包含其他模型的 AR、Self-Forcing 和蒸馏能力，**不等于本项目 H3 相机控制模型已经完成 AR 或少步蒸馏**。逐 latent 动作文本、未知 pose 门控、新辅助监督等计划文档也不代表已启用。

## 阅读入口

| 内容 | 文档 |
|---|---|
| 发布范围、依赖、续训与迁移限制 | [发布说明](docs/training/PUBLICATION_20261030_ZH.md) |
| 原生 H3 预处理基础，包含早期历史配方 | [minimax_h3_worldmodel.md](docs/training/minimax_h3_worldmodel.md) |
| QC 后的数据准备与 full-state 续训 | [cosmos_poseqc_resume1100_20261027.md](docs/training/cosmos_poseqc_resume1100_20261027.md) |
| 背景＋主体文本、编码与持久化验收 | [union_dual_text_20261023.md](docs/training/union_dual_text_20261023.md) |
| Portable 评估与四卡配置 | [cosmos_poseqc_step1700_portable_eval_20261029.md](docs/training/cosmos_poseqc_step1700_portable_eval_20261029.md) |
| 输入 HUD：原始视频与展示视频分离 | [camera_input_hud_20261022.md](docs/training/camera_input_hud_20261022.md) |
| 历次注入方式和失败原因 | [指令注入报告](docs/training/h3_fastvideowm_action_injection_report_zh_20261014.md) |
| 尚待实现和验证的改进 | [control_precision_todo_20261024.md](docs/training/control_precision_todo_20261024.md) |

本项目使用模块化训练栈 `fastvideo/train/`；上游旧栈 `fastvideo/training/` 不可随意替换。

## 安装

仓库不上传实验模型权重、训练视频与 pose、完整数据注册表、VAE/文本条件缓存、checkpoint、日志和推理输出。请自行准备并遵守各自许可证。

先核对操作系统、GPU、驱动与 PyTorch wheel 的 CUDA 兼容性；`nvidia-smi` 的 CUDA 上限不是已安装 toolkit 的版本。依赖以 [pyproject.toml](pyproject.toml) 和 [GPU 安装说明](docs/getting_started/installation/gpu.md) 为准；H3 八卡配方的验证目标是 Linux/NVIDIA，不能把上游其他平台支持视为此配方已验证。

```bash
git clone --recurse-submodules https://github.com/LeeCASC/fastvideo_world.git
cd fastvideo_world
uv venv --python 3.12 --seed
UV_TORCH_BACKEND=cu126 uv pip install -e '.[dev,worldmodel,eval,eval-fast-decode]'
```

上例选择 CUDA 12 wheel；CUDA 13 或源码扩展必须另行确认驱动、工具链、PyTorch 和 attention/kernel 依赖兼容。不要边训练边升级环境。可选 VBench 依赖/模型不一定随主包齐备，缺失维度要标记 unavailable。

保留的第三方 gitlink：

| 路径 | 固定 revision |
|---|---|
| `fastvideo-kernel/include/cutlass` | `e67e63c331d6e4b729047c95cf6b92c8454cba89` |
| `fastvideo-kernel/include/tk` | `6c27e28c8115d1839d9eeeb530913c184a75fc87` |
| `fastvideo/third_party/eval/vbench` | `45e79ec14e69a2187202c675d2dbce1a71843d53` |

来源见 [.gitmodules](.gitmodules)。需要时运行 `git submodule update --init --recursive`，不要以未固定的最新 HEAD 替换后仍宣称环境相同。

## 原生 H3 缓存契约

当前典型窗口为 124 RGB 帧、832×480，对应 37 个视频 latent 时间位置，单条视频 latent 为 `[24,37,30,52]`；batch 维由加载器添加。H3 时间采样不是 Wan 的 `4t+1` 或简单固定 stride。

每条训练数据必须绑定同一窗口的：视频 VAE latent、首帧 VAE 条件、H3 专用 Qwen 首帧视觉和实际文本 token/语义 tags、37 个 C2W/K、完整源帧索引，以及 ID/split/首帧/窗口/文本 hash 和验收 sidecar。最终由 native H3 Parquet schema 读取。

C2W 以窗口首帧为原点；像素 K 先按真实标定尺寸归一化，再处理 resize/crop。当前不做逐视频平移幅度重归一化。W/S=前进/后退，A/D=左移/右移，I/K=上转/下转，J/L=左转/右转；历史 H3 文本缓存有不同 I/K 约定，不可直接混入。

同一窗口不变可以复用已核验视频 VAE；改文本必须重新编码条件。普通 Qwen captioner 不产生可替代的 H3 embedding，Wan/UMT5 缓存也不能直接替换。数值和持久化验收不等于视觉 pose 一定正确。

## 训练入口和完整续训

通用入口是 [launch_minimax_h3_worldmodel_stage0.sh](examples/train/launch_minimax_h3_worldmodel_stage0.sh)。名称保留自早期 stage0，**不表示默认 YAML 就是当前配方**。先准备并审核独立配置，再覆盖全部路径：

```bash
MODEL_PATH=/path/to/MiniMax-H3 \
TRAIN_DATA=/path/to/verified/preprocessed/train \
VALIDATION_DATA=/path/to/verified/preprocessed/validation \
CONFIG_FILE=/path/to/audited_training_config.yaml \
LOG_DIR=/path/to/run/logs \
NUM_GPUS=8 \
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
bash examples/train/launch_minimax_h3_worldmodel_stage0.sh
```

八卡 SP2 下有 DP4，microbatch1 × DP4 × grad accumulation8 = **total BS32**；SP 不计入独立样本数。HSDP mesh、GPU 数、SP 和 batch 必须一致。

恢复现有实验时，先重建原训练的官方 base → parent model bootstrap，再恢复完整 DCP 的模型、优化器、scheduler/callback、RNG、loader 和 global step。部分 checkpoint 只保存可训练状态，不能假设是独立完整模型。不要把 full-state 续训替换为只加载 LoRA/权重并从 0 开始，也不要重复运行固定旧 step 的初始化器。换数据时仅按显式策略重置旧 loader，其他状态继续恢复。

**所有带日期的 launch/watch/stop/resume 脚本都是本机实验记录。** 可能有绝对路径、PID、tmux、锁和停止操作，必须先逐行审核并迁移，不能凭 portable/auto 的文件名直接运行。

## 最小 CPU 回归

这些测试不启动训练或 GPU 推理，不能替代原生数据验收和真实 GPU smoke test：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q \
  tests/local_tests/test_portable_h3_bootstrap_20261029.py \
  tests/local_tests/test_cosmos_qc_portable_eval_20261029.py \
  tests/local_tests/test_resume_cosmos_qc_after_eval_20261029.py
```

部分完整推理模块在 import 时初始化 CUDA/Triton，不能把 CUDA 隐藏后的导入失败直接判为训练错误。修改代码按 [AGENTS.md](AGENTS.md) 使用项目 pre-commit 与对应测试。

## 仍需迁移的外部依赖

- `tools/` 和历史 config/docs 保留本机路径，当前发布不是全量路径重构。
- HUD 从独立 Cosmos 仓库读取绘图函数；数值 QC 依赖独立 `cosmos_pose_qc_reference`。
- 部分 caption 工具依赖 H3-World 的 `code/vwm`。需按[发布说明](docs/training/PUBLICATION_20261030_ZH.md)准备路径及审核许可证。
- HUD 是输入展示，不证明生成遵循；指标使用未覆盖的 raw 视频。半幅轨迹没有自动对应半幅 GT，PSNR 改善不能直接证明幅度更准。

没有外部依赖、模型、注册表和缓存，不能宣称独立复现完整历史实验。此版只备份代码和整理文档，不改变核心 runtime 或自动实施新实验。
