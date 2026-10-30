# H3-World 与 FastVideoWM 指令注入技术报告

更新时间：2026-10-14  
适用范围：MiniMax-H3 / H3-World、FastVideoWM、minWM 的相机控制实验。  
结论性质：代码行号是当前工作区核对结果；实验结论只覆盖报告中列出的数据和 checkpoint，不能外推到所有模型。

## 1. 结论摘要

“旧 FastVideoWM 完全没有传入 action”并不准确。旧实验确实把相机轨迹送入了模型，但不同阶段的通路容量和监督强度差异很大：

1. H3-World 原始路线把 W/A/S/D/I/J/K/L 写成 37 条逐 latent 文本，借助 directed attention mask 绑定到对应视频 latent；它不是数值相机矩阵注入。
2. FastVideoWM additive 路线把 6DoF 数值轨迹编码成 hidden residual/FiLM；它与 PRoPE 是两套不同的注入。
3. 早期 FastVideoWM exact PRoPE 只改每个 attention head 的 `[96,108)` 共 12 个通道，且不少实验只训练少量末端 LoRA；几何输入存在，但有效可学习容量很小。
4. hybrid 同时启用 action adapter 和 PRoPE；post-AdaLN 版本把 action 调制放到每个 block 的 norm/AdaLN 之后，控制路径明显更强。
5. 当前 residual PRoPE 保留预训练 H3 主注意力，另外建立只处理视频 token 的相机注意力分支，并通过零初始化 `prope_o` 加回主路；当前配置的 `action_dim: 0` 只关闭额外 action embedding/adapter，**不关闭相机矩阵 PRoPE**。
6. 旧版失败的主要原因是“信号有但相对主模型太弱/监督不直接/几何与标签不一致/训练容量和数据质量不匹配”的组合，而不是 PRoPE 数学形式本身必然无效。

当前最可靠的因果验证仍是同一个 checkpoint、首帧、随机种子和噪声下，只关闭 `prope_o` 相机分支；若正确/反向轨迹的响应差异消失，才能把当前控制响应直接归因于该分支。

## 2. 三种输入契约

| 路线 | 模型收到的 action | 对齐单位 | 是否使用文本 action | 主要训练对象 |
|---|---|---|---|---|
| H3-World 文本 | 每个 latent 区间一句 camera 文本 | 37 个 H3 latent | 是 | H3 DiT LoRA/文本注入相关路径 |
| FastVideoWM 数值 | `camera_to_world [B,37,4,4]`，可选 `K [B,37,3,3]` | 37 个 latent 帧 | 否；当前 action-only 也不输入场景文本 | PRoPE、action adapter、LoRA 或指定 block |
| minWM 参考 | `viewmats` + `Ks` | 视频 latent 帧 | 否（另有 prompt/vision 条件） | 相机分支 `prope_o`，按 recipe 选择 DiT 范围 |

FastVideoWM 的数据路径先将 C2W 按窗口首帧重置，再转换为相对 W2C；37 个 camera anchor 会按空间 patch 行展开到 packed sequence。代码见 [`worldmodel.py:588-620`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:588>) 和 [`worldmodel.py:671-704`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:671>)。`normalize_camera_translation: false` 时保留测得的平移尺度；如果重新归一化，会损失速度信息。

当前 FastVideoWM action-only recipe 保留 H3 的音频模块/权重但不输入音频 token：forward 中创建 zero-length audio tensor（[`worldmodel.py:688-704`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:688>)），因此这是“音频路径空输入并冻结”，不是删除预训练 audio 模块。场景 caption 在 `vision_only_conditioning: true` 时也不作为 DiT 文本条件；这与 H3-World 的 action-text 路线是本质不同的条件协议。

## 3. H3-World：逐 latent 文本指令注入

### 3.1 文本缓存的布局

H3-World 的 injector 不修改 video/audio latent，而是重写 `prompt_embeds` 和 packed metadata。`inject_abot_text.py` 的说明和实现见 [`inject_abot_text.py:1-24`](</home/lizihao/projects/worldmodel/H3-World/code/abot/inject_abot_text.py:1>)、[`inject_abot_text.py:197-243`](</home/lizihao/projects/worldmodel/H3-World/code/abot/inject_abot_text.py:197>)：

```text
text = [首帧视觉/场景 head | action_text_0 | ... | action_text_36]
packed.action_text_rows      # 每条 action 文本的 row span
packed.action_video_start    # 视频 token 的起点
packed.action_frame_rows     # 每个 latent 帧的空间 token 数
```

每条 action 文本独立编码，缓存相同句子；`inject_abot_text.py:267-288` 的 `_encode_action_script` 保证训练与推理的句子 embedding 一致。文本在时间坐标上使用 mirrored offset，而不是直接放在对应视频帧的正上方，具体布局见 [`audio_video.py:652-724`](</home/lizihao/projects/worldmodel/H3-World/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py:652>)。

### 3.2 Directed attention mask

H3 DiT 为 action 文本建立硬绑定 mask：annotation `k` 可以和第 `k` 个视频 latent 交互，但不能直接读写其他 latent。实现和理由见 [`minimax_h3_dit.py:146-242`](</home/lizihao/projects/worldmodel/H3-World/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py:146>)。这解决了“把 37 条句子简单拼到 prompt 后，self-attention 会混合所有指令”的问题。

旧缓存曾在 Token Refiner 中让所有 action 句子进入同一段 full attention，形成上游泄漏；当前代码按 head+每条 action sentence 分段，见 [`audio_video.py:985-1023`](</home/lizihao/projects/worldmodel/H3-World/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py:985>)。

`rewrite_one` 最后显式删除 `kwargs["action_cond"]`，因此这条新文本方案不使用 FiLM 或 feature-space action residual：[`inject_abot_text.py:237-242`](</home/lizihao/projects/worldmodel/H3-World/code/abot/inject_abot_text.py:237>)。

### 3.3 H3-World 的训练和推理含义

原始 H3-World 训练使用文本条件的 flow-matching/SFT MSE；基础 loss 见 [`loss.py:66-102`](</home/lizihao/projects/worldmodel/H3-World/DiffSynth-Studio-h3-v2/diffsynth/diffusion/loss.py:66>)。文本本身没有一个“方向分类 loss”。后期增加的 instruction validation 会用相同随机状态比较有 action 和 mask action 的 video loss，见 [`train.py:197-233`](</home/lizihao/projects/worldmodel/H3-World/DiffSynth-Studio-h3-v2/examples/minimax_h3/model_training/train.py:197>)；它是诊断/验证，不是额外监督项。

`infer_camera.py` 把一个 preset 扩展为 37 条相同文本（例如 `camera pans left during this interval`），并将正/负 action script 送入 pipeline，见 [`infer_camera.py:24-43`](</home/lizihao/projects/worldmodel/H3-World/code/vwm/infer_camera.py:24>)、[`infer_camera.py:124-145`](</home/lizihao/projects/worldmodel/H3-World/code/vwm/infer_camera.py:124>)。旧 H3 文本缓存的 I/K 曾采用相反约定；当前数值 FastVideoWM 合约统一为 **I 上转、K 下转**，不能直接混读旧缓存。

## 4. FastVideoWM 的注入演进

### 4.1 additive action adapter：最早的数值 action residual

配置 [`stage0_v2.yaml:1-31`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_v2.yaml:1>) 使用 6DoF action、每通道 robust scale、clip 和 embedding scale，并只显式训练 `action_adapter`、输入/输出投影和 LoRA。`worldmodel.py:259-289` 把 action 做 scale/clip/asinh/Fourier（视配置而定），`minimax_h3.py:847-861` 建立线性 adapter 并零初始化。

在普通 additive 模式，action embedding 在 block 入口加入 hidden；代码见 [`minimax_h3.py:1162-1186`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:1162>) 和 [`minimax_h3.py:1232-1247`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:1232>)。其优点是实现简单；缺点是它在 RMSNorm/AdaLN 前后可能被归一化、残差比例小，而且 action 与每帧视觉 token 的几何对应关系不是硬约束。

### 4.2 exact PRoPE：原位几何变换

PRoPE 先用相对 W2C 和 K 构造 `P=K@w2c` 及逆矩阵，见 [`minimax_h3_prope.py:136-167`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3_prope.py:136>)；再把投影作用到 attention head 的指定通道，见 [`minimax_h3_prope.py:170-200`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3_prope.py:170>)。旧 FastVideoWM exact/hybrid 配置常用 `[96,108)`，即每头 12 个通道，而不是整个 128 维 head；例如 [`hybrid_probe.yaml:118-121`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_hybrid_probe.yaml:118>) 和 [`full_prope_lora_v5.yaml:115-121`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml:115>)。

exact 模式直接替换 Q/K/V 的指定几何子空间。它有明确的相机几何含义，但 12/128 的通道占比很小；如果只训练少量 LoRA，主网络仍可能用原有视觉捷径解释目标视频。

### 4.3 hybrid：PRoPE + action adapter

hybrid 同时计算 `action_rows` 和 `prope_projection/inverse`，见 [`worldmodel.py:645-676`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:645>)。这样一条路径负责连续 feature conditioning，另一条路径负责显式投影几何。

旧 hybrid probe 配置使用 `action_dim=6`、`relative_pose`、`[96,108)` PRoPE、rank-16 LoRA，且加入 temporal/latent-motion 辅助 loss：[`hybrid_probe.yaml:9-51`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_hybrid_probe.yaml:9>)。v5 进一步使用 31 维 Fourier action、每 8 个 block 注入一次、residual scale/clip：[`full_prope_lora_v5.yaml:11-25`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml:11>)、[`full_prope_lora_v5.yaml:115-121`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml:115>)。

### 4.4 post-AdaLN action modulation：更强的 action 旁路

多 action 实验用 `relative_pose_delta` 形成 12 维“累计 pose + pose delta”，并训练 `action_encoder`、`action_adapter`、`action_block_gates` 以及部分后层和 rank-64 LoRA：[`multiaction_postadaln_20261002.yaml:6-39`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_multiaction_postadaln_20261002.yaml:6>)。`action_modulation_mode: film`、`action_post_adaln: true`、`action_injection_interval: 1` 使每个 block 都可在 norm/AdaLN 后接收 action 调制：[`multiaction_postadaln_20261002.yaml:109-120`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_multiaction_postadaln_20261002.yaml:109>)、[`minimax_h3.py:725-736`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:725>)、[`minimax_h3.py:1248-1268`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:1248>)。

这是 action 起效较明显的版本之一，但应注意它同时改变了 action 表示、可训练参数、LoRA rank、监督和数据采样，不能把所有收益单独归功于 post-AdaLN。

### 4.5 当前 residual PRoPE：保留 H3 主路的独立相机分支

当前配置 [`full_residual_prope_20261003.yaml:1-49`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml:1>) 使用：

```yaml
camera_conditioning: prope
action_dim: 0
prope_dim_range: [0, 128]
prope_mode: residual
prope_parallel_residual_interval: 2
prope_use_camera_intrinsics: false
normalize_camera_translation: false
```

`action_dim: 0` 只使 `action_adapter` 不创建，模型仍在 [`worldmodel.py:671-676`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:671>) 计算 camera projection/inverse，并在 [`minimax_h3.py:735`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:735>) 把矩阵传进 attention。

residual attention 的关键步骤：

1. 原 H3 Q/K/V 保持主注意力；
2. 只抽取视频 rows，使用完整 `[0,128)` head 子空间重做几何 attention；
3. `prope_o` 将相机分支投影回 hidden，并加到主 attention output；
4. `prope_o` 零初始化，所以 step 0 对原 H3 是严格 no-op；训练后该矩阵成为相机信号的可学习出口。

实现见 [`minimax_h3.py:458-491`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:458>) 和 [`minimax_h3.py:448-455`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py:448>)。`prope_o`/gate 的学习率和参数解冻在 [`worldmodel.py:178-205`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:178>)、[`worldmodel.py:204-257`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py:204>)。

这里的“完整 128 维、每隔一个 block 建分支”是当前 residual 分支的设计；不能与旧 `[96,108)` 12 维 exact 分支混为一谈。当前配置虽有 rank-64 LoRA 和若干后层 attention 参数，但所有 action/camera 辅助权重设为 0，主要仍由 flow MSE 学习：[`full_residual_prope_20261003.yaml:59-66`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml:59>)。

## 5. minWM 参考实现：为什么它更容易形成有效控制通路

minWM 的 MiniMax-H3 PRoPE processor 把 camera branch 明确设计成“原 H3 attention + camera-only residual”。它先取视频 rows、做 PRoPE Q/K/V，再只做 video-to-video attention，最后使用 `prope_o` 加回主路：[`minimax_h3/prope.py:152-209`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/prope.py:152>)、[`minimax_h3/prope.py:298-320`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/prope.py:298>)。

`add_prope_parameters` 为每个 block 添加 bias-free、零初始化的 `prope_o`：[`minimax_h3/prope.py:349-370`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/prope.py:349>)。模型 forward 会拒绝在 `use_prope=False` 时静默丢弃 `viewmats`，并要求 camera frame 数和 video rows 严格整除：[`model.py:363-404`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/model.py:363>)。这两个 guard 对排查“action 根本没有进入模型”非常重要。

历史 minWM 分析中提到的“每隔一个 block、完整 head 几何子空间、约 25 个分支”描述的是 minWM 分支的容量安排；FastVideoWM 旧版 `[96,108)`、后层 LoRA 和当前 `[0,128)` residual 是不同实现，不能直接用一个数字比较全部实验。

minWM 的作用不只是 PRoPE 公式：它的 branch 有明确输出出口、camera-only token 范围、严格 shape/first-frame 对齐和训练时连续的 viewmats/Ks 传递。FastVideoWM 旧版如果只启用一小段通道、只训练末层 LoRA 或相机输出仍接近零，理论上“传入了 action”仍可能在生成视频里几乎看不到。

## 6. 监督、参数和实验事实对照

| 实验 | action/PRoPE | 可训练范围 | 监督 | 主要限制 |
|---|---|---|---|---|
| Stage0 additive v2 | action adapter，可能无 PRoPE | adapter + proj/norm/out + rank-8 LoRA | flow MSE + motion weight | 入口 residual 容易被 norm 稀释，6DoF 表示容量有限 |
| hybrid probe | 6D adapter + `[96,108)` PRoPE | adapter + rank-16 LoRA | temporal + latent motion/static | 小样本过拟合诊断，不能证明泛化 |
| full-prope v5 | 31D Fourier adapter + exact PRoPE | adapter + rank-16 LoRA | temporal + latent motion | `[96,108)` 仍很窄，8-step action injection |
| post-AdaLN | 12D pose+delta + exact PRoPE | encoder/adapter/gates + 后层 + rank-64 | ranking/sensitivity/range 等 | 同时改动太多，因果归因困难；ranking 每 4 microbatch |
| residual current | camera-only `[0,128)` + `prope_o` | rank-64、后层 attention、PRoPE branch | 当前配置 aux 全 0，flow MSE | zero-init branch 需要足够梯度和步数；非监督 action loss |

需要特别区分：

- `action_motion_loss_weight` 是按 action 幅度加权视频目标，不等于方向正确性监督；
- `camera_ranking`/`sensitivity`/`response_range` 是 counterfactual 诊断/辅助监督，早期配置可能关闭或低频计算；
- instruction validation 比较 conditioned 与 masked loss，只能说明条件敏感性，不能单独证明方向正确；
- flow MSE 会优先奖励重建主体、纹理和时间一致性，不会自动要求“J 必须左、L 必须右”。

### 6.1 按时间的修改脉络

早期纯 PRoPE 诊断采用 rank-32 LoRA、约 200 steps，目标是先证明相机矩阵能进入 H3；它的失败只能说明该容量/监督/数据组合未通过控制诊断。之后的 hybrid/post-AdaLN 实验改为 rank-64 LoRA，并解冻 action encoder/adapter/gates、proj/norm/out 以及部分后层 attention；一次启动审计显示约 2.393B 参数可训练。这个数字不能和“LoRA 参数量”混用：它包含选定的 H3 模块/分支，且不同配置的 FSDP 包装统计口径可能不同。

因此，后续比较必须同时记录：`trainable_total`、LoRA rank/target、PRoPE range、`prope_o`/gate 是否解冻、action injection interval、loss 权重和数据索引。只报告 checkpoint step 或总 flow loss 会掩盖注入容量的变化。

## 7. 为什么旧 FastVideoWM 实验看起来“不起作用”

### 7.1 注入容量太小或出口接近零

旧 exact PRoPE 的 `[96,108)` 只占每头 128 维中的 12 维；部分 run 只训练后八个 block 的 LoRA，或只训练 adapter。相比原 H3 的大容量主路，这条支路很容易成为小扰动。当前 residual 的 `prope_o` 零初始化在正确设计下有利于保护 baseline，但也意味着初期必须靠 flow gradient 学出唯一出口。

### 7.2 监督目标没有强制方向

旧实验的核心仍是 flow MSE。没有稳定的 `correct < static < reverse` margin，就可能学到“场景看起来像真实视频”而不是“相机沿给定方向运动”。历史日志中相机排序只低频计算，且有实验将所有辅助权重置 0；因此 loss 下降不等于 action 遵循。

### 7.3 数据/标签不是严格 counterfactual

真实视频通常含复合相机运动；“名称上是 J/L”不代表每帧只有 yaw。若 J/L 首帧、主体或背景不完全相同，模型可以从首帧和纹理捷径预测视频，不依赖 camera path。此前诊断还发现部分旧缓存窗口 camera 未以窗口首帧为原点、JS-easy K 曾是 672×384 pixel 坐标而不是归一化 K；这些几何误差会直接破坏 PRoPE 的投影尺度。

### 7.4 归一化和裁剪抹掉了速度/方向尺度

`normalize_camera_translation=true` 会把不同视频平移尺度压成近似同一量级；旧 action scales/clip 也可能把后半段累计位姿截断。这样模型只能感知“有运动”，难以分出快/中/慢和精确幅度。当前 canonical pipeline 保持首帧相对 C2W，并关闭独立平移归一化。

### 7.5 H3 与 Cosmos/Wan 不能直接类比

Cosmos/Wan 的有效实验通常包含全量或更大范围 DiT 微调、不同 AdaLN/flow 约定和不同 packed layout；FastVideoWM 旧实验多为 H3 baseline + LoRA/少量模块。相同数据和“2000 steps”不意味着相同的有效参数更新量、噪声覆盖或梯度强度。

### 7.6 结构、数据、steps 同时变化，无法归因

旧 pure PRoPE、hybrid、post-AdaLN 之间同时改变了：PRoPE 通道范围、action 表示、LoRA rank、可训练 block、数据规模、noise floor、aux loss 和 sampling。因而不能从某一次失败推断“PRoPE 无效”，也不能从当前响应推断“residual branch 单独解决了全部问题”。

### 7.7 语义协议曾经不一致

旧 H3 文本缓存中 I/K 方向与当前数值协议相反；如果用旧评估脚本解释新数值结果，会把正确动作判错。当前协议固定：I=上转，K=下转，J=左转，L=右转。

## 8. 推荐的可证伪验证顺序

在扩大数据前，固定同一 checkpoint、首帧、seed、noise 和 denoising steps，做以下四个推理：

1. 无 camera（静止/identity）；
2. 正确 trajectory；
3. reverse trajectory；
4. 交换 `prope_o` 为零或关闭 camera branch。

同时记录：背景位移方向、端点相机姿态、flow/PSNR/SSIM/LPIPS、`correct-static` 与 `reverse-static` 的差值，以及 `prope_o`/LoRA 梯度范数。只有第 2 比第 1、第 3 明显反向，且关闭 `prope_o` 后差异消失，才可确认相机分支真正控制了运动。

建议的最小 ablation：

```text
A: H3 baseline，camera off
B: exact PRoPE [96,108)，其它完全相同
C: residual PRoPE [0,128)，仅 prope_o
D: residual PRoPE + rank-64 LoRA
E: D + post-AdaLN action adapter
```

A–E 必须使用同一批严格首帧重合、已验证 camera path 的样本；先用 100–200 steps 做 wiring/梯度/过拟合诊断，再扩到全量。不要在这个 ablation 中同时更换数据、采样、噪声和 loss。

## 9. 对后续训练的建议

1. **先保证 geometry contract**：逐条验证首帧 C2W identity、37 帧数量、pixel-K 转 normalized-K、I/K/J/L 方向和视频可见运动一致。
2. **先用 residual PRoPE + 可学习 `prope_o`，再逐步加 adapter**：camera-only branch 先证明 viewmats 能产生可控响应；如果方向正确但幅度不足，再加 post-AdaLN adapter。
3. **加入直接但小权重的 counterfactual 监督**：同一首帧和同一噪声，正确轨迹、static、reverse 三次 forward，要求正确分数低于 static/reverse；权重应小于 flow 主损失，避免破坏画质。
4. **按方向、幅度和来源分层采样**：平移/旋转分别统计慢中快三档，避免 multicam_video 数量主导 batch；先 W/S、A/D、I/K、J/L 均衡，再扩复合轨迹。
5. **容量逐级放大**：先 `prope_o` 全 block/隔 block，随后解冻后 8–12 个 block 的 to_q/k/v/out；只有当小样本无法过拟合时，才扩大 LoRA 或解冻更多 H3 backbone。
6. **每 50–100 steps 做固定验证**：不只看总 flow loss，必须看 action-on/off、correct/static/reverse、方向准确率、幅度响应和视频质量。验证集必须场景 disjoint，训练场景成功只能算 wiring/记忆诊断。

## 10. 代码与配置索引

- FastVideoWM PRoPE 几何：[`minimax_h3_prope.py`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3_prope.py>)。
- FastVideoWM H3 world model：[`worldmodel.py`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/train/models/minimax_h3/worldmodel.py>)。
- FastVideoWM attention/action：[`minimax_h3.py`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/fastvideo/models/dits/minimax_h3.py>)。
- 早期 additive：[`minimax_h3_worldmodel_stage0_v2.yaml`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_v2.yaml>)。
- hybrid probe：[`minimax_h3_worldmodel_stage0_hybrid_probe.yaml`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_hybrid_probe.yaml>)。
- full PRoPE v5：[`minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml>)。
- post-AdaLN：[`minimax_h3_multiaction_postadaln_20261002.yaml`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_multiaction_postadaln_20261002.yaml>)。
- 当前 residual：[`minimax_h3_full_residual_prope_20261003.yaml`](</home/lizihao/projects/military_world_model_agent/FastVideoWM/examples/train/configs/minimax_h3_full_residual_prope_20261003.yaml>)。
- H3-World 文本 injector：[`inject_abot_text.py`](</home/lizihao/projects/worldmodel/H3-World/code/abot/inject_abot_text.py>)。
- H3-World camera text：[`camera_text.py`](</home/lizihao/projects/worldmodel/H3-World/code/vwm/camera_text.py>)。
- H3-World 纯文本推理：[`infer_camera.py`](</home/lizihao/projects/worldmodel/H3-World/code/vwm/infer_camera.py>)。
- minWM PRoPE：[`minimax_h3/prope.py`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/prope.py>)。
- minWM model guard：[`minimax_h3/model.py`](</home/lizihao/projects/military_world_model_agent/minWM/minwm/modeling/minimax_h3/model.py>)。

## 11. 最终判断

旧 FastVideoWM 的“action 不起作用”最合理的解释是多因素叠加：早期 PRoPE 几何子空间窄、主路/LoRA 容量占主导；`prope_o`/adapter 初始或训练后仍弱；flow MSE 没有明确方向 margin；真实视频含复合运动且部分 camera/K/首帧对齐错误；H3 LoRA 与 Cosmos/Wan 全量微调并非同一优化问题。后续应先用固定 ablation 证明“矩阵确实进入并由 prope_o 产生响应”，再加直接 counterfactual/action loss 和更大数据，而不是继续盲目堆 steps。

