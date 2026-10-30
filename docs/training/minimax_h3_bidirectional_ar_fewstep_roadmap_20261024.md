# MiniMax-H3：双向控制模型 → AR → 少步 AR 的训练路线

日期：2026-10-24。本文是代码核查后的**后续实施方案**，不是已完成的实验结果。
本次仅新增文档，未修改、停止或启动现有标注、编码及训练任务。

## 1. 结论与模型关系

推荐主线：

```text
S0 双向 H3 相机＋文本 teacher
  → S1 多步 block-causal AR（真实历史、flow matching）
  → S2 causal consistency distillation（先每块4次去噪）
  → S3 自生成历史上的 asymmetric DMD（改善滚动稳定性）
  → S4 长时滚动，再单独尝试每块2步／更细粒度交互
```

保留现有 H3 骨干、已经学到的 residual PRoPE、LoRA、首帧和文本条件。
S1 从验收最好的 S0 初始化，不回到无相机控制的官方裸模型重新学动作。
每个阶段独立 run、优化器、配置和计数；同阶段中断则恢复完整状态，不能误当新阶段重置。

“AR”指时间上按块预测；“少步”指每块的去噪次数；“小模型”指网络参数量下降。
三者不同：本方案前期不缩小约33B的骨干，4步AR也不是整段视频总共forward4次。

### 为什么采用这条路线

- CausVid 展示了双向 teacher 监督因果少步学生的路线，但其初始化不能不加区分地套给 H3。
- Self Forcing 关注真实历史训练与自生成历史推理的分布差异；只做 teacher forcing 不能自动解决长期漂移。
- Causal Forcing / Causal Forcing++ 将 AR teacher、因果 ODE/CD 初始化与后续分布匹配分开。
  本方案优先 causal CD，避免预先生成、存储巨型多步 ODE 轨迹。
- 本地 minWM 已有 H3 专用 AR/CD/DMD 参考，而当前 FastVideoWM 尚缺完整 H3 因果接口。
  应移植经过核对的契约和测试，不直接替换整个训练框架。

参考原论文：[CausVid](https://arxiv.org/abs/2412.07772)、
[Self Forcing](https://arxiv.org/abs/2506.08009)、
[Causal Forcing](https://arxiv.org/abs/2602.02214)、
[Causal Forcing++](https://arxiv.org/abs/2605.15141)。
论文的速度、收敛步数和硬件结果不等于本机 H3 的预期结果。

## 2. 当前可复用的基础与真实缺口

当前数据/自动链见 `union_dual_text_20261023.md`、`control_precision_todo_20261024.md`。

- 完整 union 注册115,988个124帧窗口，计划113,207训练／2,781验证。
  这是注册规模，不代表本文确认所有文本及条件验收已完成。
- 条件：首帧原生VAE、首帧Qwen视觉token、版本1场景＋主体外观/已观察运动文本、逐latent C2W。
- 当前训练保留 residual PRoPE、rank64注意力LoRA和部分主层，共约23.93亿可训练参数。
- K保存和审计，但当前PRoPE配置K-off；音频模块保留冻结，音频输入长度0。
- 当前额外action loss未启用。后续排序、几何监督及CFG是待实现工作，不能冒充现配方。

| 项目 | 当前事实 | 后续工作 |
| --- | --- | --- |
| 主注意力 | `minimax_h3.py` 的 `MiniMaxH3Attention` 为 `causal=False` | 主路分块因果mask |
| 相机分支 | `_residual_video_attention()` 独立执行全视频attention | 独立因果mask及KV |
| 文本/首帧prefix | 参与packed joint attention | 禁止prefix读目标视频，防止跨层未来泄漏 |
| 时间调制 | DiT已有逐行timestep索引 | wrapper需支持逐块sigma，不能继续只读scalar |
| KV | 只有文本refiner/RoPE预处理cache | 新建主路和相机路历史KV，不混称已有cache |
| 方法接口 | H3仅接受dense conditional、data latents | 增加H3因果模型接口、采样与score适配 |
| 少步训练 | 仓库有其他架构DMD/SF/CD | H3不支持直接换YAML类名运行 |

主要入口（相对FastVideoWM仓库）：

- `fastvideo/models/dits/minimax_h3.py`
- `fastvideo/train/models/minimax_h3/worldmodel.py`
- `fastvideo/pipelines/basic/minimax_h3/packing.py`
- `fastvideo/train/models/base.py` 的 `CausalModelBase`
- `fastvideo/train/methods/fine_tuning/{tfsft,dfsft,worldmodel}.py`
- `fastvideo/train/methods/consistency_model/causal_cd.py`
- `fastvideo/train/methods/distribution_matching/{dmd2,self_forcing}.py`

实现沿用新栈 `fastvideo/train/`，不从legacy `fastvideo/training/`交叉导入。
当前worldmodel `predict_noise()`会丢弃外部timestep，使用`batch.timesteps[0]`，并拒绝unconditional/VSA。
现有通用DMD/SF调用unconditional teacher、采用其他scheduler及chunk默认值；必须逐项适配。

## 3. 数据安排：先复用，再补真正需要的新信息

### 3.1 短程基线与采样

先完成当前已批准的版本1文本基线，不擅自修改排队的3000步配方。
之后冻结一个可复现的训练/评估版本，再比较改动。

建议未来训练采用“覆盖队列＋平衡控制队列”，而非只保留十几个纯J/L视频：

- 约70%–80%抽样来自完整有效训练池，用无放回基础队列记录覆盖率。
- 约20%–30%来自可靠控制样本，覆盖全部WASDIJKL、静止相机且主体可动、复合动作。
- 两队列共同约束来源：multicam目标上限30%–35%，spatialvid上限40%，
  其余按可靠供给安排；这些是抽样配额建议，不是声称已有同等数量的罕见数据。
- 方向按实际逐帧位姿增量统计，不把复合视频强行变成一个纯指令。
  在供给充分的方向间，目标有效动作帧数量最大/最小不超过约2倍；稀缺方向限幅重采样并披露实际独立视频数。
- 慢/中/快先按来源内可信尺度分层，目标大致各1/3；不逐clip除最大位移抹掉速度。
  单目单位不能直接标成米，来源间速度比较需校准。
- AR调试可冻结约1万–2万条分层工作集作为目标规模，真实数量取审计后的可用数；
  200步只用于端到端诊断，通过后扩至完整池，不将“小集合过拟合”作为最终结论。

版本2背景文本描述的是同一GT，不是“主体静止”或反向动作负例。
主体前进/倒退的样本必须来自真实行为，不通过改caption制造错误监督。

### 3.2 124帧latent可复用，但时间边界不能套Wan

当前目标`[24,37,30,52]`，首帧条件`[24,1,30,52]`，分辨率832×480。
原生VAE将连续17 RGB帧分组编码，再在完整视频尾部丢弃3个latent：

```text
124 RGB → 37 latent → 5+5+5+5+5+5+5+2
243 RGB → 72 latent → 14×5+2
```

第一版选择每块5个latent，匹配原生分组与minWM参考，124帧最后2个latent使用显式短尾块。
不能把尾块移到开头、静默丢弃，或将每个17帧块分别调用公共`encode()`：公共入口每次都丢尾3个，结果不再是5个。
正确做法是读取完整已编码tensor后分块。

本次源码核查纠正一个历史报告的混淆：

- 当前encoder是左侧时间padding的causal CNN，GroupNorm逐时间帧独立。
  “17帧整块编码必然泄漏整块未来”不适用于这份实现；5是保守的首版分块选择，不能据此证明1 latent在编码端必然不因果。
- 当前pose锚点为`17*floor(i/5)+4*(i%5)`，即`0,4,8,12,16,17,21,...`。
- 原生RoPE使用各span起点：`0,1,5,9,13,17,18,...`，再乘原生尺度并加origin。
  **pose锚点与RoPE位置不是同一个序列**，也不能用统一stride4替代。
- 源码入口：`dataset/vwm_h3.py:h3_latent_source_frame_indices()`、
  `pipelines/basic/minimax_h3/packing.py:temporal_position_grid()`、
  `models/vaes/minimax_h3_video.py`。

复用条件：完整源帧索引、裁剪、归一化、17帧phase及编码版本一致。
保留独立首帧条件，不假设它等于目标latent的第0帧。
必要的因果测试比较posterior mean/logvar，避免不同随机采样tensor形状产生误判。
修改pose派生条件不要求重做不变的视频VAE。

### 3.3 真正在线AR的数据契约

新增派生字段：原视频ID、窗口/全局帧起点、24fps时间戳、RGB索引、latent锚点、
encoder phase、块ID/有效尾长、逐帧C2W/K、坐标基准、尺度来源、caption作用时间及可信度。

有两种不同任务，必须显式标记：

1. **规划式生成**：用户提前提供整个未来文本/轨迹；未来计划可作为外生输入。
2. **在线交互**：只知道当前收到的指令；未来GT图像、未来实际位姿及未来观察事件不能泄漏给当前块。

现有caption观察了完整窗口，作为“计划条件”可用于条件生成；不能直接声称是“只见历史的观察”。
如果测试在线任务，需要首帧可见场景描述、用户给定主体计划，或按可见历史更新文本。
旧cache某些相机标定使用全窗口最大平移，当前`normalize_camera_translation=false`不会追溯删除此统计。
严格在线版本需采用固定来源/全局标定尺度，重新派生camera条件；不是重新编码视频。
额外action features若未来启用，也不能未经核对沿用整clip最大位移归一化。

### 3.4 长视频不是拼接独立短clip

现有124帧池足够做第一版AR/CD，但长时稳定性需连续原视频：

- 243f是另一套注册/QC产物，不能据目录存在就说VLM、VAE、72帧pose已训练就绪。
- 243帧@24fps约10.1秒，345帧约14.4秒，都不能冒充15秒或60秒真实GT。
- 从同一长视频构造连续多块，保留全局时间、相机坐标及无场景切换标记。
- 两个独立124帧cache不能简单拼成243帧GT，裁剪起点与17帧phase可能不同。
- 15秒/60秒合成轨迹测试可以没有GT；只能报告控制、质量、稳定性，不能制造PSNR/SSIM真值。

## 4. S0：把双向控制模型变成可靠teacher

完成当前文本基线后选“控制＋画质”最好的完整checkpoint，不机械选择最新step。
使用固定首帧/seed，保持文本和相机输入契约一致。

改进顺序：

1. 先建立相机、主体文本、画质分开的指标；核查C2W、符号、源尺度、真实文本token及分布。
2. 引入上一节分层采样，保留全量真实复合轨迹，不因评估想看纯W就删去GT其他轴。
3. 保留flow-matching MSE，对可靠样本逐项增加条件区分监督。
4. 文本仍弱时再单独比较解冻`context_embedder`，既有text-refiner LoRA已经可训练，不能说原来完全没训文本。
5. 不同时切K-on、CFG、rank、所有loss和采样，否则难以解释提升来源。

可用的loss结构：`L = L_flow + λc L_camera_rank + λt L_text_rank`。
排序固定图像、noisy latent、sigma和GT，只改变一个条件，要求正确条件的重建误差小于错误条件。
仅对可靠、有信息量的样本计算；先低频计算并校准梯度，让新增项早期不主导主loss。
具体λ由loss/梯度量级决定，不把未经测量的固定权重当成最优值。

排序能减少“忽略条件”，不能单独证明物理方向正确。
物理相机监督须看静态背景、可信姿态估计及有效梯度；不能把纯W约束成水平平均光流0，
也不能把静止相机强制成整画面静止。主体运动需区分自身朝向、相机视差与屏幕位移。
相关风险和实施入口详见`control_precision_todo_20261024.md`。

验收建议：全部八方向有响应、相反方向趋势正确，非指令轴串扰不恶化；
同轨迹更换可靠主体文本能改变主体行为而非主要改变相机。若teacher未达到这一点，少步蒸馏不会自动补出缺失语义。

## 5. S1：先得到正确的多步AR

### 5.1 Teacher forcing 与mask

概念布局：`[prefix | clean blocks | noisy blocks]`。

| query | 可见keys |
| --- | --- |
| text/首帧prefix | 仅prefix |
| clean block b | prefix＋clean blocks≤b |
| noisy block b | prefix＋clean blocks<b＋本块noisy |

预测当前块不能看同块clean target，prefix也不能先读未来再经下一层泄漏。
主attention和video-only PRoPE attention必须同时实现对应mask。
块内仍可双向，因此这是block-causal，不是每个RGB帧都即时响应。
用普通`causal=True`token三角不能替代上述语义。

clean/noisy同一物理帧共享原生位置，不将它们误当时间上连续的2T帧。
文本、首帧prefix固定时间以便缓存；当前块单独sigma。
H3原生已有逐行AdaLN索引能力，无需仅为多sigma重新设计Transformer。

### 5.2 时间、速度与采样适配

```text
z_sigma = (1-sigma)*z0 + sigma*epsilon
t_H3 = 1-sigma
native_output = z0-epsilon
wrapper_velocity = epsilon-z0
z0_prediction = z_sigma-sigma*wrapper_velocity
```

相机/位置、sigma/time运算保留正确FP32边界；不要先转BF16或把724.8等时间截成整数。
shift=12只应用一次，实际sigma表与模型调用数必须保存。
原生“50步”可能包含终点，不能直接当作50次forward。
H3已发布guidance-distilled模型不等于少步AR；第一版维持conditional-only/guidance1。
若未来做text-only CFG，需训练真实的丢文本支路并保留首帧/相机，不能伪造unconditional输入。

### 5.3 缓存与正确性测试

- prefix prefill一次；主路KV和相机路KV分别存储。
- 当前noisy块只读历史，不将各去噪步的临时KV提交为历史。
- 去噪完成后以clean历史时间refresh一次，再提交detached KV。
- 不每块重新将相机归零；保持共同坐标和全局RoPE位置。若滚动换基准，所有缓存变换必须保持一致。
- 修改文本或已提交历史动作时，识别受影响缓存并失效/重建，不继续用旧prefix KV。
- activation checkpoint重放使用不可变write offset/相机快照；反向前不释放或覆写所需cache。
- 先无cache参考forward，再证明cached/full-prefix输出一致；梯度比较必须使用相同的历史detach/截断策略，
  不能拿detached历史cache与全历史BPTT作等价比较。
- 先单rank小模型，再真实权重小批，之后SP2/多卡；保留现有重放一致性检查。

先保留rank64、已训练的相机分支和现有部分解冻，减少变量。
若通过正确性测试后AR适配仍不足，单独增加全部主block的注意力可训练覆盖，而不是先堆一个新action模块。
全量FFN解冻暂不作为默认，显存和训练成本需先测。

### 5.4 训练窗口

先完成200个optimizer steps的真实多方向端到端诊断，每100步保存，每200步固定推理。
通过后计划2000–4000步的AR适配窗口，按控制与画质曲线决定继续，而不是保证这些步数必收敛。
初期真实clean历史为主；随后可单独测试小幅context noise，并使cache refresh时间与训练一致。
任意生成历史与原GT强配重建可能产生错监督，不作为“随便混一点rollout”的捷径。

## 6. S2：因果一致性蒸馏，先追求每块4步

角色：冻结的S1 AR teacher、可训练AR student及EMA target。
同一个真实clean历史下取相邻sigma：teacher做一步ODE推进，EMA预测下一噪声状态的clean结果，
student拟合该停止梯度的一致性target。末端覆盖sigma=0。

不能用双向teacher看完整未来得到的ODE终点，直接当作严格AR学生同输入的确定性回归目标。
双向teacher仍可在S3作为分布score；这与S2因果flow-map回归是不同用途。

建议先1000–3000个student更新作为初始预算，每200步比较8/4步；
4步通过后才考虑2步。预算是本项目建议，不是抄论文步数的收敛保证。
主要优化一致性loss，保留可控的小比例真实数据锚定作为独立对照，不同时开启一堆旧motion loss。

所有teacher/student/EMA必须接收相同首帧、文本、相机和mask；不能在蒸馏缓存里只存text丢掉pose/tags。
保存EMA及完整恢复状态，恢复时不能再次用student覆盖已保存teacher/EMA。
记录每个块实际NFE、refresh次数、sigma列表、采样器，区分4次去噪与额外prefill/refresh。

## 7. S3：自生成历史上的DMD，解决滚动误差

当S2在真实历史上好、自己滚动却漂移时，引入本阶段；不把它当作修输入bug的工具。

| 角色 | 来源与职责 |
| --- | --- |
| causal generator | S2学生；使用自己的已生成块继续rollout |
| generator EMA | 可选部署权重／稳定目标，单独保存 |
| frozen real score | 合格S0双向控制teacher，不是无相机裸H3 |
| trainable fake score | 从S0初始化，在generator当前分布上学习去噪 |

用real/fake score差异给generator分布匹配梯度；fake score训练目标来自当前生成样本加噪。
这不是把整段fake强行逐像素拟合某个GT，也不是普通GAN二分类器。
双向score可以查看完整noisy fake clip，但不能把未来真实视频输入部署generator。

先按每5次critic更新对应1次G更新测试稳定性，明确分别记录两种计数。
初始预算2000–5000个critic/外循环更新，对应约400–1000次G更新；
每100次G更新做固定验证，不能把5000外循环叫作5000次generator训练。
更新频率与预算依据fake-score跟踪和rollout曲线调整，不承诺固定比例最优。
使用截断梯度、历史detach、分布式同步exit step；不能让各FSDP rank进入不同次数的forward。

DMD若在八卡完整更新/第二次更新/保存恢复的峰值验证不过，就先交付S2四步模型，
再优化sharding、角色可训练量和cache，不通过无限CPU offload重演主机OOM。

## 8. S4：长时滚动、低延迟与2步的顺序

先保留4步稳定模型，扩生成时长5秒→约10–15秒→30秒→60秒。
训练用连续长视频和短rollout到长rollout的课程；真实GT不足时明确区分无GT合成控制测试。
加入滚动历史上限、全局位置偏移、首帧锚点保留/attention sink及缓存淘汰，逐项消融。
固定horizon cache不等于已实现无限生成。

243f数据需真正实现72latent、72pose/K、长窗口caption及独立验收；不能只改num_frames。
首轮可在124f上完成AR/CD，不要求为此重做全部VAE；长数据再按可用量补齐。

之后再比较4步→2步；如果要缩到1 latent粒度，必须重新验证分块、时间映射、条件和解码延迟。
暂不把更换VAE、压缩骨干、稀疏注意力/量化与2步蒸馏一起做。

### 原生decoder的不可忽视延迟

当前decoder是full-attention ViT，`z[5c:5c+7]`才产生对应17帧的已完成输出，另有5帧blend。
即5个当前latent＋2个未来latent，**不是仅缓存过去的重叠**。
每5latent AR块约对应17/24≈0.71秒画面；decoder还需后续块提供lookahead。
因此第一版是有缓冲的chunk-AR，不承诺逐RGB帧零等待或24fps实时。

先实现有界lookahead的原生解码/并行流水；真要更低延迟，再独立训练因果/低lookahead decoder，
以原生解码作teacher加重建、感知及时间一致性验收。更换decoder前先固定latent生成器，避免归因混乱。

## 9. 八卡资源与训练预算

### 9.1 有效batch

第一版目标total BS32，按**独立视频窗口**计，不把同一个视频的8个块当8条独立视频。

| 并行候选 | 数据组 | micro/video | grad accumulation | total BS |
| --- | --- | --- | --- | --- |
| 8GPU，SP2 | 4 | 1 | 8 | 32 |
| 8GPU，SP4 | 2 | 1 | 16 | 32 |
| 8GPU，SP8 | 1 | 1 | 32 | 32 |

这是batch算式，不是已经验证的H3 AR/CD/DMD并行配方；FSDP组、heads、角色分片和通信仍需验证。
S1先测SP2，S2/S3按角色显存再测SP4/SP8。不先为追BS64牺牲稳定性或盲目降低分辨率。
同时报告video数、有效target latent帧数/token数，避免不同history/块长造成“同BS”的假比较。

### 9.2 DMD不能假设一定装得下

只读取原生safetensors头部，本机基础DiT约33.123B参数、61.73GiB权重，新增PRoPE/LoRA另算。
理想8路全分片时，单份BF16权重约7.71GiB/卡。
G/teacher/fake/EMA四份BF16约30.85GiB/卡；FP32 G/EMA加BF16双score约46.27GiB/卡，
以上尚未计入梯度、optimizer、all-gather峰值、激活、双KV、临时视频和保存峰值。
两个33B网络全量Adam训练，仅BF16梯度与两个FP32 moment另需约77GiB/卡，不能当作八卡80GB可运行方案。

因此S2/S3初期保持PEFT/部分层训练；冻结辅助网络也必须正确分片，不能每rank完整复制。
先测完整一次及第二次G/critic更新和checkpoint保存/恢复，记录GPU allocated/reserved、主机RSS与cgroup上限。
主机可用额度取物理内存与cgroup限制的更小值并预留系统/加载/保存空间，不能仅凭机器标称760GB调大阈值。
逐角色加载，限制CPU worker/prefetch，禁止八进程各自同时加载多份完整pipeline。
磁盘按实际剩余量预算，不默认存全量50步teacher轨迹；causal CD优先在线一小步teacher构造target。

### 9.3 时间只能先测后承诺

历史124f双向、八卡BS32约208–212秒/optimizer step：

- 200步约11.6–11.8小时；
- 1000步约58–59小时；
- 3000步约7.2–7.4天，不含验证/保存/故障。

这些不是AR/CD/DMD测速。AR TF可能复制clean/noisy序列，CD需teacher/EMA，DMD需rollout和score，
训练不保证比双向快；KV主要改善流式推理的历史重算。
每阶段先热身后测20–50个稳态更新，分别报告主训练、验证、prefill、cache refresh和解码开销，
再用剩余更新数×实测中位数并留10%–20%余量给ETA。
整条路线应按多阶段研发＋多轮多日训练安排，不承诺一晚完成。

## 10. 验收：不能只看loss和“两个视频有差异”

### 正确性硬门槛

- 扰动未来GT视频，早期块输出不变；prefix和PRoPE均覆盖此测试。
- noisy块看不到同块clean target；直接/间接未来泄漏测试均通过。
- 改未来在线指令不影响此前输出；若允许未来计划则单独标记任务模式。
- cached/full-prefix输出一致；相同历史detach/截断策略下梯度一致，另测AC开关梯度、SP1/SP2及完整恢复一致。
- C2W共同原点、pose锚点、RoPE全局偏移、短尾块和decoder完整输出正确。
- 因果encoder测试与有界lookahead decoder测试分开，不能把二者合成一句“VAE因果”。

### 效果测试

- 沿用JS-easy固定首帧、WASDIJKL和真实GT轨迹两套；另加未被父模型训练过的场景。
- 静态背景估计旋转/平移方向、非目标轴串扰；速度给0/0.5/1/2倍，来源单位不可靠时只看相对单调响应。
- 相机{静止,W,S}×主体{前进,倒退,不指定}的交叉测试；方向可靠的主体才判前后。
- 加入动作切换：W→A、J→L、W→停止，记录收到指令到画面响应的延迟和过冲。
- 对真实GT轨迹算PSNR/SSIM/LPIPS等，但随机生成无法逐像素对齐时联合解释；
  无GT单动作/长视频只算无参考质量、控制与长时稳定性，不伪造重建指标。
- 15/60秒记录身份漂移、背景崩坏、冻结、接缝、轨迹累计误差；多场景多seed。
- 记录TTFF、稳态输出fps、P50/P95块延迟、显存、CPU峰值。
  去噪4步不等于12.5倍端到端加速；实时要求含解码的块耗时跟得上对应视频时间。
- 原始视频用于指标；单独输出用户要求的指令/prompt HUD，不用HUD证明遵循。

初始放行建议（需先确认指标估计可靠）：AR/少步版本八方向成功率相对teacher下降不超过5个百分点，
可信背景轨迹误差中位数恶化不超过约10%–15%，画质无明显系统性退化；
达不到就停在上一阶段分析，而不是单看flow loss继续压步。

## 11. 工程工作包与参考分支

以下名称为拟新增接口/文件，不代表已经存在：

1. `MiniMaxH3CausalWorldModel`：继承因果模型契约，分块sigma、TF及streaming入口。
2. H3 causal packing/mask模块：prefix、clean/noisy块、视频索引、短尾、位置及相机时间表。
3. H3双路KV manager：prefill、read-only denoise、commit/refresh、replay snapshot、release。
4. H3 noise/score adapter：continuous sigma、速度符号、shift、conditional-only和真实model能力声明。
5. H3 causal-CD配方：teacher/student/EMA、终点约束、保存恢复。
6. H3 DMD配方：real/fake/G、独立更新计数、截断梯度和角色内存预算。
7. 逐阶段YAML/launch/eval：独立输出root、父checkpoint hash、数据/文本/几何版本及恢复命令。
8. CPU小模型测试＋真实权重GPU冒烟＋多卡/保存恢复测试，先通过再长训。

本地参考：`/home/lizihao/projects/military_world_model_agent/minWM`。

- `6dfb3d927405aae52c5c207638855e2b1ad876f3`：H3 causal adapter、mask、双cache、精度及重放修复。
- `8dfe6a1`：完整H3四阶段配置参考，接口与上一条线不完全一致。
- 使用`git show <ref>:<path>`阅读，不能切换/覆盖正在工作的checkout。
- minWM复制prefix、仅切video；FastVideoWM切整个packed序列，SP布局不同。
  迁移mask/cache不变量和测试，不直接复制其SP enter/exit。
- 用同输入双向forward核对state-dict转换，LoRA合并/未合并状态清晰，避免重复加权或丢相机权重。

可选后续研究：[DMAD](https://arxiv.org/abs/2610.02188)报告过MiniMax-H3四步视听蒸馏，
但不等于本项目相机AR已验证。待主线AR/CD稳定后才评估其共享判别骨干路线，
不为追最新方法同时重写控制、因果结构及蒸馏。

## 12. 最小可交付顺序

| 里程碑 | 最小产物 | 初始训练预算，不保证收敛 |
| --- | --- | --- |
| M0 双向teacher | 有文本/相机的可靠checkpoint及分项评测 | 已批准3000步基线；额外改进另立run |
| M1 多步AR | 5-latent分块、无未来泄漏、双cache、37帧尾块 | 200步诊断→2000–4000步 |
| M2 四步AR | causal-CD student/EMA与同条件对照 | 1000–3000次student更新 |
| M3 自历史稳定 | DMD后5秒控制不丢、rollout改善 | 2000–5000次critic，约400–1000次G更新 |
| M4 长时低延迟 | 15/60秒测试、缓存上限、4步与2步对照 | 按长数据供给/实测曲线另预算 |

优先保证M0/M1/M2可独立交付，不把M3显存风险变成“做不出可用AR”的单点阻塞。
训练、推理阶段的计数、数据版本和质量结论分别保存；没有运行过的项目保持待执行状态。
