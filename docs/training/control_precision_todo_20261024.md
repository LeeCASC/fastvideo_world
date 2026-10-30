# FastVideoWM 相机与主体文本控制：完整实施 TODO

更新日期：2026-10-24，时间统一使用日本时间 JST。

> 本文件是后续修改清单，不是已经完成的实验记录，也不是执行全部修改的授权。
> 当前只批准：补齐双文本、使用版本1验收后自动续训。不要因阅读本清单而中断现有作业、
> 修改排队配方、增加 loss、切换模型结构或启动第二个八卡任务。

## 0. 阅读入口、现状与完成定义

仓库：`/home/lizihao/projects/military_world_model_agent/FastVideoWM`。
下文代码路径均相对该仓库；H3-World 是另一条原生 DiffSynth 管线，不要混用环境与缓存。

先阅读：

- `docs/training/union_dual_text_20261023.md`：当前数据与自动训练契约。
- `docs/training/control_precision_plan_20261023.md`：已知证据、假说、监督风险。
- `docs/training/h3_fastvideowm_action_injection_report_zh_20261014.md`：历史注入方式与实验边界。
- `/home/lizihao/.codex/skills/fastvideowm-h3-handoff/SKILL.md` 及其最新交接段落。
- 改代码前阅读修改目录对应的 `AGENTS.md`，保留 dirty worktree 中已有工作。

### 已落实，不应反复重做

- [x] 已建立115,988条完整窗口索引：旧65,389＋新增50,599，源目录只读。
- [x] 已实现主体运动／仅背景双文本、原始分片保留和独立整句清理记录。
- [x] 已实现同视频、完整帧索引、窗口、caption/registry/plan 哈希绑定。
- [x] 已实现原生条件编码复用及持久化 tensor、真实文本/视觉 token、首相机原点验收。
- [x] 已实现验收通过后使用版本1自动训练，及新轮存在checkpoint时禁止从父模型误重开。
- [x] 已保留文本训练checkpoint-100、历史日志和W&B；旧run约21步未保存进度不可恢复。
- [x] 上次加固的32项CPU回归及pre-commit已通过；不代表未来GPU端到端或收敛已通过。

### 当前仍未完成

- [ ] 完整双文本全部通过；2026-10-24 00:54 JST快照仍在第三轮，第二轮最终遗留941条。
- [ ] 新合集版本1条件编码与全量验收完成。
- [ ] 新轮实际完成optimizer step，并产生完整checkpoint。
- [ ] 八方向及主体前后动作的精度验收通过。

以上是快照，不是实时监控。核对实际 `chain.log`、分片日志、marker、GPU进程和checkpoint后再改勾选。

### 当前已确认训练配方

| 项目 | 约定 |
| --- | --- |
| 数据根目录 | `/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023` |
| 训练文本 | `subject_motion`：环境＋主体外观＋有证据的主体运动 |
| 对照文本 | `background_only`：排除前景主体及行为；当前只准备描述清单 |
| 计划划分 | 113,207训练／2,781验证，按源视频隔离；父模型可能见过部分视频 |
| 初始化 | `fastvideowm_scene_text_20261022/runs/train/checkpoints/checkpoint-100`模型权重 |
| 优化器与计数 | 新轮重新初始化，不回退至旧action-only800，也不冒充旧run精确恢复 |
| 八卡batch | SP2，DP4，micro1，accum8，total BS32 |
| 训练计划 | 3000新steps，保存及loss验证每100steps，启动时先做loss验证 |
| 模型 | residual PRoPE，每隔一层设独立视频分支，每头128维，rank64 LoRA＋部分主层解冻 |
| 文本 | 实际输入；两个text refiner注意力block已有LoRA，context投影等基础参数冻结 |
| 监督 | 主要为视频去噪flow-matching MSE，额外action/camera loss均未启用 |
| 几何与音频 | K存储和审计但PRoPE K-off；不逐视频再归一化平移；音频输入长度0 |
| 稳定性 | eager模式，保留full activation checkpointing及一致性检查 |

每项TODO完成至少留下：代码/config差异、输入/权重hash、测试命令及结果、指标或审计产物、
结论及回滚点。只有写了代码或看到显存占用，不足以勾选完成。

## 1. 推荐实施顺序与依赖

| 优先级 | 工作包 | 依赖与资源 | 通过条件 |
| --- | --- | --- | --- |
| P0 | C：标注尾部、去重与恢复 | CPU规则先行；真正缺失才用GPU | 不编造描述，全部ID有明确有效/待审状态，能安全接续 |
| P0 | O：自动链状态与恢复 | 只读检查；代码测试不占训练GPU | 不重复启动、不误重置、不绕过验收 |
| P1 | I/G：输入与几何验收 | CPU全量＋少量原生GPU复编码 | 实际训练输入与清单、首帧、轨迹一致 |
| P1 | D/E：分布统计与解耦评估 | CPU统计，独立安排推理GPU | 有可信相机、主体、质量三组基线指标 |
| P2 | S：固定分层采样 | D/G可靠后，CPU冻结计划 | 来源/方向/尺度/可靠性有记录，恢复顺序一致 |
| P2 | L：单项条件排序 | I/E通过后，新增GPU训练对照 | 正确条件优势和可见控制提高，质量不明显退化 |
| P3 | M：区域与物理监督 | 可信mask/目标/梯度链验证 | 监督不混淆主体和相机，梯度真实贯通 |
| P3 | A：文本投影、CFG、K等结构对照 | 前面证据支持后，单因素实验 | 同条件下可归因改善，不依赖错误指标 |
| P4 | X/P：规模化、吞吐与交付 | 有效配方稳定后 | 质量、控制、覆盖、速度及恢复均达标 |

不要等待全部P3/P4完成才建立当前全量文本基线，也不要把尚未实现的建议写进当前配方。

## 2. P0-C：先解决标注长尾与重复工作

入口：`tools/prepare_union_dual_captions_20261023.py` 的
`caption()`、`validate_pair()`、`clean_pair_sentences()`、`finalize()`；
`tools/run_union_dual_text_20261023.sh`；测试 `tests/local_tests/test_union_dual_captions_20261023.py`。

- [ ] **C01：统一“有效结果”索引与待处理队列。**
  - 同时读取所有原始成功分片和`caption_repairs.jsonl`，按ID＋真实视频＋完整124帧＋窗口＋policy判定复用。
  - 当前worker只把原分片成功行算done，未完整复用finalizer修复结果，因此下一轮重复生成大量已可用样本。
  - 支持改变worker数量后全局查重；不能只读取同名worker历史文件。
  - 输出`pending_ids`、有效原始/修复/解码失败/真正缺失计数；测试修复后重跑零生成、重分片不重复、错误窗口不复用。

- [ ] **C02：处理“没有可观察主体”的场景。**
  - 先复核失败原始响应和抽帧，区分无主体、主体静止、遮挡、方向不确定、纯相机叙述。
  - 不让“删除相机句后只剩4个词”变成无限相同重试，也不靠降低字数门槛接受空洞的无主体声明。
  - 无主体时允许基于画面证据的环境描述，但必须显式设计状态与policy，并留出该类训练语义；不能编造人物/倒车。
  - 使用少量经审阅的无主体视频回归；普通视频不被强迫添加主体，确有主体的视频不被误标无主体。

- [ ] **C03：消除词义歧义与静态表述误删。**
  - 区分工业储罐/storage tank与前景战斗坦克，区分static scenic view与相机运动命令。
  - 背景版仍须排除前景主体及行为，不能为提高通过率直接关闭foreground/camera规则。
  - 测试工业储罐、战斗坦克、静态道路延伸、人物行走、相机跟随、混合句、无可见背景；记录清理前后文本与原因。
  - 若仅靠正则无法可靠区分，使用有针对性的VLM重写/待审，而非默许歧义。

- [ ] **C04：让重试真正改变失败原因，而非重复同一输入。**
  - 给出具体失败字段/句子与规则，不仅追加“camera/audio text”笼统错误。
  - 检查复用draft是否已经违反本轮规范；不能在新生成文本合格后又强制替换回旧的违规draft。
  - 记录响应hash、尝试次数、错误类别和无进展次数；重复相同结果时切换有证据的重写策略或停止待审。
  - 禁止通用占位句、编造运动和悄悄删除样本；明确有限重试/无进展停止条件。

- [ ] **C05：修复尾行中断及尾部调度。**
  - JSONL末尾不完整行应可审计地恢复；中间损坏必须报错，不可全部静默跳过。
  - 先CPU汇总/清理，再仅加载真正有工作量的GPU worker；对尾部按实际数量均衡分片。
  - 单ID保持明确写入所有权，避免两个worker同时补同一条。
  - 测试强制中断、重跑幂等、空队列不加载模型、跨分片重复和两个不同成功结果冲突。

交付：独立原始响应、修复记录、真正pending清单、各类失败ID、最终双版本plan及policy/hash。
规则通过不等于主体方向已经人工/几何确认，两个结论必须分别记录。

## 3. P0-O：自动验证、训练和恢复不能误触发

入口：`tools/run_union_dual_text_20261023.sh`、`tools/configure_union_dual_training_20261023.py`，
以及`tests/local_tests/test_union_dual_training_gate_20261023.py`。

- [ ] **O01：明确阶段状态机和当前有效完成标记。**
  - 记录阶段、owner、输入hash、输出数量、开始/结束时间、失败原因及下一步。
  - 三轮finalizer失败后不能仅凭旧`CAPTIONS_VERIFIED`文件存在就继续；先核对本次hash和本次成功退出状态。
  - 文本、条件编码、数据验证、启动loss验证、实际optimizer step、完整checkpoint分别表示，禁止混称“训练已开始”。
  - 测试陈旧marker、完整文本但不完整embedding、部分parquet写入、失败退出及重新进入的幂等性。

- [ ] **O02：建立按阶段恢复，而不是从头重跑的入口。**
  - 已完整编码且hash一致的部分复用，只重试缺失或确认失败的部分。
  - 新轮已有checkpoint时显式恢复其完整状态；不要再次从parent100初始化。
  - 检查checkpoint的DCP元数据、所有预期rank shard、训练metadata和RNG；不能只看到目录就算完整。
  - 用户只问状态或要文档时禁止启动恢复；现有链live时禁止第二个owner。

- [ ] **O03：GPU可用性与主机资源检查可解释。**
  - 当前idle检查遇到占用直接失败，并非一直排队；是否改为有限等待须明确配置。
  - 若实现等待，默认300秒检查，不忙轮询，不触碰其他用户进程；记录超时与退出原因。
  - `/datacc05`曾仅余约2.8TB；写大缓存前估算字节数并预留空间，不因磁盘满损坏原缓存。
  - 保持互斥锁和GPU所有权核对；不编辑已解析的live Bash循环并假设立即生效。

- [ ] **O04：恢复必须保留 eager/checkpoint 修复。**
  - 变长文本曾触发Dynamo重编译限制和activation-checkpoint重放不一致。
  - 测试短/长文本交替、前向/反向、真实refiner梯度、完整checkpoint恢复。
  - 不通过关闭一致性检查掩盖故障；若重新启用compile，另设性能和数值对照，不直接覆盖稳定配方。

交付：可重入的阶段协议、失败报告、明确的恢复命令说明及测试；不得伪造成功marker。

## 4. P1-I：从清单到真正DiT输入的完整审计

入口：`tools/prepare_text_training_20261022.py`、`tools/prepare_union_dual_captions_20261023.py`、
`tools/audit_scene_text_20261022.py`、`tools/text_condition_training_guard_20261022.py`，
以及`fastvideo/pipelines/preprocess/minimax_h3_worldmodel/{preprocess,encoding}.py`。

- [ ] **I01：检查每行实际训练组成和来源绑定。**
  - ID、源视频、124个源索引、源Parquet/行号、caption、首帧PNG、目标/首帧latent、C2W、K和embedding逐级关联。
  - 同ID错窗口、错PNG、换embedding、重复ID、train/val源视频交叉都必须拒绝。
  - 复用Parquet只新建派生sidecar，不覆盖旧sidecar；原始视频/pose/latent保持只读。
  - 当前已有哈希检查应保留并完善，不另做一套互相矛盾的完成定义。

- [ ] **I02：独立抽查“caption确实导致该embedding”。**
  - 生产端hash一致只能证明持久化了生产端给的内容，不能独立证明生产端用对文本。
  - 按来源、长短文本、复用/新编码、运动状态分层抽样，以原生Qwen重新编码实际caption＋同PNG。
  - 记录模型/权重、tokenizer、processor、预处理版本fingerprint，比较token tags、长度和embedding容差。
  - 测试caption互换且sidecar同步伪换、错误encoder、图片错配；不得使用WAN/UMT5条件缓存替代。

- [ ] **I03：校验collator到DiT的实际文本和布局。**
  - 记录实际样本ID、caption hash、文本/视觉token数、长度、截断状态、真实分支和少量tensor fingerprint。
  - 测试变长、关键方向词位于句尾、长度边界、padding、text tags与embedding同步裁剪、SP分片一致。
  - 历史缓存兼容裁剪不能成为新文本数据允许静默丢掉方向词的理由。
  - 普通text-on样本不得被vision-only/neutralize配置静默屏蔽。

交付：完整数据组成表、独立抽样复编码报告、输入审计样例与可复现测试。
现有基础测试：`test_text_training_preflight_20261022.py`、`test_scene_text_parallel_audit.py`及union gate测试。

## 5. P1-G：几何、窗口首帧与尺度

入口：`fastvideo/dataset/vwm_h3.py`、`fastvideo/models/dits/minimax_h3_prope.py`、
`fastvideo/train/models/minimax_h3/worldmodel.py`、`examples/train/verify_minimax_h3_vae_cache.py`。

- [ ] **G01：统一分散的几何validator。**
  - 验证124帧→37 latent锚点、齐次矩阵底行、旋转正交和正行列式、有限性、第一帧单位阵。
  - 验证K形状、正焦距、源单位及normalized转换；当前文本auditor不等于完整rigid/K验收。
  - 加入反射矩阵、非刚体旋转、像素K直接输入、错锚点、非首帧原点的拒绝测试。
  - 有限tensor、首帧单位阵不等于pose与画面一致，后者另做G03。

- [ ] **G02：记录分辨率、裁剪与内参转换链。**
  - 特别核对JS-easy的672×384像素K、1344×768原视频、832×480训练裁剪关系。
  - 改分辨率/裁剪会影响K与图像坐标，不应随意改物理外参；明确何处已归一化，避免重复缩放。
  - 测试中心裁剪、非等比resize、逐帧K、共享K和训练/推理一致性。
  - 当前K-off保持不变；K-on是后续单独实验，不用它掩盖尺度/标签问题。

- [ ] **G03：按来源检查方向、幅度及画面一致性。**
  - 核对OpenCV/规范坐标转换、c2w/w2c、局部/全局位移、I上/K下及J/L正负。
  - 记录原平移单位、单目尺度歧义、历史按轴因子和是否曾归一化/截断，防止再抹去幅度。
  - VGGT任意单位不能当米；速度档可先来源内校准，保留尺度来源及可信度。
  - 抽取纯/复合、慢/中/快、静止/抖动各类视频核对背景运动，保留复核ID与证据帧。

交付：共享validator、跨来源校准报告、异常ID清单和训练/推理几何回归。
修改pose派生条件时不重做不变的视频VAE，但必须重建受影响的条件、hash和索引。

## 6. P1-D / P2-S：真正可解释的分布与采样

入口：`fastvideo/train/utils/dataloader.py`、`fastvideo/dataset/parquet_dataset_map_style.py`、
`fastvideo/train/models/minimax_h3/worldmodel.py`的`sampling_index_path`，
历史辅助脚本`examples/train/prepare_minimax_h3_full_residual_sampling.py`等。

- [ ] **D01：建立canonical source别名和稳定行索引。**
  - 合并train_v2_multicam与multicam等同源命名，保留原始来源字段和映射版本。
  - 冻结ID→实际Parquet行→源视频/场景映射，不能依赖会变化的目录遍历顺序。
  - 完整合集约multicam48.6%、spatialvid41.4%；旧文本集合95%spatialvid不是当前合集比例。

- [ ] **D02：相机与主体分别统计，不把词频当真值。**
  - 相机按逐帧实测六轴方向、有效时长/幅度、单一主导/复合、来源内慢中快、可信度统计。
  - 主体分别统计前进/倒退/静止/转向/不确定、是否可见自身朝向、遮挡和前景覆盖率。
  - 同一复合窗口可有多个方向；方向计数不必等于视频总数。
  - 记录主体与相机动作相关性，找出模型靠“跟随镜头”捷径即可重建的样本。

- [ ] **S01：冻结来源×方向×速度×可靠性的采样计划。**
  - 当前motion-balanced只混排合成动作幅度分位数，并非上述联合均衡。
  - 来源上限30%–40%、核验控制样本20%–30%只能作为试验起点，按实际供给调整并写明不可满足配额。
  - 稀有类别限制重复次数；不能只用两条视频硬凑平衡。限制同源视频重叠窗口支配batch。
  - 复合视频保留完整原轨迹，不能把GT其他运动轴清零后当纯W正例。
  - 输出固定seed、输入hash、全程/前200/400步分布、唯一覆盖率及每ID重复次数。

- [ ] **S02：验证分布式与恢复一致性。**
  - SP2同组使用同样本，DP4按计划分配，accum8合成为BS32；不能误算成8×1×8=64。
  - 测试空桶、严重稀有桶、别名绕上限、数据顺序变化、固定seed和checkpoint恢复后同一采样位置。
  - 3000×32=96,000次抽取，小于113,207训练条目；均匀无放回至少3538步才覆盖一遍。
  - 如要求完整一轮，另行确认4000以上steps；加权采样不能保证有限步内每条都出现。

交付：`sampling_plan`、分布报告、不可满足条件列表及 sampler 测试，不把“多数方向”说成“全程纯单指令”。

## 7. P1-E：先建立解耦评估，再判断是否有效

入口：`scripts/evaluation/portable_js_easy_h3.py`、`portable_js_easy_h3_subject_text.py`、
`score_portable_js_easy_h3.py`、`analyze_minimax_h3_camera_control.py`、
`fastvideo/train/callbacks/worldmodel_validation.py`、`tools/visualize_h3_controls.py`。

- [ ] **E01：冻结多场景相机×主体文本交叉测试。**
  - 固定首帧、seed、采样器、sigma/shift、分辨率、帧数与时间轴。
  - 相机{静止,W,S}×主体{前进,后退,不指定动作}，覆盖不同主体朝向与场景；另保留完整WASDIJKL及速度档。
  - 同条件比较原底模、action-only800、文本100和新union文本checkpoint，检验遗忘假说而非预设成立。
  - 另做保留主体身份/外观、只删除动作短语的干净消融；当前背景版不是这个消融，也不等于主体静止。
  - 不需要重新生成完整第三版训练数据才能做少量评估条件干预；任何新增数据版本另行记录。

- [ ] **E02：相机指标只利用可信静态背景。**
  - 加前景排除、稳健运动估计和置信度，分别报旋转、平移方向、尺度对齐轨迹和非指令轴串扰。
  - 现有角点/LK/affine方法只给二维全局运动，不是完整6DoF，也不能自动排除移动坦克。
  - 单目尺度未知时不报绝对米数；纯W存在透视扩张，不能要求平均水平光流为0。
  - 测试静止背景＋移动主体、纯旋转、W/S透视、不对称景物、平移、低纹理、切镜与遮挡。

- [ ] **E03：主体方向以自身朝向和相机补偿为依据。**
  - 建立实例跟踪、可见朝向、相机补偿位移；朝向/遮挡不清时abstain，并报告有效覆盖率。
  - 不把目标变小、向画面左移或背景变化直接当作后退/左移。
  - 用可控合成样例和人工核验样例校准，避免用待评模型生成结果反过来当绝对真值。

- [ ] **E04：把质量、控制正确性和条件敏感性拆开报告。**
  - 真实/静止/反向轨迹loss、latent RMS只表示重建/响应，不能替代物理方向准确率。
  - GT类样例才算PSNR/SSIM/LPIPS等；无GT类明确标为质量/控制评估，不能混排名。
  - HUD展示实际输入，不是输出运动；指标用未加HUD的raw视频。
  - 记录请求50步与实际transformer调用数、124帧适配和输出FPS；portable duration-preserving FPS不必等于训练24fps。

交付：冻结eval manifest、原始视频/输入审计、逐case指标与抽帧、覆盖率和失败清单。
预先定义方向/串扰/质量的放行标准和统计不确定性，不能看完某组结果后临时改阈值挑例子。

## 8. P2-L：保留flow MSE，逐项验证条件排序

入口：`fastvideo/train/methods/fine_tuning/worldmodel.py`、`finetune.py`、
`fastvideo/models/dits/minimax_h3_prope.py`；文本条件涉及原生encoding与packed layout。

- [ ] **L01：相机同输入排序的最小对照。**
  - 利用已有`camera_counterfactual_same_input`，显式保持noisy latent、噪声、sigma、首帧、GT及文本相同，只改相机。
  - 最小实验使用same_input=true、static_paired_target=false，其他静态/运动辅助项仍为0。
  - 只有可靠且有辨别信息的轨迹参与；零delta、等价条件、几何不可靠样本跳过并统计比例。
  - 区分逆运动、时间反放和轴符号翻转；不能对4×4矩阵逐元素取负。
  - 测试同输入逐字节一致、异常后条件恢复、零delta有限梯度、关闭项回归纯flow。

- [ ] **L02：新增主体文本排序，而不是字符串表面替换。**
  - 仅用实际行为方向已核验的样本，构造真实相反运动短语；不确定样本仍可用于普通flow训练。
  - 为完整正/负文本生成正确原生Qwen embedding、tags和可变长度layout，不能复用旧embedding冒充新prompt。
  - 固定相机和其他输入，只改变主体运动；背景版不是负样本，错误文本不配GT当普通正例。
  - 测试不同长度、图像/相机不变、方向短语进入DiT、模型梯度非零和disabled时零额外forward。

- [ ] **L03：量化loss权重、频率、噪声分布及分布式代价。**
  - 分sigma/timestep记录正负误差、margin、胜率、有效样本比例和梯度贡献。
  - cadence改变会改变实际权重；相机ranking与其他旧辅助项的频率缩放方式不完全一致，须核对实现。
  - 全部参与FSDP/SP collective的rank保持兼容forward次数；不能某DP样本无可靠标签就少做一次collective。
  - 同一噪声/随机状态配对；排序仅证明辨别提高，不直接证明物理方向正确，必须同时通过E评估。

交付：只加一项的独立config、正确/错误条件loss曲线、梯度报告、额外算力开销与同条件视频。
不要同时打开所有loss后将改善归功于其中某一项。

## 9. P3-M：动态前景安全的区域与几何监督

入口：`fastvideo/train/methods/fine_tuning/{finetune,worldmodel}.py`、
`fastvideo/train/models/minimax_h3/camera_motion_probe.py`及待新增的mask/可微观察器模块。

- [ ] **M01：审计并隔离不适用的旧counterfactual目标。**
  - 旧项包含“static相机→全局零位移”和“reverse相机→负GT二维运动”；动态主体、旋转、遮挡、W/S视差不满足。
  - 重复首帧不是任意静止相机视频的真实GT，不能用它强制所有运动主体停止。
  - latent二维translation observer不能作为W/S、6DoF或主体前后方向真值。
  - 为适用/不适用情形写回归和显式开关，不因已有代码就默认全量启用。

- [ ] **M02：先实现可信前景/背景区域去噪MSE。**
  - mask与832×480裁剪、VAE时空感受野、37锚点和latent/token布局对齐。
  - 使用区域面积和可信度归一化，处理空mask、边界、遮挡、多个主体及失效帧。
  - 区域加权仍是去噪重建，不是物理相机方向监督；不要错命名。
  - 测试移动前景不污染背景权重、错位被拒绝、全空mask有限及关闭项回归。

- [ ] **M03：再实现背景相机、前景主体的物理目标。**
  - 静态背景监督匹配实际旋转/平移方向与轨迹；主体监督使用相机补偿后的运动和可见自身朝向。
  - 估计不可靠或尺度未知时按定义跳过；不根据整幅图平均流强制纯W左右为0。
  - 只有真实配对视频可作额外正例，不能把错误条件对应的理想视频凭空设为GT。

- [ ] **M04：证明预测→解码→观察器→loss→DiT的梯度链。**
  - 冻结VAE/pose/flow模型参数可以，但预测支路不能detach/no_grad；target观察支路可以无梯度。
  - 先核实H3当前flow参数化及干净样本重建公式，不能套用另一模型epsilon公式。
  - 先在可靠噪声区间、有限帧和受控样本验证，测显存/额外时间及估计器本身误差。
  - 用合成运动、有限差分和一次梯度更新证明loss推动正确物理方向；observer权重无梯度而DiT有梯度。
  - 离线MP4评分、普通RANSAC或VLM评语只能监控，不能宣称已提供可微训练监督。

交付：可信目标/mask索引、适用范围、梯度与物理方向测试、小权重独立实验和失败分析。

## 10. P3-A：只在证据支持时修改模块

入口：`fastvideo/models/dits/minimax_h3.py`、`fastvideo/train/utils/lora.py`、
`fastvideo/train/models/minimax_h3/{minimax_h3,worldmodel}.py`与native H3输入准备/编码stage。

- [ ] **A01：审计真实可训练参数与条件响应。**
  - 按主video block、text refiner、context投影、prope_o分组记录参数名、数量、梯度及参数更新。
  - 当前text refiner已有LoRA，不能写“文本分支全冻结”；action_dim=0也不等于关闭相机。
  - 相机分支处理视频token，但QKV来自已受joint attention影响的hidden state，不是因果独立通路。
  - 在同checkpoint/首帧/seed下关闭相机残差出口作诊断，区分分支作用和其它变化；不据此推断PRoPE理论无效。

- [ ] **A02：文本投影适配单因素实验。**
  - 保留既有refiner LoRA，单独对比context_embedder LoRA或约2750万参数的投影解冻。
  - 独立参数组/学习率、allowlist、参数计数、保存/恢复；不同时增rank和解冻整网。
  - 测变长文本、activation checkpoint与梯度恢复；同数据/初始化/步数比较E指标及质量。

- [ ] **A03：文本-only CFG完整实现与对照。**
  - 当前训练拒绝非零training_cfg_rate，预测拒绝unconditional/cfg_uncond，native输入拒绝guidance≠1；不能只调参数。
  - 在text refiner混合前移除scene/subject文本，重建tags、span和packed layout；保留首帧vision/keyframe、C2W/K、噪声/sigma及零音频。
  - 明确picture标签等固定文本的保留政策，训练/推理相同；不能把全部条件向量归零。
  - SP组同步dropout决定并记录实际text-off比例；guard允许明确合法text-off，不放松普通text-on检查。
  - 实现原生两分支flow预测组合，guidance=1严格回归当前条件推理；测额外forward、显存和质量退化。
  - 在新实验中启用，绝不偷偷改变当前排队任务。

- [ ] **A04：K、额外相机适配器与容量仅作后置独立实验。**
  - K-on先通过G，再比较是否改善方向与尺度；不要同时改变K、归一化和loss。
  - 若可信监督下仍出现低条件梯度/响应，再比较显式逐帧调制或其它残差注入；保留现有有效主路作对照。
  - 旧12维纯PRoPE失败不证明所有PRoPE无效；当前2.393B可训练参数也不支持先断言rank过小。
  - 参考Cosmos/Wan全量微调与minWM/H3残差分支时逐项列出骨干、参数、数据、噪声、监督差异，避免不等价归因。

交付：单因素配置、可训练参数/梯度证据、checkpoint兼容与恢复测试、E评估结果。

## 11. P4-X / P：实验矩阵、速度与持续训练

- [ ] **X01：建立可归因实验矩阵。**
  - B0为当前完整版本1文本＋flow基线；后续依次S采样、L01相机排序、L02文本排序、M区域/物理、A投影/CFG。
  - 改一项时固定可比数据、初始化、样本顺序、seed、sigma分布、优化步数和评估协议，记录无法固定的因素。
  - 记录父checkpoint路径/hash和本轮独立step，不能把H3-World step2000与FastVideoWM同名step混为一个模型。
  - 保留质量退化/无效结果；不得只选最好例子排名或凭总loss曲线宣称控制成功。

- [ ] **X02：制定200/400步方向验证与扩展门槛。**
  - 用户不要求重复十几条J/L；在多方向、多场景和可信控制样本上比较。
  - 同时报告相机误差、主体方向、覆盖率、串扰、质量、条件排序及置信区间。
  - 200步用于早期诊断，400步确认趋势；无改善先检查目标/梯度/数据，不机械扩大steps。
  - 新的自动推理排程或暂停训练必须单独实现并获准；当前链只有启动及每100步loss验证，不等于已有此推理。

- [ ] **P01：在BS32下做真实吞吐pilot。**
  - 当前SP2/DP4：micro1×accum8；可独立测试micro2×accum4等保持BS32的组合。
  - 记录加载、前向、反向、通信、optimizer、保存和验证时间；不要只看显存或tqdm平均。
  - GPU pilot须安排空闲卡，不与现有worker叠跑；检查loss/梯度和checkpoint一致性后再应用。
  - 固定数据形状与可训练模块，避免把数据缓存速度、数值变化当作batch优化效果。

- [ ] **P02：61f/更长窗口作为独立工程，不偷改124f协议。**
  - 若测试短窗口加速，重新核实native VAE时序、首帧条件、camera锚点、packed layout和评估时长。
  - 不能直接截37个latent的一半冒充正确61帧，也不能混入WAN latent。
  - 长视频/AR适配保留24fps逐帧轨迹与模型窗口转换，单独记录分段连续性、条件重置及评估限制。

- [ ] **P03：长期稳定性与资源恢复。**
  - checkpoint保存、数据读取重试、NFS/磁盘容量、CPU RAM、GPU显存、端口和owner均有明确失败路径。
  - 不同时加载八份完整DiffSynth H3推理模型；历史八份主机OOM不能靠显卡显存充裕推断安全。
  - 串行加载/分片模型/内存上限与并发预算分别度量；重启服务器或驱动操作不属于本TODO的默认执行步骤。
  - 状态通知仅在完成、失败或需用户操作时触发，不每几秒持续发消息。

交付：实验登记、预先固定放行门槛、吞吐表、恢复演练和可独立追溯的checkpoint/结果目录。

## 12. 最终放行与交接清单

- [ ] 所有训练ID均能定位视频、准确窗口、首帧、pose、caption、embedding和latent来源。
- [ ] 双版本分开，训练实际使用版本1；主体不确定、无主体和可靠方向样本的语义明确。
- [ ] 多来源几何/尺度/首帧审计通过；画面与pose不一致者有ID清单及处理记录。
- [ ] 八方向、速度档、主体行为统计与真实采样分布可复现，稀有类别不靠少数样本无限重复。
- [ ] 每项新增loss的适用假设、正确/错误条件、梯度链、频率和分布式一致性均有测试。
- [ ] E指标证实独立控制与质量改善，而不只是latent有差异或loss下降。
- [ ] 新轮恢复不重置父模型，断点数据位置、优化器、RNG、参数组和配方可追溯。
- [ ] 更新repo报告、handoff最新段落及完成勾选，保留无效实验和回滚点，不覆盖历史证据。

每项建议采用以下记录模板：

```text
任务ID / 负责人 / 状态 / 依赖：
要解决的问题与证据：
本次修改文件、配置、数据与checkpoint hash：
CPU测试：
GPU smoke、分布式或恢复测试：
评估输入、指标、样例与失败ID：
对比基线及结论（有效／无效／不确定）：
额外时间、显存、CPU RAM、存储成本：
是否达到预设门槛；下一步与回滚点：
```

## 13. 后续可复制的CPU检查命令

以下只验证代码/文档，不会启动新的标注、编码或训练：

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM
.venv/bin/python -m pytest -q \
  tests/local_tests/test_union_dual_captions_20261023.py \
  tests/local_tests/test_union_dual_training_gate_20261023.py \
  tests/local_tests/test_scene_text_parallel_audit.py \
  tests/local_tests/test_text_training_preflight_20261022.py
bash -n tools/run_union_dual_text_20261023.sh
.venv/bin/python /home/lizihao/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /home/lizihao/.codex/skills/fastvideowm-h3-handoff
```

修改具体模型/数据模块后，还需运行其目录下对应测试；不要把这组CPU测试当作GPU端到端或控制效果保证。
