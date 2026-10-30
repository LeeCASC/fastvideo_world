# H3-World 文本对照与 FastVideoWM 场景文本续训

## 2026-10-23 最新状态：暂停训练，补齐完整合集双版本文本

本页下方的“训练正在运行”是历史状态。用户现已要求先补齐旧 65,389 条描述，
同时为完整 115,988 条准备“场景＋主体运动”和“仅背景（无前景主体）”两个版本。
旧八卡训练已按用户授权 SIGINT 停止，保留完整 checkpoint-100（最近观察约 121 步，
约 21 步未保存进度无法续接）。源数据、历史日志及模型未删除。

当前流程与恢复说明见 [完整合集双文本流程](union_dual_text_20261023.md)。
输出根目录 `/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023`。
用户已确认全部补齐和验收后先训练主体运动版，背景版留作对照。
不要重新启动本页历史 50,598 条训练 launcher，也不要启动重复的标注或编码 worker。

## 2026-10-23 10:05 JST：js_00001短文本两用例，之后自动恢复训练

用户要求简化昨天的主体前/后运动文本，仅生成两个视频。因此沿用昨天的union checkpoint-800，
不是将刚更新数步的新文本模型冒充已保存checkpoint。首帧、相机W轨迹、alpha1、seed20261012、
124帧、832×480、原native shift12/requested50-step采样保持一致。

两个完整prompt分别为 `A tank is on a dirt road in a forest. The tank moves forward.`
和 `A tank is on a dirt road in a forest. The tank moves backward.`。
不加长篇身份/外观保持约束；主体运动不转成相机W/S。重新用原生Qwen编码，
从旧97个文本token缩短到22个，仍保留392视觉token；两个DiT入口实际文本检查均通过。

暂停时新文本训练完成3个optimizer steps，但尚无新checkpoint；已提前向用户说明，
这3个未保存step不会延续，之后从最新完整checkpoint-800重新启动同一3000-step文本轮次。
旧训练日志和W&B保留。未改训练数据、损失、BS或checkpoint/验证频率。

根目录 `/datacc05/lizihao/worldmodel/fastvideowm_step800_js00001_simple_text_20261023`。
维护脚本 `tools/js00001_simple_text_20261023.py`、
`tools/run_js00001_simple_text_then_train_20261023.sh`；tmux为
`fastvideowm_js00001_simple_text_then_train_20261023`。
四卡SP2、两路同时推理，仅两条manifest；decode自动附上HUD，观看版 `visualized/`，raw保持原样。
双视频完成及全解码检查后写 `INFERENCE_COMPLETE`，脚本自动释放GPU并启动八卡BS32训练。
即使诊断失败也会保留失败日志，并在显卡释放后恢复独立的已审核训练；不重做VAE/Qwen全量预处理。
训练依然关闭Dynamo编译，保留梯度重计算一致性检查及100-step验证。
10:16 JST两条视频及HUD已全部生成，完整解码均为124帧；`INFERENCE_COMPLETE` 和
`VISUALIZATION_COMPLETE` 已存在。两版latent RMS差0.16892764，仅证明对输入文本敏感，
不能据此认定坦克运动方向正确。脚本已自动启动八卡训练，日志
`runs/train/logs/training_config_20261023_091554.log`；仍在同一个eval-then-train tmux里。
10:17 JST新训练已连续完成2个microbatch的前后向，正在首步第3/8次梯度累积，
八卡约45～46GB显存且持续运算；尚未完成这次重启后的首个optimizer step。

## 2026-10-23 09:48 JST：首步编译/重计算错误修复并重启

50,598条全量tensor校验已于昨晚23:29 JST通过，报告为 `text_data_audit.json`，
包括50,470训练和128验证；`TEXT_DATA_VERIFIED` 已存在。不要重跑VAE、Qwen或全量审计。
自动队列随后确实进入八卡训练：模型加载、数据集初始化、12批step-0验证都完成，
实际文本guard已证明每样本有50～69个场景文本token和392个视觉token进入DiT。

但23:49 JST首个训练microbatch反向时出现
`torch.utils.checkpoint.CheckpointError: Recomputed values ... different metadata`。
此前LoRA的 `@torch.compile()` 达到Dynamo `recompile_limit`，变长文本触发不同编译图，
前向与重计算的保存tensor序列错位。没有完成optimizer step，也没有新checkpoint。
旧日志 `runs/train/logs/training_config_20261022_222938.log` 保留；不是OOM或VAE错误。

仅在本训练启动器中设置 `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`，
让编译装饰函数走eager；保留full activation checkpointing及其一致性检查，
不设置 `determinism_check=none`，不改变数据、模型、LoRA、PRoPE、优化器、有效BS。
增加启动器flock，避免重复trainer。CPU回归验证先用多种长度做no-grad预热，
再对不同长度进行checkpoint反向，确认编译backend未调用，且梯度与普通前向一致。

09:48 JST在tmux `fastvideowm_scene_text_train_20261023` 直接启动同一训练器，
不重走旧推理/编码/审计队列。此次使用 `RUN_INITIAL_VALIDATION=false`，
复用同一parent权重上已完成的step-0验证；每100步的12批验证保持不变。
新日志 `runs/train/logs/training_config_20261023_084829.log`；
根日志仍为 `chain.log`。旧失败标记归档于 `logs/TRAINING_NEEDS_ATTENTION_graph_mismatch_20261023`。
09:57 JST已实际观察到2/3000个optimizer steps完成，16次梯度累积microbatch的反向均通过。
W&B已提交第1步 `total_loss=0.1039411277`、`grad_norm/student=0.0388183594`，数值有限；
第1步216.49秒，第2步约211秒，八卡显存约47～49GB且持续运算。
loss有一个step的离线W&B提交延迟，不能仅因尚无history行就判定未训练。
配置中的辅助相机loss/响应指标权重为0，相应零值不表示PRoPE被关闭。
本次仅证实启动及早期反向正常，不等于长时间稳定性或指令遵循质量已经验证。

## 2026-10-22 22:15 JST 恢复检查：CPU审计并行化

H3文本开/关64条及三组输入控制界面已完成，50,598条新文本条件也已完成编码。
GPU空闲的直接原因是全量CPU审计仍在执行，而非新文本训练崩溃；该轮尚无训练checkpoint。
原审计单线程逐文件执行 `pq.read_table`，约19分钟仅推进到第2个worker目录。

保留原有全部tensor有限性、shape、caption绑定、SHA256、文本token、首帧原点与split覆盖检查，
将审计拆到 `tools/audit_scene_text_20261022.py`：最多16个CPU reader，先顺序读取单个
文件，再通过 `ParquetFile(BytesIO(...)).read(use_threads=False)` 解析，
避免NFS上逐列小范围读取、逐文件Dataset发现与嵌套Arrow线程池；
额外核对实际caption哈希及metadata ID。每500条输出进度及ETA，并写
`text_data_audit_progress.json`。不改原始/派生数据，不重新VAE或Qwen编码。

5个CPU回归覆盖全量payload读取、双split计数、哈希损坏、首帧原点错误、无文本token和重复ID拒绝；
真实128条试读通过。默认 `--audit-workers=16`，不改变编码的 `--workers=8`。
NFS存在冷热缓存差异，不能将热缓存小批次试读速度作为全量ETA。
已终止旧单线程审计，保留其失败标记于
`logs/NEEDS_ATTENTION_serial_audit_replaced_20261022_2115`，并恢复同名唯一tmux依赖队列。
本节记录恢复动作，不代表训练已完成首步；以 `TEXT_DATA_VERIFIED`、真实训练日志和有限loss为准。

## 最新更新：H3文本开／关64条重跑（2026-10-22）

FastVideoWM文本开启的32条已于16:13 JST（15:13 CST）全部完成。原H3单文本测试生成了4个完整124帧视频，但DiffSynth的写出函数把32.524590 FPS通过 `int(fps)` 截为32，随后在FPS校验处退出；这次不是OOM。旧目录及4个视频保留，不伪造完成标记，也不将缺少生成metadata的旧产物直接当作新成功结果。

按用户最新要求，新H3实验重新生成：`text_on`完整场景／主体forward/backward文本32条，`text_off`无场景／主体文本32条，共64条。**text_off不等于关闭相机控制**：仍保留首帧视觉token与37段独立相机动作文本；先以空prompt运行原生Qwen编码，再移除全局head中所有tag=1的文本token，仅保留tag=0视觉token，重新对齐37个action span，不用通用占位句或全零embedding冒充无文本。

每对text-on/off保持同checkpoint、首帧、相机动作、seed、分辨率和采样参数。无文本组的forward/backward本来就是相同有效输入，保留32条输出可核对重复性，不将两个标签误称为主体运动控制输入。

新结果：`/datacc05/lizihao/outputs/h3_world/step2000_subject_text_on_off_portable_20261022/`，便捷视频分别位于 `raw/text_on/single_action_text/` 和 `raw/text_off/single_action_text/`。每条保存 `latents.pt`、`metadata.json`、完整MP4和完成标记。DiT入口必须验证text_on有全局文本token、text_off恰好为0，两组都有视觉token和37个camera span。最后全量验证64条并汇总on/off latent RMS响应；RMS非零不是主体方向正确率。

写出改用PyAV有理数FPS；CPU回归已验证实际MP4的124帧、832×480、FPS为32.524590而非32，caption移除回归同时证明视觉和每个camera span embedding未被修改。另复用上一轮编译缓存以减少冷编译，不复用旧失败视频。

依赖队列保持原tmux/启动器，已切到新H3目录；H3日志改为 `logs/h3_subject_text_on_off.log`。旧训练队列失败标记归档在 `logs/NEEDS_ATTENTION_old_fps_failure_20261022_1554`。2026-10-22 16:54 JST（15:54 CST）恢复队列，仍四卡原生H3/串行加载/600G scope限制；64条完成后按下文方案编码、校验并启动八卡FastVideoWM文本续训。以下旧32条H3协议描述是历史上下文，以本节为准。

## 用户确认的执行计划

2026-10-22：先完成当前FastVideoWM checkpoint-800文本开启诊断，再选择证据最充分的H3-World版本执行相同32条主体forward/backward文本测试，之后启动FastVideoWM新一轮文本＋数值相机条件训练。用户已确认：从union checkpoint-800模型权重初始化，保留相机分支，重置优化器和步数，8卡、有效total BS=32、3000个新optimizer steps；优先已有合格VLM描述的数据，缺少有效描述的数据不混入。

## H3-World 模型选择和输入适配

选择权重：

`/datacc05/lizihao/outputs/h3_world/vwm_filtered_v1/train_v2_plus_js_26708_from_step10000_bs35_gpu0_6_2000iters/step-2000.safetensors`

证据：旧16-case同协议 `control_validation_v1_16cases/evaluation/summary.json` 中，在相同50步采样时，step-2000与baseline的简单方向代理准确率均为0.875，GT轨迹cosine分别约0.664、0.584。step-2000在20步采样下代理更强，但不能把20步结果直接当作本轮50步结果。本选择是“当前有完整对照证据的较好候选”，不是已经证明其主体文本能力最好；其他后续H3版本缺乏同协议完整优势证据。

输入完全保留portable `manifests/subject_text.json` 的32条完整prompt：首帧描述＋主体沿自身朝向前进/后退；每对首帧/相机方向/seed一致。首帧复用前次832×480实际预处理PNG，不使用WAN latent、UMT5或FastVideoWM的Qwen embedding。

H3-World原生入口是DiffSynth v2的 `prompt=完整文本`，以及37条逐latent相机动作文本 `action_script`。本轮将W/A/S/D/I/J/K/L转换成相机描述，I向上、K向下；主体forward/backward保持在全局prompt，不能误转成相机W/S。相机动作文本与numeric PRoPE不存在已标定的一一幅度等价，元数据明确 `numeric_camera_injected=false`。不能将两实现的方向代理或不同噪声空间直接当成严格可比指标。

统一输出124帧、832×480、requested50步、CFG1，按portable源FPS×124/61输出，保持与FastVideoWM测试的播放时长一致。H3保留其原生音频分支采样，但MP4不写音频；FastVideoWM使用零长度音频token，这也是跨实现差异之一。

脚本：`/home/lizihao/projects/worldmodel/H3-World/code/vwm/run_portable_subject_text.py`。

结果：`/datacc05/lizihao/outputs/h3_world/step2000_subject_text_portable_20261022/`；`raw/single_action_text/*.mp4`为32条便捷视图；`videos/<case>/metadata.json`记录实际prompt、哈希、相机文本和DiT入口文本审计。

原生H3历史八份完整pipeline曾导致主机内存OOM，因此此阶段沿用4个独立worker，串行加载模型，置于已验证可启动的user systemd scope、MemoryMax=600G、MemorySwapMax=0；不是同时运行8份H3完整模型。全局prompt文本token、37个action span以及LoRA挂载数都必须真实通过，任一失败不放行训练。

## 场景描述与旧缓存的实际问题

旧65,389条conditioned缓存抽检仍为通用 `A video scene observed from a moving camera.`，不是当前VLM场景描述。新增50,599条conditioned缓存抽检保留原生成长prompt，包含相机/音频文字；虽然同批独立VLM描述已经生成，但并未绑定进此前最终conditioned Parquet。因此不能只设 `vision_only_conditioning=false` 就称为使用新场景描述。

本轮从既有50,599条VAE中，按global_id、window_id、video路径、完整source_frame_indices和VLM抽帧索引匹配 `vlm_qwen3vl_strict_50599/gpu*/captions_shard_*.jsonl`（根目录的四个空JSONL不算完成文件）。只用 `observed_scene_subject_motion_no_camera_v2` 的成功描述，再去除残留相机/音频句子。

最终50,598条：50,470训练＋128验证，按源视频划分无交叉；1条 `ue_easy:20260318__capture_w_14_0_200_00000_00200` 因描述清理后无有效内容排除。原115,988条未删除，本轮不使用旧65,389条中性的描述缓存。

| 来源 | 训练 | 验证 |
|---|---:|---:|
| spatialvid | 47,917 | 123 |
| multicam_video | 1,935 | 5 |
| ue_easy | 179 | 0 |
| minmax | 148 | 0 |
| ue_stage3 | 133 | 0 |
| ue_stage1 | 125 | 0 |
| ue_game_video | 20 | 0 |
| ltx | 7 | 0 |
| sekai_real_drone | 6 | 0 |

这是本轮有有效描述的集合，不是方向均衡训练集，spatialvid占比较高；本轮不悄悄改变既有采样/损失以声称完成方向平衡。验证集只在本轮按源视频留出，父模型可能见过部分相同视频，不能宣称全实验链路的未见场景泛化。

## 编码、校验和训练设置

根目录：`/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/`。

数据方案：`text_data_plan.jsonl`、`data_plan_summary.json`、`excluded_ids.jsonl`。实际训练文件分别在 `preprocessed/train/`、`preprocessed/validation/`，只在编码完成后出现，不能把已生成manifest称为训练缓存已完成。

重新用H3原生Qwen3-VL编码“实际场景描述＋VAE阶段保存的首帧PNG”。不重新生成VAE：目标latent、首帧latent、数值相机路径和内参直接复用；170条原始VAE中间记录仍以原视频为原点，在新派生输出中沿用现有conditioner的首帧重设原点逻辑，不改源文件。这不意味着旧最终conditioned训练输入未修复，两者须区分。

编码每行记录caption、首帧PNG、目标latent、首帧latent、C2W、K、embedding的SHA256。全量CPU验收实际读取持久化Parquet，检查全部payload有限性、shape、哈希、C2W首帧单位矩阵、真实vision/text token、1024长度内无截断、文本/标签对齐、ID与分割覆盖；失败不启动训练。

训练使用新独立 `training_config.yaml`，由旧union配方派生：vision_only=false；不neutralize；保持residual PRoPE、rank64 LoRA和原有后层可训练部分；K不进入PRoPE、平移幅度不逐视频重归一化、零长度音频、原生视频flow MSE。本次不另加未经验证的文本loss。

从稳定硬链接快照 `fastvideowm_union_step800_portable_eval_20261022/checkpoint/checkpoint-800` 载入模型权重；不resume旧优化器。SP=2，8卡构成4个数据并行组，microbatch=1，accumulation=8，所以有效BS=4×1×8=32；3000steps约96,000次样本抽取，约1.9轮。checkpoint每100步、最多35个；独立W&B offline和训练log，验证每100步，固定12个batch。

`TextConditionTrainingGuard`在真实prepare_batch输出检查文本token确实进入DiT，不允许因vision-only、neutralization、静默截断等原因出现“配置有文本但实际未训练”。写 `runs/train/training_text_input_audit.jsonl`；该guard无额外模型前向、不修改loss。

## 自动执行链与状态边界

tmux：`h3_text_eval_then_fastvideo_text_train_20261022`。

入口：`tools/after_text_evaluations_train_20261022.sh`，日志 `chain.log`。每5分钟检查前序FastVideoWM的 `INFERENCE_COMPLETE` 以及失败标志；确认GPU释放后运行H3的32条测试；再次确认释放后8卡编码新场景文本；全量CPU验证通过 `TEXT_DATA_VERIFIED` 后才执行 `tools/launch_scene_text_training_20261022.sh`。同目录flock避免重复队列；不杀别人的进程。模型参数与数据通过前置检查，不等于GPU端到端训练已实测完成，须看未来训练首步/有限loss/梯度和checkpoint。

新训练checkpoint：`runs/train/checkpoints/checkpoint-*`；日志 `runs/train/logs/`。`TRAINING_COMPLETE`只有正常达到目标才写，训练错误写 `TRAINING_NEEDS_ATTENTION`。服务器重启不会保留tmux进程，恢复前必须查看状态并选择完整checkpoint，不自动用启动器重置已有文本轮次。

当前已验证：32条H3 manifest、同图/seed/相机配对、完整prompt保留、native v2导入、原生CPU packing文本97/vision392/action span37、shell语法、Python编译、训练配置解析（1024 context、BS32、3000steps）、6个CPU输入guard回归测试。GPU当前仍由先前FastVideoWM文本测试占用，H3新任务和新文本训练是依赖队列，不能提前声称已完成GPU推理或训练。
