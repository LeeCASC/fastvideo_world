# Cosmos pose-QC checkpoint-1700 portable评测

2026-10-29（北京时间），用户要求先停止训练并使用测试集推理。

## 最新授权：评估完成后自动恢复八卡训练

2026-10-29下午，用户追加授权：全部评估结束后继续八卡训练。覆盖下文历史“不自动恢复”的
描述，但不改变本轮评估仅GPU0–3的限制。不会修改正在运行的评估进程或扩大评估用卡。

独立协调脚本：`tools/resume_cosmos_qc_after_eval_20261029.py`。
tmux：`fastvideowm_poseqc1700_eval_then_train_20261029`；每300秒检查一次，最长等待24小时。
评估根目录保存`autoresume.log`、`autoresume_state.json`，失败另记`AUTORESUME_NEEDS_ATTENTION.json`。

触发条件：128条latent和经解码验收的视频全部完成、HUD完成、32条GT及96条无GT指标记录完整，
现有指标流程完成，且原评估`job.lock`已释放。缺依赖的可选VBench项目明确记为不可用，不冒充完整
七项成功；不为补这些依赖无期限阻塞续训。评估失败时停止自动链，不跳过失败。

恢复前检查数据proof、原config、plan和两个loader缓存hash；选择同根最新完整checkpoint，核验
metadata步数/配方、8个非空DCP分片、8份RNG及文件时间稳定。只写新的派生恢复配置，不修改
原`training_config.yaml`或其hash-bound proof。读取原训练`launch.lock`且持有到训练进程退出，
全部八卡空闲后启动，不终止任何其他任务。GPU占用时每5分钟等待。

当前最新完整checkpoint为1700，恢复1701；保留base＋parent100初始化后完整恢复1700的
模型、优化器、调度器、callback、RNG与同数据loader状态。数据仍108439条（训练105935、
验证2504），背景＋主体文本＋数值相机，八卡/SP2/DP4/micro1/accum8，总BS32，
累计目标3000步，每100步保存/验证、保留35份。不重置步数，不更换模型结构或监督。

恢复配置：`<训练root>/training_resume_after_eval_20261029.yaml`。
新stdout：`<训练root>/resume_after_eval_20261029.log`，原生logs/checkpoints仍在`runs/train/`。
历史主动停止marker保留；实际启动时写`POST_EVALUATION_RESUME_AUTHORIZED_20261029.json`。

17:12:32已实际启动上述协调tmux，17:12:33状态为`waiting_for_evaluation`，
轮询周期300秒；不是训练已经恢复。只读preflight确认最新完整checkpoint1700，原数据、
config及索引hash通过。新增35项CPU回归覆盖成功/失败启动、发布完成状态的短暂竞态、
完整checkpoint和配置不变；与原准备脚本回归合跑结果见当前测试输出。

## 最新资源覆盖：仅GPU0–3

10:09用户追加要求只用GPU0–3，覆盖下文最初八卡方案。旧启动组在模型生成前停止，
旧配置、manifest和启动日志移动到输出根目录的`startup_8gpu_before_user_override/`保存，
已有权重硬链接、16组native输入和32条已验证C条件保留，不删除视频/权重。

10:10:53同一tmux恢复四卡自动链：

- 条件编码仅GPU0、2，两worker。
- DiT仅GPU0–3，四rank、SP2、HSDP1×4、两视频并行。
- 解码仅GPU0–3，四worker；指标仅GPU0。
- 空闲预检只检查GPU0–3，不检查或使用GPU4–7。
- `freeze --inference-num-gpus 4`只改派生推理配置，训练配置和checkpoint不动。
- 完整128条协议、文本、采样、HUD和指标口径不变；不自动恢复训练。

22项CPU回归通过。以下涉及八卡的文字是最初方案/历史，不是当前资源占用。

## 停训与权重

- 训练root：`/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_108439_resume1100_20261027/subject_motion`。
- 最后完成step1719；最近完整保存step1700，08:46落盘、08:53验证完成。
- 核实step1700元数据、8份非空DCP、DCP metadata及8份RNG后，仅对本人torchrun PID3734212发送SIGINT。
- 未保存的19个已完成step不属于评测权重，不宣称已保存。
- root的`USER_PAUSED_FOR_EVALUATION_20261029.json`说明主动停止。原launcher的SignalException/NEEDS_ATTENTION不是自发崩溃，不据此自动重启训练。
- 本次没有授权评测后自动恢复训练。后续恢复应使用最新完整checkpoint1700，不能重启固定1100启动器。

## 测试范围与条件

输出：`/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_step1700_portable_eval_20261029`。

复用portable包`/datacc05/lizihao/js_easy_benchmark_portable_20261021`的完整协议：A16真实轨迹、B16单WASDIJKL、C32首帧描述＋主体forward/backward；alpha1、0.5，共128段。当前模型为真实文本微调版本，不再把C组称为vision-only负对照。

H3保持原生124帧、832×480、37latent相机锚点、shift12、requested50步、guidance1和零长度音频。输出FPS沿用portable H3适配的124/61×源FPS，保留原时长；不直接复制WAN61帧latent或UMT5条件。GT只用于独立flow/像素指标，不输入生成未来帧。

使用HSDP八卡/SP2、4条视频并行，禁止八份完整DiT同时加载。Qwen条件编码、DiT生成、VAE解码分阶段执行；raw与HUD分开保存。

## 本次适配修正

1. `scripts/evaluation/portable_js_easy_h3.py freeze --preserve-init-checkpoint`保留训练模型的parent100初始化，再加载评测checkpoint1700；只清除训练器resume，不读取旧优化器开始训练。
2. 明确记录base→parent100→checkpoint1700的加载顺序。父子DCP当前模型键相同，但仍沿用经训练验证的初始化过程，不凭假设跳过。
3. 修复旧status硬编码step800及“所有文本均未使用”的历史元数据；文本按实际配置和DiT入口token审计确认。
4. A/B根据实际manifest prompt重新编码；A必须复制GT latent所在的`gt.parquet`，不能用无GT的simple shape carrier计算flow loss。
5. C条件仅在完整prompt、首帧hash、embedding hash及token审计一致时复用；不会因checkpoint变化无端重编视频VAE。
6. 每组各case独立条件路径，避免A/B相同js ID相互覆盖。两alpha共享同一文本＋首帧条件，只有相机幅度改变。

四项CPU回归覆盖bootstrap保持/legacy默认保持不变/step元数据/文本报告。pre-commit按仓库配置执行，相关Python路径被格式钩子跳过，不能称为全量格式检查。

## 结果口径

先看`logs/`、`status.json`和实际进程确认阶段。`ALL_LATENTS_COMPLETE`、`ALL_VIDEOS_COMPLETE`、`VISUALIZATION_COMPLETE`和`CORE_METRICS_COMPLETE`独立标记，任务启动不等于完成。

A组报告native H3 flow MSE与PSNR/SSIM/LPIPS；沿用此portable适配包含首帧的124帧指标口径，不能与WAN61帧直接排名。B/C无对应GT。可用VBench adapter分数不冒充官方完整七项，缺依赖明确标记；质量指标或HUD亮键不证明相机/主体方向遵循。

本文件记录准备与运行方法；实际启动和完成时间以新输出目录日志为准。

## 启动记录与导入修复

10:06:21首次启动在freeze阶段退出：共享portable模块导入Triton kernel，
`CUDA_VISIBLE_DEVICES=`导致其查询CUDA target时报“0 active drivers”。这不是显存OOM或权重损坏。
freeze仅执行文件冻结/哈希，但其现有导入依赖要求可见CUDA设备，因此改为该阶段显式可见GPU0，
不加载推理模型。失败日志保留为`logs/freeze_before_cuda_import_fix.log`，失败marker改名保留历史。
10:07:09同一tmux名称恢复；setup/audit仍是独立CPU入口，不需要可见GPU。
