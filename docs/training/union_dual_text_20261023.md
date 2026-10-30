# 115,988 条数据的双版本文本补齐与续训

## 2026-10-24 12:17 JST：先编码显式“背景＋主体”版

用户指出旧subject_motion有时只有主体，检查确认编码没有拼入单独背景。
按新指令先编码“背景环境＋主体外观与行为”，纯背景编码暂缓。

- 新根目录：`/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024`。
- 新tmux：`fastvideowm_union_scene_subject_text_20261024`。
- 原根目录已停止编码，作为只读描述/部分缓存来源保留，没有删除。
- 工具：`python -m tools.prepare_scene_subject_text_20261024`。
  背景在前、主体在后，仅去掉完全重复句子，不重新VLM，不改VAE/首帧/pose。
- 纳入115,905条，train/val 113,126/2,779；原83条排除保持。
  111,220条prompt改变，920个已有同prompt缓存可严格复用。
- 新`subject_motion/`技术目录实际是完整合成文本，以`prompt_composition`识别。
  `background_only/`只有清单，暂不编码、不启动对照训练。
- `UNION_DUAL_ROOT=<新根目录> REQUIRE_VERIFIED_CAPTIONS=1`运行原launcher，
  gate失败即停止，不回退VLM。12:17:02 JST已启动八卡编码。
- 编码后全量审计，额外核对实际caption与背景/主体组件组合及hash；
  通过后沿用parent100、BS32、3000新步数训练。保留35份checkpoint；
  启训前预留至少512GiB，否则阻止启动，不减少保留数。
- 不同prompt使用不同embedding；只复用完全相同文本和窗口。旧缓存不改，
  新复用文件用硬链接及独立sidecar，避免旧路径清理导致悬空链接。

## 2026-10-24 11:21 JST：用户授权舍弃83条，停止长尾重试

用户明确要求这83条全部舍弃，不再反复加载八个模型重试。
排除spatialvid 68、train_v2_multicam_video 9、train_v2_context 6；
包括这83条中后来重试成功的条目，不只是最终仍失败的少数。
原始registry保留115,988条不改，`caption_selection.json`以源registry hash
绑定83个完整源视频窗口与原因，本轮纳入115,905条。
详细ID见`caption_excluded_ids.jsonl`及`user_excluded_initial83_ids_20261024.jsonl`。
源视频、pose、VAE及raw标注未删除。该授权不允许额外静默舍弃样本。

`verify_captions`验证已冻结的registry/selection/plan hash，避免恢复时
再次遍历全部raw重试输出；验收后直接进入条件编码，持久化tensor全量审计仍执行。
配置生成器仅接受完整集合或该明确授权的子集，不放宽BS32、text-on及3000新步数等约束。

最终文本验收通过115,905条，训练/验证113,126/2,779。
11:22:37 JST自动启动八卡条件编码，日志tag为`20261024T112231`；
11:23:24核验8个编码进程均存活，GPU各约51.6GB、处于H3编码器权重加载阶段。
61项CPU回归及pre-commit通过。以上不是全量编码/验证完成声明。

## 2026-10-24 11:15 JST：v2 长尾修复

10:41 恢复的三轮任务于11:08:18因剩737条未通过而退出，没有启动条件编码。
查到清理器漏识别“No foreground subjects…”；删除相机句后遗留的absence句
又被当作实际主体矛盾。还存在train station/tracks、hot dog cart等词级误判。

v2 只删除完整absence句和显式无主体时的无生物运动句，窄范围识别上述场景设施；
真实可见主体矛盾仍拒绝，不提高confidence、不编造通用文本，不改源数据或raw输出。
重检剩83条，11:14:35在原tmux重启自动链，tag为`20261024T111435`。
57项caption/gate/audit/preflight CPU测试通过，pre-commit通过。
五个真实双版本示例位于输出根目录`caption_examples_5_20261024.md/json`；
样例按来源和置信度挑选，不等价于随机人工质量审计。

以上是启动时快照；编码、审计、训练是否开始仍须查最新日志与实际进程。

## 2026-10-24 恢复说明（优先于下文旧重试描述）

旧三轮标注于02:17 JST因871条未通过而退出，未启动编码/训练。
用户授权修复后，10:41:35 JST在原tmux名恢复自动链。仍使用完整115,988条、
版本1文本、parent checkpoint-100模型权重、新计数3000步、八卡BS32；没有修改action模块或loss。

- `plan`、worker恢复与finalizer共用全局精确窗口解析器，复用所有分片及独立修复结果。
  生成`caption_pending.jsonl`及registry/队列hash，空队列不加载GPU。
- 全量语义重检发现5,060条待修，包括旧版接受过的空泛“没有主体运动”描述。
  没有通过放宽字数、通用占位句或删除视频凑齐。
- 无可见主体时要求VLM显式标记`subject_presence=none`，版本1描述真实环境。
  若同一响应的版本1只剩空泛句，可复用同次、同帧的有效环境描述并记录独立flag/mode。
  仅`not_observed`不能证明无主体；不能因此删去一个真实主体或编造主体静止。
- 区分储油/储水等基础设施tank与战斗坦克、木结构framing与镜头构图；背景仍排除前景主体。
- 失败原始输出和旧日志保留；新响应写时间戳分片。只容忍未结束的最后一条损坏记录，中间损坏报错。
- 新恢复日志带`20261024T104135`标签。第一轮worker已加载的Python不会热更新；
  后续finalizer及worker进程会读新代码，raw有效计数与可清理复用数量需区分。
- caption失败仍阻止编码；所有条件编码和持久化tensor审计完成后才启动训练。
  GPU被占用时每300秒重试，24小时超时，不终止其他用户进程。
- 用户决定继续保留35份checkpoint，亲自清理旧模型，本次未修改保留数或删除文件。
  父checkpoint约13.39GiB，35份同配方约469GiB；模型目录清单位于
  `/datacc05/lizihao/worldmodel/model_checkpoint_inventory_20261024`。

检查实际tmux、最新带时间戳日志、marker及完整checkpoint后再报告状态，以上不是实时监控结论。

## 用户确认的范围

2026-10-23，先停止现有训练，补齐旧 65,389 条数据的真实场景描述；
完整合集同时准备两个独立文本版本。补齐并验收后，先训练“场景＋主体运动版”，
背景版保留作对照。这里不更改 action 模块、辅助损失或相机尺度。

- 输出根目录：`/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023`
- 主清单：`registry.jsonl`，总计 115,988 条。
- 批次：旧 65,389 条＋新增 50,599 条（包含上一轮文本清理失败的 1 条，重新标注）。
- 本次按源视频隔离训练/验证：113,207 / 2,781。保留旧验证视频和上一轮文本验证视频；
  同源视频的所有窗口一起归验证，不能宣称父模型从未见过这些视频。
- 首轮已匹配描述：50,598 条当前文本绑定＋373 条完整窗口一致的历史动态描述。
  65,017 条无满足本次精确窗口条件的可复用动态描述。
  “无可复用描述”不是“磁盘上不存在任何相关文本”，也不是视频质量不合格。

## 两种文本的区别

1. `subject_motion/`：环境、主体身份/外观及有证据的主体运动。
   不确定自身前后运动时不猜方向，记录 `subject_motion_status=uncertain`。
2. `background_only/`：仅地形、道路、建筑、植被、天空、天气、材质和光照。
   **不仅移除运动动词，也移除前景车辆、坦克、人物等主体的身份与外观。**

两版共享 ID、视频、完整 124 个源帧索引、首帧、相机轨迹及训练/验证划分。
两版均禁止相机控制和音频描述，不能将它们拼接成同一条训练 prompt。

例子（仅解释格式，不是已核验样例）：

- 主体版：`A tan tank reverses along a dirt road in a forest under a blue sky.`
- 背景版：`A dirt road crosses a forest under a blue sky.`

## 标注及复用

脚本：`tools/prepare_union_dual_captions_20261023.py`。

1. 从已验收的原始 VAE Parquet 读 ID、124 帧索引、latent shape 等元数据。
   旧 ID 经 `source_mapping.jsonl` 及原训练/验证 ID 清单找到真实源视频。
2. 历史动态描述必须匹配真实视频路径和**全部**源帧索引，不能只看文件名、按键或抽样帧。
   不使用右 padding 已拒绝的历史标注、不使用 61 帧描述替代 124 帧描述。
3. 按每条实际 124 帧计划均匀抽 6 帧；保存抽样源索引、RGB 哈希、模型路径和原始输出。
4. 同一次 Qwen3-VL 调用生成两个版本，并检验已有描述是否与当前画面一致；
   只有声明支持且通过规则的已有主体描述才复用。全局位移和目标变大/变小不等于主体运动。
5. 背景版规则拒绝前景主体、运动、相机及音频关键词。规则检查不是人工审核，
   VLM 的运动判断也不是几何真值；保留原始响应、置信度、不确定状态和质量标记。
6. 八个独立 worker，默认每卡 batch=4；显存 OOM 时同一批退回单样本。
   每个 worker 独立追加日志；重跑跳过已成功 ID，最多三轮补失败样本。
   未完成项写 `caption_missing_ids.jsonl`，**不会用通用描述悄悄补位或自动删除失败视频**。
7. 汇总时可从失败原始响应中**删除完整的违规句子**，另存 `caption_repairs.jsonl`。
   不改运动方向、不补写事实；删除相机句后将主体运动状态降为 uncertain。
   清理后文本过短或为空仍然失败，继续重新生成。原始分片不覆盖。

## 编码和验收

- 不改源视频、pose 或现有 VAE 缓存，也不重新做 VAE 编码。
- 两套描述都输出 `text_data_plan.jsonl`，包含 ID、caption、caption 哈希、
  来源 Parquet/行号、视频路径、全部帧索引、窗口 ID、分割与复用来源。
- 本次自动训练只使用 `subject_motion`，因此只为该版本准备完整 DiT 条件缓存。
  **背景版此时是描述清单，不是已编码的训练 Parquet。**后续对照训练再独立编码背景文本。
- 未改变的已有主体版 conditioned Parquet 和 audit 文件以符号链接复用；
  其余重新编码“实际文本＋原始缓存首帧 PNG”，复用目标和首帧 VAE、C2W、K。
- 最终逐行读取持久化 Parquet，检查所有 tensor 有限性、shape、哈希，
  首帧 C2W 单位矩阵、真实文本和视觉 token、无截断、分割与 ID 全覆盖。
- 2026-10-23 加固：验收核对真实视频、完整帧索引、窗口 ID、来源 Parquet/行号；
  registry 与训练 plan 的 SHA-256 贯穿描述、编码、审计和启动配置。
  复用旧 Parquet 时新建派生 sidecar，旧 sidecar 不改写。
  新轮已经有 checkpoint 时禁止自动从父模型重新初始化，须显式恢复。
- 不足 115,988 条、任一描述/编码错误或验收失败都不启动训练，写 `NEEDS_ATTENTION`。

## 暂停与恢复训练

旧任务：`fastvideowm_js00001_simple_text_then_train_20261023`。
已确认 torchrun PID 910286 属于本次训练后，发送 SIGINT；未触碰其他用户进程。
日志最近核验约 121 步，只有 100 步有完整 checkpoint，因此约 21 步未保存进度无法恢复。

保留模型：

`/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/runs/train/checkpoints/checkpoint-100`

包含 DCP `.metadata` 和全部 8 个 rank 的 RNG 快照；原日志、W&B 和模型文件未删除。
旧 launcher 因人为停止留下异常/attention 标记，不代表新发现数值或模型错误。

新训练使用这份已训练文本的 checkpoint-100 **模型权重**，不回退到 action-only checkpoint-800。
数据量和划分变化，使用新数据加载器、优化器与独立步数计数；新轮目标 3000 步。
保持八卡、SP2、DP4、microbatch1、grad accumulation8，total BS=32。
沿用 residual PRoPE、rank64 LoRA、后层参数、K-off、非逐视频归一化平移、零长度音频。
保留 eager 模式以及 full activation checkpointing 的一致性检查。
每 100 步保存和验证，实际 DiT 入口的文本 guard 仍启用。

自动链：`tools/run_union_dual_text_20261023.sh`。
配置生成：`tools/configure_union_dual_training_20261023.py`。
所有 worker、编码和全量验收成功后才启动训练；不以时钟时间替代完成条件。

## 查看状态与恢复

```bash
ROOT=/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023
tmux list-sessions
tail -n 15 "$ROOT/chain.log"
ls "$ROOT/progress"
cat "$ROOT/inventory_summary.json"
cat "$ROOT/caption_missing_ids.jsonl"
```

标注结果：`captions/worker_00.jsonl` 至 `worker_07.jsonl`。
吞吐量：`progress/worker_*.json`，按各卡实际速率合计估算 ETA，冷启动时间另计。
完成状态：`CAPTIONS_VERIFIED` → `subject_motion/TEXT_DATA_VERIFIED` → 训练实际成功 steps。
不要仅因 tmux/显存存在而声称已训练；不要因某个 marker 陈旧就启动重复任务。

测试：`tests/local_tests/test_union_dual_captions_20261023.py`，包括背景排主体/运动/相机、
不确定行为不强制前进、完整窗口复用、异常状态等 CPU 回归。

控制精度的后续分析见 [相机与主体控制改进方案](control_precision_plan_20261023.md)。
本次自动训练仍保持已确认结构与 flow MSE；后续 loss/模型改进不等于已经实现。
