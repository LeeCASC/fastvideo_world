# Cosmos复筛后从1100步完整续训

## 用户授权与数据

运行核验（2026-10-27，北京时间）：20:25:14八卡启动，20:26:49读取新训练索引，
20:26:52读取新验证索引，20:28:59完整DCP恢复1100成功，20:29:01开始启动验证，
实际文本＋首帧视觉token进入DiT的guard检查通过。
20:35:20启动验证结束并恢复RNG快照，进入1101步计算。
截至20:39日志确认第1101步完成，首个恢复步耗时216.95秒，八卡训练继续运行。
该记录仅为当时快照，后续步数须看实时日志；下一次新保存点为1200。

2026-10-27用户要求使用新筛选数据，从checkpoint-1100继续训练。
新数据来自已完成的Cosmos参考规则审计：108,439条，训练105,935、验证2,504。
此前排除的83条、881条以及本次6,585条均不重新混入。
这是当前124帧训练窗口数量，不是独立源视频数或动作数量。

输入审计目录：`/datacc05/lizihao/worldmodel/fastvideowm_cosmos_pose_qc_20261027`。
`NUMERIC_POSE_QC_COMPLETE`绑定输入清单、完整audit和accepted文件的hash。
本次复核三项hash，确认accepted/rejected不相交、ID唯一、逐条数值规则通过。

新训练根目录：

`/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_108439_resume1100_20261027/subject_motion`

保留现有文本＋首帧条件编码、原视频VAE、相机条件，按原相对路径建立硬链接。
每个Parquet和sidecar均核对与父文件samefile，源文件stat与本次数值审计一致；
核对sidecar的ID、caption hash、视频和完整源帧索引，重新读取全部Parquet footer确认每文件1条。
最后检查训练目录精确路径集合，不仅比较数量，防止额外未通过样本混入。

这不是重新生成文本、VAE或姿态，也不是重新执行全量条件embedding编码验证。
完整条件tensor验证继承原全量审计；新增数值QC检查与全部footer/绑定验证另有记录。
原视频、原pose、旧cache、checkpoint均保留，不物理删除共享source_parquet。

## 上次启动失败与本次修复

上一版881剔除数据的续训在2026-10-27 15:57（北京时间）退出。
日志显示rank0进入`get_parquet_files_and_length`串行打开十万多个Parquet获取长度，
其他rank在Gloo barrier等待1800000ms超时，尚未进入1101步。
这不是当时模型前向/反向显存不足的错误；完整checkpoint-1100仍存在。

本次在初始化GPU训练之前，用24个有界CPU线程校验所有实际文件footer，
按原loader格式生成每个split专属`map_style_cache/file_info.pkl`：
绝对新文件路径按字典序排序，并一一配对真实num_rows。
不拷贝旧缓存路径、旧行索引或旧loader状态，不简单延长分布式超时。
原数据loader和模型代码不需要为此改变。

## 完整恢复语义

恢复点：

`/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024/subject_motion/runs/train/checkpoints/checkpoint-1100`

已检查metadata step1100、DCP metadata、8个非空DCP分片和8份RNG。
先按原配方加载官方base及parent checkpoint-100的model-only初始化，
重现可能不在trainable-only DCP中的继承冻结权重；随后加载1100的完整DCP模型、优化器和callback状态。
没有从parent100重新计步或重置优化器。

`reset_dataloader_on_data_change=true`只在旧新data_path不同的本次恢复跳过旧loader状态；
全局step从1100延续，下一次优化器step为1101。
后续同一数据根目录恢复则正常加载其新loader状态。
恢复到本次新根目录已有checkpoint后，必须显式选该目录最新完整checkpoint，不能盲目再跑1100。

现有trainer的tqdm未显式设置total，恢复时可能显示`1100/1900`：1900是剩余range长度，
不是终止global step。实际循环范围是1101至配置的3000，没有因此在1900步停止；
状态判断以配置、完整checkpoint和实际global step为准。本次未为显示问题修改正在运行的trainer。

保持原模型结构、训练模块和loss；文本为“背景＋主体外观与行为”，数值C2W控制。
八卡/SP2/DP4，microbatch1、grad accumulation8，总BS32；全局目标3000，即追加1900步。
每100步保存及loss验证，启动时也会验证；保留35份checkpoint。
保留eager环境、full activation checkpointing一致性检查、K-off、零长度音频。
新验证集较旧数据不同，val loss不可假装固定数据集合直接比较。

## 入口与产物

脚本：`tools/prepare_cosmos_qc_resume_20261027.py`，分为`prepare`和`launch`。

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u tools/prepare_cosmos_qc_resume_20261027.py prepare --workers 24
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u tools/prepare_cosmos_qc_resume_20261027.py launch
```

tmux名称：`fastvideowm_cosmos_qc_resume1100_20261027`。
启动前再次核对GPU0–7全空闲，至少512GiB磁盘空间；没有授权终止其他用户任务。
父checkpoint和旧训练目录只读，新实验checkpoint均保存在新root。

根目录文件：

- `text_data_plan.jsonl`：本次实际数据ID、prompt、video/window、现有cache路径。
- `data_plan_summary.json`：各来源及split数量。
- `resume_data_audit.json`、`RESUME_DATA_READY`：继承完整tensor审计、实际footer验证、索引及配置hash。
- `preprocessed/{train,validation}/map_style_cache/file_info.pkl`：实际loader专属索引。
- `training_config.yaml`：本轮原配方配置与明确恢复点。
- `prepare.log`、`launch.log`、`runs/train/logs/`：准备和实际训练日志。
- `runs/train/checkpoints/`：本轮新checkpoint、W&B offline跟踪日志。

目录、marker和显存占用不能证明完成了训练step。至少确认
`Checkpoint loaded; resuming from step=1100`、实际文本进入DiT、启动验证结果、1101及后续步数递增，
再报告训练已正常恢复。失败不得通过关闭一致性检查或悄悄丢弃更多数据绕过。

20项CPU回归通过，包括真实单进程DCP模型/优化器/步数恢复、仅重置变更loader、
真实loader函数在rank0/7读取离线新缓存而不扫描footer、路径/多行/源变更拒绝及原配方保持。
pre-commit通过。这不代替实际八卡训练验证，也不证明指令控制质量已经收敛。
