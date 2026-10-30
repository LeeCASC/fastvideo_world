# 外部相机质检ID剔除（2026-10-27）

用户确认按同源ID全部剔除881条，含55条验证与仅由61帧筛选命中的284条。
附件5782条记录映射到当前881条：1429条外部记录命中，4353条不在当前训练版本中。
原始附件中的SSD路径只是外部证据，绝不作为本机删除目标。

## 剔除版本

新根目录：

`/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_poseqc_20261027/subject_motion`

- 剔除：826训练＋55验证＝881条。
- 保留：112300训练＋2724验证＝115024条。
- 精确清单：`excluded_ids.jsonl`，含原ID、视频、完整帧窗口、缓存位置、外部来源及原因。
- 匹配明细：`external_id_mapping.jsonl`；未匹配：`external_unmatched.jsonl`。
- 剩余训练清单：`text_data_plan.jsonl`；来源数量：`data_plan_summary.json`。
- 授权和哈希：`selection.json`；派生验证：`subset_data_audit.json`、`SUBSET_DATA_VERIFIED`。

`excluded_ids.jsonl`中的conditioned_relative_path相对于旧数据根目录，定位实际单样本训练Parquet。
不要直接删除其中的source_parquet：那是多样本共享VAE分片，删整个文件可能误伤其他合法样本。

保留缓存通过硬链接复用全量已验收父数据。逐个核对115905个sidecar的ID、caption hash、
视频和完整帧索引；对881个排除目标另读实际Parquet对应列验证；
核对全部保留文件／sidecar与父缓存同inode、总数及精确路径集合，无额外混入文件。
这是明确的“已全量验收父数据的未变更子集继承验证”，不是新跑了一次全tensor审计。
不复制旧map_style_cache，不重新编码。原视频、pose、VAE、旧清单和旧checkpoint保留可恢复。

## 自动切换（15:34 JST时尚在等待，不代表已经换成新集）

旧训练仍在运行，协调器tmux：`fastvideowm_poseqc_subset_switch_20261027`。
入口：`tools/exclude_external_pose_qc_20261027.py switch`。
每300秒等完整checkpoint>=1100，验证8个DCP分片、8个RNG文件及稳定时间戳，
再核实旧torchrun PID2276631的用户与命令后SIGINT，只停该训练。
GPU全部释放后自动启动新配置；不强杀或停止其他用户任务。
新根目录的switch.log、switch_state.json和runs/train/logs记录实际状态。
旧根目录SUPERSEDED_BY_POSE_QC_20261027.json说明主动切换，避免误把旧失败标记当成需要重启旧任务。

完整续训保留模型、优化器、scheduler和global step，仍以3000为总目标；
八卡/SP2/DP4/micro1/grad_accu8，total BS32，每100步保存／验证，保留35份checkpoint。
不改模型、loss、相机尺度或文本内容。
切换时旧loader索引失效，新增显式配置reset_dataloader_on_data_change：
仅在新旧数据根目录不同且配置开启时不加载旧loader状态；同根目录后续恢复仍完整恢复loader。
模型／优化器DCP状态不丢弃。旧验证集减少55条，后续val loss不应假装固定集合直接比较。
轮询可能产生少量checkpoint后的未保存工作，以实际恢复点为准。

## 验证与后续恢复

10项专用CPU测试通过，含实际单进程DCP模型／优化器／步数／loader恢复测试，pre-commit通过。
原包级checkpoint测试因其conftest在隐藏CUDA时导入GPU内核失败，不能据此声称整套包测试通过。
不要重启旧caption/parent100初始化自动链。新轮已有checkpoint时，从新根目录最新完整checkpoint恢复，
不是重新从旧100或1100开始。切换日志确认前，不能声称881条已经从正在运行的batch中消失。
旧111条画廊是历史快照，其中可能含剔除样本；不是剔除后的新训练清单。
