# FastVideoWM 当前训练集的 Cosmos 参考规则复筛

## 本次范围

2026-10-27 用户要求参考 `/datacc05/lizihao/worldmodel/cosmos_pose_qc_reference`，
重新过滤当前训练集，并给出逐数据集数量。

冻结输入为已获授权剔除881条后的115,024条（112,300训练、2,724验证）：

`/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_poseqc_20261027/subject_motion/text_data_plan.jsonl`

输入清单 SHA256：`4bebf1de93f3f1f40bcb1334463d6558f633a0ecbfeb61eb7392977986c528a7`。
不再将先前881条或更早83条重新混入。单位是当前124帧训练窗口，不是动作数量。

仅做CPU审计和派生清单，不停止现有训练，不修改模型、loss或采样器。
此前等待checkpoint-1100自动切换881剔除版的协调器，是另一个已授权流程；
本次新复筛结果不会自动替换其数据根目录，也不能声称已经进入训练。

## 实现与规则

入口：`tools/filter_current_cosmos_pose_qc_20261027.py`。
直接导入参考包的矩阵、运动统计、dense抖动和动作推导函数，不凭ID写过滤规则。
没有调用旧参考入口的context排除、移动文件、停止训练或新holdout等历史操作。

检查包括：

- 实际缓存ID、来源、描述hash、124帧完整索引与冻结清单一致。
- 37个实际H3相机矩阵有限值、正交性、行列式、齐次末行、首帧单位阵。
- 原始dense姿态真实存在时，复现其生产器变换，对齐实际H3锚点，最大误差≤1e-4。
- dense平移反向≥4次、反向率>25%、路程/净位移>3同时成立时拒绝。
- 锚点单步旋转>45°、异常平移跳变、平移和旋转反向、动作状态过多等参考规则。
- 次数阈值按123/60缩放：平移反向上限6，单轴旋转反向上限14，状态上限16，稳定动作切换上限24。
- 实际H3视频latent为24×37×30×52；检查有限值和相邻latent变化比（上限4）。
- 再读视频latent和相机bytes，与已验收sidecar的hash核对；检查缓存读取前后stat不变。

H3的时间锚点为0、4、8、12、16、17、21……123，并非Wan的等间隔4帧。
只使用真实37锚点及源逐帧姿态；不通过SLERP插值伪造dense覆盖。
合法15→24fps转换会重复源帧，dense运动/动作统计先去掉重复源索引。
缺失dense来源映射时允许按实际锚点检查，并明确记录覆盖缺口；已声明pose却读不到则作为处理错误拒绝。

旧批次按历史生产器重现full-video归一化、分轴尺度及必要的Y轴基变换，
只为核实已训练条件，没有修改任何尺度或重新平滑姿态；新批次按其直接读取生产器复现。
原始文件与符号链接路径不同仅在samefile证实后认可，不仅凭文件名匹配。

## 运行与交付

输出：`/datacc05/lizihao/worldmodel/fastvideowm_cosmos_pose_qc_20261027`

tmux：`fastvideowm_cosmos_pose_qc_20261027`。

```bash
cd /home/lizihao/projects/military_world_model_agent/FastVideoWM
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u tools/filter_current_cosmos_pose_qc_20261027.py \
  --workers 32 \
  --output /datacc05/lizihao/worldmodel/fastvideowm_cosmos_pose_qc_20261027
```

提交上限每批512任务，CPU worker上限32，不加载GPU模型；只读取所需Parquet列。
完整审计后按输入顺序输出：

- `accepted.jsonl`：完整保留数据清单，含描述、split、原始video/window和现有条件缓存绝对路径。
- `rejected.jsonl`：完整剔除ID和来源，包含拒绝原因。
- `audit.jsonl`：每条数值与dense覆盖。
- `summary.json`、`过滤结果_zh.md`：23个来源的筛选前、保留训练/验证、保留合计和剔除数。
- `NUMERIC_POSE_QC_COMPLETE`：绑定冻结输入、完整审计和保留清单hash的完成标记。

运行时只看 `progress.json` 和 `run.log`；**目录或进程存在不表示完成**。
冻结输入、参考源码或适配脚本变化后拒绝在同一审计版本继续追加。
失败时保留审计和日志，不自动丢弃出错记录或伪造完成标记。
通过清单不是当前loader直接消费的新版data_path；后续启用应另建只包含保留缓存的派生数据目录，
完整核验，显式切换配置和重置旧loader索引，不要删除共享source_parquet。

截至16:15 JST，首1,024条完成，零处理错误，约9条/秒；预计19:50–20:30 JST完成。
此估时不是完成证明，后续以实际日志和完成标记为准。

## 验证与边界

参考包4项CPU回归通过；本地9项适配测试通过，覆盖实际H3非均匀锚点、
长窗口重设原点、历史W2C/坐标/尺度、dense缺失、平滑闭环、反向抖动、
实际Parquet读入、重复FPS索引、dense/锚点不一致和损坏latent。pre-commit通过。

来源抽查67条覆盖23个来源，无读取错误或dense/cache不一致；63通过、4因数值规则拒绝。
这是适配抽查，不能代替115,024条最终统计。
参考latent变化比用于不同VAE家族，只是异常变化代理；没有运行新视觉GPU质检、
语义分割、深度重投影或逐条画面/pose真值验证。数值通过不证明动作语义完全正确。
已有训练权重曾接受的错误监督也不会因清单过滤而被逆向消除。
