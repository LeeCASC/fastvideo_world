# 当前训练集可视化（2026-10-27）

用户要求每个当前训练来源展示5个未过滤掉的样本，并参考原Cosmos仓库的
third_party/vwm_datasets可视化。新增独立CPU工具，不修改训练模型、配置、输入缓存或原视频。

## 入口与输出

- 工具：`tools/visualize_training_sources_20261027.py`。
- 测试：`tests/local_tests/test_training_source_visualization_20261027.py`。
- 输出：`/datacc05/lizihao/worldmodel/fastvideowm_train_gallery_20261027/`。
- 浏览：输出目录下的`index.html`，详细协议见`README_zh.md`。
- 进度：`progress.json`及`render.log`；仅`VISUALIZATION_COMPLETE`表明全量完成。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.visualize_training_sources_20261027 \
  --output /datacc05/lizihao/worldmodel/fastvideowm_train_gallery_20261027 --workers 4
```

## 数据边界

读取当前union_scene_subject_text_20261024/subject_motion的已验收文本训练计划，
仅从train split中固定哈希抽样，优先不同源视频；对照caption_selection排除授权删除的83条。
23个技术来源合计111条：22来源各5条，train_v2_spatialvid实际仅1条。
不拿validation或原始全集中未入选的数据补数；这不是质量排行榜或人工通过声明。

训练Parquet中的ID、完整帧索引、实际prompt、相机哈希和原始首帧PNG来源均逐条核对。
展示原始视频对应的124帧训练窗口，不做VAE解码；37个实际相机锚点单独保留，
SLERP和线性插值只供124帧显示，不改变训练条件。输出仍为24fps。

通过AST加载用户指定render.py的纯绘制函数和TrajectoryPanelRenderer，
避免导入旧数据集注册表及其重型依赖；不修改Cosmos仓库源码。
轨迹面板固定FOV视锥仅作示意。HUD来自相机输入，不能当作画面运动测量值。
逐视频完整解码校验帧数和FPS，失败不替换样本。

8项CPU测试和pre-commit已通过；真实导出结果、首帧MAE及最终完成数量以输出summary.json为准。
