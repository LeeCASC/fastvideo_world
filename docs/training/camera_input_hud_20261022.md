# 相机输入控制界面（2026-10-22）

用户要求：以后此项目的推理视频默认提供控制界面，带文本输入的同时展示完整实际 prompt。界面复用
`cosmos-predict2-worldmodel/eval/methods/utils.py` 中 `draw_actions_genie3_style`
及三个纯绘制依赖，视觉上与 `bash_scripts/eval/run_eval/run_eval_noncausal.sh` 相同。
脚本通过 AST 仅加载这些绘制函数，避免引入 Cosmos 的 mediapy/GPU 依赖；不改 Cosmos 源码。

## 输出与输入真实性

- 保留 `videos/**/raw.mp4` 和 `raw/` 原样，用于 PSNR/SSIM/LPIPS/VBench 等指标。
- 观看版在 `visualized/`，包含 MP4、抽帧 JPG、每视频 `.hud.json` 以及总 `index.html`。
- 显示 Move WASD、View 上下左右、Roll；I=上转，K=下转，J=左转，L=右转。
- FastVideoWM 根据该次生成目录中实际输入的 `camera_to_world.npy` 逐帧显示控制，
  不是从生成视频估计运动，也不是只按文件名猜动作。37 个 H3 时间锚点扩展为124帧；
  局部平移/旋转差分用于显示；慢转 UI 放大8倍仅影响箭头显示，不改模型条件。
- H3-World 当前采用每 latent 的相机动作文本，不是 numeric PRoPE；其恒定指令测试
  从实际 manifest 的 `action` 生成提示，界面明确标记 `INPUT camera text`。
  新的复合动作实验须接入实际逐帧脚本，不可沿用恒定字母假装逐帧控制。
- prompt 是否开启优先读取 `scene_or_subject_text_used`，其次 `text_condition_used`。
  关闭时即使 metadata 有源描述也不显示为已输入；开启时展示完整 `scene_prompt` 或 `prompt`。
  主体 forward/backward 不是相机 W/S，不从主体文本推导相机按键。
- 底部明确写 `INPUT display, not measured motion`，不能据亮键判定控制成功。
  黑色 prompt 带增加观看版画布高度，原画面宽高、帧数和实际播放 FPS 保留，不改 raw。

## 使用与默认接入

在 FastVideoWM 仓库执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/visualize_h3_controls.py \
  --roots /absolute/inference/root --workers 4
```

适配的目录协议：根目录 `inference_manifest.json`，每个用例
`videos/**/metadata.json`、`raw.mp4`、`VIDEO_COMPLETE` 或 `COMPLETE`，
可选实际 `camera_to_world.npy`。旧版平铺结果须先明确映射元数据，不凭名字推测。
GPU生成未结束时可加 `--watch-until /absolute/inference/root`，每5分钟补一次，
只用 CPU。`visualization_summary.json` 和 `VISUALIZATION_COMPLETE` 记录完整性。
每个观看版完整解码验证帧数、尺寸、FPS，记录源/输入/renderer/output SHA256；
同一用例文件锁防止重复写入。来源变化时拒绝覆盖旧版本，需使用新的观看版目录。

当前默认接入点：

1. FastVideoWM `scripts/evaluation/portable_js_easy_h3.py` 的 decode 路径逐视频生成；
   `tools/run_portable_js_easy_h3_{step800,text_on}_20261022.sh` 汇总索引。
2. H3-World `code/vwm/run_portable_subject_text.py run` 结束前调用 CPU renderer。
3. 当前正在运行的旧 H3 进程不会热加载源码；独立 tmux
   `h3_controls_hud_backfill_20261022` 补齐历史128+32条及当前H3的64条。
4. 已有依赖队列 `h3_text_eval_then_fastvideo_text_train_20261022` 不重复启动。
   H3推理完成后先进行真实 VLM 文本条件编码和全量输入审计，再自动训练；
   `tools/launch_scene_text_training_20261022.sh` 启动前核对三组 HUD 完整性。
   非法数据或失败门禁会停止并留错误标记，不绕过检查启动训练。

## 验证

`PYTHONPATH=. .venv/bin/python -m pytest tests/local_tests/test_h3_controls_hud.py -q`
覆盖实际文本标志、I/K与J/L符号、恒定控制、复用 Cosmos 绘制且不改源图。
对真实输入已执行一次完整124帧生成与第二次断点命中检查，并人工抽帧检查版式。
