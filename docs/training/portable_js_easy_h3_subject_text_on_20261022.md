# checkpoint-800 文本开启诊断：16 × forward/backward

2026-10-22 用户明确要求：同一个最新模型在推理时加入文本，16个首帧分别测试主体 forward/backward，共32个视频，使用GPU0–7。训练继续暂停，不能自动恢复。

## 权重与输入

- 权重：`/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022/checkpoint/checkpoint-800`，来自 union 115,988 条训练轮次的完整 checkpoint-800 硬链接快照。
- 原始协议：`/datacc05/lizihao/js_easy_benchmark_portable_20261021/manifests/subject_text.json`，保留全部首帧描述和主体运动文本，不使用包中WAN VAE或UMT5缓存。
- 16个样本 `js_00001`–`js_00016`，每个样本2条完整prompt，alpha=1。forward/backward指主体沿自身朝向前进/反向后退，不是将相机轨迹分别改为W/S。每对首帧、相机路径、种子、时间轴完全一致。
- 首帧H3 VAE latent、相机数值路径直接复用前次已验证的原生H3输入。未来latent以独立噪声生成，不使用GT。没有对应主体文本GT，不报告PSNR/SSIM/LPIPS。

## 仅推理端开启文本

相较前次action-only评测，独立 `inference_config.yaml` 只修改 `models.student.vision_only_conditioning: true -> false`，不修改原始训练配置或权重。使用H3原生Qwen3-VL重新编码每个完整prompt和对应首帧，保留真实vision/text token。

本checkpoint训练时为vision-only。因此这是推理端文本开启的能力诊断，不可称为文本微调模型，也不保证主体方向一定正确。保留先前文本被过滤的结果，便于后续对照。

输入审计要求：32个实际embedding、16对不同文本产生不同embedding、首帧latent和K不变、无1024-token截断、完整caption精确匹配；DiT入口再次检查 `vision_only_conditioning=false` 且 `text_token_count_entering_dit>0`，否则立即报错，不允许仅写metadata假称文本已启用。

## 执行

启动脚本：`tools/run_portable_js_easy_h3_text_on_20261022.sh`。

1. 检查八卡空闲，冻结配置和32条用例，引用既有完整权重哈希快照。
2. GPU0/2/4/6四个独立Qwen worker，每个编码8条。完成后释放Qwen并运行CPU配对审计。
3. GPU0–7单个HSDP模型，SP=2、4路视频并行，8波完成32条。不能启动8份独立完整DiT以免主机内存OOM。原生shift12、requested50 steps、guidance1；不启用H3不支持的文本CFG。
4. 8卡并行原生VAE解码，自动核对每个124×480×832 MP4。输出FPS沿用前次124/61×源FPS，保持时长一致。
5. 汇总每对生成latent差异。非零只代表对文本敏感，不等于前后方向正确，仍需抽帧/人工审核主体相对背景的运动。

结果：`/datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022/`。

tmux：`fastvideowm_text_on_step800_20261022`，2026-10-22 14:17 CST（15:17 JST）启动。

首次启动后发现审计计数将整个Qwen条件段（392视觉token加文本token）误称为文本token数，14:19 CST中断尚未完成首波的推理并改为根据packing中的 `token_tags==1` 精确计数后重新启动。此修正不改变生成信号；32条实际文本条件已编码并通过CPU审计，恢复直接复用。首次日志保留于 `logs/infer_before_token_audit_fix.log`。

脚本：`scripts/evaluation/portable_js_easy_h3_subject_text.py`；复用通用 `portable_js_easy_h3.py` 的模型加载、采样和解码，支持每case独立文本Parquet及DiT入口审计。

重要输出：`inference_manifest.json`、`checkpoints.json`、`conditions/*/encoding.json`、`text_condition_audit.json`、`videos/alpha100/single_action_text/<case_id>/{raw.mp4,metadata.json,latents.pt}`，以及便于浏览的 `raw/alpha100/single_action_text/*.mp4`。

`TEXT_CONDITIONS_COMPLETE`、`ALL_LATENTS_COMPLETE`、`ALL_VIDEOS_COMPLETE`、`INFERENCE_COMPLETE`分别标记不同阶段；失败写 `NEEDS_ATTENTION`。查看实时 `status.json` 和日志，不能把启动等同于完成。预计耗时仅参考上一轮约6–7分钟/4视频波次，另加文本编码、模型加载和解码；启用文本后的实测速率可能变化。
