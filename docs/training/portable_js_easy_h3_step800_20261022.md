# 新 JS-easy portable 协议的 MiniMax-H3 适配

输入 skill：`/datacc05/lizihao/js_easy_benchmark_portable_20261021/skill/js-easy-benchmark-eval/SKILL.md`。该包本身是 Cosmos/WAN 协议；本记录描述实际 H3 适配，不改变源测试包。

本次使用 union 训练轮次的 `checkpoint-800`：`/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/runs/train/checkpoints/checkpoint-800`，其父模型为 multicam5k checkpoint-2100。不能将此 checkpoint-800 与早期其他实验的 step800 混为一谈。停止训练前实时约 step850；未保存的更新不属于本次评测权重。

输出根目录：`/datacc05/lizihao/worldmodel/fastvideowm_union_step800_portable_eval_20261022`。tmux：`fastvideowm_portable_step800_20261022`。启动器：`tools/run_portable_js_easy_h3_step800_20261022.sh`。训练已暂停，评测后保持暂停，不自动恢复。

## 用例和模型条件

使用 `manifests/full.json` 的 A16（GT 相机轨迹）、B16（单一 W/A/S/D/I/J/K/L）、C32（主体前进/后退）用例。每组对 alpha=1、0.5 各执行一次，共128个输出。相同首帧使用相同 seed，完整 prompt、动作、来源、alpha 均保存在输出 manifest 和各视频 metadata。

当前 checkpoint 配置是 `vision_only_conditioning=true`：只有首帧 Qwen 视觉 token、首帧 VAE condition 和数值 C2W 进入 DiT；场景/相机/主体文本均不进入前向。因此 C 组是“模型没有使用该文本条件”的负对照，不能据此声称测试了已启用的主体文本能力。相同 pose/seed 的前后运动文本对理论上会生成相同结果。不能为了测试临时把配置切成文本启用而将其当作原训练模型。

## 时间轴、空间和编码

H3 使用其原生124帧、37 latent时间位置。源61帧 GT 和逐帧 pose 使用同一 `round(linspace(0,60,124))` 对齐，再以 H3 的原生37 anchors取相机位置。记录的是等进度比较，不能把124帧指标直接与WAN61帧分数排名。

为保持源视频时长，输出 FPS = 包中 manifest FPS × 124/61。例如 B 组源16FPS，H3124帧按约32.525FPS播放。首帧和GT均使用PIL LANCZOS等比缩放及832×480中心裁剪。GT与首帧全部重新使用MiniMax-H3 VAE编码；不使用包内WAN `[48,16,30,52]` latent。首帧 Qwen 视觉条件重新编码；不使用UMT5 `[512,4096]` embedding。

相机直接读取portable包的 `inputs/*_trajectory_gt_pose.json`、`*_single_action_pose.json`，重设首帧原点，alpha通过SO(3)旋转向量及平移同时缩放。B/C使用包中总平移1.1600655和15°旋转，避免沿用旧H3 20°轨迹。归一化内参K仍读取每个case自身的标定；当前checkpoint关闭intrinsics参与PRoPE，保持训练配置一致。

生成器只使用首帧和相机路径，独立噪声初始化未来latent。GT只用于单独flow loss和像素指标。B/C的latent shape carrier为全零，无未来GT。音频token为零长度。H3使用原生shift=12 flow scheduler，50个requested inference steps，并记录实际transformer调用次数；不套用WAN UniPC/CFG。

## 执行和恢复

1. 验证完整DCP，创建硬链接快照防止训练checkpoint清理影响评测；逐文件SHA256和独立inference_config保存到结果根目录。
2. 四个独立GPU worker分别负责4个scene的VAE和Qwen首帧condition，输出gt/simple Parquet、first_frame.png、GT像素、pose和alignment。
3. 8卡只加载一个HSDP模型，SP=2组成4路，连续处理32波用例。GT flow-loss有额外forward，因此每波必须来自同一group。恢复时HSDP全部rank保持相同forward次数，不能跳过某个lane但让别的lane继续forward。
4. 模型释放后使用8卡独立VAE decode，完整解码验证128个raw MP4。raw硬链接视图在 `raw/alpha100|alpha050/<group>/*.mp4`。
5. 自动计算A组flow loss、逐帧PSNR、skimage SSIM、AlexNet LPIPS（包括全部124帧，另行注明口径），并尝试本机可用VBench adapter指标。缺包/缺权重明确写null及原因；FastVideo adapter结果不冒充官方VBench七项全覆盖。当前本机尚未安装官方vbench，官方七项完整评测需要匹配的环境/权重后补。

训练恢复只在用户要求时进行，从最新完整union checkpoint选择，不能默认用启动器旧的checkpoint-200覆盖进展。

## 文件说明

`checkpoints.json` 保存实际权重、哈希、配置、模型来源、协议差异；`inference_manifest.json` 保存128条请求；`inputs/js_*/`保存重新编码的H3输入；`videos/alpha*/<group>/<id>/`保存latent、数值相机、metadata、raw MP4及GT flow loss；`logs/`保存每阶段和每个worker日志。`ALL_LATENTS_COMPLETE`、`ALL_VIDEOS_COMPLETE`、`CORE_METRICS_COMPLETE`分开记录，失败写`NEEDS_ATTENTION`，不能只看tmux是否退出判断成功。
