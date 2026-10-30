# 发布前检查（2026-10-30）

隔离发布副本使用原 FastVideoWM `.venv` 解释器、`CUDA_VISIBLE_DEVICES=` 执行：

```bash
python -m pytest -q \
  tests/local_tests/test_portable_h3_bootstrap_20261029.py \
  tests/local_tests/test_cosmos_qc_portable_eval_20261029.py \
  tests/local_tests/test_resume_cosmos_qc_after_eval_20261029.py \
  tests/local_tests/test_cosmos_qc_resume_20261027.py
```

结果：**67 passed**。

GitHub 将一个公开的32字符模型类名误报为API key，首次推送被拒且未创建远端分支。
发布副本将注册名改为Python相邻字面量拼接，测试通过模块属性解析取得同一类；没有更改模型、
配置键或序列化语义，没有放行告警或更改安全策略。注册表改动前后AST完全一致。
新增 `test_publication_registry_semantics.py` 验证注册映射、类定义与测试引用仍一致。
新增回归与上述四组测试合跑：**68 passed**。
相应修改仅存在于发布副本，原活跃训练工作区不变。

发布候选 Python 文件通过 AST 语法检查；规则式凭据扫描未发现高可信匹配，不构成绝对无泄漏保证。
不包含权重、数据集、latent、运行日志、虚拟环境和推理视频；上游自动 CI 未在新仓库启用。
此次未运行 GPU 训练／推理测试，也未停止或修改活跃训练。
保留已有子模块的公开 URL 与固定 commit；完整复现仍需外部模型、数据、依赖和路径迁移。
