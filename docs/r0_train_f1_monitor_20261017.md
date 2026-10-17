# R0 运行中查看训练集 F1

当前正式训练的 `evaluate` 已计算训练集 Macro/Weighted-F1，但逐轮日志仅保存 Accuracy，无法从历史日志恢复 F1。训练进程已加载的 Python 代码不会因编辑源文件而更新。本补充不修改冻结代码、训练进程、检查点或 R0 数据。

`scripts/watch_r0_train_f1.py` 是独立服务器 CPU 观察器：读取 `outputs/r0_full_launch.json` 定位实际代码 worktree、清单和输出；验证源码/清单/检查点绑定，读取稳定的内存检查点快照，严格加载同一 Edge 模型，对完整 160000 张源训练图做 eval/no_grad 推断。不会加载测试图，不优化模型，不写实验目录，也不改变训练进程随机数状态。禁止把训练 F1 当测试性能。

观察器只显示完成轮次的检查点评估，无法恢复已被覆盖的旧轮权重。CPU 全训练集评估有延迟，若期间训练跨过多轮，会直接读取最新轮；不保证每轮都覆盖，不是逐 batch 的即时 F1。CPU 与原训练 GPU 的数值实现不同，结果可能有细微差异。单 CPU 线程、低进程优先级能减少争用，但仍增加 CPU/磁盘开销，不能保证训练耗时不变。进度每 100 个评估 batch 打印，完整评估完成后输出 F1。

在服务器原仓库 wx 环境的新终端执行：

```bash
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/watch_r0_train_f1.py > /tmp/xgnid_watch_r0_f1.py
python -B -u /tmp/xgnid_watch_r0_f1.py --repo . --interval 60
```

Ctrl+C 只停止观察器；R0 是独立的后台进程，继续训练。若只评估当前检查点一次，在最后一行追加 `--once`。

交付检查仅限本地语法和静态审查；没有在本地加载模型、数据或训练，也尚无服务器观察器执行结果。正式验证由上述服务器命令完成。观察器新增在原 worktree 外，不更改 R0 已通过 CPU/CUDA 冒烟的源码哈希。
