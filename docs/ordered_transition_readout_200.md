# 有序双尺度读出：200 轮等预算验证

## 延长依据

原 30 轮实验：Single 最佳为第 25 轮 97.664335%，Pointwise 为第 26 轮 97.678574%，Ordered 为第 29 轮 97.704448%。Ordered 的训练损失从第 20 轮 0.02051286 降到第 30 轮 0.01974328。最佳点接近预算末端，值得延长观察；这不保证继续训练一定提分。

## 本次固定配置

- 先只运行 seed42 的 Single、Pointwise、Ordered 三组，每组完整 200 轮，无提前早停。
- 每组 42,200 次参数更新（211 次/轮）。三组初始参数、每轮样本顺序和预算保持配对。
- 仍冻结相同原 C，仅训练原有新增读出；Adam lr=0.001、batch512、八类最终 NLL 均保持原值。
- 仍用验证 Macro-F1 选最佳，包含第 0 轮，同分取最早；同时保留最佳与末轮状态。
- 自动汇总 30、100、150、200 轮预算内各组的最佳验证结果及配对差值。150 轮是同一条训练轨迹的比较窗口，无需另起一轮。
- 不在本次延长中同时改变学习率、结构、损失或 seed。先区分延长训练的共同收益与 Ordered 的额外收益。
- 原投入参考（Ordered−Single ≥0.10 pp、Ordered−Pointwise ≥0.05 pp）保留用于计算资源决策，并结合全部八类及修复/退步分析。
- 本入口只训练并验证 seed42，不自动追加 seed43/44 或运行固定测试。后续按长预算验证结果决定。
- 新目录为 outputs/ordered_transition_readout_200；原 30 轮目录保持原样。
- 本实现不会从第 30 轮继续训练。旧 readout.pt 只保留 best/last 模型状态，没有 Adam 优化器状态及训练采样器状态。新目录从相同初始条件重跑完整 200 轮，避免把重置优化器的训练写成连续续训。入口的 --resume 只跳过已经完整保存的组。

## 服务器运行

激活原 wx 环境，在原训练服务器执行：

```sh
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-ordered-transition-200"
git fetch --no-tags origin refs/heads/codex/ordered-transition-200
git worktree add --detach "$xgnid_code" FETCH_HEAD
mkdir -p "$xgnid_repo/outputs"
nohup /bin/sh "$xgnid_code/scripts/launch_ordered_transition_readout.sh" "$xgnid_repo" 42 > "$xgnid_repo/outputs/ordered_transition_200_launcher.log" 2>&1 < /dev/null &
```

首次启动使用新的工作树与输出目录。不要把旧 v1 目录传作第三个参数。若该工作树已存在，应使用已有目录并核对 HEAD，不要删除它或覆盖已有结果。

训练入口的退出处理会运行：

```sh
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only ordered_transition_readout_200
```

该上传步骤已包含在启动入口内，正常完成和训练失败时均执行，上传本轮目录内所有生成文件（包括权重、预测、账本、报告和日志）到 codex/server-outputs。网络上传失败后可单独执行上面的上传命令重试。

## 查看进度与结果

```sh
tail -f outputs/ordered_transition_readout_200-42.log
```

三组依次运行，缓存原图特征在先。主结果为 seed42/report.md；预算窗口比较为 seed42/budget_comparison.md 与 budget_comparison.json，均位于新实验目录内。训练损失下降与验证 F1 提升分别观察；最终报告保留全部组别，不能仅将 Ordered 的 200 轮与 Single 的 30 轮比较。

此变更只准备并发布代码。正式 200 轮训练需在有原始图和主干权重的服务器运行。
