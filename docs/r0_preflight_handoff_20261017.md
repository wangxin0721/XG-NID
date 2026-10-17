# R0 下一步：E1运行期间的服务器轻量取证

用户已真实回传E1正在运行，按此状态保留任务；尚未取得PID/日志/环境等原始回传。当前本地通过既有SSH别名再次尝试只读连接，跳板banner交换超时，没有接触服务器进程。本轮未执行服务器检查、数据清单生成、CUDA冒烟或训练。

新增 `scripts/inspect_r0_server.py`：只用标准库，读取/proc内train_grouped进程的PID、命令、实际解释器、cwd/output，采集nvidia-smi、Git状态和源图文件名计数/CSV路径大小。不会导入torch、反序列化图、创建CUDA上下文、计算完整CSV哈希或启动训练。命名数量不证明原始划分相同。默认只盘点已知数据目录，失败如实记录；若真实目录不同，增加`--data /真实图目录`。

在服务器**已有仓库根目录**执行以下命令，不切分支、不覆盖当前工作树，E1保持运行。python仅需标准库，无需先猜测conda环境：

```bash
git fetch origin codex/r0-edge-preparation
R0_CHECK="outputs/r0_preflight_$(date -u +%Y%m%dT%H%M%SZ).json"
if git show origin/codex/r0-edge-preparation:scripts/inspect_r0_server.py > /dev/null; then
  git show origin/codex/r0-edge-preparation:scripts/inspect_r0_server.py | python - --repo . --output "$R0_CHECK"
fi
```

回传终端JSON或R0_CHECK文件内容。获取回传后，使用实际训练解释器、图目录和来源证据建立独立R0清单；完整清单扫描消耗I/O，GPU冒烟则等GPU空闲，均不能将轻量盘点退出0当成训练验收。

本地验证：脚本在当前Windows仓库执行，正确报告/proc不可用、nvidia-smi不可用、服务器候选图目录不存在；未将这些本地结果计为服务器信息。Python语法检查通过。此前R0的4项合成测试结论不变，没有重跑真实实验。
