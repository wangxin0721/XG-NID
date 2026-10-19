# R0-A每轮训练集Macro-F1观察

当前R0-A v2实际日志已出现第1轮2500/2500批、train_loss=0.2875083166465163、train_accuracy=0.92563125、160000样本、2500步，用时2184.080秒；第2轮继续运行。原入口仅记录loss和训练accuracy，没有逐轮Macro-F1，也没有逐轮保存预测，不能从accuracy反推F1。

新增`scripts/watch_r0a_train_f1.py`，只在服务器运行独立CPU观察进程，不修改或中断当前训练，不使用GPU、不加载测试图、不执行优化更新。Macro-F1针对完整160000条**训练集**，不是验证/测试分数，不能直接与论文约97%的测试Macro-F1比较。最终测试仍由原训练在第30轮后执行一次。

## 实际机制与限制

- 绑定当前`outputs/r0_aligned_full_s42_v2/config.json`，核验原冻结源码、原manifest、官方Training身份及基础协议；使用原Edge模型适配器。R0-A保存直接官方模型的state字典，加载时仅添加reference前缀，严格匹配参数；不改变模型定义。
- 单CPU线程、nice10、隐藏GPU。固定数字顺序、batch64、eval/no_grad，每轮完整遍历全部训练图，在CPU计算Accuracy/Macro-F1/Weighted-F1。独立评价的计算设备、遍历顺序与训练函数的shuffle训练集评价不同，属于另行标注的观察结果。
- 原训练会覆盖last.pt。独立线程每30秒检查稳定文件版本及ZIP完整性，读取不可变字节，确认binding/epoch；在独立观察目录保存捕获的轮次快照，并排队计算。CPU评价落后时继续捕获新检查点，减少错过轮次；已在观察器启动前或停止期间被覆盖的旧轮次无法恢复。结束时列出实际捕获、评价和缺失轮次，不能宣称保证捕获全部30轮。
- 快照包含原保存状态但不用于恢复或训练，只用于固定权重评价；不会写回原last.pt、修改官方日志或模型选择。新增文件全部位于`outputs/r0_aligned_train_f1_v2/`：observer绑定、排他锁、checkpoints快照及train_f1.jsonl。
- 每轮需要额外CPU全量推断、读取16万张图，会延迟显示并增加CPU/磁盘负载，即使低优先级也不能保证不影响训练速度。观察器不是无需计算的实时日志展示。
- 观察器使用文件锁防止重复启动；已保存同binding观察目录可继续，已经完成且SHA一致的观察结果跳过。相同epoch快照发生变化、binding改变或输入指纹改变时停止观察，不操作训练进程。

## 服务器执行

另开终端，保留原训练日志终端。原R0的watch_r0_train_f1.py不适配当前R0-A，不要用它直接监控v2。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/watch_r0a_train_f1.py > /tmp/xgnid_watch_r0a_f1.py
python -B -u /tmp/xgnid_watch_r0a_f1.py --repo . --interval 30
```

首先显示Captured TRAIN checkpoint，然后显示已评价样本数。每个轮次全量CPU评价完成后，输出：

```text
TRAIN epoch=<真实轮次> Macro-F1=<实际计算值> Accuracy=<实际计算值>
```

原训练日志不增加F1字段，结果独立保存在`outputs/r0_aligned_train_f1_v2/train_f1.jsonl`。只能观察和记录，不据训练或测试分数改变本轮固定预算/检查点规则。

Ctrl+C只停止这个观察器，不停止另一个进程中的R0-A训练。停止后训练仍可覆盖后续last.pt，因而可能无法事后补齐未捕获轮次。监控终端需保持运行；本工具不自行后台启动。

## 本地检查及待执行

本地仅做Python3.9 AST/编译、检查点快照读取及路径限制的标准库I/O检查，没有加载真实检查点、导入模型或执行训练/推断。服务器实际CPU评价仍待执行，当前没有新增任何真实Macro-F1结果。
