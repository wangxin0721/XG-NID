# R0 服务器真实回传记录（2026-10-17）

证据来源：用户回传的r0.preflight.v1终端JSON，采集时间2026-10-17 02:45:44 UTC（北京时间10:45:44）。以下为该时刻快照，不是本地探测结果，也不保证此后状态未变。没有执行新的本地模型检查或实验。

## 已确认事实

- 服务器仓库：`/home/nzxn/下载/WX/load/XG-NID`；工作区干净，分支`codex/a44a8ef-restore`，HEAD=`b125d5c9ac3d05966e91d41fa602dd7f75b65792`。仅fetch R0分支而未切换，符合取证命令设计；不能称R0文件已经进入服务器当前工作树。
- 检查解释器：`/home/nzxn/miniconda3/envs/wx/bin/python`；E1实际进程解释器为该环境下`bin/python3.9`。环境名确认为小写`wx`。
- 发现5个`train_grouped.py`进程：3768665、3768993、3768995、3773057、3773058；命令均指向`--experiments E1 --mode full --output outputs/grouped_full_E1_s42_v1`。
- E1参数：seed42、cuda:0、batch16、lr0.001、epochs100、patience12、workers2；split ID=`a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a`，引用既有E0完成报告。进程存在得到真实支持；最近epoch、是否持续推进、完成状态及结果尚未采集。
- 冻结训练入口给train/val各配置2个persistent workers，因此“1个主进程+4个子进程”是与当前代码相符的解释；本次回传未含PPID，不能确认为父子关系，也不能把5行同argv直接判为5次训练。不得据此终止任何进程。
- 候选图目录存在：`data/processed/CICIoT2023_processed/processed`。文件名计数：`data_N.pt`160000、`data_test_N.pt`25557、其他2（示例pre_filter.pt）。这是名称盘点，不是完整图验证、逐类统计、作者样本身份恢复或R0数据验收。
- 根目录和train/test子目录各存在一份候选CSV；对应大小分别175063419字节（test）和1027673808字节（train）。相同大小不能证明内容相同，尚未计算本次服务器文件SHA256。
- `nvidia-smi`退出18，原文：`Failed to initialize NVML: Driver/library version mismatch`，`NVML library version: 580.178`。尚无GPU占用、CUDA运行健康或E1持续更新的可用结论；进程存在不能代替这些证据。

## 下一步仅做只读核查

在既有wx服务器终端、仓库根目录执行；不切分支、不重启、不安装/卸载驱动、不启动R0 GPU任务：

```bash
date -Is
ps -p 3768665,3768993,3768995,3773057,3773058 -o pid,ppid,etime,stat,args
tail -n 3 outputs/grouped_full_E1_s42_v1/E1/epochs.jsonl
cat outputs/grouped_full_E1_s42_v1/status.json
cat outputs/grouped_full_E1_s42_v1/E1/status.json
tail -n 30 outputs/grouped_full_E1_s42_v1.log
python -c 'import sys; from importlib.metadata import version; print(sys.executable); print({p:version(p) for p in ["torch","torch-geometric","scikit-learn"]})'
```

PID来自10:45:44快照；若已退出，以status/日志和当前进程为准，不用旧PID认定仍运行。日志路径为既有启动命令候选，如果文件不存在，回传缺失错误并寻找真实重定向路径；不创建替代日志。版本命令只读取包元数据，不导入torch或创建CUDA上下文。正常运行但epoch尚未结束时日志可能暂不更新，不能仅凭一次tail判定卡死。

本轮完成的是服务器环境和文件命名的真实盘点。R0清单生成、真实图内容/类别/重叠检查、GPU冒烟及正式训练仍未执行；E0完整原始交付尚待采集。待确认E1进展后安排CPU/I/O清单扫描，GPU冒烟留待资源空闲及环境可用时进行。

## 10:49:42回传补充

用户再次回传真实ps/status/完整epoch1～17日志。主进程3768665已运行约1:38:21，其余四个进程PPID均为3768665，确认是该训练的子进程，结合train/val persistent workers配置，不是五个独立启动的E1。两级status均为running。已完成17轮，每轮实际消费107621训练图；最近训练约277～284秒、验证约33～34秒，loss有限；日志记录cuda:0与非零峰值CUDA分配内存。这支持训练已在CUDA上完成多轮，不能把nvidia-smi失败直接等同于CUDA训练不可用；当前尚无完成状态或最终测试结果。

已回传17轮中的最佳验证Macro-F1为epoch15的0.9435216988706552，Accuracy0.9635579354454856；这是中途验证，不是E1最终结果。epoch17验证Macro-F1为0.9391866840180138。不会据此调整冻结超参数或提前终止。

日志还显示torch-spline-conv导入undefined symbol后被PyG禁用，以及Can't initialize NVML警告。冻结E1源码路径使用GAT而非SplineConv；日志证明这些报警未阻止已回传17轮。不安装/重装包或驱动，不中断E1；后续独立R0 smoke仍须真实验证实际依赖和CUDA路径。

下一步新增服务器CPU包装器`prepare_r0_server.py`：计算两份候选CSV的SHA256，明确记录CSV→图映射尚未验证，再调用已交付r0_data生成源train/test图输入清单、逐类数和重叠报告。运行于独立Git worktree，避免切换E1工作树；输出仅写新目录，不用GPU，不修改原数据。该新增包装器本地仅静态审查，未运行；服务器清单扫描仍待真实执行。清单完成也不等于GPU冒烟通过或严格恢复作者样本。
