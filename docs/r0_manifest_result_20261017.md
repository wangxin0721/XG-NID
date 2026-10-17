# R0 源数据清单实际结果（服务器，2026-10-17）

依据用户真实回传的summary.json与source_evidence.json；重复粘贴的两份结果内容一致。没有本地运行任何模型或实验。本记录只验收服务器清单扫描，不宣称模型冒烟或复现训练通过。

## 实际完成项

- 服务器wx解释器，PID3798797；北京时间10:54:04开始，11:05:20结束，约11分16秒。status=completed，exit_code=0，phase=manifest_preparation_only，cuda_used=false，training_started=false。
- 清单ID：`22d027fe3d0b0ad9079da9a13614597b0e40bb883a34b2d4d825452d6129d383`。
- 运行代码提交：`9d52823dd0a20e1001b3aeab9d0074f0a1f6d115`。source_repo和source_graph_root与此前盘点相同。
- 训练160000，每类20000；测试25557，标签0/2/3/4/5/6各4000、1为1090、7为467。table4_counts_match=true。全量扫描已通过当前r0_data实现的维度、类型、有限性、图结构、反向属性及标签检查。
- CSV SHA256：train=`1b546e6c69faad482b98b4c95219ef9b13aba0c13afcfe0b4abe69c1e7f38b02`，test=`08b256997d0368dde6c85a3b43ef3d56b4ce336939fcdce41596725bbdee42ed`。与既有本地源CSV审计记录一致，只证明当前候选文件内容一致，不自动证明图的每一行对应关系。

## 重叠与限制

| 输入指纹 | 跨train/test共有输入组 | 受影响测试样本 | train内多余副本 | test内多余副本 | 标签冲突组 |
|---|---:|---:|---:|---:|---:|
| Edge：节点+连接+边属性 | 1644 | 3346（约13.09%） | 47657 | 1741 | 0 |
| No Edge：节点+连接 | 1658 | 3363（约13.16%） | 47689 | 1747 | 0 |

“分区内多余副本”是每个输入组在该分区的count−1之和，不等于受影响样本总数，也不能与跨集受影响样本直接相加。两种指纹的计数也不能相加。

R0保留现有源train/test及采样副本，明确报告重叠，不重新划分、不去重、不调整测试成员来匹配论文。该结果可以用于原方法在当前源成员协议下的可审计重实现，不能当成独立无重复测试上的泛化估计，也不能与去重分组E0直接计算算法提升。

historical_author_sample_identity_verified=false，csv_to_graph_row_mapping_verified=false：作者原始成员/capture身份、CSV到图的逐行生成证据仍缺失。不得把相同数量或相同CSV哈希改写为原作者图样本已严格恢复；也不把当前重叠直接归因于作者论文实验。

source_evidence回传的code_root含有字面量`$`和反斜线，说明实际worktree目录名与预期形式不同，但该目录已实际执行并记录正确提交。后续从JSON读取真实code_root，并始终引用路径；不手工重拼、移动或删除现有worktree/清单，不为美化路径重跑扫描。

## 下一步：服务器CPU真实图功能冒烟

E1最后已知仍运行；GPU smoke等待资源空闲。可以先在服务器CPU上使用当前清单每类4图（32图）执行模型维度、六个GAT属性输入、梯度、属性干预、优化一步、两轮小训练及last重载检查。仅加载train图计算模型输出，test=null；CPU报告不能授权CUDA full。

在保存有R0_PREP变量的原服务器wx终端执行。以下命令不会更新worktree、改变E1环境或开启GPU：

```bash
R0_CODE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["code_root"])' "$R0_PREP/source_evidence.json")"
R0_CPU="$PWD/outputs/r0_cpu_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  /home/nzxn/miniconda3/envs/wx/bin/python -u "$R0_CODE/scripts/run_r0_edge.py" \
  --mode smoke --manifest-dir "$R0_PREP/manifest" --device cpu --workers 0 \
  --output "$R0_CPU" > "$R0_CPU.log" 2>&1
R0_CPU_RC=$?
printf '%s\n' "$R0_CPU_RC" > "$R0_CPU.exitcode"
tail -n 40 "$R0_CPU.log"
test ! -f "$R0_CPU/status.json" || cat "$R0_CPU/status.json"
test ! -f "$R0_CPU/summary.json" || cat "$R0_CPU/summary.json"
```

此命令前台运行，结束前等待，不重复启动。若变量已丢失，应先用实际既有路径恢复R0_PREP，不新建时间戳冒充已有清单。失败回传外部log/error.log；成功回传summary.json/exitcode。不会根据两轮训练的Accuracy挑配置或作论文性能比较。随后再安排同代码/清单的真实CUDA smoke；正式30轮仍未启动。

## 服务器CPU冒烟与E1完成回传

用户回传了`outputs/r0_cpu_smoke_server_v1/summary.json`尾部及其头部/运行环境摘录：status=completed、exit_code=0、mode=smoke、epochs_completed=2。真实服务器环境为Python3.9.23、torch2.3.1+cu118、PyG2.5.3、CUDA构建11.8；本次device_type=cpu/gpu=null。没有本地运行模型。之前一次启动因日志路径被粘贴成字面量`$\(PWD/...`而在shell重定向阶段失败，Python没有启动；修正为简单相对路径后本次CPU运行完成。

- 参数431704；loss=1.9906264543533325，optimizer_step_passed=true，test=null；记录的流程wall_seconds约1.4427（不含全部解释器/依赖导入和清单解析时间，不能据此估算正式训练）。
- 两层Link边投影梯度最大值约2.4913e-6/4.3562e-6，Reverse Contain约1.4277e-4/4.1673e-6；Contain两层均0。
- 隔离属性干预的最大logprob差：Contain0、Link0.07998848、Reverse Contain27.01780891；重复前向抖动0、容差1e-6，支持真实边属性作用。Contain单入边允许其注意力项抵消，不据此判定属性漏传。
- last.pt SHA256=`f36c0755a19c203b3cffc8d9615fe18ccf4b87e511e45ee22e836c51499fcb51`。

同次回传E1根status=completed/exit_code=0，epoch日志最后为47轮。最佳轮次、最终测试指标和完整产物尚未回传；不能把epoch47验证Macro-F1=0.9477984714392171当成最终测试成绩。E1完成记录不等于已确认cuda:0无人占用；新的GPU任务启动前仍需检查当前进程/资源安排。

## 下一步：真实CUDA冒烟，仍不启动full

在原服务器仓库wx终端先收取E1指标并检查残留训练进程：

```bash
cat outputs/grouped_full_E1_s42_v1/E1/metrics.json
pgrep -af '[p]ython.*(train_grouped.py|run_r0_edge.py)' || true
```

确认E1进程已退出且cuda:0没有其他训练占用后，使用原有独立worktree和同一清单。只运行小规模smoke，不运行full，不改变torch/驱动：

```bash
R0_CODE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["code_root"])' "$R0_PREP/source_evidence.json")"
R0_GPU=outputs/r0_cuda_smoke_server_v1
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/nzxn/miniconda3/envs/wx/bin/python -u "$R0_CODE/scripts/run_r0_edge.py" \
  --mode smoke --manifest-dir "$R0_PREP/manifest" --device cuda:0 --workers 0 \
  --output "$R0_GPU" > "$R0_GPU.log" 2>&1
printf '%s\n' "$?" > "$R0_GPU.exitcode"
tail -n 50 "$R0_GPU.log"
```

返回日志及summary/status/exitcode。CUDA不可用或属性干预/梯度/优化检查失败均如实报告，不回退CPU冒充GPU通过，不动E1检查点。只有同代码、配置、清单及runtime的真实CUDA smoke通过、结果人工核查后，才提供下一步正式训练启动操作。本段命令尚未执行。
