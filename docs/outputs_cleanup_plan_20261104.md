# outputs清理清单（仅审查，未删除）

本地以2026-11-04实际文件核对；服务器以已上传codex/server-outputs快照及当前代码依赖为依据。
快照创建于2026-11-02，不能代表服务器当前残差任务的进程状态。实际删除前须在服务器运行文末只读审查。
大小均为MiB，路径相对于各自outputs根目录：

- 本地：`D:\XG-NID\outputs\`
- 服务器：`/home/nzxn/下载/WX/load/XG-NID/outputs/`

## 一、本地可优先清理的重复副本

| 相对路径 | 大小 | 核对结果／条件 |
|---|---:|---|
| `outputs/`（即outputs/outputs） | 136.57 | 274个文件逐个与外层同名文件核对，全部字节相同；外层原结果保留 |
| `scripts/` | 0.03 | 9个复制脚本与Git HEAD完全相同；不要删除项目根目录scripts |
| `docs/` | 0.001 | 1个复制文档与Git HEAD相同；不要删除项目根目录docs |
| `README.md`、`main.py`、`requirements.txt` | 合计约0.003 | 位于outputs内部的副本均与Git HEAD相同 |
| `.source_snapshot_manifest_20261104.json` | 0.24 | 本次核验生成的临时清单；核验摘要已进入docs，可从GitHub重新读取 |
| `_organized/` | 0.02 | 仅不再使用整理视图时可清理；本地是整理目标路径的小文本，不是原始模型 |

服务器上传快照中没有上表的outputs/outputs、scripts、docs及这些项目文件副本，不能把“本地重复”当成服务器也存在。
`outputs/src/`有与当前代码不同的旧源码；`outputs/third_party/`还含旧Git元数据，这两项本轮不列入可直接删除目录。

## 二、本地与服务器共同的条件清理项

以下文件在本地的SHA与上传快照一致，已有GitHub备份。只有对应实验结束、无需立即从缓存重新训练／核验时，才适合清理。
这不是删除指令；不要清理当前运行任务的缓存。

| 相对路径 | 大小 | 应保留的内容 |
|---|---:|---|
| `packet_pool_pair_s42_v1/train_packet_nodes.pt` | 323.49 | 保留confirmation目录中的同名规范副本 |
| `packet_pool_pair_s42_v1/val_packet_nodes.pt` | 40.72 | 同上 |
| `packet_pool_pair_s42_recovery_v1/train_packet_nodes.pt` | 323.49 | 同上 |
| `packet_pool_pair_s42_recovery_v1/val_packet_nodes.pt` | 40.72 | 同上 |
| `packet_pool_pair_s42_recovery_v2/train_packet_nodes.pt` | 323.49 | 同上；保留该目录所有结果及核验记录 |
| `packet_pool_pair_s42_recovery_v2/val_packet_nodes.pt` | 40.72 | 同上 |
| `relation_pair_s42_v1/train_pre_relation.pt` | 946.53 | 保留val_pre_relation.pt、候选权重、报告、预测及全部配置／receipt |
| `cross_layer_pair_s42_v1/train_cross_layer.pt` | 114.75 | 保留val_cross_layer.pt、两组权重及全部配置／receipt；折叠核验只读验证缓存 |

前三个池化目录的6个缓存副本，可回收合计1092.62MiB；加上关系和跨层训练缓存，共2153.90MiB（约2.10GiB）。
本地再加重复的outputs/outputs，共约2.24GiB；未把小文本和潜在正在运行的残差缓存计入。

**池化规范副本选择 `packet_pool_confirmation_s42_v1/{train,val}_packet_nodes.pt`，这两个保留。**
四个池化目录的训练缓存SHA完全相同，验证缓存SHA也完全相同，已实际核对。
这些旧缓存的原路径仍被池化恢复／确认脚本的历史inventory和SHA校验引用；删旧副本后，重跑那些脚本会要求恢复原路径。
若希望旧核验命令始终可运行，更适合对同盘副本做硬链接去重而不是直接删除，本次未执行去重。
当前残差入口会从原图重新提取自己的pre-residual缓存，不使用上表的旧关系／池化／CL训练缓存。

## 三、两端均应保留

| 路径或内容 | 原因 |
|---|---|
| `residual_pair_s42_retry_v1/`及其log／pid | 当前任务可能正在运行，当前阶段和进程必须以服务器现场为准 |
| `residual_pair_s42_v1/` | 本次失败证据先保留；空间很小，不必现在删除 |
| `cross_layer_pair_s42_v1/CL-Multi/`、`CL-Last/` | 当前最佳Multi126和Last113及其原始检查点、指标、预测 |
| `cross_layer_pair_s42_v1/val_cross_layer.pt`、`cross_layer_fold_check_v1.json` | 残差前置核验需要；禁止因“都是缓存／JSON”而批量删除 |
| `relation_pair_s42_v1/`中的候选权重及小型记录 | 保留Rel-Gate134和匹配对照证据 |
| `packet_pool_confirmation_s42_v1/` | P1第146轮、匹配P0及唯一追加150轮确认；缓存规范副本也放这里 |
| `packet_pool_pair_s42_recovery_v2/`的小型记录 | 原20轮结论、恢复验收及150轮前20轮比对来源 |
| `fusion_gate_pair_s42_v1/` | 原D32-Gate第62轮及当前结构实验直接来源 |
| `fusion_dimension_pair_s42_recovery_v1/` | 来源链；其中train_encoder64.pt和val_encoder64.pt被元数据校验读取，不能按“大缓存”删除 |
| `joint_gate_lrscale01_s42_v1/` | 来源链的历史J2检查点与summary |
| `e2_concat_pair_b512_s42_lr1e4_v1/` | 冻结N0注册来源及模型 |
| `r0_graph_rebuild_author_offline_v2/`、`r0_validation_split_s42_v1/`、`server_graph_split_audit_v1/` | 原图重建、分组和清单证据，可能用于复现，不能当纯日志删除 |
| 各正式基线、重复种子实验的best.pt、结果与预测 | 论文对照和历史最好成绩，尚无逐项退役决定 |

不要批量删除所有last.pt、epoch_zero.pt、*_preflight.pt或cache文件：现有严格来源核验不仅检查best.pt，也可能读取这些文件。
不要删除status.json、binding.json、config.json、summary.json、metrics.json、cache_receipts.json、epochs.jsonl及验证预测导出。
这些文件相对模型／缓存很小，且是证明实验完成、初始一致和原图回放的重要证据。

以下目录的复制状态仍为running或与已上传快照有差异，本轮只列待核实：
`e3_gate_screen_E3aZ_b512_s42_v1`、`r0_full_s42_20261017T065259974842Z`、
`fair_vector_five_runs_v1`、`fusion_bh_b1024_s42_v1`、`grouped_full_E3_s42_v1`、`r0_aligned_full_s42_v1`。
本地复制的running／pid不证明服务器仍有对应进程，必须现场核对后才能退役。

## 四、服务器可以小规模清理的项目

- `_organized/`只在确认不再需要该链接视图时移除；仅移除链接及视图目录，不能递归跟随链接删除目标。
- 已明确退出的任务留下的.pid可移除，但不要仅凭本地PID数字去结束服务器进程。
- 已确认不用继续定位问题的失败启动日志、smoke/probe目录可归档后移除；本次没有逐个证明这些目录均无历史依赖，不提供通配符批删建议。

这些小项目回收量很少，优先考虑已验证重复副本／训练缓存更实际。

## 五、服务器只读复核命令

使用本次提交中的scripts/audit_outputs_cleanup.py，归档scripts后运行：

```bash
/home/nzxn/miniconda3/envs/wx/bin/python -B "$bundle/scripts/audit_outputs_cleanup.py" --repo "$PWD"
```

脚本只打印当前重复副本、缓存大小／哈希、磁盘状态记录以及Linux进程信息，不含任何删除或硬链接操作。
Windows执行同一脚本不能判断服务器进程状态，输出会明确标记未检查。
本地本次审查详情见 `outputs_cleanup_audit_20261104.json`。数据文件未删除或改写。
