# 服务器结果分类入口

原文件均在原位，分类目录中的条目是链接。训练可以继续写入，链接内容同步可见。

请通过仓库根目录 XG-NID-organized.code-workspace 打开分类工作区。
状态仅来自磁盘 status.json，不能据此断定进程是否存活。类别由名称确定，不代表成功与否或最佳结果。
新实验结束或启动后可重新执行本工具，补充新增链接、刷新状态。不会自动删除失效链接。
不要在链接目录内删除或修改文件，这会作用于原始实验。

## 01 正式实验 · Batch 对照

- `e0_batch_full_b512_s42_shared_v1` — 文件记录：running；exit_code=未记录（非进程检查）
- `e2_batch_full_b1024_s42_v1` — 未核实（仅旁文件或未知类型）
- `e2_batch_full_b256_s42_shared_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e2_batch_full_b512_s42_shared_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e3_batch_full_b512_s42_shared_v2` — 文件记录：failed；exit_code=1（非进程检查）
- `e3_batch_full_b512_s42_shared_v3` — 文件记录：completed；exit_code=0（非进程检查）
- `e3_batch_full_b512_s42_v1` — 未核实（仅旁文件或未知类型）

## 02 正式实验 · 原 E0–E4

- `grouped_full_E0_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_full_E1_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_full_E2_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_full_E3_s42_resume_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_full_E3_s42_v1` — 文件记录：running；exit_code=未记录（非进程检查）
- `grouped_full_E4_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）

## 03 门控改进 · E3a 与后续版本

- `e3_gate_probe_E3a_b512_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e3_gate_probe_E3a_b512_s42_v2` — 文件记录：completed；exit_code=0（非进程检查）
- `e3_gate_screen_E3a_b512_s42_v1` — 文件记录：failed；exit_code=1（非进程检查）
- `e3_gate_screen_E3a_b512_s42_v2` — 文件记录：running；exit_code=未记录（非进程检查）

## 04 门控诊断与缓存分析

- `e3_gate_cache_analysis_b512_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e3_gate_diagnostic_b512_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）

## 05 R0 系列对照

- `r0_aligned_full_s42_v1` — 文件记录：running；exit_code=未记录（非进程检查）
- `r0_aligned_smoke_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_cpu_smoke_server_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_cuda_smoke_20261017T064602357890Z` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_full_launch.json` — 未核实（仅旁文件或未知类型）
- `r0_full_s42_20261017T065259974842Z` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_gap_audit_20261019T011605512759Z` — 文件记录：checks_passed_with_unresolved_protocol_differences；exit_code=0（非进程检查）
- `r0_graph_rebuild_author_offline_v2` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_graph_rebuild_author_v1` — 文件记录：failed；exit_code=1（非进程检查）
- `r0_targeted_audit_20261019T015739852831Z` — 文件记录：targeted_checks_complete_history_unresolved；exit_code=0（非进程检查）
- `r0_validation_full_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_validation_smoke_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `r0_validation_split_s42_v1` — 无可读根级 status.json；待核实
- `r0a_v2_cleanup_receipt.json` — 未核实（仅旁文件或未知类型）

## 06 探测、冒烟与前置核对

- `e0_batch_probe_b512_s42_shared_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e2_batch_probe_b1024_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e2_batch_probe_b256_s42_shared_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e2_batch_probe_b512_s42_shared_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `e2_e3_preflight_20261017T081529305925Z` — 无可读根级 status.json；待核实
- `e2_e3_preflight_20261019T103529458611Z` — 无可读根级 status.json；待核实
- `e3_batch_probe_b512_s42_shared_v2` — 文件记录：failed；exit_code=1（非进程检查）
- `e3_batch_probe_b512_s42_shared_v3` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_gpu_smoke_E4_s42_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_gpu_smoke_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `grouped_training_pilot_v1` — 文件记录：completed；exit_code=0（非进程检查）
- `server_graph_split_audit_v1` — 无可读根级 status.json；待核实

## 07 重启恢复快照

- `e3_restart_snapshot_20261020T102912701250Z` — 无可读根级 status.json；待核实
- `e3_restart_snapshot_20261020T103302548182Z` — 无可读根级 status.json；待核实

## 08 历史输出与归档

- `GNN4ID` — 无可读根级 status.json；待核实
- `GNN4ID_edge` — 无可读根级 status.json；待核实
- `_archive` — 无可读根级 status.json；待核实
- `baseline_rerun_paper_align_v2` — 无可读根级 status.json；待核实
- `baselines` — 无可读根级 status.json；待核实
- `good` — 无可读根级 status.json；待核实
- `innov2` — 无可读根级 status.json；待核实

## 99 其他待核实文件

