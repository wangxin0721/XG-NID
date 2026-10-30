# J0/J2严格复用恢复队列V2

基于当前工作区ce6bddb，对照f62af93定向修正。恢复代码resume_fair_vector.py在独立outputs/fair_vector_five_runs_recovery_v1下继续原五项实验；新V2仍调用原run_fair_vector.py，配置/缓存/权重绑定保持原协议。旧joint配置指向中断前目录，有读取错误产物或重复预热风险。

新配置joint_gate_recovery_v2_20261030.json明确指定outputs/fair_vector_five_runs_recovery_v1/V2，并固定completed_only、require_trained_gate、require_recovery_plan。旧配置文件不改；新启动器输出outputs/joint_gate_recovery_v2_s42_v1，不覆盖已有joint_gate_s42_v1及历史恢复结果。

入口先核对恢复计划binding、恢复队列/V2/模型三级完成状态、根summary与V2 result及metrics一致性、实际config中的V2身份/488参数/冻结融合协议、N0/缓存绑定、best/last.pt及逐样本预测哈希、逐类矩阵与指标。要求选中的V2轮次大于0，以匹配本次用户说明的训练后检查点。保留旧vector_screening字段，不因未过冻结门槛拒绝联合微调，也不修改旧结论。

预检模式不导入torch，不分配GPU，不创建训练输出；打印真实路径、最佳轮次、验证Macro-F1和SHA。目录缺失、运行未结束或校验失败即报错，没有旧目录回退、自动发现最高分模型、等待后重训或预热调用。实际运行重新核验，再反序列化best.pt，核对checkpoint内epoch/config/validation与已验证报告一致，严格恢复21→16→8门控。

真实图上的N0复现、预热前等权等价、加载预热权重后的V2复现，以及两项真实图梯度/优化更新探测均保留。全部通过后依次运行J0→J2，各一次seed42，公共模块学习率及训练预算不变。J2联合第0轮复现实际预热V2，不要求等于N0。只训练最终NLL，不扩展seed或重跑V1/V2。

stage_A_receipt.json和最终binding记录本次恢复目录、result/summary/config/权重哈希、旧筛选结果及reused=true/prewarm_training_started=false。历史预热耗时依旧缺失，不伪造估算。结果与说明不将联合流程收益全部归因于门控。

本地未保存服务器恢复V2的真实权重或新汇总，不能声称已确认其数值和真实图结果；这些在服务器预检及张量探测中强制验证。本地完成恢复来源文件/哈希夹具检查、错误产物拒绝、原联合参数分组检查和恢复入口回归检查；无torch，真实前后向仍未本地执行。

## 执行

助手推送后用固定提交号替换rev，GPU1允许共享，保留所有已有任务：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_joint_recovery.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_joint_gate_recovery.sh" 1
```

启动器先前台预检，再后台训练；若出现真实目录、绑定或权重错误，停止并检查，不要删除原产物重跑。新结果目录或日志/PID已存在时也拒绝重复启动。

```bash
tail -n 30 -F outputs/joint_gate_recovery_v2_s42_v1.log
cat outputs/joint_gate_recovery_v2_s42_v1/status.json
```

完成后summary.json、comparison.json、report.md用于比较N0、冻结V2、J0、J2。配置取消的是冻结门控追加跨seed重复，不取消本次两项联合训练。
