# Flow 上下文引导的 Packet 节点调制：S0–S3

本轮只实现和筛选第一项方向，不同时引入时序卷积或梯度均衡。实验成功与否尚待真实服务器训练验证。

## 前一版结论的适用范围

已在结果快照 `74a3299fd7d14932d97797f3379519d6057de30a` 的 `outputs/conditional_complement_v1/report.md` 核对：T2 Macro-F1 为 97.935901%，普通 MLP T1 为 97.935619%，差约 **0.000282 个百分点**；T0 继续训练对照为 97.940761%，T2 比其低约 0.004860 个百分点。T4 最佳轮为 epoch0。

应停止投入这一版“冻结表示＋条件残差交互＋协同损失”。这些结果未检验特征提取阶段的跨分支交互，不能外推为“双分支互补路线行不通”。本轮不覆盖或修改旧结果与论文文件。

## 结构与对照

| 组别 | 初始化与更新范围 | 节点模块 |
| --- | --- | --- |
| S0 | 固定验证选出的 Ordered，0 次更新 | 恒等 |
| S1 | 同一 Ordered 起点，普通微调 | 恒等 |
| S2 | 与 S1 相同 | Packet 节点残差 MLP，3,008 参数 |
| S3 | 与 S1 相同 | Flow 条件仿射调制，3,008 参数 |

S1–S3 仅解冻 Packet 第二层的 `flow→packet`、`packet→packet` 两类卷积，以及现有 cross-layer 投影、Flow-query 读出、Ordered 读出和融合模块。Flow 编码器、Packet 第一层、第二层 `packet→flow` 卷积及所有 BN 参数与统计均固定；编码器保持 eval。不能将“解冻第二层”本身当作新贡献。

S3 使用冻结 Flow 编码器**第一层 BN/激活后**的流节点均值作为 64 维上下文 f。用 `64→15→128` 的轻量 MLP 生成 gamma、beta，在 Packet 第一层 BN/激活之后、第二层异构图卷积之前应用 `(1+gamma(f))*h+beta(f)`。上下文按每个节点所属图索引广播，不跨流混合；调制结果作为第二层消息的输入。原 cross-layer 第一层旁路使用未调制节点，只使交互进入第二层计算。

S2 使用 `h + MLP(h)`，MLP 为 `64→23→64`，输入不含 Flow 上下文。两个 MLP 的第一层均无偏置、末层有偏置，分别为 `64*23+23*64+64=3008` 与 `64*15+15*128+128=3008`；没有通过未使用参数凑数。该配平不是相同 FLOPs 的承诺：S2 按节点计算，S3 按流计算再广播。两者末层均零初始化，初始为恒等映射，上游层从后续更新开始获得梯度。

本方法借鉴 [FiLM 的条件逐维仿射变换](https://arxiv.org/abs/1709.07871)，FiLM 本身不是新方法。待验证点是流/报文粒度下的交互位置与信息传递方式，不能提前写为已证实收益。

## 锁定的首轮协议

- seed42 单主干开发验证筛选；源为固定清单中的 `joint_ordered`，源结果提交 `b6ca40cfb1abef54c669bdcf932ce95bdd272a66`，验证最佳 epoch191。通过旧只读来源锁恢复 C 编码器和 Ordered 读出/融合状态，不用前一版 T0–T4 的训练后权重。
- 沿用已有 200 轮完整预算：107,621 条训练、13,446 条验证；batch512，211 更新/轮，S1–S3 各42,200次、合计126,600次；不早停。S0 不训练。
- 三组逐 batch 共用原始图、样本顺序和独立 Adam；仅最终八类 NLL，无蒸馏、保持或其他辅助损失。eval batch16，workers2；原始图训练按事务直接加载，workers 只用于验证加载。
- Packet 第二层 LR `1e-6`；cross-layer / query 读出 `1e-4`；融合 `1e-4`；Ordered 读出 `1e-3`；新增模块 `1e-3`。Adam betas=(0.9,0.999)，eps=1e-8，weight_decay=0；固定新模块与采样 seed42。
- 验证从 epoch0 开始记录，按未舍入 Macro-F1 选择最佳，同分保留最早轮。正式训练前完整回放四组 epoch0 验证预测，对齐样本、标签与锁定 Ordered。
- **只有 S3 最佳验证 Macro-F1 严格同时高于 S1、S2，才记为条件调制的 seed42 初步证据。** 相同或任一未超过均不通过。该规则没有宣称统计显著性；不另外套用上一轮 0.1/0.05 pp 门槛。
- 保留 S0 比较、全部八类指标、混淆矩阵、修复/退步/错误间变化、每组最佳轮、末轮指标与训练耗时。筛选通过也不自动扩展独立主干或测试；后续需要另行锁定协议。本轮始终禁止测试图加载。

## 续训、发布和核验

每个完整配对 batch 原子保存 S1–S3 模型、各自 Adam、Python/NumPy/Torch/CUDA RNG、采样顺序与游标、NLL 累积和最佳验证记录。半批失败只恢复上一完整事务；不把失败后的部分内存状态发布为可续训状态。模型/配置/代码/来源绑定不一致时拒绝恢复。

运行器和启动器分别加锁。每轮及最佳模型回放检查冻结权重/BN 不变；来源权重和输入文件哈希在训练前后复核。完整日志位于 `publication/training.log`，它与上传收据不参与计算产物封印，避免启动器关闭日志时破坏封印。`recovery.pt` 作为恢复权威，组内 `last.pt` 为最后完整轮视图。

启动器默认在正常结束、可捕获异常及 SIGINT/SIGTERM 后选择性发布 `outputs/flow_packet_node_modulation_v1` 到 `codex/server-outputs`，包含失败现场和恢复状态；大文件沿用 LFS。SIGKILL/断电无法执行 finally，重启后用上一事务恢复。

成功发布后从精确 GitHub 提交读取 Git/LFS 实际载荷，独立复算八类混淆矩阵、指标、配对计数、选择轮、样本身份和严格比较结论，并核对训练样本顺序及预算；关闭的核验收据再次发布。失败退出码与上传失败分别保留。未完成训练时只核对来源/配置快照，不声称验证训练结果。

## 服务器命令

使用现有 `wx` 环境，不升级依赖。将 `CODE_SHA` 换成交付时的完整 SHA。工作目录不同则只修改 `xgnid_repo`。

```bash
conda activate wx
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
cd "$xgnid_repo" || exit 1
git fetch --no-tags origin codex/flow-packet-node-modulation-v1 || exit 1
xgnid_revision='CODE_SHA'
git show "$xgnid_revision":scripts/launch_flow_packet_node_modulation.sh \
  | bash -s -- "$xgnid_repo" "$xgnid_revision"
```

脚本在相邻目录创建该 SHA 的独立 detached worktree；已有工作树 SHA 不同或代码有修改时停止保留现场，不切换原仓库分支、不覆盖历史结果。已有进程则拒绝重复启动。

同一命令即真正续训；仅锁定来源可在 `bash -s -- ...` 最后加 `--lock-only`；仅原图 epoch0/梯度探测加 `--probe-only`。两者均不启动200轮。数据根目录需覆盖时加 `--data-root '/实际图目录'`。

```bash
tail -F "$xgnid_repo/outputs/flow_packet_node_modulation_v1/publication/training.log"
cat "$xgnid_repo/outputs/flow_packet_node_modulation_v1/status.json"
tail -F "$xgnid_repo/tmp/flow_packet_node_modulation_v1.launcher.log"
```

训练进程停止后的单独重传及读回核验：

```bash
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-flow-packet-node-modulation-v1"
PYTHONPATH="$xgnid_code/src:$xgnid_code/scripts" python -B \
  "$xgnid_code/scripts/sync_server_outputs.py" publish \
  --repo "$xgnid_repo" --only flow_packet_node_modulation_v1
PYTHONPATH="$xgnid_code/src:$xgnid_code/scripts" python -B \
  "$xgnid_code/scripts/audit_flow_packet_node_archive.py" \
  --repo "$xgnid_repo" --commit '<结果完整SHA>'
```

结果最小报告为 `report.md`、`summary.json`、`final_verification.json`、`publication/*_github_audit.json`；逐样本预测导出用于自行诊断，报告任务不必手动下载模型和恢复状态。
