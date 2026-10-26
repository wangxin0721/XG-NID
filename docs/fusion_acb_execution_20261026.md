# A／C／B 实现与服务器执行（2026-10-26）

状态：独立入口已实现，本机仅做标准库/AST验证。真实GPU联合探测和三项训练尚未执行。E3a/Z训练、诊断及缓存分析已完成，不重跑。原研究配置与训练工具字节保持不变，新增fusion_acb_execution_20261026.json引用冻结设计快照；旧设计中的design-only状态是历史登记状态。

## 冻结范围

共享原双异构图编码器、64→8投影、非线性共享头H=Linear(8,32)→LeakyReLU→Linear(32,8)。两路均读取完整异构图，不称为隔离模态。最终预测为log_softmax(H(wf*zf+wp*zp))，去掉最终代理混合。

统一损失：NLL(融合log_softmax,y)+0.05/2×[CE(H(zf),y)+CE(H(zp),y)]，第一项等价于融合logits的CE。监督H(z)，不是原始z；不调用旧model.loss()或其他辅助头损失。B保留z的熵、间隔等表示统计输入，不称为校准置信度。

| 方案 | 权重 | 初态 |
|---|---|---|
| A_equal | 恒定0.5/0.5 | 等权 |
| C_global | 一个全局标量s，wf=sigmoid(s) | s=0 |
| B_dynamic | 原21→16→2门控，逐样本softmax | 末层权重/偏置为0 |

统一seed42、batch512、验证batch16、初始化batch16、Adam lr.001/weight_decay0、workers2、最多100轮、patience12、严格验证Macro-F1选模。无GPU小时上限，不做测试推断。

公共参数由同一冻结流程初始化、克隆回载并核对hash；保存恢复Python/CPU/CUDA RNG。联合探测验证公共初态与训练起始RNG摘要相同，训练逐轮保存/比较样本顺序摘要。保留历史未参与损失的辅助头，A保留未使用的原gate；报告总参数、requires_grad参数、实际参与损失参数及融合器参数，不假称容量相同。

## 检查与输出

入口为scripts/run_fusion_acb_server.py，公共控制在scripts/fusion_acb.py，启动器为scripts/launch_fusion_acb.sh。每项先用3个真实训练batch探测：初始权重逐样本恰为.5/.5；两路辅助损失分别到达z、投影、H及编码器；C/B有非零梯度和参数更新，B后续batch上游层收到梯度。单邻居GAT的某个注意力张量梯度可能合法为0，因此按编码器整体检查。

联合探测通过后按A→C→B顺序筛选，正式训练全部重新初始化、重置Adam/采样器，不从探测权重继续。一项失败停止后续排程。已有输出拒绝覆盖，不自动重试或换目录重跑。

每轮保存分解损失、样本顺序、验证Macro、权重统计、best/last检查点及优化器/RNG状态。入口当前没有resume命令，保存状态不代表支持自动续训。每项结束恢复最佳检查点，验证矩阵必须复现；输出H(z)两路及融合的P/R/F1、混淆矩阵、逐类四分组挽救/破坏计数和私有逐样本预测压缩文件，不再扫权重网格。

队列汇总给出B−A/B−C百分点差值，核对A/C/B公共轮次顺序，并与旧E3a-Z日志核对B重叠轮次。B复用Z完整前向/零初始化，优化器和选模规则不变，预定优化目标差异只有lambda=0→.05。额外H(z)调用仅含Linear/LeakyReLU，不使用随机数或改变运行缓冲区。初态和顺序检查通过后，可复用Z作为B的无分支辅助监督对照；不同早停轮数意味着总曝光可能不同。A/C没有现成匹配的lambda=0对照，不将三者全部收益归给监督。

forward_seconds沿用batch16/CUDA events，不计额外诊断头；GPU允许共享并记录其他进程，耗时不是独占速度比较。逐样本预测只保留outputs，不公开提交。seed42用于筛选，有希望后再冻结确认配置、补配对seed43/44；当前入口只允许42。

## 服务器启动

使用git archive只提取所需工具到独立/tmp目录，不切换或覆盖服务器训练源码。发布时优先使用交付的固定提交号替代下面的FETCH_HEAD。

```bash
(
  set -e
  cd /home/nzxn/下载/WX/load/XG-NID
  git fetch origin codex/r0-edge-preparation
  rev=$(git rev-parse FETCH_HEAD)
  bundle=$(mktemp -d /tmp/xgnid_acb.XXXXXX)
  git archive "$rev" \
    scripts/fusion_acb.py scripts/run_fusion_acb_server.py scripts/launch_fusion_acb.sh \
    scripts/e3_gate_study.py scripts/run_e3_gate_ablation_server.py \
    scripts/diagnose_gate_candidate.py scripts/diagnose_e3_gate_server.py \
    configs/fusion_acb_execution_20261026.json configs/fusion_acb_study_20261023.json \
    configs/e3_gate_study.json > "$bundle/bundle.tar"
  tar -xf "$bundle/bundle.tar" -C "$bundle"
  bash "$bundle/scripts/launch_fusion_acb.sh" 0
)
```

该命令会提交探测和三项正式训练队列，物理GPU0允许共享。当前对话环境不能连接服务器，用户运行后以实际日志为准。

```bash
tail -n 30 -F outputs/fusion_acb_b512_s42_v1.log
```

Ctrl+C只退出tail，不停止后台队列。查看总状态与完成汇总：

```bash
cat outputs/fusion_acb_b512_s42_v1/status.json
cat outputs/fusion_acb_b512_s42_v1/summary.json
```

联合探测在probe/summary.json；正式训练分别在A_equal/、C_global/、B_dynamic/，包含status.json、metrics.json、diagnostics.json、epochs.jsonl、best.pt和last.pt。probe完成不代表三项正式训练完成。
