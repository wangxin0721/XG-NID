# R1–R3：最多20轮的训练策略诊断

本轮不新增结构，不重新跑200轮，不改历史结果或论文。三组从锁定 seed42 Ordered（验证 epoch191）重新开始，继承相同权重，不续用上一轮已退化的末轮参数。固定原划分、最终八类 NLL、采样 seed42、new-module seed42、batch512、验证 batch16。每组20轮、每轮211次更新，共12,660次，不早停、不自动延长、不访问测试图。

## 先整理的已有证据

来源结果提交 `1469f7190e7077db7afb639d2efa1347d9e261f1`。指定的0、1、2、5、10、20、50、100、200轮已在 [report.md](archive_trajectory/report.md) 与 `archive_trajectory/trajectory.json` 提取，JSON包含原始 history.json 字节 SHA256。提取脚本只读取该历史文件，不重训，不加载测试预测。

S1训练NLL从第1轮0.013502降至第200轮0.010085，总体下降但非单调；验证F1由97.914033%起点降到第1轮97.541118%，第200轮仍为97.627762%。S2、S3有相近趋势。这与过拟合或训练目标/指标不一致的解释相容，但不是原因证明。历史没有epoch0训练NLL、验证NLL或固定训练子集NLL，不能据此判断训练和验证是否从起点同时变差，也不能认定学习率太大。新实验用于补足这些缺口。

## 三组的严格边界

| 组别 | 原参数 | 新增模块 | 学习率 |
|---|---|---|---|
| R1 | 与S1完全相同的解冻范围 | 无 | S1各组LR乘0.1 |
| R2 | 全冻结，包括BN参数和统计 | 只训练S2原有节点MLP，3,008参数 | 1e-4 |
| R3 | 全冻结，包括BN参数和统计 | 只训练S3原有Flow调制，3,008参数 | 1e-4 |

R1：Packet第二层 `flow→packet`、`packet→packet` 卷积 LR=1e-7；cross-layer及query读出=1e-5；融合=1e-5；Ordered读出=1e-4。Adam均从空状态开始，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；每组优化器只创建一次并完整恢复，不在每轮重建。

R2、R3仅设置原参数 `requires_grad=False`，保留原节点调制后的第二层GNN、读出和融合运算的autograd路径。不能用 `no_grad()` 包住这条后续路径。Flow上下文编码本来就被冻结，可继续无梯度计算；Packet第二层运算不能整体detach。两步真实图探测检查新增模块末层和随后上游层获得非零梯度、第二层输入保留梯度、原参数无梯度且权重和BN不变；探测更新全部丢弃并恢复RNG。

CPU拓扑解析始终在PyG原地CUDA转移之前进行，沿用已修复的设备边界。

## 每轮观测和恢复

- 固定训练诊断子集：每类64条，共512条，按 `sha256(420064:sample_id)` 排序选取；同哈希时按ID排序。保存完整身份清单及其绑定SHA。它是类别均衡的诊断样本，不是原分布训练风险的无偏估计；正式训练仍使用全部107,621条，不改变采样。
- 从epoch0起，每轮在eval模式对相同512条训练样本和全部13,446条验证样本计算NLL。在线训练epoch均值NLL另记，区分更新中损失与固定权重损失。
- 每轮导出身份、标签、源Ordered和三组预测、真实类别的log probability；据此可独立重算NLL、八类指标、混淆矩阵、源正确样本退步数及源错误样本修复数。WebBased、Recon、Spoofing、BruteForce的指标包含在完整八类结果中。
- 每轮记录Packet第二层、其余Packet编码器、Flow编码器、cross-layer、query、Ordered、融合及新增模块的绝对L2变化、相对L2变化和变化元素数，均相对于每组epoch0。基准范数为0时相对值为null，不伪造比值。
- 共用原图batch和样本顺序，三组独立Adam。每个完整三组batch原子保存模型、Adam、RNG、采样游标、loss账本、最佳与历史记录；半批失败回到上一完整事务。epoch观测在恢复后可重建，不额外计入优化更新。
- 每组按包含epoch0的验证Macro-F1选最佳，同分保留最早轮；完整保留末轮数据。最终对最佳权重作原图回放，并复核全部输入哈希、原参数/BN冻结状态和0次测试图加载。

## 事前解释与停止规则

结构比较以R3−R2为主，不能把R3−R1解释为纯结构作用。**R3对源Ordered及R2均至少提升0.05个百分点**才通过延长投入门槛；它不是统计显著性标准，也不是自动延长命令。

| 观察 | 后续方向 |
|---|---|
| R1明显减轻退化 | 后续以稳定设置重建匹配对照 |
| R3超过源Ordered及R2，并达到两项0.05pp投入门槛 | 才考虑继续Flow调制 |
| R2、R3都提升，但彼此接近 | 支持节点适配，尚不支持Flow条件的额外收益 |
| 三组均无改善 | 结束这一版调制探索，后续考虑Packet时序专门化 |

“明显减轻”和“彼此接近”未由用户提供额外数值定义，因此报告R1与旧S1同轮次1/2/5/10/20轮差值、完整轨迹和R3−R2原始差值，不偷偷新增自动门槛。结合轨迹解释这些分支；所有分支本轮均只报告，不自动调参、开新结构、访问测试或扩展训练。20轮未达标不能解释成方法永久无效。

## 上传与独立读回

独立输出目录 `outputs/node_training_diagnostic_v1`。启动器在正常结束、可捕获异常、SIGINT/SIGTERM后自动上传配置、历史提取、全部逐轮观测/预测、检查点、恢复状态和失败现场至 `codex/server-outputs`，不覆盖上一轮目录。SIGKILL或断电只能由上一事务续训。

上传成功后从精确Git提交读取Git/LFS实际字节，逐轮独立复算NLL、指标、修复/退步、样本身份、最早最佳选择、采样顺序和预算、投入门槛与R1历史差值；再上传已关闭的审计收据。参数张量变化由服务器记录与冻结摘要复核，轻量读回审计明确不声称重新加载所有逐轮权重独立计算其范数。训练退出码与上传/审计失败分别保存。

服务器沿用现有wx环境，CODE_SHA替换为交付的完整提交：

```bash
conda activate wx
set -o pipefail
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
cd "$xgnid_repo" || exit 1
git fetch --no-tags origin codex/node-training-diagnostic-v1 || exit 1
xgnid_revision='CODE_SHA'
git show "$xgnid_revision":scripts/launch_node_training_diagnostic.sh \
  | bash -s -- "$xgnid_repo" "$xgnid_revision"
tail -F "$xgnid_repo/outputs/node_training_diagnostic_v1/publication/training.log"
```

同一命令为真正续训。首次先验证真实环境可在bash命令末尾加 `--probe-only`；下一次去掉即可从同一来源正式训练。仅来源锁与历史整理加 `--lock-only`。新工作树使用提交前12位后缀，不改原仓库和旧实验工作树。

最小结果：`report.md`、`summary.json`、`history.json`、`final_verification.json`、`publication/*_github_audit.json`；全部逐样本诊断位于 `observations/`，源轨迹位于 `historical/`。
